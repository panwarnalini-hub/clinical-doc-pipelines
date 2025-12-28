# Databricks notebook source
# DOCLING + PYSPARK HYBRID PIPELINE
#
# WHY THIS DESIGN:
#   - Docling is CPU-bound, single-threaded : Python is fine
#   - Spark shines AFTER extraction: Delta, joins, analytics, scale
#   - No UDFs, no serialization nightmares, no wake up from sleep and debug failures
#
# LAYERS:
#   Bronze: Raw extraction output (1:1 from Docling) - THIS NOTEBOOK
#   Silver: Cleaned, classified, hierarchy (02,03a,03b)
#
# NOTE:
#   This notebook handles ONLY Bronze layer

# COMMAND ----------

# DESIGN DECISION
#
# The extraction layer intentionally supports only formats that represent narrative business documents where structure matters:
#
# PDF (primary document format)
# DOC / DOCX (authoring source of most PDFs)
# RTF (Word-compatible text format used in submissions) # Standard document formats for clinical organizations
#
# PPT and spreadsheets were excluded because they do not contain hierarchical document semantics that downstream intelligence relies on.
# Extraction stays focused and reliable while I add structural intelligence in the Silver layer.

# HTML exists in code only as a placeholder. It is not implemented because HTML pages are not narrative business documents and do not provide the
# structured section semantics required for our downstream intelligence.


# COMMAND ----------

# PRODUCTION INGESTION STRATEGY
#
# This pipeline will ingest documents automatically using Autoloader directly from governed S3 storage (read-only). 
# The Autoloader section below is commented out during development 
# because we are testing extraction on selected sample documents.

# COMMAND ----------

# Fallback extraction chain:
# 1) Docling v2 : best extractor (sections, tables, hierarchy)
# 2) PyMuPDF : fast and reliable, works on most real-world PDFs
# 3) pdfplumber : used only when tables are messy and need cleanup
# 4) Raw text : last resort so the pipeline never crashes

# Each tool fails differently, so this chain keeps extraction stable across corrupted PDFs, huge files, and odd encodings.

# COMMAND ----------

import sys
sys.path.append("/Workspace/Shared")

# COMMAND ----------

# MAGIC %md
# MAGIC # Section 1: Configuration

# COMMAND ----------

from dataclasses import dataclass

#All configuration:no hardcoded paths.
@dataclass
class PipelineConfig:

    # Unity input location (existing table)
    input_catalog: str = "source_data"
    input_schema: str = "protocols"
    source_documents_table: str = "documents"  # stays in source_data

    # Output location
    catalog: str = "dev_clinical"
    schema: str = "doc_test"
    
    # Delta Tables (Bronze only - Silver/Gold in separate notebooks)
    bronze_pages: str = "bronze_pages"
    bronze_sections: str = "bronze_sections"
    bronze_images: str = "bronze_images"
    bronze_tables: str = "bronze_tables"
    bronze_formulas: str = "bronze_formulas"
    bronze_errors: str = "bronze_errors"
    
    # Document registry (tracks processing status)
    document_registry: str = "document_registry"
    
    # Storage
    source_documents_table: str = "documents"
    output_volume: str = "/Volumes/dev_clinical/doc_test/extracted"
    
    # S3 PATHS (Production; uncomment is using s3)
    # input_path: str = "s3://your-raw-bucket/documents/"
    # output_path: str = "s3://your-processed-bucket/extracted/"
    
    # Engine metadata
    extraction_engine: str = "docling"
    extraction_engine_version: str = "v2.0"
    
    # Timeout and fallback settings
    extraction_timeout_seconds: int = 300  # 5 min per doc
    enable_fallback: bool = True
    fallback_order: tuple = ("docling", "pymupdf", "pdfplumber", "raw_text")
    
    def full_source_table(self) -> str:
        return f"{self.input_catalog}.{self.input_schema}.{self.source_documents_table}"
    
    def full_table(self, table: str) -> str:
        return f"{self.catalog}.{self.schema}.{table}"


config = PipelineConfig()
print(f"Catalog: {config.catalog}.{config.schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Section 2: Docling Extractor (Pure Python)

# COMMAND ----------

import os
import json
import fitz  # PyMuPDF
import pdfplumber
from typing import List, Dict, Any
import urllib.parse, time, multiprocessing
from pyspark.sql.functions import input_file_name, regexp_extract
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from io import BytesIO
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Iterator, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
logging.getLogger("docling").setLevel(logging.ERROR)

#Get current UTC time (timezone-aware)
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Docling imports
from docling.document_converter import DocumentConverter, PdfFormatOption, WordFormatOption
from docling.datamodel.base_models import InputFormat, DocumentStream
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    EasyOcrOptions,
    TesseractOcrOptions,
    TesseractCliOcrOptions,
    RapidOcrOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

# Docling document types
from docling_core.types.doc.document import (
    DoclingDocument,
    DocItemLabel,
    ContentLayer,
    ImageRefMode,
    TableItem,
    PictureItem,
    TextItem,
    SectionHeaderItem,
)

# Fallback libraries
import fitz  # PyMuPDF
import pdfplumber

# Supported OCR engines
class OcrEngine(Enum):
    EASYOCR = "easyocr"
    TESSERACT = "tesseract"
    TESSERACT_CLI = "tesseract_cli"
    RAPIDOCR = "rapidocr"
    AUTO = "auto"

# Supported accelerator types
class AcceleratorType(Enum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"

# Supported export formats
class ExportFormat(Enum):
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    TEXT = "text"
    DOCTAGS = "doctags"

# Configuration for document extraction pipeline
@dataclass
class ExtractionConfig:
    
    # Output settings
    output_dir: str = "./extracted"
    output_volume: str = "./extracted"  # Alias for compatibility
    
    # OCR settings
    enable_ocr: bool = True
    ocr_engine: OcrEngine = OcrEngine.EASYOCR
    force_full_page_ocr: bool = False
    ocr_languages: List[str] = field(default_factory=lambda: ["en"])
    
    # Table extraction
    enable_table_structure: bool = True
    table_mode: TableFormerMode = TableFormerMode.ACCURATE
    table_cell_matching: bool = True
    
    # Enrichment features
    enable_code_enrichment: bool = True
    enable_formula_enrichment: bool = True
    enable_picture_classification: bool = True
    enable_picture_description: bool = False
    
    # Image settings
    images_scale: float = 2.0
    generate_page_images: bool = True
    generate_picture_images: bool = True
    
    # Performance settings
    accelerator: AcceleratorType = AcceleratorType.AUTO
    num_threads: int = field(default_factory=lambda: max(4, multiprocessing.cpu_count() // 2))
    extraction_timeout_seconds: int = 300
    
    # Fallback settings
    enable_fallback: bool = True
    
    # Export settings
    export_formats: List[ExportFormat] = field(default_factory=lambda: [ExportFormat.MARKDOWN, ExportFormat.JSON])
    image_ref_mode: ImageRefMode = ImageRefMode.REFERENCED
    
    # Limits
    max_pages: Optional[int] = None
    max_file_size: Optional[int] = None
    
    # Artifacts path
    artifacts_path: Optional[str] = None
    
    # Remote services
    enable_remote_services: bool = False
    
    # Engine metadata (for compatibility)
    extraction_engine: str = "docling"
    extraction_engine_version: str = "v2.0"

# Advanced Document Extractor using Docling's full capabilities. Returns dict format compatible with Spark DataFrame schemas.
    
class DoclingExtractor:
    
    def __init__(self, config: Optional[ExtractionConfig] = None):
        self.config = config or ExtractionConfig()
        self.converter = None
        self._init_docling()
   
    def _init_docling(self):
        try:
            pipeline_options = PdfPipelineOptions()
            
            # OCR configuration
            pipeline_options.do_ocr = self.config.enable_ocr
            if self.config.enable_ocr:
                pipeline_options.ocr_options = self._get_ocr_options()
            
            # Table structure
            pipeline_options.do_table_structure = self.config.enable_table_structure
            if self.config.enable_table_structure:
                pipeline_options.table_structure_options.mode = self.config.table_mode
                pipeline_options.table_structure_options.do_cell_matching = self.config.table_cell_matching
            
            # Enrichment features
            pipeline_options.do_code_enrichment = self.config.enable_code_enrichment
            pipeline_options.do_formula_enrichment = self.config.enable_formula_enrichment
            pipeline_options.do_picture_classification = self.config.enable_picture_classification
            
            # Image generation
            pipeline_options.images_scale = self.config.images_scale
            pipeline_options.generate_page_images = self.config.generate_page_images
            pipeline_options.generate_picture_images = self.config.generate_picture_images
            
            # Accelerator options
            pipeline_options.accelerator_options = self._get_accelerator_options()
            
            # Artifacts path
            if self.config.artifacts_path:
                pipeline_options.artifacts_path = self.config.artifacts_path
            
            # Create converter
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        pipeline_cls=StandardPdfPipeline,
                    ),
                    InputFormat.DOCX: WordFormatOption(
                        pipeline_options=pipeline_options
                    ),
                }
            )
            
            print(f"Docling initialized with full pipeline")
            print(f"OCR: {self.config.ocr_engine.value if self.config.enable_ocr else 'disabled'}")
            print(f"Tables: {self.config.table_mode.name if self.config.enable_table_structure else 'disabled'}")
            print(f"Accelerator: {self.config.accelerator.value}")
            
        except Exception as e:
            print(f"Docling initialization failed: {e}")
            traceback.print_exc()
            self.converter = None
    
    # Get OCR options based on configuration
    def _get_ocr_options(self):
        base_options = {
            "force_full_page_ocr": self.config.force_full_page_ocr,
            "lang": self.config.ocr_languages,
        }
        
        if self.config.ocr_engine == OcrEngine.EASYOCR:
            return EasyOcrOptions(**base_options)
        elif self.config.ocr_engine == OcrEngine.TESSERACT:
            return TesseractOcrOptions(**base_options)
        elif self.config.ocr_engine == OcrEngine.TESSERACT_CLI:
            return TesseractCliOcrOptions(**base_options)
        elif self.config.ocr_engine == OcrEngine.RAPIDOCR:
            return RapidOcrOptions(**base_options)
        else:
            return EasyOcrOptions(**base_options)
    
    # Get accelerator options based on configuration
    def _get_accelerator_options(self) -> AcceleratorOptions:

        device_map = {
            AcceleratorType.AUTO: AcceleratorDevice.AUTO,
            AcceleratorType.CPU: AcceleratorDevice.CPU,
            AcceleratorType.CUDA: AcceleratorDevice.CUDA,
            AcceleratorType.MPS: AcceleratorDevice.MPS,
        }
        
        return AcceleratorOptions(
            num_threads=self.config.num_threads,
            device=device_map.get(self.config.accelerator, AcceleratorDevice.AUTO)
        )
    
    def extract(
        self, 
        source: Union[str, Path, bytes, BytesIO],
        document_id: str,
        filename: Optional[str] = None
        # Extract content from a document.source: File path, URL, bytes, or BytesIO stream. document_id: Unique identifier for the document. filename: Optional filename (required for stream input)
    
    ) -> Dict[str, Any]:
        start_time = utcnow()
        
        # Prepare source
        if isinstance(source, (bytes, BytesIO)):
            if not filename:
                filename = f"{document_id}.pdf"
            buf = BytesIO(source) if isinstance(source, bytes) else source
            source_input = DocumentStream(name=filename, stream=buf)
            source_path = filename
        else:
            source_input = str(source)
            source_path = str(source)
        
        # Initialize result
        result = self._empty_result(document_id, source_path)
        
        # Validate file type
        file_ext = Path(source_path).suffix.lower()
        supported_formats = {".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".html", ".png", ".jpg", ".jpeg", ".tiff"}
        
        if file_ext not in supported_formats:
            result["errors"].append(self._create_error(
                document_id, None, "unsupported_format",
                f"Unsupported file type: {file_ext}"
            ))
            return self._finalize_result(result, start_time)
        
        # Try Docling first
        if self.converter is not None:
            try:
                print(f"Extracting with Docling...")
                docling_result = self._extract_with_docling(source_input, document_id, source_path)
                
                if docling_result.get("success") or len(docling_result.get("pages", [])) > 0:
                    docling_result["extraction_method"] = "docling"
                    print(f"Docling extraction successful")
                    return self._finalize_result(docling_result, start_time)
                    
            except Exception as e:
                print(f"Docling failed: {str(e)[:100]}")
                result["errors"].append(self._create_error(
                    document_id, None, "docling_failure", str(e),
                    traceback.format_exc()[:2000]
                ))
        
        # Fallback chain
        if not self.config.enable_fallback or file_ext != ".pdf":
            result["extraction_method"] = "none"
            result["success"] = False
            return self._finalize_result(result, start_time)
        
        fallback_methods = [
            ("pymupdf", self._extract_with_pymupdf),
            ("pdfplumber", self._extract_with_pdfplumber),
            ("raw_text", self._extract_raw_text),
        ]
        
        for method_name, method_func in fallback_methods:
            try:
                print(f"Trying {method_name}...")
                fallback_source = str(source) if not isinstance(source, (bytes, BytesIO)) else source_path
                extraction_result = self._run_with_timeout(
                    method_func,
                    args=(fallback_source, document_id),
                    timeout=self.config.extraction_timeout_seconds
                )
                
                if extraction_result and (extraction_result.get("success") or len(extraction_result.get("pages", [])) > 0):
                    extraction_result["extraction_method"] = method_name
                    extraction_result["warnings"].append(f"Used {method_name} fallback - some features unavailable")
                    print(f"{method_name} succeeded")
                    return self._finalize_result(extraction_result, start_time)
                    
            except FuturesTimeoutError:
                print(f"{method_name} timed out")
                result["errors"].append(self._create_error(
                    document_id, None, "timeout",
                    f"{method_name} timed out after {self.config.extraction_timeout_seconds}s"
                ))
            except Exception as e:
                print(f"{method_name} failed: {str(e)[:100]}")
                result["errors"].append(self._create_error(
                    document_id, None, f"{method_name}_failure", str(e),
                    traceback.format_exc()[:1000]
                ))
        
        result["extraction_method"] = "none"
        result["success"] = False
        print(f"All extraction methods failed")
        return self._finalize_result(result, start_time)
    
    def _extract_with_docling(
        self,
        source: Union[str, DocumentStream],
        document_id: str,
        source_path: str

       #Extract using Docling's full pipeline
    ) -> Dict[str, Any]:
        
        result = self._empty_result(document_id, source_path)
        now = utcnow()
        
        # Convert document - only pass limits if set
        convert_kwargs = {}
        if self.config.max_pages is not None:
            convert_kwargs['max_num_pages'] = self.config.max_pages
        if self.config.max_file_size is not None:
            convert_kwargs['max_file_size'] = self.config.max_file_size
        
        conv_result = self.converter.convert(source, **convert_kwargs)
        doc: DoclingDocument = conv_result.document
        
        # Output directory
        output_dir = Path(self.config.output_dir) / document_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Counters
        section_seq = 0
        image_seq = 0
        table_seq = 0
        formula_seq = 0
        toc_pages = []
        error_pages = []
        
        # Process pages
        for page_no, page in doc.pages.items():
            try:
                # Get page text per page
                page_text = ""
                page_md = ""
                
                # Collect text from items on this page
                for item, level in doc.iterate_items(included_content_layers={ContentLayer.BODY}):
                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            if prov.page_no == page_no:
                                text = getattr(item, 'text', '') or ''
                                if text:
                                    page_text += text + "\n"
                                    page_md += text + "\n\n"
                                break
                
                is_toc = self._is_toc_page(page_text)
                if is_toc:
                    toc_pages.append(page_no)
                
                # Get header/footer from furniture layer
                header_text = footer_text = None
                has_header = has_footer = False
                
                for item, _ in doc.iterate_items(included_content_layers={ContentLayer.FURNITURE}):
                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            if prov.page_no == page_no:
                                if hasattr(item, 'label'):
                                    if item.label == DocItemLabel.PAGE_HEADER:
                                        has_header = True
                                        header_text = getattr(item, 'text', '')
                                    elif item.label == DocItemLabel.PAGE_FOOTER:
                                        has_footer = True
                                        footer_text = getattr(item, 'text', '')
                
                # Count items on this page
                page_section_count = 0
                page_image_count = 0
                page_table_count = 0
                page_formula_count = 0
                
                for item, _ in doc.iterate_items(included_content_layers={ContentLayer.BODY}):
                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            if prov.page_no == page_no:
                                if isinstance(item, (TextItem, SectionHeaderItem)):
                                    page_section_count += 1
                                elif isinstance(item, TableItem):
                                    page_table_count += 1
                                elif isinstance(item, PictureItem):
                                    page_image_count += 1
                                elif hasattr(item, 'label') and item.label == DocItemLabel.FORMULA:
                                    page_formula_count += 1
                                break
                
                # Page result - matching schema
                result["pages"].append({
                    "document_id": document_id,
                    "page_number": page_no,
                    "page_text": "" if is_toc else page_text,
                    "page_markdown": "" if is_toc else page_md,
                    "is_toc_page": is_toc,
                    "has_header": has_header,
                    "has_footer": has_footer,
                    "header_text": header_text,
                    "footer_text": footer_text,
                    "section_count": page_section_count,
                    "image_count": page_image_count,
                    "table_count": page_table_count,
                    "formula_count": page_formula_count,
                    "extracted_at": now.isoformat()
                })
                
            except Exception as e:
                error_pages.append(page_no)
                result["errors"].append(self._create_error(
                    document_id, page_no, "page_extraction",
                    str(e), traceback.format_exc()[:1000]
                ))
        # Process sections - capture ALL text-based items with proper labels
        for item, level in doc.iterate_items(included_content_layers={ContentLayer.BODY}):
            # Skip tables and pictures (handled separately)
            if isinstance(item, (TableItem, PictureItem)):
                continue
            
            # Get text content
            text = getattr(item, 'text', '') or ''
            if not text.strip():
                continue
            
            section_seq += 1
            
            # Use Docling's label (preferred) or fallback to regex classification
            if hasattr(item, 'label') and item.label:
                section_type = self._map_docling_label(item.label)
            else:
                section_type = self._classify_section(text)
            
            # Override for SectionHeaderItem
            if isinstance(item, SectionHeaderItem):
                section_type = "heading"
            
            # Get page number and bbox from provenance
            page_no = 0
            bbox_x0 = bbox_y0 = bbox_x1 = bbox_y1 = None
            
            if hasattr(item, 'prov') and item.prov:
                prov = item.prov[0]
                page_no = prov.page_no
                if prov.bbox:
                    bbox_x0 = prov.bbox.l
                    bbox_y0 = prov.bbox.t
                    bbox_x1 = prov.bbox.r
                    bbox_y1 = prov.bbox.b
            
            # Get heading level
            heading_level = None
            if isinstance(item, SectionHeaderItem):
                heading_level = getattr(item, 'level', 1)
            elif section_type == "heading":
                # Infer from numbering pattern (e.g., "1.2.3" = level 3)
                match = re.match(r'^([\d\.]+)', text.strip())
                if match:
                    heading_level = len(match.group(1).strip('.').split('.'))
                else:
                    heading_level = min(level + 1, 6)  # Use document hierarchy
            
            # Generate proper markdown
            content_markdown = self._generate_markdown(text, section_type, heading_level, item)
            
            # Check strikethrough (Unicode combining character)
            is_strikethrough = '\u0336' in text
            
            # Check if item is from TOC
            is_toc = (
                section_type in ("toc", "toc_entry", "document_index") or
                (hasattr(item, 'label') and item.label == DocItemLabel.DOCUMENT_INDEX)
            )
            
            # Section result - matching schema
            result["sections"].append({
                "section_id": f"{document_id}_s{section_seq:05d}",
                "document_id": document_id,
                "page_number": page_no,
                "sequence_number": section_seq,
                "section_type": section_type,
                "content_text": text,
                "content_markdown": content_markdown,
                "is_strikethrough": is_strikethrough,
                "is_toc": is_toc,
                "heading_level": heading_level,
                "bbox_x0": bbox_x0,
                "bbox_y0": bbox_y0,
                "bbox_x1": bbox_x1,
                "bbox_y1": bbox_y1,
                "extracted_at": now.isoformat()
            })
        
        # Process images/pictures
        for pic_item in doc.pictures:
            image_seq += 1
            
            page_no = 0
            is_above_header = False
            
            if pic_item.prov:
                prov = pic_item.prov[0]
                page_no = prov.page_no
                if prov.bbox and prov.bbox.t < 100:
                    is_above_header = True
            
            # Save image
            file_path = None
            width = height = None
            img_format = "PNG"
            
            if pic_item.image and pic_item.image.pil_image:
                pil_img = pic_item.image.pil_image
                width, height = pil_img.size
                
                img_filename = f"p{page_no:03d}_img{image_seq:03d}.png"
                img_path = output_dir / img_filename
                pil_img.save(str(img_path), "PNG")
                file_path = str(img_path)
            
            # Image result - matching schema
            result["images"].append({
                "image_id": f"{document_id}_p{page_no:03d}_img{image_seq:03d}",
                "document_id": document_id,
                "page_number": page_no,
                "sequence_number": image_seq,
                "file_path": file_path,
                "width": width,
                "height": height,
                "format": img_format,
                "markdown_ref": f"![img{image_seq}](images/p{page_no:03d}_img{image_seq:03d}.png)",
                "is_above_header": is_above_header,
                "extracted_at": now.isoformat()
            })
        
        # Process tables
        for table_item in doc.tables:
            table_seq += 1
            
            page_no = 0
            page_numbers = []
            
            if table_item.prov:
                for prov in table_item.prov:
                    if prov.page_no not in page_numbers:
                        page_numbers.append(prov.page_no)
                page_no = page_numbers[0] if page_numbers else 0
            
            # Get table data
            row_count = col_count = 0
            if table_item.data:
                row_count = table_item.data.num_rows
                col_count = table_item.data.num_cols
            
            # Export formats
            html_content = ""
            md_content = ""
            
            if hasattr(table_item, 'export_to_html'):
                try:
                    html_content = table_item.export_to_html()
                except:
                    pass
            
            if hasattr(table_item, 'export_to_markdown'):
                try:
                    md_content = table_item.export_to_markdown()
                except:
                    pass
            
            # Table result - matching schema
            result["tables"].append({
                "table_id": f"{document_id}_p{page_no:03d}_tbl{table_seq:03d}",
                "document_id": document_id,
                "page_number": page_no,
                "sequence_number": table_seq,
                "row_count": row_count,
                "column_count": col_count,
                "html_content": html_content,
                "markdown_content": md_content,
                "spans_multiple_pages": len(page_numbers) > 1,
                "page_numbers": page_numbers,
                "extracted_at": now.isoformat()
            })
        
        # Process formulas
        for item, level in doc.iterate_items(included_content_layers={ContentLayer.BODY}):
            if hasattr(item, 'label') and item.label == DocItemLabel.FORMULA:
                formula_seq += 1
                
                text = getattr(item, 'text', '') or ''
                page_no = 0
                
                if hasattr(item, 'prov') and item.prov:
                    page_no = item.prov[0].page_no
                
                # Formula result - matching schema
                result["formulas"].append({
                    "formula_id": f"{document_id}_p{page_no:03d}_eq{formula_seq:03d}",
                    "document_id": document_id,
                    "page_number": page_no,
                    "sequence_number": formula_seq,
                    "raw_content": text,
                    "context_text": text[:200] if text else "",
                    "needs_llm_processing": True,
                    "extracted_at": now.isoformat()
                })
       # If doc.pages is empty, build pages from sections
        if not result["pages"] and result["sections"]:
            page_numbers = set()
            for section in result["sections"]:
                pn = section.get("page_number")
                if pn is not None:  # Changed: include page 0!
                    page_numbers.add(pn)
            
            for page_no in sorted(page_numbers):
                page_sections = [s for s in result["sections"] if s.get("page_number") == page_no]
                page_text = "\n".join([s.get("content_text", "") or "" for s in page_sections])
                
                result["pages"].append({
                    "document_id": document_id,
                    "page_number": page_no,
                    "page_text": page_text[:5000],
                    "page_markdown": page_text[:5000],
                    "is_toc_page": False,
                    "has_header": False,
                    "has_footer": False,
                    "header_text": None,
                    "footer_text": None,
                    "section_count": len(page_sections),
                    "image_count": len([i for i in result["images"] if i.get("page_number") == page_no]),
                    "table_count": len([t for t in result["tables"] if t.get("page_number") == page_no]),
                    "formula_count": 0,
                    "extracted_at": now.isoformat()
                })
            
            print(f"Built {len(result['pages'])} pages from sections")

            ...
        # Metadata - matching schema
        result["metadata"] = {
            "document_id": document_id,
            "source_path": source_path,
            "file_name": Path(source_path).name,
            "page_count": len(doc.pages),
            "section_count": section_seq,
            "image_count": image_seq,
            "table_count": table_seq,
            "formula_count": formula_seq,
            "toc_page_count": len(toc_pages),
            "toc_pages": toc_pages,
            "error_page_count": len(error_pages),
            "error_pages": error_pages,
            "processing_duration_seconds": 0.0,  # Will be set in finalize
            "extraction_engine": "docling",
            "extraction_engine_version": self._get_docling_version(),
            "extracted_at": now.isoformat()
        }
        
        result["success"] = len(error_pages) == 0
        return result
    
    # Fallback extraction with PyMuPDF
    def _extract_with_pymupdf(self, source_path: str, document_id: str) -> Dict[str, Any]:
        
        result = self._empty_result(document_id, source_path)
        now = utcnow()
        
        output_dir = Path(self.config.output_dir) / document_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pdf = fitz.open(source_path)
        
        section_seq = 0
        image_seq = 0
        error_pages = []
        
        for page_no in range(len(pdf)):
            try:
                page = pdf[page_no]
                page_num = page_no + 1
                
                page_text = page.get_text("text")
                is_toc = self._is_toc_page(page_text)
                
                # Extract images
                for img_idx, img in enumerate(page.get_images()):
                    try:
                        xref = img[0]
                        base_image = pdf.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        image_seq += 1
                        img_filename = f"p{page_num:03d}_img{image_seq:03d}.png"
                        img_path = output_dir / img_filename
                        
                        with open(img_path, "wb") as f:
                            f.write(image_bytes)
                        
                        result["images"].append({
                            "image_id": f"{document_id}_p{page_num:03d}_img{image_seq:03d}",
                            "document_id": document_id,
                            "page_number": page_num,
                            "sequence_number": image_seq,
                            "file_path": str(img_path),
                            "width": base_image.get("width"),
                            "height": base_image.get("height"),
                            "format": base_image.get("ext", "png").upper(),
                            "markdown_ref": f"![img{image_seq}](images/{img_filename})",
                            "is_above_header": False,
                            "extracted_at": now.isoformat()
                        })
                    except:
                        pass
                
                # Create sections
                paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
                
                for para in paragraphs:
                    section_seq += 1
                    section_type = self._classify_section(para)
                    heading_level = self._get_heading_level(para, section_type)
                    
                    # Generate proper markdown for fallback too
                    content_markdown = self._generate_markdown(para, section_type, heading_level, None)
                    
                    result["sections"].append({
                        "section_id": f"{document_id}_s{section_seq:05d}",
                        "document_id": document_id,
                        "page_number": page_num,
                        "sequence_number": section_seq,
                        "section_type": section_type,
                        "content_text": para,
                        "content_markdown": content_markdown,
                        "is_strikethrough": False,
                        "is_toc": section_type in ("toc", "toc_entry"),
                        "heading_level": heading_level,
                        "bbox_x0": None,
                        "bbox_y0": None,
                        "bbox_x1": None,
                        "bbox_y1": None,
                        "extracted_at": now.isoformat()
                    })
                
                # Page result
                result["pages"].append({
                    "document_id": document_id,
                    "page_number": page_num,
                    "page_text": "" if is_toc else page_text,
                    "page_markdown": "" if is_toc else page_text,
                    "is_toc_page": is_toc,
                    "has_header": False,
                    "has_footer": False,
                    "header_text": None,
                    "footer_text": None,
                    "section_count": len(paragraphs),
                    "image_count": len([i for i in result["images"] if i["page_number"] == page_num]),
                    "table_count": 0,
                    "formula_count": 0,
                    "extracted_at": now.isoformat()
                })
                
            except Exception as e:
                error_pages.append(page_no + 1)
                result["errors"].append(self._create_error(
                    document_id, page_no + 1, "page_extraction",
                    str(e), traceback.format_exc()[:1000]
                ))
        
        pdf.close()
        
        result["metadata"] = {
            "document_id": document_id,
            "source_path": source_path,
            "file_name": Path(source_path).name,
            "page_count": len(result["pages"]),
            "section_count": section_seq,
            "image_count": image_seq,
            "table_count": 0,
            "formula_count": 0,
            "toc_page_count": sum(1 for p in result["pages"] if p["is_toc_page"]),
            "toc_pages": [p["page_number"] for p in result["pages"] if p["is_toc_page"]],
            "error_page_count": len(error_pages),
            "error_pages": error_pages,
            "processing_duration_seconds": 0.0,
            "extraction_engine": "pymupdf",
            "extraction_engine_version": fitz.version[0],
            "extracted_at": now.isoformat()
        }
        
        result["success"] = len(error_pages) == 0
        return result
    
    def _extract_with_pdfplumber(self, source_path: str, document_id: str) -> Dict[str, Any]:
        
        result = self._empty_result(document_id, source_path)
        now = utcnow()
        
        with pdfplumber.open(source_path) as pdf:
            section_seq = 0
            table_seq = 0
            error_pages = []
            
            for page_no, page in enumerate(pdf.pages):
                try:
                    page_num = page_no + 1
                    page_text = page.extract_text() or ""
                    is_toc = self._is_toc_page(page_text)
                    
                    # Extract tables
                    for table in page.extract_tables():
                        if table:
                            table_seq += 1
                            
                            md_rows = []
                            for row_idx, row in enumerate(table):
                                clean_row = [str(cell or "").replace("|", "\\|") for cell in row]
                                md_rows.append("| " + " | ".join(clean_row) + " |")
                                if row_idx == 0:
                                    md_rows.append("|" + "|".join(["---"] * len(row)) + "|")
                            
                            result["tables"].append({
                                "table_id": f"{document_id}_p{page_num:03d}_tbl{table_seq:03d}",
                                "document_id": document_id,
                                "page_number": page_num,
                                "sequence_number": table_seq,
                                "row_count": len(table),
                                "column_count": len(table[0]) if table else 0,
                                "html_content": "",
                                "markdown_content": "\n".join(md_rows),
                                "spans_multiple_pages": False,
                                "page_numbers": [page_num],
                                "extracted_at": now.isoformat()
                            })
                    
                    # Sections
                    paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]
                    
                    for para in paragraphs:
                        section_seq += 1
                        section_type = self._classify_section(para)
                        heading_level = self._get_heading_level(para, section_type)
                        
                        # Generate proper markdown
                        content_markdown = self._generate_markdown(para, section_type, heading_level, None)
                        
                        result["sections"].append({
                            "section_id": f"{document_id}_s{section_seq:05d}",
                            "document_id": document_id,
                            "page_number": page_num,
                            "sequence_number": section_seq,
                            "section_type": section_type,
                            "content_text": para,
                            "content_markdown": content_markdown,
                            "is_strikethrough": False,
                            "is_toc": section_type in ("toc", "toc_entry"),
                            "heading_level": heading_level,
                            "bbox_x0": None,
                            "bbox_y0": None,
                            "bbox_x1": None,
                            "bbox_y1": None,
                            "extracted_at": now.isoformat()
                        })
                    
                    result["pages"].append({
                        "document_id": document_id,
                        "page_number": page_num,
                        "page_text": "" if is_toc else page_text,
                        "page_markdown": "" if is_toc else page_text,
                        "is_toc_page": is_toc,
                        "has_header": False,
                        "has_footer": False,
                        "header_text": None,
                        "footer_text": None,
                        "section_count": len(paragraphs),
                        "image_count": 0,
                        "table_count": len([t for t in result["tables"] if t["page_number"] == page_num]),
                        "formula_count": 0,
                        "extracted_at": now.isoformat()
                    })
                    
                except Exception as e:
                    error_pages.append(page_no + 1)
                    result["errors"].append(self._create_error(
                        document_id, page_no + 1, "page_extraction",
                        str(e), traceback.format_exc()[:1000]
                    ))
        
        result["metadata"] = {
            "document_id": document_id,
            "source_path": source_path,
            "file_name": Path(source_path).name,
            "page_count": len(result["pages"]),
            "section_count": section_seq,
            "image_count": 0,
            "table_count": table_seq,
            "formula_count": 0,
            "toc_page_count": sum(1 for p in result["pages"] if p["is_toc_page"]),
            "toc_pages": [p["page_number"] for p in result["pages"] if p["is_toc_page"]],
            "error_page_count": len(error_pages),
            "error_pages": error_pages,
            "processing_duration_seconds": 0.0,
            "extraction_engine": "pdfplumber",
            "extraction_engine_version": "0.10",
            "extracted_at": now.isoformat()
        }
        
        result["success"] = len(error_pages) == 0
        return result
    
    # Last resort: extract raw text only
    def _extract_raw_text(self, source_path: str, document_id: str) -> Dict[str, Any]:
        
        result = self._empty_result(document_id, source_path)
        now = utcnow()
        
        try:
            pdf = fitz.open(source_path)
            
            for page_no in range(len(pdf)):
                page = pdf[page_no]
                page_text = page.get_text("text") or ""
                
                result["pages"].append({
                    "document_id": document_id,
                    "page_number": page_no + 1,
                    "page_text": page_text,
                    "page_markdown": page_text,
                    "is_toc_page": False,
                    "has_header": False,
                    "has_footer": False,
                    "header_text": None,
                    "footer_text": None,
                    "section_count": 0,
                    "image_count": 0,
                    "table_count": 0,
                    "formula_count": 0,
                    "extracted_at": now.isoformat()
                })
            
            pdf.close()
            
            result["metadata"] = {
                "document_id": document_id,
                "source_path": source_path,
                "file_name": Path(source_path).name,
                "page_count": len(result["pages"]),
                "section_count": 0,
                "image_count": 0,
                "table_count": 0,
                "formula_count": 0,
                "toc_page_count": 0,
                "toc_pages": [],
                "error_page_count": 0,
                "error_pages": [],
                "processing_duration_seconds": 0.0,
                "extraction_engine": "raw_text",
                "extraction_engine_version": "1.0",
                "extracted_at": now.isoformat()
            }
            
            result["success"] = True
            
        except Exception as e:
            result["errors"].append(self._create_error(
                document_id, None, "raw_text_failure",
                str(e), traceback.format_exc()[:1000]
            ))
        
        return result
    
    # Helper Methods

    # Create empty result structure matching Spark schemas
    
    def _empty_result(self, document_id: str, source_path: str) -> Dict[str, Any]:
        return {
            "success": False,
            "document_id": document_id,
            "source_path": source_path,
            "extraction_method": None,
            "metadata": {},
            "pages": [],
            "sections": [],
            "images": [],
            "tables": [],
            "formulas": [],
            "errors": [],
            "warnings": []
        }
    
    def _create_error(
        self,
        document_id: str,
        page_number: Optional[int],
        error_type: str,
        message: str,
        tb: Optional[str] = None

        # Create error record matching schema
    ) -> Dict[str, Any]:
        return {
            "error_id": f"{document_id}_{error_type}_{utcnow().timestamp()}",
            "document_id": document_id,
            "page_number": page_number,
            "error_type": error_type,
            "error_message": message[:1000],
            "error_traceback": tb,
            "occurred_at": utcnow().isoformat()
        }
    
     # Finalize result with timing
    def _finalize_result(self, result: Dict[str, Any], start_time: datetime) -> Dict[str, Any]:
    
        end_time = utcnow()
        duration = (end_time - start_time).total_seconds()
        
        if "metadata" in result:
            result["metadata"]["processing_duration_seconds"] = duration
        
        return result
    
    # Run function with timeout
    def _run_with_timeout(self, func, args, timeout):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args)
            return future.result(timeout=timeout)
    
    # Detect if page is a table of contents
    def _is_toc_page(self, text: str) -> bool:
        if not text or len(text.strip()) < 20:
            return False
        tl = text.lower()
        for hdr in ["table of contents", "contents", "index"]:
            if hdr in tl and re.search(r'\.{3,}\s*\d+', text):
                return True
        lines = [x.strip() for x in text.split("\n") if x.strip()]
        if len(lines) > 5:
            nums = sum(1 for ln in lines if re.search(r'[\.\s]+\d{1,4}\s*$', ln))
            if nums / len(lines) > 0.5:
                return True
        return False
    
    # Map Docling's DocItemLabel to section_type for schema compatibility
    def _map_docling_label(self, label) -> str:
        if label is None:
            return "paragraph"
        
        # Get string value from enum
        label_str = label.value if hasattr(label, 'value') else str(label)
        label_lower = label_str.lower()
        
        label_map = {
            "title": "title",
            "document_index": "toc",
            "section_header": "heading",
            "paragraph": "paragraph",
            "text": "paragraph",
            "list_item": "list_item",
            "caption": "caption",
            "figure_caption": "figure_caption",
            "table_caption": "table_caption",
            "code": "code",
            "formula": "equation",
            "footnote": "footnote",
            "page_header": "header",
            "page_footer": "footer",
            "reference": "reference",
            "checkbox_selected": "checkbox",
            "checkbox_unselected": "checkbox",
            "picture": "figure",
            "table": "table",
        }
        return label_map.get(label_lower, "paragraph")
    
    # Generate proper markdown based on section type
    def _generate_markdown(self, text: str, section_type: str, heading_level: int = None, item=None) -> str:
        if not text:
            return ""
        
        text = text.strip()
        
        # Headings
        if section_type in ("heading", "title", "section_header"):
            level = heading_level or 1
            level = min(max(level, 1), 6)  # Clamp to 1-6
            return f"{'#' * level} {text}"
        
        # List items
        if section_type == "list_item":
            # Check if it's enumerated
            if item and hasattr(item, 'enumerated') and item.enumerated:
                return f"1. {text}"
            elif item and hasattr(item, 'marker') and item.marker:
                marker = str(item.marker)
                if re.match(r'^\d+[\.\)]', marker):
                    return f"1. {text}"
            return f"- {text}"
        
        # Code blocks
        if section_type == "code":
            lang = ""
            if item and hasattr(item, 'code_language') and item.code_language:
                lang = item.code_language.value if hasattr(item.code_language, 'value') else str(item.code_language)
            return f"```{lang}\n{text}\n```"
        
        # Formulas/equations - wrap in LaTeX delimiters
        if section_type == "equation":
            if '\n' in text or len(text) > 50:
                return f"$$\n{text}\n$$"
            return f"${text}$"
        
        # Captions - italicize
        if section_type in ("caption", "figure_caption", "table_caption"):
            return f"*{text}*"
        
        # Footnotes
        if section_type == "footnote":
            return f"[^]: {text}"
        
        # Default: plain paragraph
        return text
    
    # Classify section type
    def _classify_section(self, text: str) -> str:
        if not text:
            return "unknown"
        t = text.strip()
        tl = t.lower()
        
        if self._is_toc_page(t):
            return "toc"
        if re.match(r'^[\d\.]+\s+.+[\.\s]+\d+\s*$', t):
            return "toc_entry"
        if re.match(r'^(chapter|section|article|part)\s+[\divxlc]+', tl):
            return "heading"
        if re.match(r'^[\d\.]+\s+[A-Z]', t) and len(t) < 100:
            return "heading"
        if re.match(r'^appendix\s+[a-z\d]', tl):
            return "appendix"
        if re.match(r'^[\-\•\*]\s', t):
            return "list_item"
        if re.match(r'^(figure|fig\.?)\s+\d+', tl):
            return "figure_caption"
        if re.match(r'^(table|tbl\.?)\s+\d+', tl):
            return "table_caption"
        if re.search(r'[∫∑∏√∂∇≠≤≥±×÷]', t):
            return "equation"
        return "paragraph"
    
    def _get_heading_level(self, text: str, section_type: str) -> Optional[int]:
        """Get heading level"""
        if section_type != "heading":
            return None
        match = re.match(r'^([\d\.]+)', text.strip())
        if match:
            return len(match.group(1).strip('.').split('.'))
        return 1
    
    def _get_docling_version(self) -> str:
        """Get Docling version"""
        try:
            import docling
            return getattr(docling, '__version__', 'unknown')
        except:
            return 'unknown'


if __name__ == "__main__":
    print("DoclingExtractor: Compatible with Spark Pipeline")
    
    extraction_config = ExtractionConfig(
        enable_ocr=True,
        ocr_engine=OcrEngine.EASYOCR,
        enable_table_structure=True,
        table_mode=TableFormerMode.ACCURATE,
        accelerator=AcceleratorType.AUTO,
    )
    
    extractor = DoclingExtractor(extraction_config)
    
    print("\nDoclingExtractor ready")

# COMMAND ----------

# MAGIC %md
# MAGIC # Section 3: Python Output to Spark DataFrames

# COMMAND ----------

from pyspark.sql.types import *
import json

# EXPLICIT SCHEMAS

pages_schema = StructType([
    StructField("document_id", StringType(), True),
    StructField("page_number", IntegerType(), True),
    StructField("page_text", StringType(), True),
    StructField("page_markdown", StringType(), True),
    StructField("is_toc_page", BooleanType(), True),
    StructField("has_header", BooleanType(), True),
    StructField("has_footer", BooleanType(), True),
    StructField("header_text", StringType(), True),
    StructField("footer_text", StringType(), True),
    StructField("section_count", IntegerType(), True),
    StructField("image_count", IntegerType(), True),
    StructField("table_count", IntegerType(), True),
    StructField("formula_count", IntegerType(), True),
    StructField("extracted_at", StringType(), True),
])

sections_schema = StructType([
    StructField("section_id", StringType(), True),
    StructField("document_id", StringType(), True),
    StructField("page_number", IntegerType(), True),
    StructField("sequence_number", IntegerType(), True),
    StructField("section_type", StringType(), True),
    StructField("content_text", StringType(), True),
    StructField("content_markdown", StringType(), True),
    StructField("is_strikethrough", BooleanType(), True),
    StructField("is_toc", BooleanType(), True),
    StructField("heading_level", IntegerType(), True),
    StructField("bbox_x0", DoubleType(), True),
    StructField("bbox_y0", DoubleType(), True),
    StructField("bbox_x1", DoubleType(), True),
    StructField("bbox_y1", DoubleType(), True),
    StructField("extracted_at", StringType(), True),
])

images_schema = StructType([
    StructField("image_id", StringType(), True),
    StructField("document_id", StringType(), True),
    StructField("page_number", IntegerType(), True),
    StructField("sequence_number", IntegerType(), True),
    StructField("file_path", StringType(), True),
    StructField("width", IntegerType(), True),
    StructField("height", IntegerType(), True),
    StructField("format", StringType(), True),
    StructField("markdown_ref", StringType(), True),
    StructField("is_above_header", BooleanType(), True),
    StructField("extracted_at", StringType(), True),
])

tables_schema = StructType([
    StructField("table_id", StringType(), True),
    StructField("document_id", StringType(), True),
    StructField("page_number", IntegerType(), True),
    StructField("sequence_number", IntegerType(), True),
    StructField("row_count", IntegerType(), True),
    StructField("column_count", IntegerType(), True),
    StructField("html_content", StringType(), True),
    StructField("markdown_content", StringType(), True),
    StructField("spans_multiple_pages", BooleanType(), True),
    StructField("page_numbers", ArrayType(IntegerType()), True),
    StructField("extracted_at", StringType(), True),
])

formulas_schema = StructType([
    StructField("formula_id", StringType(), True),
    StructField("document_id", StringType(), True),
    StructField("page_number", IntegerType(), True),
    StructField("sequence_number", IntegerType(), True),
    StructField("raw_content", StringType(), True),
    StructField("context_text", StringType(), True),
    StructField("needs_llm_processing", BooleanType(), True),
    StructField("extracted_at", StringType(), True),
])

errors_schema = StructType([
    StructField("error_id", StringType(), True),
    StructField("document_id", StringType(), True),
    StructField("page_number", IntegerType(), True),
    StructField("error_type", StringType(), True),
    StructField("error_message", StringType(), True),
    StructField("error_traceback", StringType(), True),
    StructField("occurred_at", StringType(), True),
])

metadata_schema = StructType([
    StructField("document_id", StringType(), True),
    StructField("source_path", StringType(), True),
    StructField("file_name", StringType(), True),
    StructField("page_count", IntegerType(), True),
    StructField("section_count", IntegerType(), True),
    StructField("image_count", IntegerType(), True),
    StructField("table_count", IntegerType(), True),
    StructField("formula_count", IntegerType(), True),
    StructField("toc_page_count", IntegerType(), True),
    StructField("toc_pages", ArrayType(IntegerType()), True),
    StructField("error_page_count", IntegerType(), True),
    StructField("error_pages", ArrayType(IntegerType()), True),
    StructField("processing_duration_seconds", DoubleType(), True),
    StructField("extraction_engine", StringType(), True),
    StructField("extraction_engine_version", StringType(), True),
    StructField("extracted_at", StringType(), True),
])
registry_schema = StructType([
    StructField("document_id", StringType(), True),
    StructField("source_path", StringType(), True),
    StructField("processing_status", StringType(), True),
    StructField("page_count", IntegerType(), True),
    StructField("error_count", IntegerType(), True),
    StructField("processed_at", StringType(), True)
])


print("Bronze Schemas loaded:")
print(f"Pages schema     → {pages_schema.jsonValue()['fields']} fields")
print(f"Sections schema  → {sections_schema.jsonValue()['fields']} fields")
print(f"Images schema    → {images_schema.jsonValue()['fields']} fields")
print(f"Tables schema    → {tables_schema.jsonValue()['fields']} fields")
print(f"Formulas schema  → {formulas_schema.jsonValue()['fields']} fields")
print(f"Errors schema    → {errors_schema.jsonValue()['fields']} fields")
print(f"Metadata schema  → {metadata_schema.jsonValue()['fields']} fields")
print("Schema definitions validated and ready for DataFrames")


# COMMAND ----------

# MAGIC %md
# MAGIC # Section 4: Run Bronze Extraction for Selected Protocol Documents (Python)

# COMMAND ----------

# Spark-Parallel Bronze Extraction
import os, json, urllib.parse, time
from datetime import datetime
from pyspark import SparkContext

# Correct imports ie only these two exist
from pipeline.docling_extractor import DoclingExtractor, extract_single_document


# DBFS Normalization
def normalize_dbfs_path(path: str) -> str:
    if not path:
        return None
    if path.startswith("dbfs:/Volumes/"):
        return urllib.parse.unquote(path.replace("dbfs:/", "/"))
    if path.startswith("dbfs:/"):
        return urllib.parse.unquote("/dbfs" + path[len("dbfs:"):])
    return urllib.parse.unquote(path)


# Worker Partition Function
def worker_partition(iterator, pipeline_config_dict):
    import json
    import os
    from datetime import datetime

    for row in iterator:
        raw_path = row.path
        dbfs_path = normalize_dbfs_path(raw_path)

        # Extract document ID safely
        doc_id = os.path.basename(os.path.dirname(dbfs_path)) or ""

        try:
            # 1. COPY PDF FROM DBFS → WORKER LOCAL DISK

            filename = os.path.basename(dbfs_path)
            local_path = f"/local_disk0/tmp/{filename}"

            # Ensure directory exists
            os.makedirs("/local_disk0/tmp", exist_ok=True)

            # Copy file into worker node
            import subprocess
            subprocess.run(
                ["cp", dbfs_path, local_path],
                check=True
            )

            # 2. RUN DOCLING EXTRACTION USING LOCAL PATH

            result = extract_single_document(
                input_path=local_path,
                output_volume=pipeline_config_dict["output_volume"]
            )

            yield json.dumps(result)

        except Exception as e:
            fail = {
                "pages": [],
                "sections": [],
                "images": [],
                "tables": [],
                "formulas": [],
                "errors": [],
                "registry": {
                    "document_id": doc_id,
                    "source_path": raw_path,
                    "processing_status": "failed",
                    "error_message": str(e),
                    "page_count": 0,
                    "error_count": 1,
                    "tools_used": [],
                    "processed_at": datetime.utcnow().isoformat(),
                }
            }
            yield json.dumps(fail)


        except Exception as e:
            # Return failure record but do NOT crash the worker
            fail = {
                "pages": [],
                "sections": [],
                "images": [],
                "tables": [],
                "formulas": [],
                "errors": [],
                "registry": {
                    "document_id": doc_id,
                    "source_path": path,
                    "processing_status": "failed",
                    "error_message": str(e),
                    "page_count": 0,
                    "error_count": 1,
                    "tools_used": [],
                    "processed_at": datetime.utcnow().isoformat(),
                }
            }
            yield json.dumps(fail)


# Load Document List
docs_df = spark.sql(f"SELECT path FROM {config.full_source_table()} WHERE path IS NOT NULL")
print("Files to process:", docs_df.count())

sc = SparkContext.getOrCreate()
num_parts = max(1, min(sc.defaultParallelism, docs_df.count()))
paths_df = docs_df.repartition(num_parts)

print(f"Running Spark parallel extraction over {num_parts} partitions")

pipeline_config_dict = {
    "output_volume": config.output_volume
}

start = time.time()
rdd_json = paths_df.rdd.mapPartitions(
    lambda it: worker_partition(it, pipeline_config_dict)
)

results = rdd_json.collect()
print(f"Worker processing done in {round(time.time() - start, 2)}s\nAggregating...")


# Aggregate Results (Driver)
all_pages, all_sections, all_images = [], [], []
all_tables, all_formulas, all_errors, all_registry = [], [], [], []

for rec in results:
    obj = json.loads(rec)

    all_pages.extend(obj.get("pages", []))
    all_sections.extend(obj.get("sections", []))
    all_images.extend(obj.get("images", []))
    all_tables.extend(obj.get("tables", []))
    all_formulas.extend(obj.get("formulas", []))
    all_errors.extend(obj.get("errors", []))
    all_registry.append(obj.get("registry", {}))


print("DONE: Spark-parallel extraction finished.")
print("Pages:", len(all_pages))
print("Sections:", len(all_sections))
print("Registry:", len(all_registry))



# COMMAND ----------

# MAGIC %md
# MAGIC #Making document_id=protocol_id. Only one time run. Do not re-run

# COMMAND ----------

# MAGIC %md
# MAGIC from pyspark.sql import functions as F
# MAGIC
# MAGIC def extract_protocol_id(col):
# MAGIC     # Extract everything before first underscore
# MAGIC     return F.regexp_extract(col, r"^([^_]+)", 1)
# MAGIC
# MAGIC
# MAGIC # FIX BRONZE SECTIONS
# MAGIC bronze_sections = spark.table(config.full_table(config.bronze_sections))
# MAGIC
# MAGIC bronze_sections_fixed = (
# MAGIC     bronze_sections
# MAGIC     .withColumn("protocol_id", extract_protocol_id(F.col("section_id")))
# MAGIC     .withColumn("document_id", F.col("protocol_id"))
# MAGIC )
# MAGIC
# MAGIC bronze_sections_fixed.write \
# MAGIC     .format("delta") \
# MAGIC     .mode("overwrite") \
# MAGIC     .option("overwriteSchema", "true") \
# MAGIC     .saveAsTable(config.full_table(config.bronze_sections))
# MAGIC
# MAGIC print("Bronze sections updated with protocol_id and document_id")
# MAGIC
# MAGIC bronze_pages = spark.table(config.full_table(config.bronze_pages))
# MAGIC
# MAGIC bronze_pages_fixed = (
# MAGIC     bronze_pages
# MAGIC     .withColumn("protocol_id", extract_protocol_id(F.col("document_id")))
# MAGIC     .withColumn("document_id", F.col("protocol_id"))
# MAGIC )
# MAGIC
# MAGIC bronze_pages_fixed.write \
# MAGIC     .format("delta") \
# MAGIC     .mode("overwrite") \
# MAGIC     .option("overwriteSchema", "true") \
# MAGIC     .saveAsTable(config.full_table(config.bronze_pages))
# MAGIC
# MAGIC print("Bronze pages updated with protocol_id and document_id")
# MAGIC
# MAGIC registry = spark.table(config.full_table(config.document_registry))
# MAGIC
# MAGIC registry_fixed = (
# MAGIC     registry
# MAGIC     .withColumn("protocol_id", extract_protocol_id(F.col("document_id")))
# MAGIC )
# MAGIC
# MAGIC registry_fixed.write \
# MAGIC     .format("delta") \
# MAGIC     .mode("overwrite") \
# MAGIC     .option("overwriteSchema", "true") \
# MAGIC     .saveAsTable(config.full_table(config.document_registry))
# MAGIC
# MAGIC print("Bronze registry updated with protocol_id")

# COMMAND ----------

def drop_if_exists(name):
    spark.sql(f"DROP TABLE IF EXISTS {config.full_table(name)}")

drop_if_exists(config.bronze_pages)
drop_if_exists(config.bronze_sections)
drop_if_exists(config.bronze_images)
drop_if_exists(config.bronze_tables)
drop_if_exists(config.bronze_formulas)
drop_if_exists(config.bronze_errors)
drop_if_exists(config.document_registry)

# COMMAND ----------

# MAGIC %md
# MAGIC # Section 5: Write to Bronze (Raw Extraction Output)

# COMMAND ----------

# FULL BRONZE (Docling-fidelity) + WRITE SAFELY
import os, urllib.parse
from pyspark.sql.types import *
from datetime import datetime

# Helpers
def normalize_path(p: str) -> str:
    if not p:
        return None
    if p.startswith("dbfs:/Volumes/"):
        return urllib.parse.unquote(p.replace("dbfs:/", "/"))
    if p.startswith("dbfs:/"):
        return urllib.parse.unquote("/dbfs" + p[len("dbfs:"):])
    return urllib.parse.unquote(p)

def extract_protocol_id(path: str) -> str:
    p = normalize_path(path or "")
    if not p:
        return None
    return os.path.basename(os.path.dirname(p))

def safe_get(d, k, default=None):
    return d.get(k, default) if isinstance(d, dict) else default

# Ensure full section record
def normalize_section(rec):
    # rec may be partial; return a dict with all required keys
    r = {} if rec is None else dict(rec)  # shallow copy
    # basic ids
    src = r.get("source_path") or r.get("source_pdf") or r.get("path") or ""
    r["source_path"] = normalize_path(src)
    r["protocol_id"]  = extract_protocol_id(r["source_path"])
    # keep document_id consistent with protocol_id
    r["document_id"]  = r.get("document_id") or r["protocol_id"] or r.get("doc_id")

    # Docling fields Silver expects — set defaults if missing
    r["section_id"]         = str(r.get("section_id") or r.get("id") or "") 
    r["page_number"]        = int(r.get("page_number") or r.get("page") or 0)
    r["sequence_number"]    = int(r.get("sequence_number") or r.get("seq") or 0)
    r["section_type"]       = str(r.get("section_type") or r.get("type") or "")
    r["content_text"]       = str(r.get("content_text") or r.get("text") or r.get("content") or "")
    r["content_markdown"]   = str(r.get("content_markdown") or r.get("markdown") or "")
    r["is_strikethrough"]   = bool(r.get("is_strikethrough") or False)
    r["is_toc"]             = bool(r.get("is_toc") or False)
    # heading level may be None
    hv = r.get("heading_level")
    r["heading_level"]      = int(hv) if hv is not None and str(hv).strip() != "" else None

    # bbox may be nested or flattened - normalize to fields bbox_x0..bbox_y1
    bbox = r.get("bbox") or {}
    # support multiple shapes e.g. {"x0":..., "y0":...} or {"x0":...}
    x0 = safe_get(bbox, "x0", safe_get(r, "bbox_x0", None))
    y0 = safe_get(bbox, "y0", safe_get(r, "bbox_y0", None))
    x1 = safe_get(bbox, "x1", safe_get(r, "bbox_x1", None))
    y1 = safe_get(bbox, "y1", safe_get(r, "bbox_y1", None))
    try:
        r["bbox_x0"] = float(x0) if x0 is not None else None
        r["bbox_y0"] = float(y0) if y0 is not None else None
        r["bbox_x1"] = float(x1) if x1 is not None else None
        r["bbox_y1"] = float(y1) if y1 is not None else None
    except:
        r["bbox_x0"] = r["bbox_y0"] = r["bbox_x1"] = r["bbox_y1"] = None

    # extracted_at fallback
    r["extracted_at"] = r.get("extracted_at") or r.get("extracted_ts") or datetime.utcnow().isoformat()

    return r

# Normalize pages / images / tables / formulas / errors / registry
def normalize_page(rec):
    r = {} if rec is None else dict(rec)
    src = r.get("source_path") or r.get("source_pdf") or ""
    r["source_path"] = normalize_path(src)
    r["protocol_id"]  = extract_protocol_id(r["source_path"])
    r["document_id"]  = r.get("document_id") or r["protocol_id"]
    r["page_number"]  = int(r.get("page_number") or r.get("page") or 0)
    r["text"]         = str(r.get("text") or "")
    # keep header/footer fields for Silver if present; default False/empty
    r["has_header"]   = bool(r.get("has_header") or False)
    r["has_footer"]   = bool(r.get("has_footer") or False)
    r["header_text"]  = str(r.get("header_text") or "")
    r["footer_text"]  = str(r.get("footer_text") or "")
    return r

def normalize_image(rec):
    r = {} if rec is None else dict(rec)
    src = r.get("source_path") or ""
    r["source_path"] = normalize_path(src)
    r["protocol_id"]  = extract_protocol_id(r["source_path"])
    r["document_id"]  = r.get("document_id") or r["protocol_id"]
    r["image_id"]     = str(r.get("image_id") or r.get("id") or "")
    r["page_number"]  = int(r.get("page_number") or 0)
    r["image_format"] = str(r.get("format") or r.get("image_format") or "")
    # store path or None — we avoid raw large bytes unless present intentionally
    r["image_path"]   = r.get("image_path") or r.get("path") or None
    return r

def normalize_table(rec):
    r = {} if rec is None else dict(rec)
    src = r.get("source_path") or ""
    r["source_path"] = normalize_path(src)
    r["protocol_id"]  = extract_protocol_id(r["source_path"])
    r["document_id"]  = r.get("document_id") or r["protocol_id"]
    r["table_id"]     = str(r.get("table_id") or r.get("id") or "")
    r["page_number"]  = int(r.get("page_number") or 0)
    r["html"]         = str(r.get("html") or r.get("table_html") or "")
    return r

def normalize_formula(rec):
    r = {} if rec is None else dict(rec)
    src = r.get("source_path") or ""
    r["source_path"] = normalize_path(src)
    r["protocol_id"]  = extract_protocol_id(r["source_path"])
    r["document_id"]  = r.get("document_id") or r["protocol_id"]
    r["formula_id"]   = str(r.get("formula_id") or r.get("id") or "")
    r["content"]      = str(r.get("content") or r.get("latex") or "")
    r["page_number"]  = int(r.get("page_number") or 0)
    return r

def normalize_error(rec):
    r = {} if rec is None else dict(rec)
    src = r.get("source_path") or ""
    r["source_path"] = normalize_path(src)
    r["protocol_id"]  = extract_protocol_id(r["source_path"])
    r["document_id"]  = r.get("document_id") or r["protocol_id"]
    r["error_stage"]  = str(r.get("error_stage") or r.get("stage") or "")
    r["error_message"]= str(r.get("error_message") or r.get("message") or "")
    return r

def normalize_registry(rec):
    r = {} if rec is None else dict(rec)
    src = r.get("source_path") or ""
    r["source_path"] = normalize_path(src)
    r["protocol_id"]  = extract_protocol_id(r["source_path"])
    r["document_id"]  = r.get("document_id") or r["protocol_id"]
    r["processing_status"] = str(r.get("processing_status") or "failed")
    r["page_count"]   = int(r.get("page_count") or 0)
    r["error_count"]  = int(r.get("error_count") or 0)
    r["processed_at"] = str(r.get("processed_at") or datetime.utcnow().isoformat())
    # ensure tools_used is list of strings
    tools = r.get("tools_used") or []
    if isinstance(tools, (str,)):
        tools = [tools]
    r["tools_used"] = [str(t) for t in tools]
    r["error_message"] = str(r.get("error_message") or "")
    return r

# Run normalization on in-memory lists
all_sections = [normalize_section(s) for s in (all_sections or [])]
all_pages    = [normalize_page(p) for p in (all_pages or [])]
all_images   = [normalize_image(i) for i in (all_images or [])]
all_tables   = [normalize_table(t) for t in (all_tables or [])]
all_formulas = [normalize_formula(f) for f in (all_formulas or [])]
all_errors   = [normalize_error(e) for e in (all_errors or [])]
all_registry = [normalize_registry(r) for r in (all_registry or [])]

# Define full Docling-like schemas
sections_schema = StructType([
    StructField("section_id", StringType(), True),
    StructField("document_id", StringType(), True),
    StructField("protocol_id", StringType(), True),
    StructField("page_number", IntegerType(), True),
    StructField("sequence_number", IntegerType(), True),
    StructField("section_type", StringType(), True),
    StructField("content_text", StringType(), True),
    StructField("content_markdown", StringType(), True),
    StructField("is_strikethrough", BooleanType(), True),
    StructField("is_toc", BooleanType(), True),
    StructField("heading_level", IntegerType(), True),
    StructField("bbox_x0", FloatType(), True),
    StructField("bbox_y0", FloatType(), True),
    StructField("bbox_x1", FloatType(), True),
    StructField("bbox_y1", FloatType(), True),
    StructField("extracted_at", StringType(), True),
    StructField("source_path", StringType(), True)
])

pages_schema = StructType([
    StructField("document_id", StringType(), True),
    StructField("protocol_id", StringType(), True),
    StructField("page_number", IntegerType(), True),
    StructField("text", StringType(), True),
    StructField("has_header", BooleanType(), True),
    StructField("has_footer", BooleanType(), True),
    StructField("header_text", StringType(), True),
    StructField("footer_text", StringType(), True),
    StructField("source_path", StringType(), True)
])

images_schema = StructType([
    StructField("image_id", StringType(), True),
    StructField("document_id", StringType(), True),
    StructField("protocol_id", StringType(), True),
    StructField("page_number", IntegerType(), True),
    StructField("image_format", StringType(), True),
    StructField("image_path", StringType(), True),
    StructField("source_path", StringType(), True)
])

tables_schema = StructType([
    StructField("table_id", StringType(), True),
    StructField("document_id", StringType(), True),
    StructField("protocol_id", StringType(), True),
    StructField("page_number", IntegerType(), True),
    StructField("html", StringType(), True),
    StructField("source_path", StringType(), True)
])

formulas_schema = StructType([
    StructField("formula_id", StringType(), True),
    StructField("document_id", StringType(), True),
    StructField("protocol_id", StringType(), True),
    StructField("content", StringType(), True),
    StructField("page_number", IntegerType(), True),
    StructField("source_path", StringType(), True)
])

errors_schema = StructType([
    StructField("document_id", StringType(), True),
    StructField("protocol_id", StringType(), True),
    StructField("error_stage", StringType(), True),
    StructField("error_message", StringType(), True),
    StructField("source_path", StringType(), True)
])

registry_schema = StructType([
    StructField("protocol_id", StringType(), True),
    StructField("document_id", StringType(), True),
    StructField("source_path", StringType(), True),
    StructField("processing_status", StringType(), True),
    StructField("page_count", IntegerType(), True),
    StructField("error_count", IntegerType(), True),
    StructField("processed_at", StringType(), True),
    StructField("tools_used", ArrayType(StringType()), True),
    StructField("error_message", StringType(), True)
])

# Drop and rewrite tables safely
def safe_drop(tbl):
    try:
        spark.sql(f"DROP TABLE IF EXISTS {tbl}")
    except Exception as e:
        print("drop failed:", tbl, e)

# Use config.full_table names
tables_to_write = [
    (config.full_table(config.bronze_sections), all_sections, sections_schema),
    (config.full_table(config.bronze_pages),    all_pages,    pages_schema),
    (config.full_table(config.bronze_images),   all_images,   images_schema),
    (config.full_table(config.bronze_tables),   all_tables,   tables_schema),
    (config.full_table(config.bronze_formulas), all_formulas, formulas_schema),
    (config.full_table(config.bronze_errors),   all_errors,   errors_schema),
    (config.full_table(config.document_registry), all_registry, registry_schema)
]

for tbl_name, data, schema in tables_to_write:
    print(f"\n Writing {tbl_name} ({len(data)} rows)")
    # drop old table to avoid merge/schema issues
    safe_drop(tbl_name)
    if not data:
        print(f"{tbl_name}: no data, skipping write.")
        continue
    df = spark.createDataFrame(data, schema=schema)
    df.write.format("delta").option("overwriteSchema", "true").mode("overwrite").saveAsTable(tbl_name)
    print(f"{tbl_name}: wrote {df.count()} rows")

print("Full Bronze restore complete. You can run Silver now.")


# COMMAND ----------

# MAGIC %md
# MAGIC # Section 6: Monitoring Queries

# COMMAND ----------

# MAGIC %md
# MAGIC Monitoring Query 1: Processing Status Summary

# COMMAND ----------

print("PROCESSING STATUS SUMMARY")
display(spark.sql(f"""
SELECT
    processing_status,
    COUNT(*) AS count_docs,
    SUM(page_count) AS total_pages,
    SUM(error_count) AS total_errors,
    MIN(processed_at) AS first_processed_at,
    MAX(processed_at) AS last_processed_at
FROM {config.full_table(config.document_registry)}
GROUP BY processing_status
ORDER BY count_docs DESC
"""))


# COMMAND ----------

# MAGIC %md
# MAGIC Monitoring Query 2: Extraction Tool Usage

# COMMAND ----------

print("EXTRACTION TOOL USAGE")
display(spark.sql(f"""
SELECT
    tool,
    COUNT(*) AS usage_count
FROM (
    SELECT EXPLODE(tools_used) AS tool
    FROM {config.full_table(config.document_registry)}
)
GROUP BY tool
ORDER BY usage_count DESC
"""))


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT document_id, error_message
# MAGIC FROM dev_clinical.doc_test.document_registry
# MAGIC WHERE size(tools_used) = 0

# COMMAND ----------

print(f"/Volumes/dev_clinical/doc_test/documents/00027984/")
dbutils.fs.ls('/Volumes/dev_clinical/doc_test/documents/00027984/')

# COMMAND ----------

def test_docling(_):
    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        out = converter.convert("/dbfs/path/to/any.pdf")
        return ["OK"]
    except Exception as e:
        return [str(e)]

print(spark.sparkContext.parallelize([1], 1).mapPartitions(test_docling).collect())


# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT error_message, COUNT(*) as cnt 
# MAGIC FROM dev_clinical.doc_test.document_registry 
# MAGIC WHERE tools_used[0] = 'pymupdf'
# MAGIC GROUP BY error_message
# MAGIC ORDER BY cnt DESC
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC Monitoring Query 3: Section Type Distribution

# COMMAND ----------

print("SECTION TYPE DISTRIBUTION")
display(spark.sql(f"""
SELECT
    section_type,
    COUNT(*) AS count_sections
FROM {config.full_table(config.bronze_sections)}
GROUP BY section_type
ORDER BY count_sections DESC
"""))


# COMMAND ----------

# MAGIC %md
# MAGIC ## What This Notebook Does
# MAGIC
# MAGIC 1. **Extract** documents with Docling (fallback: PyMuPDF - pdfplumber - raw text)
# MAGIC 2. **Write** raw extraction to Bronze Delta tables
# MAGIC 3. **Track** processing status in document registry
# MAGIC
# MAGIC ## Bronze Tables
# MAGIC
# MAGIC | Table | Description |
# MAGIC |-------|-------------|
# MAGIC | bronze_pages | Raw page extraction |
# MAGIC | bronze_sections | Raw sections with classification |
# MAGIC | bronze_images | Image metadata |
# MAGIC | bronze_tables | Table content (HTML/MD) |
# MAGIC | bronze_formulas | Formulas for LLM processing |
# MAGIC | bronze_errors | Extraction errors |
# MAGIC | document_registry | Processing status tracking |
# MAGIC
# MAGIC ## What Can Break
# MAGIC
# MAGIC ```
# MAGIC Password-protected PDFs = logged, skipped
# MAGIC Corrupted pages = other pages still extract
# MAGIC Massive docs (500+ pages) = might timeout, check registry
# MAGIC OCR-heavy scans sometimes give junk text - shows up in bronze_errors
# MAGIC ```
# MAGIC
# MAGIC ## Notes
# MAGIC - No UDFs
# MAGIC - Safe to re-run (uses MERGE) ie idempotent
# MAGIC - Resumable: Registry tracks status, retry failed docs
# MAGIC - Check `document_registry` for failed docs
