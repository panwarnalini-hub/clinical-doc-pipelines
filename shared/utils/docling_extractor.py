# docling_extractor.py v0.0.10
"""
Document extraction pipeline:
- DIGITAL PDFs: Docling → PyMuPDF → pdfplumber → raw_text
- SCANNED PDFs: Tesseract → PyMuPDF (with image extraction)

Designed for Databricks workers with read-only site-packages.
"""

import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import fitz  # PyMuPDF


def is_scanned_pdf(path: str, sample_pages: int = 3) -> bool:
    """
    Detect if PDF is scanned (images only, minimal text).
    Samples first few pages to decide.
    """
    try:
        doc = fitz.open(path)
        if len(doc) == 0:
            doc.close()
            return False
        
        pages_to_check = min(sample_pages, len(doc))
        total_text_chars = 0
        total_images = 0
        
        for i in range(pages_to_check):
            page = doc[i]
            text = page.get_text().strip()
            images = page.get_images()
            total_text_chars += len(text)
            total_images += len(images)
        
        doc.close()
        
        # If very little text but has images, likely scanned
        avg_chars_per_page = total_text_chars / pages_to_check
        return avg_chars_per_page < 100 and total_images > 0
        
    except Exception:
        return False


class DoclingExtractor:
    """
    Extract DIGITAL PDFs using Docling.
    OCR is disabled to avoid RapidOCR read-only filesystem issues.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.converter = None
        self._init_error = None
        
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat
            
            # Configure for DIGITAL PDFs only - no OCR
            pipeline_options = PdfPipelineOptions()
            pipeline_options.images_scale = 2.0
            pipeline_options.generate_page_images = True
            pipeline_options.generate_picture_images = True
            pipeline_options.do_ocr = False  # Disabled for digital PDFs
            pipeline_options.do_table_structure = True
            
            self.converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        except Exception as e:
            self._init_error = f"Docling init failed: {str(e)}"

    def extract(self, path: str, doc_id: str) -> dict:
        """Extract document and return structured data."""
        
        if self._init_error:
            return {
                "success": False,
                "error": self._init_error,
                "pages": [], "sections": [], "images": [], 
                "tables": [], "formulas": [], "errors": [],
                "tools_used": ["docling_init_failed"],
            }
        
        try:
            from docling_core.types.doc.document import (
                DoclingDocument, TableItem, PictureItem,
                SectionHeaderItem
            )
            
            doc_output_dir = os.path.join(self.output_dir, doc_id)
            os.makedirs(doc_output_dir, exist_ok=True)
            
            conv_result = self.converter.convert(path)
            doc: DoclingDocument = conv_result.document
            
            now = datetime.utcnow().isoformat()
            pages, sections, images, tables, formulas, errors = [], [], [], [], [], []
            section_seq = image_seq = table_seq = 0
            
            # === PAGES ===
            for page_no, page in doc.pages.items():
                page_text = ""
                for item, level in doc.iterate_items():
                    if hasattr(item, 'prov') and item.prov:
                        for prov in item.prov:
                            if prov.page_no == page_no:
                                text = getattr(item, 'text', '') or ''
                                if text:
                                    page_text += text + "\n"
                                break
                
                pages.append({
                    "document_id": doc_id,
                    "protocol_id": doc_id,
                    "page_number": int(page_no),
                    "text": page_text,
                    "source_path": path,
                })
            
            # === SECTIONS ===
            for item, level in doc.iterate_items():
                if isinstance(item, (TableItem, PictureItem)):
                    continue
                
                text = getattr(item, 'text', '') or ''
                if not text.strip():
                    continue
                
                section_seq += 1
                
                if isinstance(item, SectionHeaderItem):
                    section_type = "heading"
                    heading_level = getattr(item, 'level', 1)
                elif hasattr(item, 'label'):
                    section_type = self._map_label(item.label)
                    heading_level = None
                else:
                    section_type = "paragraph"
                    heading_level = None
                
                page_no = 0
                bbox = [None, None, None, None]
                if hasattr(item, 'prov') and item.prov:
                    prov = item.prov[0]
                    page_no = prov.page_no
                    if prov.bbox:
                        bbox = [prov.bbox.l, prov.bbox.t, prov.bbox.r, prov.bbox.b]
                
                sections.append({
                    "section_id": f"{doc_id}_s{section_seq:05d}",
                    "document_id": doc_id,
                    "protocol_id": doc_id,
                    "page_number": int(page_no),
                    "sequence_number": section_seq,
                    "section_type": section_type,
                    "content_text": text,
                    "heading_level": heading_level,
                    "bbox_x0": bbox[0],
                    "bbox_y0": bbox[1],
                    "bbox_x1": bbox[2],
                    "bbox_y1": bbox[3],
                    "extracted_at": now,
                    "source_path": path,
                })
            
            # === IMAGES ===
            for pic_item in doc.pictures:
                image_seq += 1
                page_no = pic_item.prov[0].page_no if pic_item.prov else 0
                
                file_path = None
                width = height = None
                
                if pic_item.image and pic_item.image.pil_image:
                    pil_img = pic_item.image.pil_image
                    width, height = pil_img.size
                    img_filename = f"p{page_no:03d}_img{image_seq:03d}.png"
                    img_path = os.path.join(doc_output_dir, img_filename)
                    pil_img.save(img_path, "PNG")
                    file_path = img_path
                
                images.append({
                    "image_id": f"{doc_id}_p{page_no:03d}_img{image_seq:03d}",
                    "document_id": doc_id,
                    "protocol_id": doc_id,
                    "page_number": int(page_no),
                    "image_format": "PNG",
                    "image_path": file_path,
                    "width": width,
                    "height": height,
                    "source_path": path,
                })
            
            # === TABLES ===
            for table_item in doc.tables:
                table_seq += 1
                page_no = table_item.prov[0].page_no if table_item.prov else 0
                
                html_content = md_content = ""
                row_count = col_count = 0
                
                if table_item.data:
                    row_count = table_item.data.num_rows
                    col_count = table_item.data.num_cols
                
                try:
                    html_content = table_item.export_to_html() if hasattr(table_item, 'export_to_html') else ""
                except:
                    pass
                try:
                    md_content = table_item.export_to_markdown() if hasattr(table_item, 'export_to_markdown') else ""
                except:
                    pass
                
                tables.append({
                    "table_id": f"{doc_id}_p{page_no:03d}_tbl{table_seq:03d}",
                    "document_id": doc_id,
                    "protocol_id": doc_id,
                    "page_number": int(page_no),
                    "html": html_content,
                    "markdown": md_content,
                    "row_count": row_count,
                    "column_count": col_count,
                    "source_path": path,
                })
            
            # Save exports
            md_path = os.path.join(doc_output_dir, f"{doc_id}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(doc.export_to_markdown())
            
            return {
                "success": True,
                "pages": pages,
                "sections": sections,
                "images": images,
                "tables": tables,
                "formulas": formulas,
                "errors": errors,
                "tools_used": ["docling"],
                "output_path": doc_output_dir,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()[:1500],
                "pages": [], "sections": [], "images": [],
                "tables": [], "formulas": [], "errors": [],
                "tools_used": ["docling_failed"],
            }

    def _map_label(self, label) -> str:
        if label is None:
            return "paragraph"
        label_str = label.value if hasattr(label, 'value') else str(label)
        mapping = {
            "title": "title", "document_index": "toc", "section_header": "heading",
            "paragraph": "paragraph", "text": "paragraph", "list_item": "list_item",
            "caption": "caption", "formula": "equation", "footnote": "footnote",
        }
        return mapping.get(label_str.lower(), "paragraph")


def _extract_with_tesseract(source_path: str, doc_id: str, output_dir: str) -> dict:
    """
    Extract SCANNED PDF using Tesseract OCR.
    Fallback for scanned documents.
    """
    try:
        import pytesseract
        from PIL import Image
        
        doc_output_dir = os.path.join(output_dir, doc_id)
        os.makedirs(doc_output_dir, exist_ok=True)
        
        pdf = fitz.open(source_path)
        pages, sections, images = [], [], []
        section_seq = image_seq = 0
        now = datetime.utcnow().isoformat()
        
        for page_idx in range(len(pdf)):
            page = pdf[page_idx]
            page_num = page_idx + 1
            
            # Render page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # OCR the image
            text = pytesseract.image_to_string(img)
            
            pages.append({
                "document_id": doc_id,
                "protocol_id": doc_id,
                "page_number": page_num,
                "text": text,
                "source_path": source_path,
            })
            
            # Save page image
            image_seq += 1
            img_filename = f"p{page_num:03d}_page.png"
            img_path = os.path.join(doc_output_dir, img_filename)
            img.save(img_path, "PNG")
            
            images.append({
                "image_id": f"{doc_id}_p{page_num:03d}_img{image_seq:03d}",
                "document_id": doc_id,
                "protocol_id": doc_id,
                "page_number": page_num,
                "image_format": "PNG",
                "image_path": img_path,
                "width": pix.width,
                "height": pix.height,
                "source_path": source_path,
            })
            
            # Create section from OCR text
            if text.strip():
                section_seq += 1
                sections.append({
                    "section_id": f"{doc_id}_s{section_seq:05d}",
                    "document_id": doc_id,
                    "protocol_id": doc_id,
                    "page_number": page_num,
                    "sequence_number": section_seq,
                    "section_type": "ocr_text",
                    "content_text": text,
                    "extracted_at": now,
                    "source_path": source_path,
                })
        
        pdf.close()
        
        return {
            "success": True,
            "pages": pages,
            "sections": sections,
            "images": images,
            "tables": [],
            "formulas": [],
            "errors": [],
            "tools_used": ["tesseract"],
        }
        
    except ImportError:
        return {"success": False, "error": "pytesseract not installed", "tools_used": ["tesseract_missing"]}
    except Exception as e:
        return {"success": False, "error": str(e), "tools_used": ["tesseract_failed"]}


def _extract_with_pymupdf(source_path: str, doc_id: str, output_dir: str) -> dict:
    """
    Extract using PyMuPDF - fast fallback for digital PDFs.
    Also extracts images.
    """
    try:
        doc_output_dir = os.path.join(output_dir, doc_id)
        os.makedirs(doc_output_dir, exist_ok=True)
        
        pdf = fitz.open(source_path)
        pages, sections, images = [], [], []
        section_seq = image_seq = 0
        now = datetime.utcnow().isoformat()
        
        for page_idx in range(len(pdf)):
            page = pdf[page_idx]
            page_num = page_idx + 1
            page_text = page.get_text("text") or ""
            
            pages.append({
                "document_id": doc_id,
                "protocol_id": doc_id,
                "page_number": page_num,
                "text": page_text,
                "source_path": source_path,
            })
            
            # Extract images
            for img_idx, img in enumerate(page.get_images()):
                try:
                    image_seq += 1
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    image_bytes = base_image["image"]
                    img_ext = base_image.get("ext", "png")
                    
                    img_filename = f"p{page_num:03d}_img{image_seq:03d}.{img_ext}"
                    img_path = os.path.join(doc_output_dir, img_filename)
                    with open(img_path, "wb") as f:
                        f.write(image_bytes)
                    
                    images.append({
                        "image_id": f"{doc_id}_p{page_num:03d}_img{image_seq:03d}",
                        "document_id": doc_id,
                        "protocol_id": doc_id,
                        "page_number": page_num,
                        "image_format": img_ext.upper(),
                        "image_path": img_path,
                        "width": base_image.get("width"),
                        "height": base_image.get("height"),
                        "source_path": source_path,
                    })
                except:
                    pass
            
            # Extract sections from text blocks
            blocks = page.get_text("dict").get("blocks", [])
            for block in blocks:
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        text = " ".join([span["text"] for span in line.get("spans", [])])
                        if text.strip():
                            section_seq += 1
                            sections.append({
                                "section_id": f"{doc_id}_s{section_seq:05d}",
                                "document_id": doc_id,
                                "protocol_id": doc_id,
                                "page_number": page_num,
                                "sequence_number": section_seq,
                                "section_type": "paragraph",
                                "content_text": text,
                                "extracted_at": now,
                                "source_path": source_path,
                            })
        
        pdf.close()
        
        return {
            "success": True,
            "pages": pages,
            "sections": sections,
            "images": images,
            "tables": [],
            "formulas": [],
            "errors": [],
            "tools_used": ["pymupdf"],
        }
        
    except Exception as e:
        return {"success": False, "error": str(e), "tools_used": ["pymupdf_failed"]}


def _extract_with_pdfplumber(source_path: str, doc_id: str) -> dict:
    """Extract using pdfplumber - good for tables."""
    try:
        import pdfplumber
        
        pages, tables = [], []
        table_seq = 0
        
        with pdfplumber.open(source_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                
                pages.append({
                    "document_id": doc_id,
                    "protocol_id": doc_id,
                    "page_number": page_num,
                    "text": page.extract_text() or "",
                    "source_path": source_path,
                })
                
                # Extract tables
                for tbl in page.extract_tables() or []:
                    if tbl:
                        table_seq += 1
                        tables.append({
                            "table_id": f"{doc_id}_p{page_num:03d}_tbl{table_seq:03d}",
                            "document_id": doc_id,
                            "protocol_id": doc_id,
                            "page_number": page_num,
                            "row_count": len(tbl),
                            "column_count": len(tbl[0]) if tbl else 0,
                            "source_path": source_path,
                        })
        
        return {
            "success": True,
            "pages": pages,
            "sections": [],
            "images": [],
            "tables": tables,
            "formulas": [],
            "errors": [],
            "tools_used": ["pdfplumber"],
        }
        
    except Exception as e:
        return {"success": False, "error": str(e), "tools_used": ["pdfplumber_failed"]}


def extract_single_document(
    input_path: str = None,
    output_dir: str = None,
    path: str = None,
    output_volume: str = None,
    document_id: str = None,
) -> dict:
    """
    Extract a single document with appropriate method based on type.
    
    DIGITAL PDFs: Docling → PyMuPDF → pdfplumber → raw_text
    SCANNED PDFs: Tesseract → PyMuPDF
    
    Args:
        input_path: Path to source PDF
        output_dir: Output directory
        document_id: Explicit document ID (recommended)
    """
    source_path = input_path or path
    output_path = output_dir or output_volume
    
    if not source_path:
        raise ValueError("Must provide input_path or path")
    if not output_path:
        raise ValueError("Must provide output_dir or output_volume")

    # Determine doc_id
    if document_id:
        doc_id = document_id
    else:
        doc_id = os.path.basename(os.path.dirname(source_path))
        if not doc_id or doc_id in ("tmp", "uploads", "documents", ""):
            doc_id = os.path.splitext(os.path.basename(source_path))[0]
    
    def _safe(o):
        if o is None:
            return None
        if isinstance(o, (str, int, float, bool)):
            return o
        if isinstance(o, dict):
            return {str(k): _safe(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_safe(x) for x in o]
        return str(o)

    record = {
        "protocol_id": doc_id,
        "document_id": doc_id,
        "source_path": source_path,
        "processing_status": "failed",
        "page_count": 0,
        "error_count": 0,
        "processed_at": datetime.utcnow().isoformat(),
        "tools_used": [],
        "error_message": "",
    }

    # === DETECT SCANNED vs DIGITAL ===
    scanned = is_scanned_pdf(source_path)
    
    if scanned:
        # === SCANNED PDF CHAIN: Tesseract → PyMuPDF ===
        
        # Try Tesseract first
        result = _extract_with_tesseract(source_path, doc_id, output_path)
        if result.get("success"):
            record.update({
                "processing_status": "success",
                "page_count": len(result.get("pages", [])),
                "tools_used": result.get("tools_used", ["tesseract"]),
            })
            return {"registry": _safe(record), **{k: _safe(v) for k, v in result.items() if k != "success"}}
        
        # Fallback to PyMuPDF (will just get images, minimal text)
        result = _extract_with_pymupdf(source_path, doc_id, output_path)
        if result.get("success"):
            record.update({
                "processing_status": "success",
                "page_count": len(result.get("pages", [])),
                "tools_used": result.get("tools_used", ["pymupdf"]),
                "error_message": "Scanned PDF - Tesseract failed, using PyMuPDF",
            })
            return {"registry": _safe(record), **{k: _safe(v) for k, v in result.items() if k != "success"}}
    
    else:
        # === DIGITAL PDF CHAIN: Docling → PyMuPDF → pdfplumber → raw_text ===
        
        # Try Docling first (with HARD KILL timeout using multiprocessing)
        try:
            import multiprocessing
            
            DOCLING_TIMEOUT_SECONDS = 90  # Enterprise standard: 90s per document
            
            def _run_docling_worker(result_queue, src_path, out_path, d_id):
                """Worker function that runs in separate process."""
                try:
                    extractor = DoclingExtractor(out_path)
                    result = extractor.extract(src_path, d_id)
                    result_queue.put(result)
                except Exception as e:
                    result_queue.put({"success": False, "error": str(e)})
            
            # Use multiprocessing for hard kill capability
            result_queue = multiprocessing.Queue()
            process = multiprocessing.Process(
                target=_run_docling_worker,
                args=(result_queue, source_path, output_path, doc_id)
            )
            process.start()
            process.join(timeout=DOCLING_TIMEOUT_SECONDS)
            
            if process.is_alive():
                # HARD KILL - terminate the hung process
                process.terminate()
                process.join(timeout=5)  # Give it 5s to terminate gracefully
                if process.is_alive():
                    process.kill()  # Force kill if still alive
                    process.join()
                result = {"success": False, "error": f"Docling timeout after {DOCLING_TIMEOUT_SECONDS}s (terminated)"}
            else:
                # Process completed - get result
                try:
                    result = result_queue.get_nowait()
                except:
                    result = {"success": False, "error": "Docling process completed but no result returned"}
            
            if result.get("success"):
                record.update({
                    "processing_status": "success",
                    "page_count": len(result.get("pages", [])),
                    "tools_used": result.get("tools_used", ["docling"]),
                })
                return {"registry": _safe(record), **{k: _safe(v) for k, v in result.items() if k != "success"}}
            else:
                record["error_message"] = f"Docling: {result.get('error', 'unknown')}"
        except Exception as e:
            record["error_message"] = f"Docling exception: {str(e)}"
        
        # Fallback to PyMuPDF
        result = _extract_with_pymupdf(source_path, doc_id, output_path)
        if result.get("success"):
            record.update({
                "processing_status": "success",
                "page_count": len(result.get("pages", [])),
                "tools_used": result.get("tools_used", ["pymupdf"]),
            })
            return {"registry": _safe(record), **{k: _safe(v) for k, v in result.items() if k != "success"}}
        
        # Fallback to pdfplumber
        result = _extract_with_pdfplumber(source_path, doc_id)
        if result.get("success"):
            record.update({
                "processing_status": "success",
                "page_count": len(result.get("pages", [])),
                "tools_used": result.get("tools_used", ["pdfplumber"]),
            })
            return {"registry": _safe(record), **{k: _safe(v) for k, v in result.items() if k != "success"}}
        
        # Last resort: raw text
        try:
            with open(source_path, "rb") as f:
                raw = f.read().decode("latin-1", "ignore")
            
            record.update({
                "processing_status": "success",
                "page_count": 1,
                "tools_used": ["raw_text"],
            })
            return {
                "registry": _safe(record),
                "pages": [{"document_id": doc_id, "protocol_id": doc_id, "page_number": 1, "text": raw}],
                "sections": [], "images": [], "tables": [], "formulas": [], "errors": [],
            }
        except:
            pass

    # Total failure
    record["error_message"] = "All extraction methods failed"
    return {
        "registry": _safe(record),
        "pages": [], "sections": [], "images": [], "tables": [], "formulas": [],
        "errors": [{"error_type": "total_failure", "message": record["error_message"]}],
    }
