# Bronze Ingestion V2 - Clinical Document Processing Pipeline

**Author:** Nalini Panwar  
**Branch:** `feature/nalini-prototype`  
**Last Updated:** December 12, 2025

---

## Overview

End-to-end clinical protocol document processing pipeline for the Advarra Document Intelligence project. This pipeline extracts, structures, and classifies clinical trial protocol documents using a multi-tool extraction approach with biomedical NLP classification.

---

## Directory Structure

```
bronze_ingestion_v2/
├── pipeline/                          # Python wheel package
│   └── pipeline-0.0.10-py3-none-any.whl
├── init/                              # Cluster initialization scripts
│   └── install_doc_tools.sh           # Tesseract OCR, dependencies
├── 01_Bronze_Document_Extraction.py   # PDF extraction
├── 02_Bronze_Structure_Sections.py    # Document hierarchy
├── 03a_Bronze_Embeddings.py           # SapBERT embeddings
├── 03b_Bronze_Classification.py       # Section classification
├── 04_QA_Bronze_Validation.py         # Quality assurance
└── README.md
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DOCUMENT EXTRACTION (01)                             │
├─────────────────────────────────────────────────────────────────────────┤
│  Source: POC Flywheel (protocols.documents)                            │
│  Extraction Chain:                                                      │
│    Digital PDFs: Docling (300s timeout) → PyMuPDF → pdfplumber          │
│    Scanned PDFs: Tesseract OCR → PyMuPDF                               │
│  Output: bronze_documents, bronze_pages, bronze_sections               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    STRUCTURE PROCESSING (02)                            │
├─────────────────────────────────────────────────────────────────────────┤
│  Hierarchy Reconstruction:                                              │
│    - Parent-child relationships from heading levels                     │
│    - Document order sequencing                                         │
│    - Broken paragraph merging                                          │
│  Output: bronze_sections_structured                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    EMBEDDINGS GENERATION (03a)                          │
├─────────────────────────────────────────────────────────────────────────┤
│  Model: SapBERT (cambridgeltl/SapBERT-from-PubMedBERT-fulltext)        │
│  Embedding Types: Section title, Section content                       │
│  Output: bronze_section_embeddings                                     │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    SECTION CLASSIFICATION (03b)                         │
├─────────────────────────────────────────────────────────────────────────┤
│  Method: Cosine similarity against category prototypes                  │
│  Categories: 109 (from Domain_and_Insight_Requirements_Oct_2025.xlsx)  │
│  Domains: Demographics, Reproductive, Lifestyle, Vitals, Labs, etc.    │
│  Output: bronze_section_classifications                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    QUALITY ASSURANCE (04)                               │
├─────────────────────────────────────────────────────────────────────────┤
│  Validation: Embedding normalization, classification completeness       │
│  Status Update: DOCUMENT_PREPROCESSING_COMPLETE                        │
│  Output: document_registry (status update)                             │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Notebooks

| Notebook | Description | Estimated Runtime |
|----------|-------------|-------------------|
| `01_Bronze_Document_Extraction` | Extracts text, tables, and structure from PDF documents using Docling, PyMuPDF, and Tesseract OCR | 45-60 min |
| `02_Bronze_Structure_Sections` | Builds document hierarchy, assigns parent-child relationships, merges broken paragraphs | 30-45 min |
| `03a_Bronze_Embeddings` | Generates SapBERT embeddings for section titles and content | 15-20 min |
| `03b_Bronze_Classification` | Classifies sections into 109 categories using embedding similarity | 10-15 min |
| `04_QA_Bronze_Validation` | Validates processing completeness and updates document status | 2-5 min |

---

## Output Tables

All tables are stored in Unity Catalog: `dev_braid_nalini.doc_test`

| Table | Description | Key Columns |
|-------|-------------|-------------|
| `bronze_documents` | Document registry with metadata | document_id, protocol_id, processing_status |
| `bronze_pages` | Page-level content and images | document_id, page_number, content_text |
| `bronze_sections` | Raw extracted sections | section_id, document_id, content_text, heading_level |
| `bronze_sections_structured` | Sections with hierarchy | section_id, parent_id, doc_order, is_heading |
| `bronze_section_embeddings` | SapBERT vector embeddings | section_id, title_embedding, content_embedding |
| `bronze_section_classifications` | Category assignments | section_id, classification, confidence, domain |

---

## Category Taxonomy

109 categories organized into domains:

| Domain | Categories | Examples |
|--------|------------|----------|
| Demographics | 12 | GENDER, BMI_DISCRETE, WEIGHT_RANGE, ETHNICITY |
| Reproductive | 17 | PREGNANCY_TEST_PRE, CONTRACEPTION_BARRIER, BREASTFEEDING |
| Lifestyle | 13 | SMOKING_STATUS, ALCOHOL_LIMIT, SUBSTANCE_SCREENING |
| Measurements | 5 | UNITS_WEIGHT, UNITS_HEIGHT, UNITS_LAB |
| Informed Consent | 4 | INFORMED_CONSENT, CONSENT_CAPACITY, CONSENT_WITHDRAWAL |
| Vitals | 8 | TEMPERATURE, HEART_RATE, BLOOD_PRESSURE_SYSTOLIC |
| Clinical Labs | 28 | HEMOGLOBIN, PLATELET_COUNT, CREATININE_CLEARANCE |
| Assessments | 6 | ECOG, ECG, ECG_QT, RECIST |
| Document Structure | 16 | INCLUSION_CRITERIA, EXCLUSION_CRITERIA, STUDY_DESIGN |

Source: `Domain_and_Insight_Requirements_Oct_2025.xlsx`

---

## Dependencies

### Cluster Init Script (`init/install_doc_tools.sh`)

```bash
#!/bin/bash
# Install Tesseract OCR
apt-get update
apt-get install -y tesseract-ocr tesseract-ocr-eng

# Install Python dependencies
pip install pytesseract pdfplumber
```

### Pipeline Wheel (`pipeline/`)

```
pipeline-0.0.10-py3-none-any.whl
├── docling_extractor.py    # Multi-tool extraction with 300s timeout
└── __init__.py
```

Install in notebook:
```python
%pip install /Volumes/dev_braid_nalini/doc_test/packages/pipeline-0.0.10-py3-none-any.whl --force-reinstall
```

---

## Configuration

### Cluster Requirements

| Setting | Value |
|---------|-------|
| Runtime | DBR 14.3 LTS ML |
| Node Type | Standard_DS3_v2 (or equivalent) |
| Workers | 3-8 (recommended: 8 for production) |
| Init Script | `init/install_doc_tools.sh` |

### Docling Timeout

The extraction pipeline uses a 300-second timeout per document for Docling processing. Documents exceeding this threshold fall back to PyMuPDF.

```python
DOCLING_TIMEOUT_SECONDS = 300  # Configurable in pipeline wheel
```

---

## Usage

### Run Full Pipeline

Execute notebooks in order:

```
01_Bronze_Document_Extraction  →  02_Bronze_Structure_Sections  →  03a_Bronze_Embeddings  →  03b_Bronze_Classification  →  04_QA_Bronze_Validation
```

### Run as Databricks Workflow

Create a job with task dependencies:

1. Task 1: `01_Bronze_Document_Extraction`
2. Task 2: `02_Bronze_Structure_Sections` (depends on Task 1)
3. Task 3: `03a_Bronze_Embeddings` (depends on Task 2)
4. Task 4: `03b_Bronze_Classification` (depends on Task 3)
5. Task 5: `04_QA_Bronze_Validation` (depends on Task 4)

---

## References

- SapBERT: Self-Alignment Pretraining for Biomedical Entity Representations (Liu et al., 2021)
- Docling: Document conversion library (github.com/DS4SD/docling)
- Category taxonomy: Domain_and_Insight_Requirements_Oct_2025.xlsx

---

## Contact

Nalini Panwar  (nalini.panwar@advarra.com)
Lead - Technology  
Iris Software Inc.