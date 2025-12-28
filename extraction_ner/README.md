# Extraction NER Pipeline

Named Entity Recognition for clinical trial protocols. Extracts 8 entity types from classified document sections.

## Pipeline Architecture

```
bronze_from_classification/     # Input from classification pipeline
        │
        ▼
    silver/
    01_apply_ner.py            # Apply trained NER model
        │
        ▼
     gold/
    02_write_entities.py       # Write gold_clinical_entities table
```

## Entity Types

| Entity | Description | Example |
|--------|-------------|---------|
| CONDITION | Disease or medical condition | metastatic breast cancer |
| DRUG | Medication or therapeutic agent | Pembrolizumab |
| DOSAGE | Drug dosage and administration | 200mg IV every 3 weeks |
| STUDY_PHASE | Clinical trial phase | Phase III |
| ENDPOINT | Outcome measure | Overall Survival, PFS |
| PATIENT_CRITERIA | Inclusion/exclusion criteria | Age >= 18 years |
| BIOMARKER | Biological marker | HER2, PD-L1 |
| ENDPOINT_TYPE | Primary or secondary | Primary endpoint |

## Model

- **Base:** SapBERT (cambridgeltl/SapBERT-from-PubMedBERT-fulltext)
- **Training:** 100 annotated samples from ClinicalTrials.gov
- **Technique:** Class weighting for imbalanced data
- **F1 Score:** 74.1%

## Structure

```
extraction_ner/
├── bronze_from_classification/   # Consumes classification output
├── silver/
│   └── 01_apply_ner.py          # Apply NER to sections
├── gold/
│   └── 02_write_entities.py     # Write final entities table
├── training/                     # Model training (not in medallion DAG)
│   ├── 01_convert_annotations.py
│   └── 02_train_model.py
├── models/
│   └── sapbert_ner/final/       # Trained model weights
├── data/
│   ├── raw/                     # ClinicalTrials.gov API responses
│   ├── annotations/             # Label Studio export
│   └── processed/               # BIO-tagged training data
├── outputs/
│   └── ner_visualization.html   # Results visualization
└── tests/
```

## Usage

### Training (one-time)
```bash
cd training
python 01_convert_annotations.py
python 02_train_model.py
```

### Inference
```bash
python silver/01_apply_ner.py
python gold/02_write_entities.py
```

### Visualization
```bash
python scripts/04_visualize.py
# Open outputs/ner_visualization.html
```

## Input/Output

**Input:** `gold_classified_sections` from ingestion_classification pipeline

**Output:** `gold_clinical_entities` table with schema:
- entity_id
- document_id
- section_id
- entity_text
- entity_type
- start_position
- end_position

## Author

Nalini Panwar  
December 2025
