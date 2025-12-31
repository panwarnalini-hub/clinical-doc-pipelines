# Training Pipeline

## Overview

Training is **NOT part of the medallion data path**. It produces model artifacts consumed by `extraction_ner/silver/01_apply_ner.py`.

## Purpose

Local development workflow for fine-tuning the clinical trial NER model. Training is performed on a local workstation for faster iteration and cost efficiency.

## Files

- `01_convert_annotations.py` - Convert Label Studio exports to training format
- `02_train_model.py` - Fine-tune SapBERT on annotated clinical trial data
- `03_test_local.py` - Validate trained model locally before Databricks deployment

## Output

**Model artifacts** saved to `models/sapbert_ner/final/`:
- `config.json`
- `pytorch_model.bin`
- `tokenizer_config.json`
- `id2label.json`

These artifacts are uploaded to Databricks for production inference at scale.

## Workflow

1. Annotate protocols in Label Studio (8 entity types, 91 protocols)
2. Export annotations and run `01_convert_annotations.py`
3. Train model with `02_train_model.py`
4. Validate locally with `03_test_local.py`
5. Upload trained model to Databricks `/dbfs/mnt/models/clinical_ner/`
6. Deploy via `extraction_ner/silver/01_apply_ner.py`

## Results

- **F1 Score:** 74.1%
- **Entity Types:** 8 (CONDITION, DRUG, DOSAGE, STUDY_PHASE, ENDPOINT, PATIENT_CRITERIA, BIOMARKER, ENDPOINT_TYPE)
- **Training Data:** 91 annotated clinical trial protocols
- **Base Model:** SapBERT (cambridgeltl/SapBERT-from-PubMedBERT-fulltext)

## Tests

See `tests/` directory for API exploration and model validation scripts.
