# Migration Guide: Reorganizing Your Files

## Step 1: Create the new structure

```powershell
cd C:\Users\Admin\dev
mkdir clinical-doc-pipelines
cd clinical-doc-pipelines
python setup_structure.py
```

## Step 2: Copy files from ingestion_classification (Advarra work)

```powershell
# Bronze
copy ..\path-to-advarra\01_Document_Extraction_to_Bronze.py ingestion_classification\bronze\01_document_extraction.py

# Silver
copy ..\path-to-advarra\02_Structure_to_Silver_Sections.py ingestion_classification\silver\02_structure_sections.py
copy ..\path-to-advarra\03a_Silver_Embeddings.py ingestion_classification\silver\03a_embeddings.py
copy ..\path-to-advarra\03b_Silver_Classification_onlySapBert.py ingestion_classification\silver\03b_classification.py

# Tests
copy ..\path-to-advarra\Tests.py ingestion_classification\tests\test_classification.py

# Model comparison report
copy ..\path-to-advarra\Model_Comparison_Report.pdf ingestion_classification\models\
```

## Step 3: Copy files from extraction_ner (your NER project)

```powershell
# From clinical_ner_pipeline to new structure

# Training (NOT in medallion path)
copy ..\clinical_ner_pipeline\scripts\01_convert_labelstudio_to_ner.py extraction_ner\training\01_convert_annotations.py
copy ..\clinical_ner_pipeline\scripts\02_train_sapbert_ner.py extraction_ner\training\02_train_model.py

# Silver (inference)
# Create new file: 01_apply_ner.py (inference on classified sections)

# Gold
copy ..\clinical_ner_pipeline\scripts\03_inference.py extraction_ner\gold\02_write_entities.py

# Data
xcopy ..\clinical_ner_pipeline\data\* extraction_ner\data\ /E

# Models
xcopy ..\clinical_ner_pipeline\models\* extraction_ner\models\ /E

# Outputs
copy ..\clinical_ner_pipeline\outputs\ner_visualization.html extraction_ner\outputs\

# Tests
copy ..\clinical_ner_pipeline\tests\test_api.py extraction_ner\tests\
```

## Step 4: Copy shared files

```powershell
# Visualization script can go to demo
copy ..\clinical_ner_pipeline\scripts\04_visualize.py extraction_ner\outputs\visualize.py

# Streamlit app
copy ..\clinical_ner_pipeline\app.py app.py
```

## Step 5: Create the Silver NER inference script

You need a NEW script that applies NER to classification output.
This bridges the two pipelines.

`extraction_ner/silver/01_apply_ner.py` should:
1. Read from `gold_classified_sections` (classification pipeline output)
2. Apply NER model to each section
3. Write structured entities to silver table

## File Mapping Summary

| Old Location | New Location |
|--------------|--------------|
| **Advarra Pipeline** | |
| 01_Document_Extraction_to_Bronze.py | ingestion_classification/bronze/01_document_extraction.py |
| 02_Structure_to_Silver_Sections.py | ingestion_classification/silver/02_structure_sections.py |
| 03a_Silver_Embeddings.py | ingestion_classification/silver/03a_embeddings.py |
| 03b_Silver_Classification_onlySapBert.py | ingestion_classification/silver/03b_classification.py |
| Tests.py | ingestion_classification/tests/test_classification.py |
| Model_Comparison_Report.pdf | ingestion_classification/models/ |
| **NER Pipeline** | |
| 01_convert_labelstudio_to_ner.py | extraction_ner/training/01_convert_annotations.py |
| 02_train_sapbert_ner.py | extraction_ner/training/02_train_model.py |
| 03_inference.py | extraction_ner/gold/02_write_entities.py |
| 04_visualize.py | extraction_ner/outputs/visualize.py |
| test_api.py | extraction_ner/tests/test_api.py |
| data/* | extraction_ner/data/* |
| models/* | extraction_ner/models/* |
| app.py | app.py (root) |

## What This Communicates

When someone opens this repo, they see:
- ✅ Two independent, composable pipelines
- ✅ Medallion architecture used correctly
- ✅ ML training treated as artifact generation, not data transformation
- ✅ Clear data contracts between stages
- ✅ Databricks as orchestrator

This is senior-level data engineering structure.
