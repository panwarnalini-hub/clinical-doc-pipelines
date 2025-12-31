# Bronze: Input from Classification Pipeline

## Overview

This Bronze layer does **not perform data ingestion**. It directly consumes the Gold output from the classification pipeline as its input.

## Data Source

**Input Table:** `dev_clinical.doc_test.gold_classified_sections`

This table is produced by `ingestion_classification/gold/04_gold_classified_sections.py`

## Why No Bronze Code?

The classification pipeline's Gold layer already provides NER-ready data with:
- High-confidence classified sections
- NER processing hints (priority, expected entity types)
- Protocol-level context
- Quality filters applied

**No additional transformation needed** - the Silver NER layer (`extraction_ner/silver/01_apply_ner.py`) reads this table directly.

## Pipeline Flow

```
ingestion_classification/gold/
  └── gold_classified_sections (Delta table)
        |
        | (consumed directly as Bronze input)
        |
extraction_ner/silver/
  └── 01_apply_ner.py (reads gold_classified_sections)
```

## Schema

The input table contains:

```python
protocol_id: string
section_id: string
classification: string               # e.g., "INCLUSION_CRITERIA"
classification_confidence: double
section_text: string                 # Text for NER processing
ner_priority: string                 # HIGH/MEDIUM/STANDARD
expected_entity_types: array<string> # Hint for NER models
ready_for_ner: boolean              # Quality flag
...
```

## Usage in Silver Layer

```python
# In extraction_ner/silver/01_apply_ner.py:

# Read directly from classification Gold
classified_sections = spark.table("dev_clinical.doc_test.gold_classified_sections")

# Filter for NER processing
ner_input = classified_sections.filter(col("ready_for_ner") == True)

# Apply NER model
...
```

## Design Principle

**Composable pipelines** - each pipeline's Gold output can serve as another pipeline's Bronze input, avoiding duplicate ingestion and maintaining data lineage.
