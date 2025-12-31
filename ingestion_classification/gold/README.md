# Gold Layer: NER-Ready Classified Sections

## Purpose

This Gold layer transforms classified sections from Silver into an analytics-ready format optimized for consumption by the downstream **NER/Entity Extraction pipeline** (`extraction_ner`).

## Pipeline Flow

```
Classification Pipeline:
  Bronze (Raw Documents)
    - Silver (Classified Sections)
    - Gold (NER-Ready Sections) ← YOU ARE HERE
         |
         | (consumed as Bronze input)
         |
  NER/Extraction Pipeline:
    Bronze (from classification/gold)
    - Silver (Extracted Entities)
    - Gold (Entity Analytics)
```

## What This Layer Does

### Input
- **Source:** `dev_clinical.doc_test.silver_section_classifications`
- **Content:** Sections classified into 87 biomedical categories initially (it will increase as I add more)

### Transformations
1. **Quality Filtering**
   - Keep only high-confidence classifications (≥72%)
   - Filter out very short sections (<10 characters)
   - Remove ambiguous/low-quality classifications

2. **NER Processing Hints**
   - Priority flags (HIGH/MEDIUM/STANDARD) based on category importance
   - Expected entity types per category (LAB_TEST, MEDICATION, CONDITION, etc.)
   - Section complexity indicators (SHORT/MEDIUM/LONG)
   - Batch keys for efficient parallel processing

3. **Protocol Enrichment**
   - Add protocol-level statistics (total sections, category diversity)
   - Include document context for entity resolution
   - Metadata for tracking and debugging

4. **Partitioning**
   - Partition by `category_domain` and `ner_priority`
   - Optimized for selective NER model application
   - Efficient filtering by entity type

### Output
- **Table:** `dev_clinical.doc_test.gold_classified_sections`
- **Format:** Delta Lake (partitioned)
- **Schema:**
  ```
  protocol_id: string
  section_id: string
  classification: string (e.g., "INCLUSION_CRITERIA")
  classification_confidence: double
  section_text: string
  ner_priority: string (HIGH/MEDIUM/STANDARD)
  expected_entity_types: array<string>
  ner_batch_key: string
  ready_for_ner: boolean
  ...
  ```

## Usage

### For NER Pipeline Consumption

```python
# Read from classification/gold as Bronze input for NER
classified_sections = spark.table("dev_clinical.doc_test.gold_classified_sections")

# Filter by priority for selective processing
high_priority = classified_sections.filter(col("ner_priority") == "HIGH")

# Process by category domain for specialized NER models
lab_sections = classified_sections.filter(col("category_domain") == "CLINICAL_LABS")
```

### Quality Metrics

- **Section Count:** ~8,500+ classified sections
- **Confidence Threshold:** ≥72%
- **Partition Strategy:** category_domain × ner_priority
- **Coverage:** 87 biomedical categories across all protocols

## Key Features

1. **NER-Optimized Schema**
   - Pre-classified sections reduce NER model scope
   - Expected entity types guide model selection
   - Priority flags enable phased processing

2. **Quality Assurance**
   - Only high-confidence classifications included
   - Protocol-level context for entity resolution
   - Processing metadata for debugging

3. **Performance Optimization**
   - Partitioned by domain for parallel processing
   - Batch keys for efficient job scheduling
   - Text length/complexity for resource allocation

## Files in This Directory

- **`04_gold_classified_sections.py`** - Main Gold layer transformation notebook

## Next Steps

After this Gold layer is created, the NER pipeline consumes it as Bronze input:
```
extraction_ner/
  ├── bronze_from_classification/  - Reads this Gold table
  ├── silver_entities/
  └── gold_entity_analytics/
```

## Example Queries

```sql
-- High-priority sections for immediate NER processing
SELECT protocol_id, classification, section_text, expected_entity_types
FROM dev_clinical.doc_test.gold_classified_sections
WHERE ner_priority = 'HIGH'
  AND category_domain IN ('CLINICAL_LABS', 'DEMOGRAPHICS')

-- Sections by category for specialized NER models
SELECT classification, COUNT(*) as section_count, AVG(text_length) as avg_length
FROM dev_clinical.doc_test.gold_classified_sections
GROUP BY classification
ORDER BY section_count DESC

-- Protocol coverage summary
SELECT protocol_id, 
       protocol_total_sections,
       protocol_unique_categories,
       protocol_avg_confidence
FROM dev_clinical.doc_test.gold_classified_sections
GROUP BY protocol_id, protocol_total_sections, protocol_unique_categories, protocol_avg_confidence
```

## Integration with NER Pipeline

This Gold layer serves as the **Bronze input** for the NER extraction pipeline:

```python
# In extraction_ner/bronze_from_classification/README.md:
# "This Bronze layer reads from ingestion_classification/gold"
```

The separation allows:
- **Classification pipeline** focuses on category assignment
- **NER pipeline** focuses on entity extraction
- Clean interface via Gold - Bronze handoff
- Independent scaling and optimization of each pipeline