# Databricks notebook source
# MAGIC %md
# MAGIC # Clinical Document Intelligence Pipeline - Demo
# MAGIC 
# MAGIC This notebook demonstrates the end-to-end document processing pipeline:
# MAGIC 1. **Ingestion & Classification** - Extract and classify document sections
# MAGIC 2. **Entity Extraction (NER)** - Extract clinical entities from classified sections

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Architecture
# MAGIC 
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │           INGESTION & CLASSIFICATION PIPELINE               │
# MAGIC │                                                             │
# MAGIC │   BRONZE          SILVER              SILVER         GOLD   │
# MAGIC │  ┌───────┐      ┌─────────┐        ┌──────────┐   ┌───────┐│
# MAGIC │  │ Raw   │─────▶│Structure│───────▶│ Classify │──▶│Sections│
# MAGIC │  │ Docs  │      │Sections │        │  (109)   │   │ Table ││
# MAGIC │  └───────┘      └─────────┘        └──────────┘   └───┬───┘│
# MAGIC └───────────────────────────────────────────────────────┼────┘
# MAGIC                                                         │
# MAGIC                                                         ▼
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │              EXTRACTION NER PIPELINE                        │
# MAGIC │                                                             │
# MAGIC │   BRONZE              SILVER                    GOLD        │
# MAGIC │  ┌───────────┐      ┌─────────┐            ┌──────────┐    │
# MAGIC │  │Classified │─────▶│Apply NER│───────────▶│ Entities │    │
# MAGIC │  │ Sections  │      │ Model   │            │  Table   │    │
# MAGIC │  └───────────┘      └─────────┘            └──────────┘    │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# Configuration
CATALOG = "dev_clinical"
SCHEMA = "doc_intelligence"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. View Classified Sections (Gold Output from Classification Pipeline)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     document_id,
# MAGIC     section_id,
# MAGIC     classification,
# MAGIC     classification_confidence,
# MAGIC     LEFT(content_text, 100) as content_preview
# MAGIC FROM gold_classified_sections
# MAGIC ORDER BY classification_confidence DESC
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Classification Distribution

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     classification,
# MAGIC     COUNT(*) as count
# MAGIC FROM gold_classified_sections
# MAGIC GROUP BY classification
# MAGIC ORDER BY count DESC
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. View Extracted Entities (Gold Output from NER Pipeline)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     entity_id,
# MAGIC     document_id,
# MAGIC     entity_text,
# MAGIC     entity_type,
# MAGIC     section_type
# MAGIC FROM gold_clinical_entities
# MAGIC ORDER BY entity_type
# MAGIC LIMIT 30

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Entity Distribution by Type

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     entity_type,
# MAGIC     COUNT(*) as count
# MAGIC FROM gold_clinical_entities
# MAGIC GROUP BY entity_type
# MAGIC ORDER BY count DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Entities by Section Type

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     section_type,
# MAGIC     entity_type,
# MAGIC     COUNT(*) as count
# MAGIC FROM gold_clinical_entities
# MAGIC GROUP BY section_type, entity_type
# MAGIC ORDER BY count DESC
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Sample: Drug Entities with Dosages

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC     d.document_id,
# MAGIC     d.entity_text as drug,
# MAGIC     dos.entity_text as dosage
# MAGIC FROM gold_clinical_entities d
# MAGIC JOIN gold_clinical_entities dos 
# MAGIC     ON d.document_id = dos.document_id 
# MAGIC     AND d.section_id = dos.section_id
# MAGIC WHERE d.entity_type = 'DRUG' 
# MAGIC     AND dos.entity_type = 'DOSAGE'
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Performance Summary
# MAGIC 
# MAGIC ### Classification Model (SapBERT)
# MAGIC - **Categories:** 109 biomedical section types
# MAGIC - **Approach:** SapBERT embeddings + cosine similarity
# MAGIC - **Finding:** SapBERT outperforms dual-model approach by 1.3%
# MAGIC 
# MAGIC ### NER Model (Fine-tuned SapBERT)
# MAGIC - **Entity Types:** 8 (CONDITION, DRUG, DOSAGE, STUDY_PHASE, ENDPOINT, PATIENT_CRITERIA, BIOMARKER, ENDPOINT_TYPE)
# MAGIC - **F1 Score:** 74.1%
# MAGIC - **Training:** 100 annotated samples with class weighting

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline Execution
# MAGIC 
# MAGIC To run the full pipeline:
# MAGIC 
# MAGIC ```python
# MAGIC # Classification Pipeline
# MAGIC dbutils.notebook.run("../ingestion_classification/bronze/01_document_extraction", 3600)
# MAGIC dbutils.notebook.run("../ingestion_classification/silver/02_structure_sections", 3600)
# MAGIC dbutils.notebook.run("../ingestion_classification/silver/03a_embeddings", 3600)
# MAGIC dbutils.notebook.run("../ingestion_classification/silver/03b_classification", 3600)
# MAGIC 
# MAGIC # NER Pipeline
# MAGIC dbutils.notebook.run("../extraction_ner/silver/01_apply_ner", 3600)
# MAGIC dbutils.notebook.run("../extraction_ner/gold/02_write_entities", 3600)
# MAGIC ```
# MAGIC 
# MAGIC Or use the Databricks Workflows defined in `/databricks/workflows/`
