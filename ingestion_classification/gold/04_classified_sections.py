# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: NER-Ready Classified Sections
# MAGIC 
# MAGIC **Purpose:** Transform classified sections from Silver into analytics-ready format optimized for downstream NER/entity extraction pipeline.
# MAGIC 
# MAGIC **Input:** `silver_section_classifications` (from 03b_classification)  
# MAGIC **Output:** `gold_classified_sections` → consumed as Bronze by extraction_ner pipeline
# MAGIC 
# MAGIC **Key Transformations:**
# MAGIC - Filter to high-confidence classifications only
# MAGIC - Enrich with protocol-level metadata
# MAGIC - Prepare text for NER models
# MAGIC - Create category-specific partitions for efficient extraction
# MAGIC - Add quality flags and processing hints

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window

# Configuration
CATALOG = "dev_clinical"
SCHEMA = "doc_test"

# Input
SILVER_CLASSIFICATIONS = f"{CATALOG}.{SCHEMA}.silver_section_classifications"
SILVER_SECTIONS = f"{CATALOG}.{SCHEMA}.silver_sections"

# Output
GOLD_CLASSIFIED_SECTIONS = f"{CATALOG}.{SCHEMA}.gold_classified_sections"

# Quality thresholds
MIN_CONFIDENCE = 0.72  # Only include high-confidence classifications
MIN_TEXT_LENGTH = 10   # Minimum characters for meaningful extraction

print(f"Input: {SILVER_CLASSIFICATIONS}")
print(f"Output: {GOLD_CLASSIFIED_SECTIONS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Classified Sections

# COMMAND ----------

# Load classifications
classifications = spark.table(SILVER_CLASSIFICATIONS)

# Load original sections for text content
sections = spark.table(SILVER_SECTIONS)

print(f"Total classifications: {classifications.count():,}")
print(f"Total sections: {sections.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality Filtering

# COMMAND ----------

# Filter to high-quality classifications only
high_quality = classifications.filter(
    (F.col("classification_status") == "CLASSIFIED") &
    (F.col("classification_confidence") >= MIN_CONFIDENCE)
)

print(f"High-quality classifications: {high_quality.count():,}")
print(f"Filtered out: {classifications.count() - high_quality.count():,}")

# Show quality distribution
print("\nQuality distribution:")
classifications.groupBy("classification_status").count().orderBy(F.desc("count")).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enrich with Section Content

# COMMAND ----------

# Join with original sections to get full text content
enriched = high_quality.join(
    sections.select(
        "protocol_id",
        "section_id", 
        "section_title",
        "section_text",
        "page_num",
        "section_level"
    ),
    on=["protocol_id", "section_id"],
    how="inner"
)

# Filter out very short sections (likely headers/artifacts)
enriched = enriched.filter(F.length("section_text") >= MIN_TEXT_LENGTH)

print(f"Enriched sections: {enriched.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add NER Processing Hints

# COMMAND ----------

# Add flags to guide NER processing
gold_sections = enriched.withColumn(
    "ner_priority",
    F.when(F.col("classification").isin([
        "INCLUSION_CRITERIA", 
        "EXCLUSION_CRITERIA",
        "PRIMARY_ENDPOINT",
        "SECONDARY_ENDPOINT"
    ]), "HIGH")
    .when(F.col("category_domain").isin([
        "DEMOGRAPHICS",
        "CLINICAL_LABS", 
        "REPRODUCTIVE"
    ]), "MEDIUM")
    .otherwise("STANDARD")
).withColumn(
    "expected_entity_types",
    F.when(F.col("classification").like("%LAB%"), F.array(F.lit("BIOMARKER"), F.lit("PATIENT_CRITERIA")))
    .when(F.col("classification").like("%DEMOGRAPHIC%"), F.array(F.lit("PATIENT_CRITERIA")))
    .when(F.col("classification").like("%DRUG%"), F.array(F.lit("DRUG"), F.lit("DOSAGE")))
    .when(F.col("classification").like("%DISEASE%"), F.array(F.lit("CONDITION")))
    .when(F.col("classification").like("%ENDPOINT%"), F.array(F.lit("ENDPOINT"), F.lit("ENDPOINT_TYPE")))
    .when(F.col("classification").like("%PHASE%"), F.array(F.lit("STUDY_PHASE")))
    .otherwise(F.array(F.lit("CONDITION"), F.lit("DRUG"), F.lit("BIOMARKER")))
).withColumn(
    "text_length",
    F.length("section_text")
).withColumn(
    "word_count",
    F.size(F.split("section_text", "\\s+"))
).withColumn(
    "section_complexity",
    F.when(F.col("word_count") > 200, "LONG")
    .when(F.col("word_count") > 50, "MEDIUM")
    .otherwise("SHORT")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Protocol-Level Metadata

# COMMAND ----------

# Calculate protocol-level statistics
protocol_stats = gold_sections.groupBy("protocol_id").agg(
    F.count("*").alias("total_sections"),
    F.countDistinct("classification").alias("unique_categories"),
    F.avg("classification_confidence").alias("avg_confidence"),
    F.sum(F.when(F.col("ner_priority") == "HIGH", 1).otherwise(0)).alias("high_priority_sections")
)

# Join protocol stats back
gold_sections = gold_sections.join(
    protocol_stats.select(
        "protocol_id",
        F.col("total_sections").alias("protocol_total_sections"),
        F.col("unique_categories").alias("protocol_unique_categories"),
        F.col("avg_confidence").alias("protocol_avg_confidence")
    ),
    on="protocol_id",
    how="left"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Processing Metadata

# COMMAND ----------

from datetime import datetime

gold_sections = gold_sections.withColumn(
    "gold_created_at",
    F.current_timestamp()
).withColumn(
    "gold_version",
    F.lit("1.0")
).withColumn(
    "ready_for_ner",
    F.lit(True)
).withColumn(
    # Partition key for NER processing
    "ner_batch_key",
    F.concat(
        F.col("category_domain"),
        F.lit("_"),
        F.col("ner_priority")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Schema

# COMMAND ----------

# Select final columns in logical order
final_columns = [
    # Identifiers
    "protocol_id",
    "section_id",
    
    # Classification Results
    "classification",
    "classification_confidence",
    "classification_status",
    "category_domain",
    "category_level1",
    
    # Section Content
    "section_title",
    "section_text",
    "text_length",
    "word_count",
    "section_complexity",
    "section_level",
    "page_num",
    
    # NER Processing Hints
    "ner_priority",
    "expected_entity_types",
    "ner_batch_key",
    
    # Protocol Context
    "protocol_total_sections",
    "protocol_unique_categories", 
    "protocol_avg_confidence",
    
    # Metadata
    "ready_for_ner",
    "gold_created_at",
    "gold_version"
]

gold_final = gold_sections.select(*final_columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Gold Table

# COMMAND ----------

# Write with partitioning for efficient NER processing
gold_final.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .partitionBy("category_domain", "ner_priority") \
    .saveAsTable(GOLD_CLASSIFIED_SECTIONS)

print(f"Written to {GOLD_CLASSIFIED_SECTIONS}")
print(f"Total sections: {gold_final.count():,}")
print(f"Partitioned by: category_domain, ner_priority")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality Checks

# COMMAND ----------

print("=" * 80)
print("GOLD LAYER QUALITY REPORT")
print("=" * 80)
print()

# Overall stats
total_sections = gold_final.count()
total_protocols = gold_final.select("protocol_id").distinct().count()
unique_categories = gold_final.select("classification").distinct().count()

print(f"Total Sections:     {total_sections:,}")
print(f"Total Protocols:    {total_protocols:,}")
print(f"Unique Categories:  {unique_categories}")
print(f"Avg Confidence:     {gold_final.agg(F.avg('classification_confidence')).collect()[0][0]:.3f}")
print()

# Priority distribution
print("NER PRIORITY DISTRIBUTION:")
gold_final.groupBy("ner_priority").count().orderBy(F.desc("count")).show()

# Category domain distribution  
print("CATEGORY DOMAIN DISTRIBUTION:")
gold_final.groupBy("category_domain").count().orderBy(F.desc("count")).show()

# Section complexity
print("SECTION COMPLEXITY DISTRIBUTION:")
gold_final.groupBy("section_complexity").count().orderBy(F.desc("count")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Output for NER Pipeline

# COMMAND ----------

print("Sample high-priority sections ready for NER extraction:")
gold_final.filter(F.col("ner_priority") == "HIGH").select(
    "protocol_id",
    "classification",
    "section_title",
    "text_length",
    "classification_confidence",
    "expected_entity_types"
).limit(10).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Statistics for NER Team

# COMMAND ----------

# Create summary for NER pipeline consumption
ner_batch_summary = gold_final.groupBy("ner_batch_key", "category_domain", "ner_priority").agg(
    F.count("*").alias("section_count"),
    F.avg("text_length").alias("avg_text_length"),
    F.avg("word_count").alias("avg_word_count"),
    F.countDistinct("protocol_id").alias("protocol_count")
).orderBy("ner_batch_key")

print("NER Batch Summary (for extraction pipeline planning):")
ner_batch_summary.display()

# COMMAND ----------

print("=" * 80)
print("GOLD LAYER COMPLETE")
print("=" * 80)
print()
print(f"Output table: {GOLD_CLASSIFIED_SECTIONS}")
print(f"Ready for consumption by extraction_ner pipeline")
print()
print("Next step: Run extraction_ner/bronze_from_classification")
print("=" * 80)
