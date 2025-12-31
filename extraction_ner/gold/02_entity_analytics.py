# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: Entity Analytics & Catalog
# MAGIC 
# MAGIC **Purpose:** Transform extracted entities from Silver into analytics-ready datasets for reporting and downstream consumption.
# MAGIC 
# MAGIC **Input:** `dev_clinical.doc_intelligence.silver_ner_extractions` (from extraction_ner/silver)  
# MAGIC **Output:** Multiple Gold tables for entity analytics
# MAGIC 
# MAGIC **Gold Outputs:**
# MAGIC - `gold_entity_catalog` - Deduplicated entity catalog with statistics
# MAGIC - `gold_protocol_entities` - Protocol-level entity summaries
# MAGIC - `gold_entity_cooccurrence` - Entity co-occurrence matrix
# MAGIC - `gold_category_entities` - Entity distribution by classification category

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import *

# Configuration
CATALOG = "dev_clinical"
SCHEMA = "doc_intelligence"

# Input
SILVER_NER = f"{CATALOG}.{SCHEMA}.silver_ner_extractions"

# Outputs
GOLD_ENTITY_CATALOG = f"{CATALOG}.{SCHEMA}.gold_entity_catalog"
GOLD_PROTOCOL_ENTITIES = f"{CATALOG}.{SCHEMA}.gold_protocol_entities"
GOLD_ENTITY_COOCCURRENCE = f"{CATALOG}.{SCHEMA}.gold_entity_cooccurrence"
GOLD_CATEGORY_ENTITIES = f"{CATALOG}.{SCHEMA}.gold_category_entities"

print(f"Input:  {SILVER_NER}")
print(f"Outputs:")
print(f"  - {GOLD_ENTITY_CATALOG}")
print(f"  - {GOLD_PROTOCOL_ENTITIES}")
print(f"  - {GOLD_ENTITY_COOCCURRENCE}")
print(f"  - {GOLD_CATEGORY_ENTITIES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Silver NER Extractions

# COMMAND ----------

silver_ner = spark.table(SILVER_NER)

print(f"Total sections: {silver_ner.count():,}")
print(f"Sections with entities: {silver_ner.filter(F.col('entity_count') > 0).count():,}")

# Sample
silver_ner.limit(5).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold 1: Entity Catalog

# COMMAND ----------

# Explode entities to individual rows
entities_exploded = (
    silver_ner
    .filter(F.col("entity_count") > 0)
    .select(
        "protocol_id",
        "section_id",
        "classification",
        "category_domain",
        F.explode("entities").alias("entity")
    )
    .select(
        "protocol_id",
        "section_id",
        "classification",
        "category_domain",
        F.col("entity.text").alias("entity_text"),
        F.col("entity.entity_type").alias("entity_type"),
        F.col("entity.confidence").alias("confidence")
    )
)

# Normalize entity text (lowercase, trim)
entities_exploded = entities_exploded.withColumn(
    "entity_text_normalized",
    F.lower(F.trim(F.col("entity_text")))
)

# Create entity catalog with statistics
entity_catalog = (
    entities_exploded
    .groupBy("entity_text_normalized", "entity_type")
    .agg(
        # Statistics
        F.count("*").alias("occurrence_count"),
        F.countDistinct("protocol_id").alias("protocol_count"),
        F.countDistinct("section_id").alias("section_count"),
        F.avg("confidence").alias("avg_confidence"),
        F.min("confidence").alias("min_confidence"),
        F.max("confidence").alias("max_confidence"),
        
        # Context
        F.collect_set("classification").alias("found_in_classifications"),
        F.collect_set("category_domain").alias("found_in_domains"),
        F.collect_list("protocol_id").alias("protocol_list"),
        
        # Original text variants
        F.collect_set("entity_text").alias("text_variants"),
        F.first("entity_text").alias("canonical_text")
    )
    .withColumn(
        "entity_id",
        F.concat(
            F.col("entity_type"),
            F.lit("_"),
            F.md5(F.col("entity_text_normalized"))
        )
    )
    .withColumn(
        "variant_count",
        F.size("text_variants")
    )
)

# Add ranking by frequency
window_by_type = Window.partitionBy("entity_type").orderBy(F.desc("occurrence_count"))
entity_catalog = entity_catalog.withColumn(
    "rank_in_type",
    F.row_number().over(window_by_type)
)

# Write entity catalog
entity_catalog.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .partitionBy("entity_type") \
    .saveAsTable(GOLD_ENTITY_CATALOG)

print(f"✓ Entity catalog created: {entity_catalog.count():,} unique entities")

# Show top entities by type
print("\nTop 10 entities by type:")
for entity_type in entity_catalog.select("entity_type").distinct().limit(5).collect():
    etype = entity_type["entity_type"]
    print(f"\n{etype}:")
    entity_catalog.filter(F.col("entity_type") == etype) \
        .orderBy(F.desc("occurrence_count")) \
        .select("canonical_text", "occurrence_count", "protocol_count") \
        .limit(10) \
        .display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold 2: Protocol Entity Summaries

# COMMAND ----------

# Aggregate entities by protocol
protocol_entities = (
    entities_exploded
    .groupBy("protocol_id")
    .agg(
        # Overall statistics
        F.count("*").alias("total_entities"),
        F.countDistinct("entity_text_normalized").alias("unique_entities"),
        F.countDistinct("entity_type").alias("unique_entity_types"),
        F.avg("confidence").alias("avg_entity_confidence"),
        
        # Entity type distribution
        F.collect_set("entity_type").alias("entity_types_present"),
        
        # Category context
        F.countDistinct("classification").alias("unique_classifications"),
        F.countDistinct("category_domain").alias("unique_domains")
    )
)

# Add entity type counts as map
@F.udf(MapType(StringType(), IntegerType()))
def count_by_type(entity_list):
    """Count entities by type"""
    if not entity_list:
        return {}
    counts = {}
    for entity_type in entity_list:
        counts[entity_type] = counts.get(entity_type, 0) + 1
    return counts

protocol_entities = protocol_entities.withColumn(
    "entity_type_counts",
    count_by_type(F.col("entity_types_present"))
)

# Calculate entity density
protocol_entities = protocol_entities.withColumn(
    "entity_density",
    F.col("unique_entities") / F.col("total_entities")
)

# Write protocol summaries
protocol_entities.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(GOLD_PROTOCOL_ENTITIES)

print(f"Protocol entity summaries created: {protocol_entities.count():,} protocols")

# Show statistics
print("\nProtocol entity statistics:")
protocol_entities.select(
    F.avg("total_entities").alias("avg_total_entities"),
    F.avg("unique_entities").alias("avg_unique_entities"),
    F.avg("unique_entity_types").alias("avg_entity_types"),
    F.avg("entity_density").alias("avg_density")
).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold 3: Entity Co-occurrence Matrix

# COMMAND ----------

# Get entities per section
section_entities = (
    entities_exploded
    .groupBy("protocol_id", "section_id", "classification")
    .agg(
        F.collect_list(
            F.struct(
                "entity_text_normalized",
                "entity_type"
            )
        ).alias("entities_in_section")
    )
)

# Create co-occurrence pairs
from itertools import combinations

@F.udf(ArrayType(StructType([
    StructField("entity1", StringType()),
    StructField("entity2", StringType()),
    StructField("type1", StringType()),
    StructField("type2", StringType())
])))
def create_pairs(entities):
    """Create entity co-occurrence pairs"""
    if not entities or len(entities) < 2:
        return []
    
    pairs = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            e1 = entities[i]
            e2 = entities[j]
            
            # Sort alphabetically for consistent pairing
            if e1['entity_text_normalized'] < e2['entity_text_normalized']:
                pairs.append({
                    'entity1': e1['entity_text_normalized'],
                    'entity2': e2['entity_text_normalized'],
                    'type1': e1['entity_type'],
                    'type2': e2['entity_type']
                })
            else:
                pairs.append({
                    'entity1': e2['entity_text_normalized'],
                    'entity2': e1['entity_text_normalized'],
                    'type1': e2['entity_type'],
                    'type2': e1['entity_type']
                })
    
    return pairs

# Generate pairs and explode
cooccurrence = (
    section_entities
    .withColumn("pairs", create_pairs(F.col("entities_in_section")))
    .select(
        "protocol_id",
        "section_id",
        "classification",
        F.explode("pairs").alias("pair")
    )
    .select(
        "protocol_id",
        "section_id",
        "classification",
        F.col("pair.entity1").alias("entity1"),
        F.col("pair.entity2").alias("entity2"),
        F.col("pair.type1").alias("type1"),
        F.col("pair.type2").alias("type2")
    )
)

# Aggregate co-occurrence counts
entity_cooccurrence = (
    cooccurrence
    .groupBy("entity1", "entity2", "type1", "type2")
    .agg(
        F.count("*").alias("cooccurrence_count"),
        F.countDistinct("protocol_id").alias("protocol_count"),
        F.countDistinct("section_id").alias("section_count"),
        F.collect_set("classification").alias("cooccurring_in_classifications")
    )
    .orderBy(F.desc("cooccurrence_count"))
)

# Write co-occurrence matrix
entity_cooccurrence.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable(GOLD_ENTITY_COOCCURRENCE)

print(f"Entity co-occurrence matrix created: {entity_cooccurrence.count():,} pairs")

# Show top co-occurring entities
print("\nTop 20 co-occurring entity pairs:")
entity_cooccurrence.select(
    "entity1",
    "entity2",
    "type1",
    "type2",
    "cooccurrence_count",
    "protocol_count"
).limit(20).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Gold 4: Category-Entity Distribution

# COMMAND ----------

# Aggregate entities by classification category
category_entities = (
    entities_exploded
    .groupBy("classification", "category_domain", "entity_type")
    .agg(
        F.count("*").alias("entity_count"),
        F.countDistinct("entity_text_normalized").alias("unique_entities"),
        F.countDistinct("protocol_id").alias("protocol_count"),
        F.avg("confidence").alias("avg_confidence"),
        
        # Top entities in this category
        F.collect_list(
            F.struct(
                "entity_text_normalized",
                "confidence"
            )
        ).alias("entity_samples")
    )
)

# Extract top 10 entities per category-type
@F.udf(ArrayType(StringType()))
def get_top_entities(entity_samples):
    """Get top entities by frequency"""
    if not entity_samples:
        return []
    
    # Count occurrences
    counts = {}
    for sample in entity_samples:
        entity = sample['entity_text_normalized']
        counts[entity] = counts.get(entity, 0) + 1
    
    # Sort by count
    sorted_entities = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [entity for entity, count in sorted_entities[:10]]

category_entities = category_entities.withColumn(
    "top_entities",
    get_top_entities(F.col("entity_samples"))
).drop("entity_samples")

# Write category-entity distribution
category_entities.write \
    .format("delta") \
    .mode("overwrite") \
    .partitionBy("category_domain") \
    .saveAsTable(GOLD_CATEGORY_ENTITIES)

print(f"✓ Category-entity distribution created")

# Show distribution
print("\nEntity distribution by classification:")
category_entities.groupBy("classification").agg(
    F.sum("entity_count").alias("total_entities"),
    F.sum("unique_entities").alias("total_unique")
).orderBy(F.desc("total_entities")).limit(20).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analytics Summary

# COMMAND ----------

print("=" * 80)
print("GOLD LAYER - ENTITY ANALYTICS SUMMARY")
print("=" * 80)
print()

# Entity catalog stats
catalog_stats = entity_catalog.agg(
    F.count("*").alias("unique_entities"),
    F.sum("occurrence_count").alias("total_occurrences"),
    F.countDistinct("entity_type").alias("entity_types")
).collect()[0]

print(f"Entity Catalog:")
print(f"  Unique entities:     {catalog_stats['unique_entities']:,}")
print(f"  Total occurrences:   {catalog_stats['total_occurrences']:,}")
print(f"  Entity types:        {catalog_stats['entity_types']}")
print()

# Protocol stats
protocol_stats = protocol_entities.agg(
    F.count("*").alias("protocols"),
    F.avg("total_entities").alias("avg_entities"),
    F.avg("unique_entity_types").alias("avg_types")
).collect()[0]

print(f"Protocol Summaries:")
print(f"  Total protocols:     {protocol_stats['protocols']:,}")
print(f"  Avg entities/protocol: {protocol_stats['avg_entities']:.1f}")
print(f"  Avg entity types:    {protocol_stats['avg_types']:.1f}")
print()

# Co-occurrence stats
cooccur_count = entity_cooccurrence.count()
print(f"Co-occurrence Matrix:")
print(f"  Entity pairs:        {cooccur_count:,}")
print()

# Category distribution
category_count = category_entities.count()
print(f"Category Distribution:")
print(f"  Category-type combos: {category_count:,}")
print()

print("Gold Tables Created:")
print(f"  {GOLD_ENTITY_CATALOG}")
print(f"  {GOLD_PROTOCOL_ENTITIES}")
print(f"  {GOLD_ENTITY_COOCCURRENCE}")
print(f"  {GOLD_CATEGORY_ENTITIES}")
print()
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Analytics Queries

# COMMAND ----------

# Example 1: Most common entities across all protocols
print("Top 20 most common entities (all types):")
spark.sql(f"""
    SELECT 
        canonical_text,
        entity_type,
        occurrence_count,
        protocol_count,
        ROUND(avg_confidence, 3) as avg_conf
    FROM {GOLD_ENTITY_CATALOG}
    ORDER BY occurrence_count DESC
    LIMIT 20
""").display()

# COMMAND ----------

# Example 2: Entity diversity by protocol
print("Protocols with highest entity diversity:")
spark.sql(f"""
    SELECT 
        protocol_id,
        total_entities,
        unique_entities,
        unique_entity_types,
        ROUND(entity_density, 3) as diversity
    FROM {GOLD_PROTOCOL_ENTITIES}
    ORDER BY entity_density DESC
    LIMIT 20
""").display()

# COMMAND ----------

# Example 3: Entity co-occurrence patterns
print("Most frequently co-occurring entities:")
spark.sql(f"""
    SELECT 
        entity1,
        type1,
        entity2,
        type2,
        cooccurrence_count,
        protocol_count
    FROM {GOLD_ENTITY_COOCCURRENCE}
    WHERE type1 != type2
    ORDER BY cooccurrence_count DESC
    LIMIT 20
""").display()

# COMMAND ----------

print("=" * 80)
print("GOLD LAYER COMPLETE - NER PIPELINE")
print("=" * 80)
print()
print("All entity analytics tables created and ready for consumption!")
print()
print("=" * 80)
