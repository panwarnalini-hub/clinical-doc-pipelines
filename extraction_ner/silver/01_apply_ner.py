# Databricks notebook source
# MAGIC %md
# MAGIC # Silver: NER Entity Extraction
# MAGIC 
# MAGIC **Purpose:** Apply fine-tuned NER model to classified sections from the classification pipeline's Gold layer.
# MAGIC 
# MAGIC **Input:** `dev_clinical.doc_test.gold_classified_sections` (from ingestion_classification/gold)  
# MAGIC **Output:** `dev_clinical.doc_intelligence.silver_ner_extractions`
# MAGIC 
# MAGIC **Pipeline Integration:**
# MAGIC ```
# MAGIC ingestion_classification/gold (gold_classified_sections)
# MAGIC   |
# MAGIC extraction_ner/silver (THIS NOTEBOOK)- Apply NER model
# MAGIC   |
# MAGIC extraction_ner/gold (entity analytics)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

import json
import torch
from pathlib import Path
from typing import List, Dict
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType, FloatType
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Configuration
class NERConfig:
    """Configuration for NER extraction"""
    # Model path (relative to notebook or absolute)
    model_path = "/dbfs/mnt/models/clinical_ner/sapbert_ner/final"
    
    # Unity Catalog - Input (from classification pipeline)
    input_catalog = "dev_clinical"
    input_schema = "doc_test"
    input_table = "gold_classified_sections"
    
    # Unity Catalog - Output (this pipeline)
    output_catalog = "dev_clinical"
    output_schema = "doc_intelligence"
    output_table = "silver_ner_extractions"
    
    # Processing
    batch_size = 32
    max_length = 512
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def input_full_table(self):
        return f"{self.input_catalog}.{self.input_schema}.{self.input_table}"
    
    @property
    def output_full_table(self):
        return f"{self.output_catalog}.{self.output_schema}.{self.output_table}"

config = NERConfig()

print(f"Input:  {config.input_full_table}")
print(f"Output: {config.output_full_table}")
print(f"Device: {config.device}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Classified Sections

# COMMAND ----------

# Read from classification pipeline's Gold layer
classified_sections = spark.table(config.input_full_table)

print(f"Total sections from classification pipeline: {classified_sections.count():,}")

# Show schema
print("\nInput schema:")
classified_sections.printSchema()

# Sample
print("\nSample sections:")
classified_sections.select(
    "protocol_id",
    "section_id", 
    "classification",
    "ner_priority",
    "section_text"
).limit(5).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filter for NER Processing

# COMMAND ----------

# Filter to sections that need NER extraction
# Based on ner_priority and expected_entity_types from Gold layer

ner_input = classified_sections.filter(
    F.col("ready_for_ner") == True
).select(
    "protocol_id",
    "section_id",
    "classification",
    "classification_confidence",
    "category_domain",
    "section_title",
    "section_text",
    "ner_priority",
    "expected_entity_types",
    "text_length",
    "word_count"
)

print(f"Sections ready for NER: {ner_input.count():,}")

# Priority distribution
print("\nNER Priority Distribution:")
ner_input.groupBy("ner_priority").count().orderBy(F.desc("count")).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load NER Model

# COMMAND ----------

class ClinicalNER:
    """NER model wrapper for clinical entity extraction"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device
        model_path = Path(model_path)
        
        print(f"Loading NER model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_path))
        self.model.to(device)
        self.model.eval()
        
        # Load label mapping
        label_path = model_path / 'id2label.json'
        if label_path.exists():
            with open(label_path, 'r') as f:
                self.id2label = {int(k): v for k, v in json.load(f).items()}
        else:
            # Fallback to model config
            self.id2label = self.model.config.id2label
        
        print(f"✓ Model loaded on {device}")
        print(f"  Entity types: {len(set(v.split('-')[1] for v in self.id2label.values() if '-' in v))}")
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities from text with confidence scores"""
        if not text or not text.strip():
            return []
        
        # Tokenize with word alignment
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True,
            return_offsets_mapping=True
        )
        
        offset_mapping = encoding.pop('offset_mapping')[0]
        input_dict = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**input_dict)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()
            
            # Get confidence scores
            probs = torch.softmax(outputs.logits, dim=2)[0].cpu()
            confidences = [probs[i][pred].item() for i, pred in enumerate(predictions)]
        
        # Extract entities with BIO tagging
        entities = []
        current_entity = None
        
        for idx, (pred_id, conf, (start, end)) in enumerate(zip(predictions, confidences, offset_mapping)):
            # Skip special tokens
            if start == end:
                continue
            
            label = self.id2label[pred_id]
            
            if label.startswith('B-'):
                # Save previous entity
                if current_entity:
                    entities.append(current_entity)
                
                # Start new entity
                entity_type = label[2:]
                current_entity = {
                    'text': text[start:end],
                    'entity_type': entity_type,
                    'start_pos': int(start),
                    'end_pos': int(end),
                    'confidence': float(conf)
                }
            
            elif label.startswith('I-'):
                entity_type = label[2:]
                
                # Continue current entity if same type
                if current_entity and current_entity['entity_type'] == entity_type:
                    current_entity['end_pos'] = int(end)
                    current_entity['text'] = text[current_entity['start_pos']:current_entity['end_pos']]
                    # Average confidence
                    current_entity['confidence'] = (current_entity['confidence'] + conf) / 2
                else:
                    # Type mismatch - save and start new
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        'text': text[start:end],
                        'entity_type': entity_type,
                        'start_pos': int(start),
                        'end_pos': int(end),
                        'confidence': float(conf)
                    }
            
            else:  # O tag
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Save final entity
        if current_entity:
            entities.append(current_entity)
        
        return entities

# Load model (will be broadcast to workers)
ner_model = ClinicalNER(config.model_path, config.device)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define UDF for Entity Extraction

# COMMAND ----------

# Define entity schema
entity_schema = ArrayType(StructType([
    StructField("text", StringType(), True),
    StructField("entity_type", StringType(), True),
    StructField("start_pos", IntegerType(), True),
    StructField("end_pos", IntegerType(), True),
    StructField("confidence", FloatType(), True)
]))

# Create UDF
@F.udf(entity_schema)
def extract_entities_udf(text):
    """UDF to extract entities from section text"""
    if not text:
        return []
    
    try:
        entities = ner_model.extract_entities(text)
        return entities
    except Exception as e:
        print(f"Error extracting entities: {e}")
        return []

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply NER Model

# COMMAND ----------

# Apply NER to section_text
ner_extractions = ner_input.withColumn(
    "entities",
    extract_entities_udf(F.col("section_text"))
).withColumn(
    "entity_count",
    F.size("entities")
).withColumn(
    "extraction_timestamp",
    F.current_timestamp()
).withColumn(
    "model_version",
    F.lit("sapbert_ner_v1.0")
)

print(f"NER extraction complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add Entity Type Statistics

# COMMAND ----------

# Extract entity type counts
from pyspark.sql.types import MapType

@F.udf(MapType(StringType(), IntegerType()))
def count_entity_types(entities):
    """Count entities by type"""
    if not entities:
        return {}
    
    counts = {}
    for entity in entities:
        entity_type = entity['entity_type']
        counts[entity_type] = counts.get(entity_type, 0) + 1
    
    return counts

ner_extractions = ner_extractions.withColumn(
    "entity_type_counts",
    count_entity_types(F.col("entities"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quality Checks

# COMMAND ----------

print("=" * 80)
print("NER EXTRACTION QUALITY REPORT")
print("=" * 80)
print()

total_sections = ner_extractions.count()
sections_with_entities = ner_extractions.filter(F.col("entity_count") > 0).count()
total_entities = ner_extractions.agg(F.sum("entity_count")).collect()[0][0]

print(f"Total Sections Processed: {total_sections:,}")
print(f"Sections with Entities:   {sections_with_entities:,} ({sections_with_entities/total_sections*100:.1f}%)")
print(f"Total Entities Extracted: {total_entities:,}")
print(f"Avg Entities per Section: {total_entities/total_sections:.1f}")
print()

# Distribution by priority
print("Extraction by NER Priority:")
ner_extractions.groupBy("ner_priority").agg(
    F.count("*").alias("section_count"),
    F.sum("entity_count").alias("total_entities"),
    F.avg("entity_count").alias("avg_entities")
).orderBy(F.desc("total_entities")).display()

# Distribution by category
print("\nTop Categories by Entity Count:")
ner_extractions.groupBy("classification").agg(
    F.count("*").alias("section_count"),
    F.sum("entity_count").alias("total_entities")
).orderBy(F.desc("total_entities")).limit(20).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Results

# COMMAND ----------

print("Sample extractions (high-priority sections with entities):")
ner_extractions.filter(
    (F.col("ner_priority") == "HIGH") &
    (F.col("entity_count") > 0)
).select(
    "protocol_id",
    "classification",
    "section_title",
    "entity_count",
    "entities"
).limit(5).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Silver Table

# COMMAND ----------

# Select final columns
final_columns = [
    # Identifiers
    "protocol_id",
    "section_id",
    
    # Classification context (from Gold input)
    "classification",
    "classification_confidence",
    "category_domain",
    "section_title",
    "section_text",
    
    # NER processing hints (from Gold input)
    "ner_priority",
    "expected_entity_types",
    
    # NER results
    "entities",
    "entity_count",
    "entity_type_counts",
    
    # Metadata
    "extraction_timestamp",
    "model_version",
    "text_length",
    "word_count"
]

silver_output = ner_extractions.select(*final_columns)

# Write to Delta table
silver_output.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .partitionBy("category_domain", "ner_priority") \
    .saveAsTable(config.output_full_table)

print(f"Written to {config.output_full_table}")
print(f"Total records: {silver_output.count():,}")
print(f"Partitioned by: category_domain, ner_priority")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Entity Type Distribution

# COMMAND ----------

# Explode entities to analyze distribution
entity_distribution = (
    ner_extractions
    .select("protocol_id", "classification", F.explode("entities").alias("entity"))
    .select(
        "protocol_id",
        "classification",
        F.col("entity.entity_type").alias("entity_type"),
        F.col("entity.text").alias("entity_text"),
        F.col("entity.confidence").alias("confidence")
    )
)

print("Entity Type Distribution:")
entity_distribution.groupBy("entity_type").agg(
    F.count("*").alias("count"),
    F.avg("confidence").alias("avg_confidence"),
    F.countDistinct("entity_text").alias("unique_entities")
).orderBy(F.desc("count")).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Top Entities by Type

# COMMAND ----------

print("Most Common Entities by Type:")

# Get top 5 entity types
top_types = entity_distribution.groupBy("entity_type").count().orderBy(F.desc("count")).limit(5)

for row in top_types.collect():
    entity_type = row['entity_type']
    print(f"\n{entity_type}:")
    
    entity_distribution.filter(F.col("entity_type") == entity_type) \
        .groupBy("entity_text").count() \
        .orderBy(F.desc("count")) \
        .limit(10) \
        .display()

# COMMAND ----------

print("=" * 80)
print("SILVER NER EXTRACTION COMPLETE")
print("=" * 80)
print()
print(f"Input:  {config.input_full_table} (from classification/gold)")
print(f"Output: {config.output_full_table}")
print(f"Total sections: {total_sections:,}")
print(f"Total entities: {total_entities:,}")
print()
print("Next step: Run gold/02_entity_analytics.py")
print("=" * 80)
