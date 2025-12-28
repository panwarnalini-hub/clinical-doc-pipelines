# Databricks notebook source
# MAGIC %sql
# MAGIC USE CATALOG dev_clinical;
# MAGIC USE SCHEMA doc_test;

# COMMAND ----------

# SILVER EMBEDDINGS PIPELINE
#
# WHAT THIS NOTEBOOK DOES:
#   Generate embeddings from silver_sections and write to a new embeddings table.
#   Different models for different content types:
#   - SapBERT: headings (optimized for clinical entity names/titles)
#   - PubMedBERT: narrative content (optimized for clinical text)
#
# WHY TWO MODELS:
#   Headings are short, entity-like strings ("Adverse Events", "Dosage and Administration")
#   SapBERT excels at these because it was trained on UMLS concept names.
#   
#   Paragraphs are longer narrative text with context and relationships.
#   PubMedBERT captures semantic meaning better for full sentences.
#
# INPUT:
#   silver_sections (from 02_Structure_to_Silver_Sections)
#
# OUTPUT:
#   silver_section_embeddings (new Delta table)
#
# NOT DONE HERE:
#   - No hierarchy changes
#   - No merging or chunking (already done in 02)
#   - No edits to silver_sections
#   - No ML classification

# COMMAND ----------

# MAGIC %md
# MAGIC # Configuration

# COMMAND ----------

from dataclasses import dataclass, field
from typing import List

#Configuration for embedding pipeline
@dataclass
class EmbeddingConfig:
    
    # Unity Catalog
    catalog: str = "dev_clinical"
    schema: str = "doc_test"
    
    # Tables
    silver_sections: str = "silver_sections"
    silver_embeddings: str = "silver_section_embeddings"
    
    # Model configuration
    sapbert_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    pubmedbert_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    
    # Model versions (for tracking)
    sapbert_version: str = "1.0"
    pubmedbert_version: str = "1.0"
    
    # Processing
    batch_size: int = 32
    max_sequence_length: int = 512
    normalize_embeddings: bool = True
    
    # Write mode: "overwrite" for full refresh, "append" for incremental
    write_mode: str = "overwrite"
    
    def full_table(self, table: str) -> str:
        return f"{self.catalog}.{self.schema}.{table}"


config = EmbeddingConfig()
print(f"Pipeline: {config.catalog}.{config.schema}")
print(f"Input:  {config.full_table(config.silver_sections)}")
print(f"Output: {config.full_table(config.silver_embeddings)}")
print(f"SapBERT model: {config.sapbert_model}")
print(f"PubMedBERT model: {config.pubmedbert_model}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Install Dependencies
# MAGIC
# MAGIC These models require transformers and torch. I havent added them in my cluster yet.

# COMMAND ----------

try:
    import transformers, torch, sentence_transformers
    print("Dependencies already installed.")
except ImportError:
    print("Installing dependencies...")
    %pip install transformers torch sentence-transformers --quiet
    print("Dependencies installed. Please restart the kernel.")


# COMMAND ----------

# MAGIC %md
# MAGIC # Load Models

# COMMAND ----------

import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load SapBERT (for headings)
print(f"\nLoading SapBERT: {config.sapbert_model}")
sapbert_tokenizer = AutoTokenizer.from_pretrained(config.sapbert_model)
sapbert_model = AutoModel.from_pretrained(config.sapbert_model).to(device)
sapbert_model.eval()
print("SapBERT loaded")

# Load PubMedBERT (for content)
print(f"\nLoading PubMedBERT: {config.pubmedbert_model}")
pubmedbert_tokenizer = AutoTokenizer.from_pretrained(config.pubmedbert_model)
pubmedbert_model = AutoModel.from_pretrained(config.pubmedbert_model).to(device)
pubmedbert_model.eval()
print("PubMedBERT loaded")

# Get embedding dimensions
with torch.no_grad():
    test_input = sapbert_tokenizer("test", return_tensors="pt").to(device)
    sapbert_dim = sapbert_model(**test_input).last_hidden_state.shape[-1]
    
    test_input = pubmedbert_tokenizer("test", return_tensors="pt").to(device)
    pubmedbert_dim = pubmedbert_model(**test_input).last_hidden_state.shape[-1]

print(f"\nSapBERT embedding dimension: {sapbert_dim}")
print(f"PubMedBERT embedding dimension: {pubmedbert_dim}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Embedding Functions

# COMMAND ----------

def normalize_vector(vector):
    """L2 normalize a vector."""
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector

# Generate embedding for text using specified model.

def get_embedding(text, tokenizer, model, max_length=512, normalize=True):
    if not text or not text.strip():
        # Return zero vector for empty text
        return [0.0] * model.config.hidden_size
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=True
    ).to(device)
    
    # Get embedding
    with torch.no_grad():
        outputs = model(**inputs)
        # Use [CLS] token embedding (first token)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    
    # Normalize if requested
    if normalize:
        embedding = normalize_vector(embedding)
    
    return embedding.tolist()

# Generate SapBERT embedding for heading/title text.

def get_sapbert_embedding(text):
    return get_embedding(
        text, 
        sapbert_tokenizer, 
        sapbert_model,
        max_length=config.max_sequence_length,
        normalize=config.normalize_embeddings
    )

#Generate PubMedBERT embedding for content text.

def get_pubmedbert_embedding(text):
    return get_embedding(
        text,
        pubmedbert_tokenizer,
        pubmedbert_model,
        max_length=config.max_sequence_length,
        normalize=config.normalize_embeddings
    )


# Test embeddings
print("Testing embedding functions...")
test_heading = "Adverse Events"
test_content = "Patients experienced mild to moderate headaches during the study period."

heading_emb = get_sapbert_embedding(test_heading)
content_emb = get_pubmedbert_embedding(test_content)

print(f"Heading embedding: {len(heading_emb)} dimensions")
print(f"Content embedding: {len(content_emb)} dimensions")
print(f"Heading L2 norm: {np.linalg.norm(heading_emb):.4f}")
print(f"Content L2 norm: {np.linalg.norm(content_emb):.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Silver Sections

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *

# Load silver sections
silver_sections_df = spark.sql(f"""
    SELECT 
        section_id,
        document_id,
        content_text,
        is_heading,
        word_count
    FROM {config.full_table(config.silver_sections)}
    WHERE content_text IS NOT NULL 
      AND TRIM(content_text) != ''
""")

total_sections = silver_sections_df.count()
heading_count = silver_sections_df.filter(F.col("is_heading") == True).count()
content_count = silver_sections_df.filter(F.col("is_heading") == False).count()

print(f"Total sections to embed: {total_sections}")
print(f"Headings (SapBERT):  {heading_count}")
print(f"Content (PubMedBERT): {content_count}")

# Cache for processing
silver_sections_df.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC # Split by Embedding Type

# COMMAND ----------

# Split into headings and content sections
headings_df = silver_sections_df.filter(F.col("is_heading") == True)
content_df = silver_sections_df.filter(F.col("is_heading") == False)

print(f"Headings to process: {headings_df.count()}")
print(f"Content sections to process: {content_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Generate Embeddings
# MAGIC
# MAGIC Process in batches using Pandas UDFs for efficiency.

# COMMAND ----------

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import ArrayType, FloatType
import pandas as pd

# Broadcast models to workers (for distributed processing)
# Note: For very large clusters, consider using Spark ML pipelines or model serving

# Define Pandas UDFs for batch embedding generation

#Pandas UDF to generate SapBERT embeddings in batches.
@pandas_udf(ArrayType(FloatType()))
def sapbert_embed_udf(texts: pd.Series) -> pd.Series:
    embeddings = []
    for text in texts:
        if text and str(text).strip():
            emb = get_sapbert_embedding(str(text))
        else:
            emb = [0.0] * sapbert_dim
        embeddings.append(emb)
    return pd.Series(embeddings)

#Pandas UDF to generate PubMedBERT embeddings in batches.
@pandas_udf(ArrayType(FloatType()))
def pubmedbert_embed_udf(texts: pd.Series) -> pd.Series:
    embeddings = []
    for text in texts:
        if text and str(text).strip():
            emb = get_pubmedbert_embedding(str(text))
        else:
            emb = [0.0] * pubmedbert_dim
        embeddings.append(emb)
    return pd.Series(embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process Headings with SapBERT

# COMMAND ----------

from pyspark.sql import functions as F

# Generate embeddings for headings
print("Generating SapBERT embeddings for headings...")

headings_embedded_df = headings_df.withColumn(
    "embedding_vector",
    sapbert_embed_udf(F.col("content_text"))
).withColumn(
    "embedding_type",
    F.lit("title")
).withColumn(
    "model_name",
    F.lit("SapBERT")
).withColumn(
    "model_version",
    F.lit(config.sapbert_version)
).withColumn(
    "vector_length",
    F.lit(sapbert_dim)
).withColumn(
    "embedded_at",
    F.current_timestamp()
)

# Select final columns
headings_final_df = headings_embedded_df.select(
    "section_id",
    "document_id",
    "embedding_type",
    "embedding_vector",
    "model_name",
    "model_version",
    "word_count",
    "vector_length",
    "embedded_at"
)

print(f"Heading embeddings generated: {headings_final_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Process Content with PubMedBERT

# COMMAND ----------

# Generate embeddings for content sections
print("Generating PubMedBERT embeddings for content sections...")

content_embedded_df = content_df.withColumn(
    "embedding_vector",
    pubmedbert_embed_udf(F.col("content_text"))
).withColumn(
    "embedding_type",
    F.lit("content")
).withColumn(
    "model_name",
    F.lit("PubMedBERT")
).withColumn(
    "embedding_type",
    F.lit("content")
).withColumn(
    "model_version",
    F.lit(config.pubmedbert_version)
).withColumn(
    "vector_length",
    F.lit(pubmedbert_dim)
).withColumn(
    "embedded_at",
    F.current_timestamp()
)

# Select final columns
content_final_df = content_embedded_df.select(
    "section_id",
    "document_id",
    "embedding_type",
    "embedding_vector",
    "model_name",
    "model_version",
    "word_count",
    "vector_length",
    "embedded_at"
)

print(f"Content embeddings generated: {content_final_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Combine and Deduplicate

# COMMAND ----------

# Union all embeddings
all_embeddings_df = headings_final_df.unionByName(content_final_df)

# Verify no duplicate section_ids
duplicate_check = all_embeddings_df.groupBy("section_id").count().filter(F.col("count") > 1)
duplicate_count = duplicate_check.count()

if duplicate_count > 0:
    print(f"WARNING: Found {duplicate_count} duplicate section_ids!")
    display(duplicate_check.limit(10))
    
    # Deduplicate by keeping first occurrence
    all_embeddings_df = all_embeddings_df.dropDuplicates(["section_id"])
    print(f"Deduplicated to: {all_embeddings_df.count()} rows")
else:
    print(f"No duplicate section_ids found")

print(f"\nTotal embeddings: {all_embeddings_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Write to Delta Table

# COMMAND ----------

# Define schema for embeddings table
embeddings_table = config.full_table(config.silver_embeddings)

# Write embeddings
print(f"Writing embeddings to: {embeddings_table}")
print(f"Mode: {config.write_mode}")

all_embeddings_df.write \
    .format("delta") \
    .mode(config.write_mode) \
    .saveAsTable(embeddings_table)

print(f"Embeddings written successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC # Validation

# COMMAND ----------

print("EMBEDDING SUMMARY")

# Count by embedding type
print("\nEMBEDDINGS BY TYPE")
display(spark.sql(f"""
    SELECT 
        embedding_type,
        model_name,
        COUNT(*) as count,
        ROUND(AVG(word_count), 1) as avg_word_count,
        MIN(vector_length) as vector_dim
    FROM {embeddings_table}
    GROUP BY embedding_type, model_name
    ORDER BY embedding_type
"""))

# Count by document
print("\nEMBEDDINGS BY DOCUMENT")
display(spark.sql(f"""
    SELECT 
        document_id,
        COUNT(*) as total_embeddings,
        SUM(CASE WHEN embedding_type = 'title' THEN 1 ELSE 0 END) as title_embeddings,
        SUM(CASE WHEN embedding_type = 'content' THEN 1 ELSE 0 END) as content_embeddings
    FROM {embeddings_table}
    GROUP BY document_id
"""))

# Vector statistics
print("\nVECTOR STATISTICS")
display(spark.sql(f"""
    SELECT 
        model_name,
        vector_length,
        COUNT(*) as count,
        -- Check for zero vectors (embedding failures)
        SUM(CASE WHEN SIZE(FILTER(embedding_vector, x -> x != 0.0)) = 0 THEN 1 ELSE 0 END) as zero_vectors
    FROM {embeddings_table}
    GROUP BY model_name, vector_length
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC # Sample Output

# COMMAND ----------

# Show sample embeddings (first few dimensions only)
print("\nSAMPLE EMBEDDINGS")
display(spark.sql(f"""
    SELECT 
        section_id,
        document_id,
        embedding_type,
        model_name,
        word_count,
        vector_length,
        -- Show first 5 dimensions of vector
        SLICE(embedding_vector, 1, 5) as vector_preview,
        embedded_at
    FROM {embeddings_table}
    ORDER BY document_id, section_id
    LIMIT 10
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC # Join Example: Silver Sections + Embeddings

# COMMAND ----------

# Example: How to join sections with embeddings for downstream use
print("\nEXAMPLE JOIN: Silver Sections + Embeddings")
display(spark.sql(f"""
    SELECT 
        s.section_id,
        s.document_id,
        s.section_type,
        SUBSTRING(s.content_text, 1, 80) as content_preview,
        e.embedding_type,
        e.model_name,
        e.vector_length,
        SLICE(e.embedding_vector, 1, 3) as vector_sample
    FROM {config.full_table(config.silver_sections)} s
    JOIN {embeddings_table} e ON s.section_id = e.section_id
    ORDER BY s.document_id, s.section_order
    LIMIT 10
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Output Schema:
# MAGIC ```
# MAGIC section_id        - Links back to silver_sections
# MAGIC document_id       - Source document
# MAGIC embedding_type    - "title" or "content"
# MAGIC embedding_vector  - ARRAY<FLOAT>, L2 normalized
# MAGIC model_name        - "SapBERT" or "PubMedBERT"
# MAGIC model_version     - Version string for tracking
# MAGIC word_count        - From source section
# MAGIC vector_length     - 768 (BERT base dimension)
# MAGIC embedded_at       - Processing timestamp
# MAGIC ```
# MAGIC
# MAGIC ## Models Used:
# MAGIC - **SapBERT** (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`): 
# MAGIC   Self-alignment pre-training for clinical entity representations.
# MAGIC   Optimized for short, entity-like text (headings, titles, concepts).
# MAGIC   
# MAGIC - **PubMedBERT** (`microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`):
# MAGIC   Pre-trained on PubMed abstracts and PMC full-text articles.
# MAGIC   Optimized for clinical narrative text understanding.

# COMMAND ----------

# Cleanup: uncache DataFrames
silver_sections_df.unpersist()
print("Pipeline complete!")
