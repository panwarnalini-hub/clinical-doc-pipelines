"""
Extraction NER Pipeline - Silver Layer
01_apply_ner.py

Apply fine-tuned NER model to classified sections from ingestion pipeline.

INPUT:  gold_classified_sections (from ingestion_classification pipeline)
OUTPUT: silver_ner_extractions

This script bridges the two pipelines:
  ingestion_classification/gold → extraction_ner/silver
"""

import json
import torch
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForTokenClassification

# For Databricks
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


@dataclass
class NERConfig:
    """Configuration for NER extraction"""
    # Model
    model_path: str = "extraction_ner/models/sapbert_ner/final"
    
    # Unity Catalog (Databricks)
    catalog: str = "dev_clinical"
    schema: str = "doc_intelligence"
    
    # Input table (from classification pipeline)
    input_table: str = "gold_classified_sections"
    
    # Output table
    output_table: str = "silver_ner_extractions"
    
    # Processing
    batch_size: int = 32
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


config = NERConfig()


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
        
        with open(model_path / 'id2label.json', 'r') as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}
        
        print(f"Model loaded. Device: {device}")
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract entities from text"""
        if not text or not text.strip():
            return []
        
        words = text.split()
        if not words:
            return []
        
        # Track word positions
        word_spans = []
        current_pos = 0
        for word in words:
            start = text.find(word, current_pos)
            end = start + len(word)
            word_spans.append((start, end))
            current_pos = end
        
        # Tokenize
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            padding=True
        )
        
        word_ids = encoding.word_ids(batch_index=0)
        input_dict = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**input_dict)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()
        
        # Extract entities
        entities = []
        current_entity = None
        prev_word_idx = None
        
        for idx, (pred_id, word_idx) in enumerate(zip(predictions, word_ids)):
            if word_idx is None or word_idx == prev_word_idx:
                continue
            prev_word_idx = word_idx
            
            label = self.id2label[pred_id]
            
            if label.startswith('B-') or label.startswith('I-'):
                entity_type = label[2:]
                
                if current_entity and current_entity['entity_type'] == entity_type:
                    start, end = word_spans[word_idx]
                    current_entity['end_pos'] = end
                    current_entity['text'] = text[current_entity['start_pos']:current_entity['end_pos']]
                else:
                    if current_entity:
                        entities.append(current_entity)
                    
                    start, end = word_spans[word_idx]
                    current_entity = {
                        'text': text[start:end],
                        'entity_type': entity_type,
                        'start_pos': start,
                        'end_pos': end
                    }
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities


def run_spark_pipeline():
    """Run NER extraction on Databricks/Spark"""
    if not SPARK_AVAILABLE:
        raise RuntimeError("Spark not available. Use run_local() instead.")
    
    spark = SparkSession.builder.getOrCreate()
    
    # Load model (broadcast to workers)
    ner_model = ClinicalNER(config.model_path, config.device)
    
    # Read classified sections from gold
    input_table = f"{config.catalog}.{config.schema}.{config.input_table}"
    print(f"Reading from: {input_table}")
    
    sections_df = spark.table(input_table)
    
    # Define UDF for entity extraction
    entity_schema = ArrayType(StructType([
        StructField("text", StringType(), True),
        StructField("entity_type", StringType(), True),
        StructField("start_pos", IntegerType(), True),
        StructField("end_pos", IntegerType(), True)
    ]))
    
    @F.udf(entity_schema)
    def extract_entities_udf(text):
        if not text:
            return []
        entities = ner_model.extract_entities(text)
        return entities
    
    # Apply NER to each section
    result_df = sections_df.withColumn(
        "entities",
        extract_entities_udf(F.col("content_text"))
    ).withColumn(
        "entity_count",
        F.size("entities")
    ).withColumn(
        "extracted_at",
        F.current_timestamp()
    )
    
    # Write to silver table
    output_table = f"{config.catalog}.{config.schema}.{config.output_table}"
    print(f"Writing to: {output_table}")
    
    result_df.write \
        .format("delta") \
        .mode("overwrite") \
        .saveAsTable(output_table)
    
    # Summary
    print("\nExtraction Summary:")
    print(f"  Total sections: {result_df.count()}")
    print(f"  Sections with entities: {result_df.filter(F.col('entity_count') > 0).count()}")
    
    return result_df


def run_local(input_texts: List[str] = None):
    """Run NER extraction locally (for testing/demo)"""
    ner_model = ClinicalNER(config.model_path, config.device)
    
    if input_texts is None:
        # Demo texts
        input_texts = [
            "Patients with metastatic breast cancer will receive Pembrolizumab 200mg IV every 3 weeks.",
            "Inclusion: Age >= 18 years, ECOG performance status 0-2.",
            "Primary endpoint: Progression-Free Survival (PFS) at 12 months.",
        ]
    
    print("\nExtracting entities from texts...")
    results = []
    
    for text in input_texts:
        entities = ner_model.extract_entities(text)
        results.append({
            "text": text,
            "entities": entities,
            "entity_count": len(entities)
        })
        
        print(f"\nText: {text[:60]}...")
        if entities:
            for ent in entities:
                print(f"  [{ent['entity_type']}] {ent['text']}")
        else:
            print("  No entities found")
    
    return results


def main():
    print("=" * 60)
    print("Extraction NER Pipeline - Silver Layer")
    print("=" * 60)
    print(f"Input:  {config.input_table} (from classification pipeline)")
    print(f"Output: {config.output_table}")
    print(f"Device: {config.device}")
    
    if SPARK_AVAILABLE:
        print("\nRunning Spark pipeline...")
        run_spark_pipeline()
    else:
        print("\nSpark not available. Running local demo...")
        run_local()
    
    print("\n" + "=" * 60)
    print("Silver NER extraction complete!")
    print("Next: Run gold/02_write_entities.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
