"""
Shared entity type definitions used across pipelines.
"""

# Classification categories (ingestion_classification)
SECTION_CATEGORIES = [
    "Inclusion Criteria",
    "Exclusion Criteria", 
    "Primary Endpoint",
    "Secondary Endpoint",
    # ... 109 total categories
]

# NER entity types (extraction_ner)
NER_ENTITIES = [
    "CONDITION",
    "DRUG",
    "DOSAGE",
    "STUDY_PHASE",
    "ENDPOINT",
    "PATIENT_CRITERIA",
    "BIOMARKER",
    "ENDPOINT_TYPE",
]

# BIO tags for NER
BIO_TAGS = ["O"] + [f"{prefix}-{entity}" for entity in NER_ENTITIES for prefix in ["B", "I"]]
