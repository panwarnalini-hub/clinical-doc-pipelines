"""
Clinical Doc Pipelines - Folder Structure Setup
Run this script to create the recommended folder structure.

Architecture decisions:
1. NER consumes classification output (Option A)
2. Training separated from medallion path
3. Gold layer is consumer-oriented
4. Separate Databricks workflows per pipeline
"""

import os
from pathlib import Path

# Define the structure
STRUCTURE = {
    "clinical-doc-pipelines": {
        "ingestion_classification": {
            "README.md": None,
            "bronze": {
                "__init__.py": "",
                "01_document_extraction.py": "# Bronze: Raw document extraction from source"
            },
            "silver": {
                "__init__.py": "",
                "02_structure_sections.py": "# Silver: Structured sections from documents",
                "03a_embeddings.py": "# Silver: SapBERT embeddings generation",
                "03b_classification.py": "# Silver: Section classification (87 categories)"
            },
            "gold": {
                "__init__.py": "",
                "04_classified_sections.py": "# Gold: Analytics-ready classified sections"
            },
            "models": {
                "model_comparison_report.pdf": None  # Copy manually
            },
            "tests": {
                "__init__.py": "",
                "test_classification.py": "# Tests for classification pipeline"
            }
        },
        "extraction_ner": {
            "README.md": None,
            "bronze_from_classification": {
                "__init__.py": "",
                "README.md": "# Bronze input comes from ingestion_classification/gold\n\nThis layer consumes the output of the classification pipeline.\nNo separate data ingestion - we extract entities from classified sections."
            },
            "silver": {
                "__init__.py": "",
                "01_apply_ner.py": "# Silver: Apply NER model to extract entities",
            },
            "gold": {
                "__init__.py": "",
                "02_write_entities.py": "# Gold: Write gold_clinical_entities table"
            },
            "training": {
                "__init__.py": "",
                "01_convert_annotations.py": "# Training: Convert Label Studio annotations",
                "02_train_model.py": "# Training: Fine-tune SapBERT with class weighting",
                "README.md": "# Training Pipeline\n\nTraining is NOT part of the medallion data path.\nIt produces model artifacts consumed by silver/01_apply_ner.py"
            },
            "models": {
                "sapbert_ner": {}  # Model weights go here
            },
            "data": {
                "raw": {},
                "annotations": {},
                "processed": {}
            },
            "outputs": {},
            "tests": {
                "__init__.py": "",
                "test_api.py": "# Tests for ClinicalTrials.gov API",
                "test_ner.py": "# Tests for NER inference"
            }
        },
        "shared": {
            "schemas": {
                "__init__.py": "",
                "entity_types.py": '''"""
Shared entity type definitions used across pipelines.
"""

# Classification categories (ingestion_classification)
SECTION_CATEGORIES = [
    "Inclusion Criteria",
    "Exclusion Criteria", 
    "Primary Endpoint",
    "Secondary Endpoint",
    # ... 87 total categories
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
'''
            },
            "utils": {
                "__init__.py": "",
                "embeddings.py": '''"""
Shared SapBERT embedding utilities.
Used by both classification and NER pipelines.
"""

import torch
from transformers import AutoTokenizer, AutoModel

def load_sapbert(model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"):
    """Load SapBERT model and tokenizer."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model, device

def get_embedding(text: str, tokenizer, model, device, normalize: bool = True):
    """Generate SapBERT embedding for text."""
    import numpy as np
    
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
        embedding = (sum_embeddings / sum_mask).cpu().numpy()[0]
    
    if normalize:
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
    
    return embedding.tolist()
'''
            },
            "config": {
                "__init__.py": "",
                "settings.py": '''"""
Shared configuration settings.
"""

from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent
    
    # Models
    SAPBERT_MODEL: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    
    # Unity Catalog (Databricks)
    CATALOG: str = "dev_clinical"
    SCHEMA: str = "doc_intelligence"
    
    # Tables
    BRONZE_DOCUMENTS: str = "bronze_documents"
    SILVER_SECTIONS: str = "silver_sections"
    SILVER_EMBEDDINGS: str = "silver_embeddings"
    SILVER_CLASSIFICATIONS: str = "silver_classifications"
    GOLD_CLASSIFIED_SECTIONS: str = "gold_classified_sections"
    GOLD_CLINICAL_ENTITIES: str = "gold_clinical_entities"

config = Config()
'''
            }
        },
        "databricks": {
            "workflows": {
                "ingestion_classification.yml": '''# Databricks Workflow: Ingestion & Classification Pipeline
#
# DAG: bronze -> silver (structure) -> silver (embed) -> silver (classify) -> gold
#
# Schedule: Daily or on new document upload

name: ingestion_classification_pipeline

tasks:
  - task_key: bronze_extraction
    notebook_task:
      notebook_path: /Repos/clinical-doc-pipelines/ingestion_classification/bronze/01_document_extraction
    
  - task_key: silver_structure
    depends_on:
      - task_key: bronze_extraction
    notebook_task:
      notebook_path: /Repos/clinical-doc-pipelines/ingestion_classification/silver/02_structure_sections

  - task_key: silver_embeddings
    depends_on:
      - task_key: silver_structure
    notebook_task:
      notebook_path: /Repos/clinical-doc-pipelines/ingestion_classification/silver/03a_embeddings

  - task_key: silver_classification
    depends_on:
      - task_key: silver_embeddings
    notebook_task:
      notebook_path: /Repos/clinical-doc-pipelines/ingestion_classification/silver/03b_classification

  - task_key: gold_output
    depends_on:
      - task_key: silver_classification
    notebook_task:
      notebook_path: /Repos/clinical-doc-pipelines/ingestion_classification/gold/04_classified_sections
''',
                "extraction_ner.yml": '''# Databricks Workflow: NER Extraction Pipeline
#
# DAG: gold_classified_sections (from classification) -> silver (NER) -> gold (entities)
#
# Schedule: Triggered after ingestion_classification completes
# Dependency: Requires ingestion_classification/gold output

name: extraction_ner_pipeline

tasks:
  - task_key: silver_apply_ner
    notebook_task:
      notebook_path: /Repos/clinical-doc-pipelines/extraction_ner/silver/01_apply_ner
    # Input: gold_classified_sections from classification pipeline

  - task_key: gold_entities
    depends_on:
      - task_key: silver_apply_ner
    notebook_task:
      notebook_path: /Repos/clinical-doc-pipelines/extraction_ner/gold/02_write_entities
    # Output: gold_clinical_entities
'''
            },
            "notebooks": {
                "demo.py": "# Demo notebook for portfolio presentation"
            }
        },
        "app.py": "# Streamlit demo app",
        "requirements.txt": '''# Clinical Doc Pipelines - Requirements

# Core ML
torch>=2.0.0
transformers>=4.30.0

# Data processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Visualization
streamlit>=1.28.0

# Databricks (optional, for cloud deployment)
# databricks-sdk>=0.12.0
''',
        "README.md": None,  # Will create separately
        ".gitignore": '''# Python
__pycache__/
*.py[cod]
*$py.class
.Python
*.so
.eggs/
*.egg-info/
*.egg

# Environments
.env
.venv
env/
venv/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Models (too large for git)
**/models/sapbert_ner/final/
**/models/*.bin
**/models/*.safetensors

# Data
**/data/raw/*.json
**/data/processed/*.json

# Outputs
**/outputs/*.html

# Databricks
.databricks/

# OS
.DS_Store
Thumbs.db
'''
    }
}


def create_structure(base_path: Path, structure: dict):
    """Recursively create folder structure."""
    for name, content in structure.items():
        path = base_path / name
        
        if content is None:
            # Placeholder file - create empty or skip
            if name.endswith('.pdf'):
                print(f"  [MANUAL] {path} - copy manually")
            else:
                path.touch()
                print(f"  [CREATED] {path}")
        elif isinstance(content, dict):
            # Directory
            path.mkdir(parents=True, exist_ok=True)
            print(f"  [DIR] {path}/")
            create_structure(path, content)
        elif isinstance(content, str):
            # File with content
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  [CREATED] {path}")


def main():
    print("=" * 60)
    print("Clinical Doc Pipelines - Structure Setup")
    print("=" * 60)
    
    base_path = Path(".")
    
    print("\nCreating folder structure...\n")
    create_structure(base_path, STRUCTURE)
    
    print("\n" + "=" * 60)
    print("Structure created!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Copy your existing files into the new structure")
    print("2. Create the main README.md")
    print("3. Copy model weights to extraction_ner/models/sapbert_ner/")
    print("4. Initialize git repo")


if __name__ == "__main__":
    main()
