"""
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
