"""
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
