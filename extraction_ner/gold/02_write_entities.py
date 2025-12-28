"""
Clinical Trial NER Pipeline - Step 3: Inference (Fixed v3)
Author: Nalini Panwar
Date: December 2025

Run inference on new clinical trial text using the fine-tuned SapBERT NER model.
"""

import json
import torch
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForTokenClassification
from dataclasses import dataclass

# Get project root (parent of scripts folder)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR


@dataclass
class InferenceConfig:
    model_path: Path = PROJECT_ROOT / "models" / "sapbert_ner" / "final"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


config = InferenceConfig()


class ClinicalNERInference:
    """Inference class for Clinical Trial NER"""
    
    def __init__(self, model_path: Path, device: str = "cpu"):
        self.device = device
        model_path = Path(model_path)
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_path))
        self.model.to(device)
        self.model.eval()
        
        with open(model_path / 'id2label.json', 'r') as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}
        
        print(f"Model loaded on {device}. {len(self.id2label)} labels.")
    
    def predict(self, text: str) -> List[Dict]:
        """Extract entities from text."""
        words = text.split()
        
        # Track word positions in original text
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
        
        # Get word_ids BEFORE moving to device
        word_ids = encoding.word_ids(batch_index=0)
        
        # Move to device
        input_dict = {k: v.to(self.device) for k, v in encoding.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**input_dict)
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()
        
        # Extract entities with relaxed B/I handling
        entities = []
        current_entity = None
        prev_word_idx = None
        
        for idx, (pred_id, word_idx) in enumerate(zip(predictions, word_ids)):
            if word_idx is None or word_idx == prev_word_idx:
                continue
            prev_word_idx = word_idx
            
            label = self.id2label[pred_id]
            
            # Extract entity type (handles both B- and I-)
            if label.startswith('B-') or label.startswith('I-'):
                entity_type = label[2:]
                
                # If we have a current entity of the same type, extend it
                if current_entity and current_entity['label'] == entity_type:
                    start, end = word_spans[word_idx]
                    current_entity['end'] = end
                    current_entity['text'] = text[current_entity['start']:current_entity['end']]
                else:
                    # Save previous entity if exists
                    if current_entity:
                        entities.append(current_entity)
                    
                    # Start new entity (treat I- as B- if no current entity)
                    start, end = word_spans[word_idx]
                    current_entity = {
                        'text': text[start:end],
                        'label': entity_type,
                        'start': start,
                        'end': end
                    }
            
            else:  # O label
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def predict_with_scores(self, text: str) -> List[Dict]:
        """Predict with confidence scores"""
        words = text.split()
        
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
        
        with torch.no_grad():
            outputs = self.model(**input_dict)
            probs = torch.softmax(outputs.logits, dim=2)[0].cpu()
            predictions = torch.argmax(probs, dim=1).tolist()
        
        results = []
        prev_word_idx = None
        for idx, (pred_id, word_idx) in enumerate(zip(predictions, word_ids)):
            if word_idx is None or word_idx == prev_word_idx:
                continue
            prev_word_idx = word_idx
            
            label = self.id2label[pred_id]
            confidence = probs[idx][pred_id].item()
            
            results.append({
                'word': words[word_idx],
                'label': label,
                'confidence': round(confidence, 3)
            })
        
        return results
    
    def format_entities(self, text: str, entities: List[Dict]) -> str:
        """Format text with highlighted entities"""
        if not entities:
            return text
        
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        result = text
        for ent in sorted_entities:
            before = result[:ent['start']]
            entity_text = result[ent['start']:ent['end']]
            after = result[ent['end']:]
            result = f"{before}[{entity_text}]({ent['label']}){after}"
        
        return result


def demo():
    """Run demo inference"""
    print("=" * 60)
    print("Clinical Trial NER - Inference Demo")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model path: {config.model_path}")
    
    if not config.model_path.exists():
        print(f"\nERROR: Model not found at {config.model_path}")
        return
    
    ner = ClinicalNERInference(config.model_path, config.device)
    
    test_texts = [
        "Patients with metastatic breast cancer will receive Pembrolizumab 200mg IV every 3 weeks.",
        "This Phase III trial evaluates Overall Survival in HER2-negative patients.",
        "Inclusion: Age >= 18 years, ECOG performance status 0-2, signed informed consent.",
        "Primary endpoint: Progression-Free Survival (PFS) at 12 months.",
        "Durvalumab combined with Carboplatin for advanced NSCLC treatment.",
        "Exclusion: Prior treatment with checkpoint inhibitors.",
        "Phase I dose escalation study of pembrolizumab in solid tumors.",
    ]
    
    print("\n" + "-" * 60)
    print("PREDICTIONS")
    print("-" * 60)
    
    for text in test_texts:
        print(f"\nInput: {text}")
        
        entities = ner.predict(text)
        
        if entities:
            print("Entities:")
            for ent in entities:
                print(f"  - '{ent['text']}' [{ent['label']}]")
            print(f"Formatted: {ner.format_entities(text, entities)}")
        else:
            print("  No entities found")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
