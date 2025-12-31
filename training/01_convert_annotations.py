"""
Clinical Trial NER Pipeline - Step 1: Convert Label Studio JSON to Training Format
Author: Nalini Panwar
Date: December 2025

Converts Label Studio annotations to HuggingFace token classification format
for fine-tuning SapBERT on clinical trial NER.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
import random

@dataclass
class NERConfig:
    """Configuration for NER data conversion"""
    input_file: str = "data/labelstudio_export.json"
    output_dir: str = "data/processed"
    train_ratio: float = 0.8
    seed: int = 42
    
    # Entity labels (from your Label Studio schema)
    labels: Tuple[str, ...] = (
        "O",  # Outside any entity
        "B-CONDITION", "I-CONDITION",
        "B-DRUG", "I-DRUG", 
        "B-DOSAGE", "I-DOSAGE",
        "B-STUDY_PHASE", "I-STUDY_PHASE",
        "B-ENDPOINT", "I-ENDPOINT",
        "B-PATIENT_CRITERIA", "I-PATIENT_CRITERIA",
        "B-BIOMARKER", "I-BIOMARKER",
        "B-ENDPOINT_TYPE", "I-ENDPOINT_TYPE",
    )

config = NERConfig()


def load_labelstudio_json(filepath: str) -> List[Dict]:
    """Load Label Studio export JSON"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} annotated samples")
    return data


def extract_annotations(task: Dict) -> Tuple[str, List[Dict]]:
    """
    Extract text and entity annotations from a Label Studio task.
    
    Returns:
        text: The original text
        entities: List of {start, end, label} dicts
    """
    text = task['data']['text']
    entities = []
    
    # Get annotations (completed labels)
    annotations = task.get('annotations', [])
    if not annotations:
        return text, []
    
    # Use the first annotation (most recent completed)
    result = annotations[0].get('result', [])
    
    for item in result:
        if item.get('type') == 'labels' and 'value' in item:
            value = item['value']
            entities.append({
                'start': value['start'],
                'end': value['end'],
                'label': value['labels'][0]  # Take first label
            })
    
    # Sort by start position
    entities.sort(key=lambda x: x['start'])
    return text, entities


def tokenize_with_labels(text: str, entities: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    Simple whitespace tokenization with BIO label alignment.
    
    For SapBERT fine-tuning, we'll re-tokenize with the model's tokenizer,
    but this gives us a clean intermediate format.
    """
    tokens = []
    labels = []
    
    # Build character-to-entity mapping
    char_labels = ['O'] * len(text)
    for ent in entities:
        label = ent['label']
        start, end = ent['start'], ent['end']
        
        # First character gets B- tag
        if start < len(char_labels):
            char_labels[start] = f"B-{label}"
        
        # Rest get I- tags
        for i in range(start + 1, min(end, len(char_labels))):
            char_labels[i] = f"I-{label}"
    
    # Tokenize and assign labels
    current_pos = 0
    for word in text.split():
        # Find word position in original text
        word_start = text.find(word, current_pos)
        if word_start == -1:
            word_start = current_pos
        word_end = word_start + len(word)
        
        # Get label for this token (use first character's label)
        token_label = char_labels[word_start] if word_start < len(char_labels) else 'O'
        
        # Handle case where token starts in middle of entity
        if token_label.startswith('I-') and (not labels or labels[-1] == 'O'):
            token_label = 'B-' + token_label[2:]
        
        tokens.append(word)
        labels.append(token_label)
        current_pos = word_end
    
    return tokens, labels


def convert_to_ner_format(data: List[Dict]) -> List[Dict]:
    """Convert all Label Studio tasks to NER training format"""
    ner_samples = []
    
    entity_counts = {}
    skipped = 0
    
    for task in data:
        text, entities = extract_annotations(task)
        
        # Skip if no text
        if not text.strip():
            skipped += 1
            continue
        
        tokens, labels = tokenize_with_labels(text, entities)
        
        # Count entities
        for label in labels:
            if label != 'O':
                entity_type = label.split('-')[1]
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
        
        ner_samples.append({
            'id': task.get('id', len(ner_samples)),
            'text': text,
            'tokens': tokens,
            'ner_tags': labels,
            'entities': entities  # Keep original spans for reference
        })
    
    print(f"\nConverted {len(ner_samples)} samples (skipped {skipped})")
    print(f"\nEntity distribution:")
    for entity, count in sorted(entity_counts.items(), key=lambda x: -x[1]):
        print(f"  {entity}: {count}")
    
    return ner_samples


def create_label_mappings(labels: Tuple[str, ...]) -> Tuple[Dict, Dict]:
    """Create label-to-id and id-to-label mappings"""
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    return label2id, id2label


def split_data(samples: List[Dict], train_ratio: float, seed: int) -> Tuple[List, List]:
    """Split data into train and validation sets"""
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)
    
    split_idx = int(len(shuffled) * train_ratio)
    train = shuffled[:split_idx]
    val = shuffled[split_idx:]
    
    print(f"\nData split:")
    print(f"  Train: {len(train)} samples")
    print(f"  Validation: {len(val)} samples")
    
    return train, val


def save_processed_data(train: List, val: List, label2id: Dict, id2label: Dict, output_dir: str):
    """Save processed data and label mappings"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save train/val splits
    with open(output_path / 'train.json', 'w') as f:
        json.dump(train, f, indent=2)
    
    with open(output_path / 'val.json', 'w') as f:
        json.dump(val, f, indent=2)
    
    # Save label mappings
    with open(output_path / 'label2id.json', 'w') as f:
        json.dump(label2id, f, indent=2)
    
    with open(output_path / 'id2label.json', 'w') as f:
        json.dump(id2label, f, indent=2)
    
    print(f"\nSaved to {output_path}/")
    print(f"  - train.json ({len(train)} samples)")
    print(f"  - val.json ({len(val)} samples)")
    print(f"  - label2id.json")
    print(f"  - id2label.json")


def main():
    print("=" * 60)
    print("Clinical Trial NER - Label Studio to Training Format")
    print("=" * 60)
    
    # Load data
    data = load_labelstudio_json(config.input_file)
    
    # Convert to NER format
    ner_samples = convert_to_ner_format(data)
    
    # Create label mappings
    label2id, id2label = create_label_mappings(config.labels)
    print(f"\nLabels: {len(label2id)} ({len(config.labels) // 2} entity types + O)")
    
    # Split data
    train, val = split_data(ner_samples, config.train_ratio, config.seed)
    
    # Save
    save_processed_data(train, val, label2id, id2label, config.output_dir)
    
    print("\n" + "=" * 60)
    print("Conversion complete! Next: Run 02_train_sapbert_ner.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
