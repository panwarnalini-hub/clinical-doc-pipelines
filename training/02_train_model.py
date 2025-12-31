"""
Clinical Trial NER Pipeline - Step 2: Fine-tune SapBERT with Class Weighting
Author: Nalini Panwar
Date: December 2025

Handles class imbalance by weighting rare entity classes higher.
"""

import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from collections import Counter
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from sklearn.metrics import classification_report
import warnings
import os

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR


@dataclass
class TrainingConfig:
    model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    data_dir: Path = PROJECT_ROOT / "data" / "processed"
    output_dir: Path = PROJECT_ROOT / "models" / "sapbert_ner"
    
    max_length: int = 256
    batch_size: int = 16
    learning_rate: float = 3e-5  # Slightly higher
    num_epochs: int = 15  # More epochs
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    eval_steps: int = 25  # More frequent eval
    save_steps: int = 50
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


config = TrainingConfig()


class ClinicalNERDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, label2id: Dict, max_length: int = 256):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = sample['tokens']
        ner_tags = sample['ner_tags']
        
        label_ids = [self.label2id.get(tag, 0) for tag in ner_tags]
        
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                if word_idx < len(label_ids):
                    aligned_labels.append(label_ids[word_idx])
                else:
                    aligned_labels.append(-100)
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }


def compute_class_weights(train_data: List[Dict], label2id: Dict) -> torch.Tensor:
    """Compute inverse frequency weights for each class"""
    # Count all labels
    label_counts = Counter()
    for sample in train_data:
        for tag in sample['ner_tags']:
            label_counts[tag] += 1
    
    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")
    
    # Compute weights (inverse frequency)
    total = sum(label_counts.values())
    num_classes = len(label2id)
    
    weights = torch.ones(num_classes)
    for label, idx in label2id.items():
        count = label_counts.get(label, 1)
        # Inverse frequency with smoothing
        weights[idx] = total / (num_classes * count)
    
    # Cap extreme weights
    weights = torch.clamp(weights, min=0.5, max=10.0)
    
    # Boost B- tags specifically (they're rarer than I- tags)
    for label, idx in label2id.items():
        if label.startswith('B-'):
            weights[idx] *= 1.5
    
    print("\nClass weights:")
    for label, idx in sorted(label2id.items(), key=lambda x: x[1]):
        print(f"  {label}: {weights[idx]:.2f}")
    
    return weights


class WeightedTrainer(Trainer):
    """Custom trainer with class weights"""
    
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss


def load_data(data_dir: Path):
    print(f"Loading data from: {data_dir}")
    
    with open(data_dir / 'train.json', 'r') as f:
        train_data = json.load(f)
    
    with open(data_dir / 'val.json', 'r') as f:
        val_data = json.load(f)
    
    with open(data_dir / 'label2id.json', 'r') as f:
        label2id = json.load(f)
    
    with open(data_dir / 'id2label.json', 'r') as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    
    print(f"Loaded {len(train_data)} train, {len(val_data)} val samples")
    
    return train_data, val_data, label2id, id2label


def compute_metrics(eval_pred, id2label: Dict):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    
    true_labels = []
    pred_labels = []
    
    for pred_seq, label_seq in zip(predictions, labels):
        for pred, label in zip(pred_seq, label_seq):
            if label != -100:
                true_labels.append(id2label[label])
                pred_labels.append(id2label[pred])
    
    report = classification_report(
        true_labels, 
        pred_labels, 
        output_dict=True,
        zero_division=0
    )
    
    return {
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1': report['weighted avg']['f1-score'],
        'accuracy': report['accuracy']
    }


def train_model(
    train_data: List[Dict],
    val_data: List[Dict],
    label2id: Dict,
    id2label: Dict,
    config: TrainingConfig
):
    print(f"\n{'='*60}")
    print(f"Training SapBERT NER with Class Weighting")
    print(f"{'='*60}")
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute class weights
    class_weights = compute_class_weights(train_data, label2id)
    
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        config.model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    model.to(config.device)
    
    print("Creating datasets...")
    train_dataset = ClinicalNERDataset(train_data, tokenizer, label2id, config.max_length)
    val_dataset = ClinicalNERDataset(val_data, tokenizer, label2id, config.max_length)
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        report_to="none",
        save_total_limit=2,
    )
    
    def metrics_fn(eval_pred):
        return compute_metrics(eval_pred, id2label)
    
    # Use weighted trainer
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metrics_fn
    )
    
    print("\nStarting training with class weighting...")
    trainer.train()
    
    print("\nFinal evaluation...")
    results = trainer.evaluate()
    print(f"\nResults:")
    print(f"  Precision: {results['eval_precision']:.4f}")
    print(f"  Recall: {results['eval_recall']:.4f}")
    print(f"  F1: {results['eval_f1']:.4f}")
    print(f"  Accuracy: {results['eval_accuracy']:.4f}")
    
    final_path = config.output_dir / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving model to: {final_path}")
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    with open(final_path / 'label2id.json', 'w') as f:
        json.dump(label2id, f, indent=2)
    with open(final_path / 'id2label.json', 'w') as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)
    
    print(f"Model saved successfully!")
    
    return trainer, results


def main():
    print("=" * 60)
    print("Clinical Trial NER - SapBERT with Class Weighting")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    
    train_data, val_data, label2id, id2label = load_data(config.data_dir)
    
    trainer, results = train_model(
        train_data, val_data, label2id, id2label, config
    )
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {config.output_dir / 'final'}")
    print("Next: Run 03_inference.py to test")
    print("=" * 60)


if __name__ == "__main__":
    main()
