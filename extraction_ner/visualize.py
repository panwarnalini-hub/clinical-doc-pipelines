"""
Clinical Trial NER - Visualization
Author: Nalini Panwar
Date: December 2025

Generates HTML visualization of NER predictions.
"""

import json
from pathlib import Path
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

# Entity colors
ENTITY_COLORS = {
    'CONDITION': '#ff6b6b',
    'DRUG': '#4ecdc4',
    'DOSAGE': '#45b7d1',
    'STUDY_PHASE': '#96ceb4',
    'ENDPOINT': '#ffeaa7',
    'PATIENT_CRITERIA': '#dfe6e9',
    'BIOMARKER': '#a29bfe',
    'ENDPOINT_TYPE': '#fd79a8',
}


class NERVisualizer:
    def __init__(self, model_path: Path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        with open(model_path / 'id2label.json', 'r') as f:
            self.id2label = {int(k): v for k, v in json.load(f).items()}
    
    def predict(self, text: str) -> List[Dict]:
        words = text.split()
        
        word_spans = []
        current_pos = 0
        for word in words:
            start = text.find(word, current_pos)
            end = start + len(word)
            word_spans.append((start, end))
            current_pos = end
        
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
            predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()
        
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
                
                if current_entity and current_entity['label'] == entity_type:
                    start, end = word_spans[word_idx]
                    current_entity['end'] = end
                    current_entity['text'] = text[current_entity['start']:current_entity['end']]
                else:
                    if current_entity:
                        entities.append(current_entity)
                    
                    start, end = word_spans[word_idx]
                    current_entity = {
                        'text': text[start:end],
                        'label': entity_type,
                        'start': start,
                        'end': end
                    }
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def render_html(self, text: str, entities: List[Dict]) -> str:
        if not entities:
            return f'<span>{text}</span>'
        
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        
        html_parts = []
        last_end = 0
        
        for ent in sorted_entities:
            if ent['start'] > last_end:
                html_parts.append(text[last_end:ent['start']])
            
            color = ENTITY_COLORS.get(ent['label'], '#e9ecef')
            entity_html = f'<span class="entity" style="background-color: {color}">{ent["text"]}<span class="entity-label">{ent["label"]}</span></span>'
            html_parts.append(entity_html)
            
            last_end = ent['end']
        
        if last_end < len(text):
            html_parts.append(text[last_end:])
        
        return ''.join(html_parts)
    
    def generate_visualization(self, examples: List[str], output_path: Path):
        # Generate legend
        legend_items = []
        for entity_type, color in ENTITY_COLORS.items():
            legend_items.append(
                f'<div class="legend-item">'
                f'<div class="legend-color" style="background-color: {color}"></div>'
                f'<span>{entity_type}</span>'
                f'</div>'
            )
        legend_html = '\n'.join(legend_items)
        
        # Generate examples
        examples_html = []
        for i, text in enumerate(examples, 1):
            entities = self.predict(text)
            rendered = self.render_html(text, entities)
            
            example_html = f'''
            <div class="example">
                <div class="example-label">Example {i}</div>
                <div class="example-text">{rendered}</div>
            </div>
            '''
            examples_html.append(example_html)
        
        # Build HTML manually (avoiding .format() issues with CSS)
        final_html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Clinical Trial NER - Results</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 40px 20px;
            background: #f8f9fa;
            color: #2d3436;
        }}
        h1 {{ text-align: center; color: #2d3436; margin-bottom: 10px; }}
        .subtitle {{ text-align: center; color: #636e72; margin-bottom: 40px; }}
        .metrics {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 40px;
            flex-wrap: wrap;
        }}
        .metric {{
            background: white;
            padding: 20px 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #0984e3; }}
        .metric-label {{ color: #636e72; font-size: 0.9em; }}
        .legend {{
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 15px;
            margin-bottom: 40px;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; font-size: 0.9em; }}
        .legend-color {{ width: 20px; height: 20px; border-radius: 4px; }}
        .example {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .example-label {{ font-size: 0.8em; color: #636e72; margin-bottom: 10px; }}
        .example-text {{ font-size: 1.1em; line-height: 2; }}
        .entity {{
            padding: 3px 8px;
            border-radius: 4px;
            margin: 0 2px;
            white-space: nowrap;
        }}
        .entity-label {{
            font-size: 0.7em;
            font-weight: bold;
            margin-left: 5px;
            vertical-align: super;
            opacity: 0.8;
        }}
        .footer {{ text-align: center; margin-top: 50px; color: #636e72; font-size: 0.9em; }}
        .tech-stack {{
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        .tech-badge {{ background: #e9ecef; padding: 5px 12px; border-radius: 15px; font-size: 0.85em; }}
    </style>
</head>
<body>
    <h1>Clinical Trial NER</h1>
    <p class="subtitle">Named Entity Recognition for Clinical Trial Protocols</p>
    
    <div class="metrics">
        <div class="metric">
            <div class="metric-value">74.1%</div>
            <div class="metric-label">F1 Score</div>
        </div>
        <div class="metric">
            <div class="metric-value">76.5%</div>
            <div class="metric-label">Precision</div>
        </div>
        <div class="metric">
            <div class="metric-value">73.8%</div>
            <div class="metric-label">Recall</div>
        </div>
        <div class="metric">
            <div class="metric-value">8</div>
            <div class="metric-label">Entity Types</div>
        </div>
    </div>
    
    <div class="legend">
        {legend_html}
    </div>
    
    {''.join(examples_html)}
    
    <div class="footer">
        <p><strong>Model:</strong> Fine-tuned SapBERT with class weighting</p>
        <p><strong>Training Data:</strong> 100 annotated clinical trial sentences</p>
        <div class="tech-stack">
            <span class="tech-badge">SapBERT</span>
            <span class="tech-badge">PyTorch</span>
            <span class="tech-badge">HuggingFace</span>
            <span class="tech-badge">Label Studio</span>
            <span class="tech-badge">ClinicalTrials.gov</span>
        </div>
        <p style="margin-top: 20px;">Built by Nalini Panwar | December 2025</p>
    </div>
</body>
</html>'''
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
        
        print(f"Visualization saved to: {output_path}")
        return output_path


def main():
    model_path = PROJECT_ROOT / "models" / "sapbert_ner" / "final"
    output_path = PROJECT_ROOT / "outputs" / "ner_visualization.html"
    
    visualizer = NERVisualizer(model_path)
    
    examples = [
        "Patients with metastatic breast cancer will receive Pembrolizumab 200mg IV every 3 weeks.",
        "This Phase III trial evaluates Overall Survival in HER2-negative patients.",
        "Inclusion: Age >= 18 years, ECOG performance status 0-2, signed informed consent.",
        "Primary endpoint: Progression-Free Survival (PFS) at 12 months.",
        "Durvalumab combined with Carboplatin for advanced NSCLC treatment.",
        "Exclusion: Prior treatment with checkpoint inhibitors or immunotherapy.",
        "Phase I dose escalation study of pembrolizumab in solid tumors.",
        "Secondary endpoints include Duration of Response (DOR) and Disease Control Rate (DCR).",
    ]
    
    visualizer.generate_visualization(examples, output_path)
    print(f"\nOpen the HTML file in your browser to view the results!")


if __name__ == "__main__":
    main()
