"""
Clinical Trial NER - Streamlit Demo
Author: Nalini Panwar
December 2025

Run with: streamlit run app.py
"""

import streamlit as st
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import random

st.set_page_config(
    page_title="Clinical Trial NER",
    page_icon="🏥",
    layout="wide"
)

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

ENTITY_DESCRIPTIONS = {
    'CONDITION': 'Disease or medical condition',
    'DRUG': 'Medication or therapeutic agent',
    'DOSAGE': 'Drug dosage and administration',
    'STUDY_PHASE': 'Clinical trial phase (I, II, III, IV)',
    'ENDPOINT': 'Outcome measure (OS, PFS, ORR)',
    'PATIENT_CRITERIA': 'Inclusion/exclusion criteria',
    'BIOMARKER': 'Biological marker (HER2, PD-L1)',
    'ENDPOINT_TYPE': 'Primary or secondary endpoint',
}


def find_project_paths():
    """Find model and data paths - works from either project location"""
    app_dir = Path(__file__).parent.resolve()
    
    # Possible model paths
    model_candidates = [
        app_dir / "models" / "sapbert_ner" / "final",
        app_dir / "extraction_ner" / "models" / "sapbert_ner" / "final",
    ]
    
    # Possible data paths
    data_candidates = [
        app_dir / "data" / "processed",
        app_dir / "extraction_ner" / "data" / "processed",
    ]
    
    model_path = None
    for p in model_candidates:
        if p.exists():
            model_path = p
            break
    
    data_path = None
    for p in data_candidates:
        if p.exists():
            data_path = p
            break
    
    return model_path, data_path


def load_real_examples(data_path, num_examples=8):
    """Load examples from actual training data"""
    if data_path is None:
        return []
    
    examples = []
    
    # Try val.json first, then train.json
    for filename in ['val.json', 'train.json']:
        filepath = data_path / filename
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Get samples that have entities
            samples_with_entities = [
                d for d in data 
                if any(t != 'O' for t in d.get('ner_tags', []))
            ]
            
            # Pick diverse samples
            if len(samples_with_entities) >= num_examples:
                selected = random.sample(samples_with_entities, num_examples)
            else:
                selected = samples_with_entities
            
            for sample in selected:
                text = sample.get('text', ' '.join(sample.get('tokens', [])))
                if len(text) > 20:  # Skip very short texts
                    examples.append(text)
            
            if examples:
                break
    
    return examples[:num_examples]


@st.cache_resource
def load_model():
    """Load model once and cache"""
    model_path, _ = find_project_paths()
    
    if model_path is None:
        st.error("Model not found. Check that models/sapbert_ner/final exists.")
        return None, None, None, None
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path_str = str(model_path).replace("\\", "/")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path_str)
    model = AutoModelForTokenClassification.from_pretrained(model_path_str)
    model.to(device)
    model.eval()
    
    with open(model_path / 'id2label.json', 'r') as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    
    return tokenizer, model, id2label, device


def predict(text, tokenizer, model, id2label, device):
    """Run NER prediction"""
    words = text.split()
    
    word_spans = []
    current_pos = 0
    for word in words:
        start = text.find(word, current_pos)
        end = start + len(word)
        word_spans.append((start, end))
        current_pos = end
    
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors='pt',
        truncation=True,
        max_length=256,
        padding=True
    )
    
    word_ids = encoding.word_ids(batch_index=0)
    input_dict = {k: v.to(device) for k, v in encoding.items()}
    
    with torch.no_grad():
        outputs = model(**input_dict)
        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().tolist()
    
    entities = []
    current_entity = None
    prev_word_idx = None
    
    for idx, (pred_id, word_idx) in enumerate(zip(predictions, word_ids)):
        if word_idx is None or word_idx == prev_word_idx:
            continue
        prev_word_idx = word_idx
        
        label = id2label[pred_id]
        
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


def render_entities(text, entities):
    """Render text with highlighted entities"""
    if not entities:
        return text
    
    sorted_entities = sorted(entities, key=lambda x: x['start'])
    
    html_parts = []
    last_end = 0
    
    for ent in sorted_entities:
        if ent['start'] > last_end:
            html_parts.append(text[last_end:ent['start']])
        
        color = ENTITY_COLORS.get(ent['label'], '#e9ecef')
        entity_html = f'<mark style="background-color: {color}; padding: 2px 6px; border-radius: 4px; margin: 0 2px;">{ent["text"]} <sup style="font-size: 0.7em; opacity: 0.8;">{ent["label"]}</sup></mark>'
        html_parts.append(entity_html)
        
        last_end = ent['end']
    
    if last_end < len(text):
        html_parts.append(text[last_end:])
    
    return ''.join(html_parts)


def main():
    st.title("Clinical Trial NER")
    st.markdown("Named Entity Recognition for Clinical Trial Protocols")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading model..."):
        tokenizer, model, id2label, device = load_model()
    
    if tokenizer is None:
        return
    
    # Load real examples from data
    _, data_path = find_project_paths()
    real_examples = load_real_examples(data_path)
    
    # Fallback if no data found
    if not real_examples:
        real_examples = [
            "Patients with metastatic breast cancer will receive Pembrolizumab 200mg IV every 3 weeks.",
            "This Phase III trial evaluates Overall Survival in HER2-negative patients.",
            "Inclusion: Age >= 18 years, ECOG performance status 0-2, signed informed consent.",
            "Primary endpoint: Progression-Free Survival (PFS) at 12 months.",
        ]
    
    # Sidebar
    with st.sidebar:
        st.header("Entity Types")
        for entity, color in ENTITY_COLORS.items():
            st.markdown(
                f'<div style="display: flex; align-items: center; margin-bottom: 10px;">'
                f'<div style="width: 20px; height: 20px; background-color: {color}; border-radius: 4px; margin-right: 10px;"></div>'
                f'<div><strong>{entity}</strong><br><span style="font-size: 0.8em; color: gray;">{ENTITY_DESCRIPTIONS[entity]}</span></div>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        st.header("Model")
        st.markdown(f"**Device:** {device}")
        st.markdown("**Base:** SapBERT")
        st.markdown("**Training:** 100 samples")
        st.markdown("**F1:** 74.1%")
    
    # Main area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Input Text")
        
        # Example selector
        st.markdown("**Load example from training data:**")
        num_buttons = min(4, len(real_examples))
        example_cols = st.columns(num_buttons)
        for i in range(num_buttons):
            if example_cols[i].button(f"Sample {i+1}", key=f"ex_{i}"):
                st.session_state.input_text = real_examples[i]
        
        # Text input
        default_text = st.session_state.get('input_text', real_examples[0] if real_examples else "")
        text_input = st.text_area(
            "Enter clinical trial text:",
            value=default_text,
            height=100,
            label_visibility="collapsed"
        )
        
        if st.button("Extract Entities", type="primary"):
            if text_input.strip():
                with st.spinner("Processing..."):
                    entities = predict(text_input, tokenizer, model, id2label, device)
                
                st.subheader("Results")
                
                rendered = render_entities(text_input, entities)
                st.markdown(
                    f'<div style="background: white; padding: 20px; border-radius: 10px; font-size: 1.1em; line-height: 2;">{rendered}</div>',
                    unsafe_allow_html=True
                )
                
                if entities:
                    st.markdown("**Extracted Entities:**")
                    for ent in entities:
                        color = ENTITY_COLORS.get(ent['label'], '#e9ecef')
                        st.markdown(
                            f'<span style="background-color: {color}; padding: 3px 8px; border-radius: 4px; margin-right: 10px;">{ent["label"]}</span> {ent["text"]}',
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No entities found.")
            else:
                st.warning("Enter text to analyze.")
    
    with col2:
        st.subheader("About")
        st.markdown("""
        Extracts 8 entity types from clinical trial text:
        
        - Medical conditions
        - Drug names
        - Dosage information
        - Trial phases
        - Clinical endpoints
        - Patient criteria
        - Biomarkers
        - Endpoint types
        
        **Technical:**
        - Fine-tuned SapBERT
        - Class weighting for imbalance
        - BIO tagging scheme
        """)
        
        st.markdown("---")
        st.markdown("Nalini Panwar")
        st.markdown("December 2025")


if __name__ == "__main__":
    main()
