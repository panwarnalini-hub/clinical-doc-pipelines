"""
Clinical Trial NER - Streamlit Demo
Author: Nalini Panwar
December 2025

Demonstrates NER results on clinical trial protocols.
"""

import streamlit as st
import json
from pathlib import Path

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


@st.cache_data
def load_examples():
    """Load examples from JSON file"""
    examples_path = Path(__file__).parent / "examples.json"
    
    if examples_path.exists():
        with open(examples_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Fallback examples if file not found
    return [
        {
            "text": "Patients with metastatic breast cancer will receive Pembrolizumab 200mg IV every 3 weeks.",
            "entities": [
                {"text": "metastatic breast cancer", "label": "CONDITION", "start": 14, "end": 38},
                {"text": "Pembrolizumab", "label": "DRUG", "start": 52, "end": 65},
                {"text": "200mg IV every 3 weeks", "label": "DOSAGE", "start": 66, "end": 88}
            ]
        }
    ]


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
    st.title("Clinical Trial NER Demo")
    st.markdown("Named Entity Recognition for Clinical Trial Protocols")
    st.markdown("---")
    
    # Load examples
    examples = load_examples()
    
    # Info box
    st.info("""
    **Demo Mode:** This app shows pre-computed NER results from the fine-tuned SapBERT model.  
    For live inference and full pipeline, see the [GitHub repository](https://github.com/panwarnalini-hub/clinical-doc-pipelines).
    """)
    
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
        st.header("Model Performance")
        st.metric("F1 Score", "74.1%")
        st.metric("Precision", "76.5%")
        st.metric("Recall", "73.8%")
        
        st.markdown("---")
        st.markdown("**Technical Details:**")
        st.markdown("- Base: SapBERT (PubMedBERT)")
        st.markdown("- Training: 91 protocols")
        st.markdown("- Annotations: ~2,500 entities")
        st.markdown("- Framework: HuggingFace Transformers")
    
    # Main area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Example Results")
        
        # Example selector
        st.markdown("**Select an example:**")
        num_buttons = min(6, len(examples))
        button_cols = st.columns(num_buttons)
        
        # Initialize session state
        if 'selected_example' not in st.session_state:
            st.session_state.selected_example = 0
        
        # Create buttons
        for i in range(num_buttons):
            if button_cols[i].button(f"Example {i+1}", key=f"ex_{i}"):
                st.session_state.selected_example = i
        
        # Get selected example
        example = examples[st.session_state.selected_example]
        
        # Display
        st.markdown("**Input Text:**")
        st.code(example["text"], language=None)
        
        st.markdown("**Extracted Entities:**")
        rendered = render_entities(example["text"], example["entities"])
        st.markdown(
            f'<div style="background: white; padding: 20px; border-radius: 10px; font-size: 1.1em; line-height: 2; border: 1px solid #ddd;">{rendered}</div>',
            unsafe_allow_html=True
        )
        
        if example["entities"]:
            st.markdown("---")
            st.markdown("**Entity Breakdown:**")
            
            for ent in example["entities"]:
                color = ENTITY_COLORS.get(ent['label'], '#e9ecef')
                st.markdown(
                    f'<div style="margin: 8px 0;"><span style="background-color: {color}; padding: 3px 8px; border-radius: 4px; margin-right: 10px; font-weight: bold;">{ent["label"]}</span> {ent["text"]}</div>',
                    unsafe_allow_html=True
                )
    
    with col2:
        st.subheader("About This Project")
        st.markdown(f"""
        This demo showcases a Named Entity Recognition (NER) system for clinical trial protocols.
        
        **Pipeline Overview:**
        1. Document extraction (Docling)
        2. Section classification (87 categories)
        3. Entity extraction (8 types)
        
        **Use Cases:**
        - Automated protocol analysis
        - Clinical data extraction
        - Trial matching systems
        - Regulatory compliance
        
        **Architecture:**
        - Medallion (Bronze/Silver/Gold)
        - Azure Databricks + Delta Lake
        - Unity Catalog governance
        
        **Examples shown:** {len(examples)}
        
        ---
        
        **Full Pipeline:**  
        [GitHub Repository](https://github.com/panwarnalini-hub/clinical-doc-pipelines)
        """)
        
        st.markdown("---")
        st.markdown("**Nalini Panwar**")
        st.markdown("Lead Data Engineer")
        st.markdown("December 2025")
    
    # Show all examples at bottom
    with st.expander("View All Examples"):
        for i, ex in enumerate(examples):
            st.markdown(f"**Example {i+1}:**")
            rendered = render_entities(ex["text"], ex["entities"])
            st.markdown(
                f'<div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0; font-size: 1em; line-height: 1.8;">{rendered}</div>',
                unsafe_allow_html=True
            )


if __name__ == "__main__":
    main()
