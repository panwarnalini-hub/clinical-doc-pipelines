# Databricks notebook source
# SILVER SECTION CLASSIFICATION PIPELINE v10
#
# WHAT THIS NOTEBOOK DOES:
#   Classify silver_sections into clinical document categories using 
#   SapBERT embedding similarity against team-defined category prototypes.
#
# APPROACH (Per the direction 2025-12-11):
#   - Single model: SapBERT only (no PubMedBERT - avoiding unjustified complexity)
#   - Categories from team requirements sheet (Domain_and_Insight_Requirements_Oct_2025.xlsx)
#   - Section title embeddings → cosine similarity → category assignment
#
# INPUT:
#   silver_sections (from 02)
#   silver_section_embeddings (from 03)
#
# OUTPUT:
#   silver_section_classifications (Delta table)

# COMMAND ----------

# MAGIC %md
# MAGIC # Configuration

# COMMAND ----------

from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

@dataclass
class ClassificationConfig:
    # Unity Catalog
    catalog: str = "dev_clinical"
    schema: str = "doc_test"
    
    # Input Tables
    silver_sections: str = "silver_sections"
    silver_embeddings: str = "silver_section_embeddings"
    
    # Output Table
    silver_classifications: str = "silver_section_classifications"
    
    # Classification thresholds
    conf_threshold: float = 0.72       # Minimum score to accept (lowered for broader taxonomy)
    margin_gate: float = 0.02          # Minimum gap between #1 and #2
    multilabel_threshold: float = 0.68 # Score to include secondary labels
    within_delta: float = 0.04         # Or within this of top-1 score
    top_k: int = 5                     # Top classifications for audit
    
    # Model (SapBERT only - per the direction)
    embedding_model: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    
    # Versioning
    classifier_version: str = "2.0"
    classification_method: str = "SAPBERT_SIMILARITY"
    
    def full_table(self, table: str) -> str:
        return f"{self.catalog}.{self.schema}.{table}"

config = ClassificationConfig()

print(f"Input: {config.full_table(config.silver_sections)}")
print(f"Input: {config.full_table(config.silver_embeddings)}")
print(f"Output: {config.full_table(config.silver_classifications)}")
print(f"Model: {config.embedding_model}")
print(f"Method: {config.classification_method} (single-model approach)")

# COMMAND ----------

# MAGIC %md
# MAGIC # Category Definitions (Team Requirements Sheet)
# MAGIC 
# MAGIC Categories extracted from Domain_and_Insight_Requirements_Oct_2025.xlsx
# MAGIC Covers: Demographics, Reproductive, Lifestyle, Measurements, Informed Consent,
# MAGIC         Triage, Clinical Labs, Assessment domains

# COMMAND ----------

# Team-defined categories from requirements sheet
# Each category maps to matching terms for prototype embedding

CATEGORY_DEFINITIONS: Dict[str, List[str]] = {
    # === DEMOGRAPHICS ===
    "GENDER": ["gender", "sex", "male", "female"],
    "BMI_DISCRETE": ["bmi", "body mass index", "bmi at screening", "bmi measurement"],
    "BMI_PERCENTILE": ["bmi percentile", "bmi_percentile"],
    "BMI_RANGE": ["bmi between", "bmi above", "bmi below", "bmi range", "bmi threshold"],
    "BMI_QUALITATIVE": ["obesity", "overweight", "obese subjects", "obesity status"],
    "WEIGHT_MINIMUM": ["minimum weight", "must weigh at least", "weighs at least"],
    "WEIGHT_MAXIMUM": ["maximum weight", "must weigh no more than", "weight limit"],
    "WEIGHT_RANGE": ["weight range", "weight between", "weight from to"],
    "WEIGHT_GENDER_SPECIFIC": ["weight male", "weight female", "gender-specific weight"],
    "WEIGHT_PEDIATRIC": ["pediatric weight", "birth weight", "child weight", "infant weight"],
    "WEIGHT_STABILITY": ["weight stability", "stable weight", "weight change"],
    "ETHNICITY": ["ethnicity", "ethnic group", "ethnic origin"],
    
    # === REPRODUCTIVE ===
    "PREGNANCY_TEST_PRE": ["pregnancy test", "serum hcg", "urine hcg", "negative pregnancy"],
    "PREGNANCY_TEST_ON": ["monthly pregnancy test", "pregnancy test per cycle"],
    "PREGNANCY_TEST_POST": ["follow-up pregnancy test", "post-treatment pregnancy"],
    "PREGNANCY_ART": ["ivf", "icsi", "assisted reproduction", "art", "fertility treatment"],
    "PREGNANCY_REPORTING": ["pregnancy registry", "report pregnancy", "pregnancy exposure"],
    "PARTNER_CONTRACEPTION": ["partner contraception", "partner agrees to contraception"],
    "PARTNER_NOTIFICATION": ["notify if partner pregnant", "partner pregnancy notification"],
    "FERTILITY_NON_CHILDBEARING": ["postmenopausal", "hysterectomy", "non-childbearing potential"],
    "FERTILITY_CHILDBEARING": ["childbearing potential", "fertile", "wocbp"],
    "FERTILITY_STERILITY": ["surgically sterile", "tubal ligation", "vasectomy"],
    "CONTRACEPTION_BARRIER": ["condom", "diaphragm", "barrier method"],
    "CONTRACEPTION_HORMONAL": ["oral contraceptive", "ocp", "hormonal contraception"],
    "CONTRACEPTION_IUCD": ["iud", "intrauterine device", "iucd"],
    "CONTRACEPTION_ABSTINENCE": ["abstinence", "sexual abstinence"],
    "CONTRACEPTION_DURATION": ["contraception duration", "continue contraception"],
    "BREASTFEEDING": ["breastfeeding", "lactating", "nursing mothers"],
    "DRUG_WASHOUT": ["washout period", "drug washout", "medication washout"],
    
    # === LIFESTYLE ===
    "DIETARY_RESTRICTIONS": ["dietary restrictions", "diet requirements", "food restrictions"],
    "DIETARY_FASTING": ["fasting", "fasting requirements", "fasting status"],
    "DIETARY_SUPPLEMENT": ["dietary supplement", "vitamin", "herbal supplement"],
    "SMOKING_STATUS": ["smoking status", "current smoker", "non-smoker", "smoking history"],
    "SMOKING_CESSATION": ["smoking cessation", "quit smoking", "stop smoking"],
    "SMOKING_PACK_YEARS": ["pack years", "smoking pack-years"],
    "ALCOHOL_STATUS": ["alcohol use", "alcohol consumption", "drinking status"],
    "ALCOHOL_ABSTINENCE": ["alcohol abstinence", "no alcohol", "abstain from alcohol"],
    "ALCOHOL_LIMIT": ["alcohol limit", "drinks per week", "units per week"],
    "SUBSTANCE_USE": ["drug use", "substance use", "recreational drug", "illicit drug"],
    "SUBSTANCE_SCREENING": ["drug screen", "urine drug screen", "toxicology screen"],
    "PHYSICAL_ACTIVITY": ["physical activity", "exercise", "activity level"],
    "SLEEP_DISORDERS": ["sleep disorder", "insomnia", "sleep apnea", "sleep disturbance"],
    
    # === MEASUREMENTS / UNITS ===
    "UNITS_WEIGHT": ["kg", "kilograms", "lbs", "pounds"],
    "UNITS_HEIGHT": ["cm", "centimeters", "meters", "inches", "feet"],
    "UNITS_TEMPERATURE": ["celsius", "fahrenheit", "degrees c", "degrees f"],
    "UNITS_BLOOD_PRESSURE": ["mmhg", "mm hg"],
    "UNITS_LAB": ["mg/dl", "mmol/l", "g/l", "iu/ml", "miu/ml"],
    
    # === INFORMED CONSENT ===
    "INFORMED_CONSENT": ["informed consent", "written consent", "consent form", "icf"],
    "CONSENT_CAPACITY": ["consent capacity", "able to consent", "legally authorized"],
    "CONSENT_ASSENT": ["assent", "pediatric assent", "child assent"],
    "CONSENT_WITHDRAWAL": ["withdrawal of consent", "withdraw consent"],
    
    # === TRIAGE / VITALS ===
    "TEMPERATURE": ["temperature", "body temperature", "fever", "afebrile"],
    "HEART_RATE": ["heart rate", "pulse", "bpm", "resting heart rate"],
    "RESPIRATORY_RATE": ["respiratory rate", "breathing rate", "breaths per minute"],
    "BLOOD_PRESSURE_SYSTOLIC": ["systolic blood pressure", "sbp", "systolic bp"],
    "BLOOD_PRESSURE_DIASTOLIC": ["diastolic blood pressure", "dbp", "diastolic bp"],
    "BLOOD_PRESSURE_RANGE": ["blood pressure range", "bp between", "normotensive"],
    "BLOOD_PRESSURE_HYPERTENSION": ["hypertension", "high blood pressure", "elevated bp"],
    "BLOOD_PRESSURE_HYPOTENSION": ["hypotension", "low blood pressure"],
    
    # === CLINICAL LABS ===
    "HEMOGLOBIN": ["hemoglobin", "hgb", "hb level", "hemoglobin level"],
    "HEMOGLOBIN_RANGE": ["hemoglobin range", "hb between", "hemoglobin threshold"],
    "HEMATOCRIT": ["hematocrit", "hct", "packed cell volume"],
    "WBC": ["white blood cell", "wbc", "leukocyte count", "wbc count"],
    "WBC_DIFFERENTIAL": ["differential count", "neutrophils", "lymphocytes", "monocytes"],
    "RBC": ["red blood cell", "rbc", "erythrocyte count", "rbc count"],
    "RBC_TRANSFUSION": ["rbc transfusion", "packed rbc", "transfusion requirement"],
    "PLATELET_COUNT": ["platelet count", "platelets", "thrombocyte count"],
    "PLATELET_TRANSFUSION": ["platelet transfusion", "platelet support"],
    "PLATELET_FUNCTION": ["platelet function", "platelet aggregation", "bleeding time"],
    "LDL_C": ["ldl cholesterol", "ldl-c", "low density lipoprotein"],
    "LDL_C_FASTING": ["fasting ldl", "fasting ldl-c"],
    "LDL_C_TREATMENT": ["ldl therapy", "ldl treatment", "pcsk9", "statin therapy"],
    "LDL_APHERESIS": ["ldl apheresis", "lipid apheresis"],
    "HDL_C": ["hdl cholesterol", "hdl-c", "high density lipoprotein"],
    "HDL_C_FASTING": ["fasting hdl", "fasting hdl-c"],
    "ALT": ["alt", "alanine aminotransferase", "sgpt", "alt level"],
    "ALT_ELEVATED": ["elevated alt", "alt elevation", "alt above uln"],
    "AST": ["ast", "aspartate aminotransferase", "sgot", "ast level"],
    "AST_ELEVATED": ["elevated ast", "ast elevation", "ast above uln"],
    "GLUCOSE_FASTING": ["fasting glucose", "fasting blood glucose", "fpg"],
    "GLUCOSE_POSTPRANDIAL": ["postprandial glucose", "ogtt", "2-hour glucose"],
    "GLUCOSE_RANDOM": ["random glucose", "random blood glucose", "rpg"],
    "GLUCOSE_SELF_MONITORING": ["self-monitoring glucose", "smbg", "fingerstick glucose"],
    "GLUCOSE_THERAPY": ["glucose-lowering therapy", "antidiabetic medication"],
    "CREATININE": ["creatinine", "serum creatinine", "plasma creatinine"],
    "CREATININE_CLEARANCE": ["creatinine clearance", "crcl", "cockcroft-gault", "egfr"],
    "CREATININE_RATIO": ["protein creatinine ratio", "albumin creatinine ratio", "acr"],
    
    # === ASSESSMENTS ===
    "ECOG": ["ecog", "ecog performance status", "ecog score", "performance status"],
    "ECOG_TIMING": ["ecog at screening", "ecog at baseline", "ecog assessment"],
    "ECG": ["ecg", "electrocardiogram", "ekg", "12-lead ecg"],
    "ECG_QT": ["qt interval", "qtc", "qtcf", "qtcb", "corrected qt"],
    "ECG_RHYTHM": ["ecg rhythm", "sinus rhythm", "arrhythmia", "atrial fibrillation"],
    "RECIST": ["recist", "recist criteria", "tumor response", "measurable disease"],
    
    # === DOCUMENT STRUCTURE (preserved from original) ===
    "STUDY_POPULATION": ["study population", "participant selection", "patient population"],
    "INCLUSION_CRITERIA": ["inclusion criteria", "patient inclusion", "inclusion requirements"],
    "EXCLUSION_CRITERIA": ["exclusion criteria", "patient exclusion", "exclusion requirements"],
    "OBJECTIVES": ["study objectives", "trial objectives", "aims"],
    "PRIMARY_OBJECTIVE": ["primary objective", "primary aim", "main objective"],
    "SECONDARY_OBJECTIVE": ["secondary objective", "secondary aim"],
    "ENDPOINTS": ["endpoints", "outcome measures", "study outcomes"],
    "PRIMARY_ENDPOINT": ["primary endpoint", "primary outcome"],
    "SECONDARY_ENDPOINT": ["secondary endpoint", "secondary outcome"],
    "STUDY_DESIGN": ["study design", "trial design", "design overview"],
    "ADVERSE_EVENTS": ["adverse events", "safety", "adverse reactions", "side effects"],
    "SAFETY_ASSESSMENTS": ["safety assessments", "safety evaluations", "safety monitoring"],
    "SCHEDULE_OF_ACTIVITIES": ["schedule of activities", "soa", "visit schedule"],
    "STATISTICAL_METHODS": ["statistical analysis", "statistical methods", "sample size"],
    "INFORMED_CONSENT_SECTION": ["informed consent", "consent process", "icf"],
    "DISCONTINUATION": ["discontinuation", "withdrawal criteria", "early termination"],
}

print(f"Loaded {len(CATEGORY_DEFINITIONS)} categories from team requirements")

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Output Table

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {config.full_table(config.silver_classifications)} (
    -- Keys
    document_id                 STRING,
    section_id                  STRING,
    
    -- Section info
    section_title               STRING,
    section_type                STRING,
    
    -- Primary classification
    classification              STRING,
    classification_status       STRING,      -- SUCCESS, UNMAPPED, AMBIGUOUS
    classification_confidence   FLOAT,
    classification_margin       FLOAT,
    
    -- Hierarchy (from team taxonomy)
    category_domain             STRING,      -- e.g., DEMOGRAPHICS, CLINICAL_LABS
    category_level1             STRING,      -- e.g., WEIGHT, HEMOGLOBIN
    
    -- Multi-label
    secondary_labels            ARRAY<STRING>,
    secondary_scores            ARRAY<FLOAT>,
    
    -- Audit
    top_k_labels                ARRAY<STRING>,
    top_k_scores                ARRAY<FLOAT>,
    
    -- Metadata
    classification_method       STRING,
    classification_version      STRING,
    embedding_model             STRING,
    created_at                  TIMESTAMP
)
USING DELTA
PARTITIONED BY (document_id)
""")

print(f"Table ready: {config.full_table(config.silver_classifications)}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Load SapBERT Model (Single Model Approach)

# COMMAND ----------

import torch
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load SapBERT - the ONLY embedding model (per the direction)
print(f"\nLoading model: {config.embedding_model}")
tokenizer = AutoTokenizer.from_pretrained(config.embedding_model)
model = AutoModel.from_pretrained(config.embedding_model).to(device)
model.eval()
print("✓ SapBERT loaded (single-model approach)")

def embed_texts(texts: List[str], normalize: bool = True) -> np.ndarray:
    """Embed text using SapBERT with mean pooling."""
    if not texts:
        return np.array([])
    
    with torch.no_grad():
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        outputs = model(**inputs)
        attention_mask = inputs["attention_mask"]
        embeddings = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        if normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        
        return pooled.cpu().numpy()

# Verify
test_emb = embed_texts(["test embedding"])
print(f"Embedding dimension: {test_emb.shape[1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Build Category Prototypes

# COMMAND ----------

def build_category_prototypes(category_defs: Dict[str, List[str]]) -> Tuple[List[str], np.ndarray]:
    """Create prototype embeddings by averaging synonym embeddings for each category."""
    labels = []
    prototypes = []
    
    print("Building category prototypes...")
    for category, synonyms in category_defs.items():
        # Embed all synonyms
        syn_embeddings = embed_texts(synonyms, normalize=True)
        
        # Average to create prototype
        prototype = syn_embeddings.mean(axis=0)
        prototype = prototype / np.linalg.norm(prototype)  # Re-normalize
        
        labels.append(category)
        prototypes.append(prototype)
        
    prototype_matrix = np.stack(prototypes, axis=0)
    print(f"✓ Built {len(labels)} category prototypes, shape: {prototype_matrix.shape}")
    
    return labels, prototype_matrix

category_labels, category_prototypes = build_category_prototypes(CATEGORY_DEFINITIONS)

# COMMAND ----------

# MAGIC %md
# MAGIC # Classification Functions

# COMMAND ----------

def classify_embedding(
    embedding: np.ndarray,
    proto_matrix: np.ndarray,
    labels: List[str],
    conf_threshold: float = 0.72,
    margin_gate: float = 0.02,
    multilabel_threshold: float = 0.68,
    within_delta: float = 0.04,
    top_k: int = 5
) -> Dict:
    """Classify an embedding against category prototypes using cosine similarity."""
    
    # Ensure normalized
    embedding = embedding / np.linalg.norm(embedding)
    
    # Cosine similarity (dot product of normalized vectors)
    scores = proto_matrix @ embedding
    
    # Sort descending
    sorted_indices = np.argsort(scores)[::-1]
    sorted_labels = [labels[i] for i in sorted_indices]
    sorted_scores = scores[sorted_indices]
    
    top1_label = sorted_labels[0]
    top1_score = float(sorted_scores[0])
    top2_score = float(sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
    margin = top1_score - top2_score
    
    # Determine status
    if top1_score < conf_threshold:
        status = "UNMAPPED"
        classification = "UNMAPPED"
    elif margin < margin_gate:
        status = "AMBIGUOUS"
        classification = top1_label
    else:
        status = "SUCCESS"
        classification = top1_label
    
    # Secondary labels (multi-label support)
    secondary_labels = []
    secondary_scores = []
    for i in range(1, len(sorted_labels)):
        score = float(sorted_scores[i])
        if score >= multilabel_threshold or (top1_score - score) <= within_delta:
            secondary_labels.append(sorted_labels[i])
            secondary_scores.append(score)
        else:
            break
    
    # Top-K for audit
    top_k_labels = sorted_labels[:top_k]
    top_k_scores = [float(s) for s in sorted_scores[:top_k]]
    
    return {
        "classification": classification,
        "status": status,
        "confidence": top1_score,
        "margin": margin,
        "secondary_labels": secondary_labels,
        "secondary_scores": secondary_scores,
        "top_k_labels": top_k_labels,
        "top_k_scores": top_k_scores,
    }

# Test
test_emb = embed_texts(["Inclusion Criteria"])[0]
test_result = classify_embedding(
    test_emb, 
    category_prototypes, 
    category_labels,
    conf_threshold=config.conf_threshold,
    margin_gate=config.margin_gate
)
print(f"Test: 'Inclusion Criteria' → {test_result['classification']} ({test_result['confidence']:.3f})")
print(f"  Status: {test_result['status']}, Margin: {test_result['margin']:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Category Domain Mapping

# COMMAND ----------

# Map categories to their domain hierarchy (for reporting)
CATEGORY_DOMAINS = {
    # Demographics
    "GENDER": ("DEMOGRAPHICS", "GENDER"),
    "BMI_DISCRETE": ("DEMOGRAPHICS", "BMI"),
    "BMI_PERCENTILE": ("DEMOGRAPHICS", "BMI"),
    "BMI_RANGE": ("DEMOGRAPHICS", "BMI"),
    "BMI_QUALITATIVE": ("DEMOGRAPHICS", "BMI"),
    "WEIGHT_MINIMUM": ("DEMOGRAPHICS", "WEIGHT"),
    "WEIGHT_MAXIMUM": ("DEMOGRAPHICS", "WEIGHT"),
    "WEIGHT_RANGE": ("DEMOGRAPHICS", "WEIGHT"),
    "WEIGHT_GENDER_SPECIFIC": ("DEMOGRAPHICS", "WEIGHT"),
    "WEIGHT_PEDIATRIC": ("DEMOGRAPHICS", "WEIGHT"),
    "WEIGHT_STABILITY": ("DEMOGRAPHICS", "WEIGHT"),
    "ETHNICITY": ("DEMOGRAPHICS", "ETHNICITY"),
    
    # Reproductive
    "PREGNANCY_TEST_PRE": ("REPRODUCTIVE", "PREGNANCY"),
    "PREGNANCY_TEST_ON": ("REPRODUCTIVE", "PREGNANCY"),
    "PREGNANCY_TEST_POST": ("REPRODUCTIVE", "PREGNANCY"),
    "PREGNANCY_ART": ("REPRODUCTIVE", "PREGNANCY"),
    "PREGNANCY_REPORTING": ("REPRODUCTIVE", "PREGNANCY"),
    "PARTNER_CONTRACEPTION": ("REPRODUCTIVE", "PARTNER_OBLIGATIONS"),
    "PARTNER_NOTIFICATION": ("REPRODUCTIVE", "PARTNER_OBLIGATIONS"),
    "FERTILITY_NON_CHILDBEARING": ("REPRODUCTIVE", "FERTILITY_STATUS"),
    "FERTILITY_CHILDBEARING": ("REPRODUCTIVE", "FERTILITY_STATUS"),
    "FERTILITY_STERILITY": ("REPRODUCTIVE", "FERTILITY_STATUS"),
    "CONTRACEPTION_BARRIER": ("REPRODUCTIVE", "CONTRACEPTION"),
    "CONTRACEPTION_HORMONAL": ("REPRODUCTIVE", "CONTRACEPTION"),
    "CONTRACEPTION_IUCD": ("REPRODUCTIVE", "CONTRACEPTION"),
    "CONTRACEPTION_ABSTINENCE": ("REPRODUCTIVE", "CONTRACEPTION"),
    "CONTRACEPTION_DURATION": ("REPRODUCTIVE", "CONTRACEPTION"),
    "BREASTFEEDING": ("REPRODUCTIVE", "BREASTFEEDING"),
    "DRUG_WASHOUT": ("REPRODUCTIVE", "DRUG_WASHOUT"),
    
    # Lifestyle
    "DIETARY_RESTRICTIONS": ("LIFESTYLE", "DIETARY"),
    "DIETARY_FASTING": ("LIFESTYLE", "DIETARY"),
    "DIETARY_SUPPLEMENT": ("LIFESTYLE", "DIETARY"),
    "SMOKING_STATUS": ("LIFESTYLE", "SMOKING"),
    "SMOKING_CESSATION": ("LIFESTYLE", "SMOKING"),
    "SMOKING_PACK_YEARS": ("LIFESTYLE", "SMOKING"),
    "ALCOHOL_STATUS": ("LIFESTYLE", "ALCOHOL"),
    "ALCOHOL_ABSTINENCE": ("LIFESTYLE", "ALCOHOL"),
    "ALCOHOL_LIMIT": ("LIFESTYLE", "ALCOHOL"),
    "SUBSTANCE_USE": ("LIFESTYLE", "SUBSTANCE_USE"),
    "SUBSTANCE_SCREENING": ("LIFESTYLE", "SUBSTANCE_USE"),
    "PHYSICAL_ACTIVITY": ("LIFESTYLE", "PHYSICAL_ACTIVITY"),
    "SLEEP_DISORDERS": ("LIFESTYLE", "SLEEP"),
    
    # Clinical Labs
    "HEMOGLOBIN": ("CLINICAL_LABS", "HEMOGLOBIN"),
    "HEMOGLOBIN_RANGE": ("CLINICAL_LABS", "HEMOGLOBIN"),
    "HEMATOCRIT": ("CLINICAL_LABS", "HCT"),
    "WBC": ("CLINICAL_LABS", "WBC"),
    "WBC_DIFFERENTIAL": ("CLINICAL_LABS", "WBC"),
    "RBC": ("CLINICAL_LABS", "RBC"),
    "RBC_TRANSFUSION": ("CLINICAL_LABS", "RBC"),
    "PLATELET_COUNT": ("CLINICAL_LABS", "PLATELET"),
    "PLATELET_TRANSFUSION": ("CLINICAL_LABS", "PLATELET"),
    "PLATELET_FUNCTION": ("CLINICAL_LABS", "PLATELET"),
    "LDL_C": ("CLINICAL_LABS", "LDL_C"),
    "LDL_C_FASTING": ("CLINICAL_LABS", "LDL_C"),
    "LDL_C_TREATMENT": ("CLINICAL_LABS", "LDL_C"),
    "LDL_APHERESIS": ("CLINICAL_LABS", "LDL_C"),
    "HDL_C": ("CLINICAL_LABS", "HDL_C"),
    "HDL_C_FASTING": ("CLINICAL_LABS", "HDL_C"),
    "ALT": ("CLINICAL_LABS", "ALT"),
    "ALT_ELEVATED": ("CLINICAL_LABS", "ALT"),
    "AST": ("CLINICAL_LABS", "AST"),
    "AST_ELEVATED": ("CLINICAL_LABS", "AST"),
    "GLUCOSE_FASTING": ("CLINICAL_LABS", "GLUCOSE"),
    "GLUCOSE_POSTPRANDIAL": ("CLINICAL_LABS", "GLUCOSE"),
    "GLUCOSE_RANDOM": ("CLINICAL_LABS", "GLUCOSE"),
    "GLUCOSE_SELF_MONITORING": ("CLINICAL_LABS", "GLUCOSE"),
    "GLUCOSE_THERAPY": ("CLINICAL_LABS", "GLUCOSE"),
    "CREATININE": ("CLINICAL_LABS", "CREATININE"),
    "CREATININE_CLEARANCE": ("CLINICAL_LABS", "CREATININE"),
    "CREATININE_RATIO": ("CLINICAL_LABS", "CREATININE"),
    
    # Assessment
    "ECOG": ("ASSESSMENT", "ECOG"),
    "ECOG_TIMING": ("ASSESSMENT", "ECOG"),
    "ECG": ("ASSESSMENT", "ECG"),
    "ECG_QT": ("ASSESSMENT", "ECG"),
    "ECG_RHYTHM": ("ASSESSMENT", "ECG"),
    "RECIST": ("ASSESSMENT", "RECIST"),
    
    # Document Structure
    "STUDY_POPULATION": ("ELIGIBILITY", "POPULATION"),
    "INCLUSION_CRITERIA": ("ELIGIBILITY", "INCLUSION"),
    "EXCLUSION_CRITERIA": ("ELIGIBILITY", "EXCLUSION"),
    "OBJECTIVES": ("STUDY_DESIGN", "OBJECTIVES"),
    "PRIMARY_OBJECTIVE": ("STUDY_DESIGN", "OBJECTIVES"),
    "SECONDARY_OBJECTIVE": ("STUDY_DESIGN", "OBJECTIVES"),
    "ENDPOINTS": ("STUDY_DESIGN", "ENDPOINTS"),
    "PRIMARY_ENDPOINT": ("STUDY_DESIGN", "ENDPOINTS"),
    "SECONDARY_ENDPOINT": ("STUDY_DESIGN", "ENDPOINTS"),
    "STUDY_DESIGN": ("STUDY_DESIGN", "DESIGN"),
    "ADVERSE_EVENTS": ("SAFETY", "ADVERSE_EVENTS"),
    "SAFETY_ASSESSMENTS": ("SAFETY", "ASSESSMENTS"),
    "SCHEDULE_OF_ACTIVITIES": ("OPERATIONS", "SOA"),
    "STATISTICAL_METHODS": ("ANALYSIS", "STATISTICS"),
    "INFORMED_CONSENT_SECTION": ("REGULATORY", "CONSENT"),
    "DISCONTINUATION": ("OPERATIONS", "DISCONTINUATION"),
}

def get_category_hierarchy(classification: str) -> Tuple[str, str]:
    """Get domain and level1 for a classification."""
    return CATEGORY_DOMAINS.get(classification, ("OTHER", "OTHER"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Silver Data

# COMMAND ----------

from pyspark.sql import functions as F

silver_sections_df = spark.table(config.full_table(config.silver_sections))
print(f"Silver sections: {silver_sections_df.count()} rows")

silver_embeddings_df = spark.table(config.full_table(config.silver_embeddings))
print(f"Silver embeddings: {silver_embeddings_df.count()} rows")

# Join sections with embeddings
sections_with_embeddings = (
    silver_sections_df.alias("s")
    .join(
        silver_embeddings_df.alias("e"),
        (F.col("s.document_id") == F.col("e.document_id")) &
        (F.col("s.section_id") == F.col("e.section_id")),
        "inner"
    )
    .select(
        F.col("s.document_id"),
        F.col("s.section_id"),
        F.col("s.content_text").alias("section_title"),
        F.col("s.section_type"),
        F.col("e.embedding_vector").alias("embedding"),
    )
)

print(f"Sections with embeddings: {sections_with_embeddings.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Classification

# COMMAND ----------

from datetime import datetime
import pandas as pd

sections_pdf = sections_with_embeddings.toPandas()
print(f"Classifying {len(sections_pdf)} sections...")

now = datetime.utcnow()
results = []

for idx, row in sections_pdf.iterrows():
    # Get pre-computed SapBERT embedding
    embedding = np.array(row["embedding"], dtype=np.float32)
    
    # Classify using single-model approach
    classification_result = classify_embedding(
        embedding,
        category_prototypes,
        category_labels,
        conf_threshold=config.conf_threshold,
        margin_gate=config.margin_gate,
        multilabel_threshold=config.multilabel_threshold,
        within_delta=config.within_delta,
        top_k=config.top_k
    )
    
    # Get hierarchy
    domain, level1 = get_category_hierarchy(classification_result["classification"])
    
    results.append({
        "document_id": row["document_id"],
        "section_id": row["section_id"],
        "section_title": row["section_title"],
        "section_type": row["section_type"],
        "classification": classification_result["classification"],
        "classification_status": classification_result["status"],
        "classification_confidence": classification_result["confidence"],
        "classification_margin": classification_result["margin"],
        "category_domain": domain,
        "category_level1": level1,
        "secondary_labels": classification_result["secondary_labels"],
        "secondary_scores": classification_result["secondary_scores"],
        "top_k_labels": classification_result["top_k_labels"],
        "top_k_scores": classification_result["top_k_scores"],
        "classification_method": config.classification_method,
        "classification_version": config.classifier_version,
        "embedding_model": config.embedding_model,
        "created_at": now,
    })
    
    if (idx + 1) % 100 == 0:
        print(f"  Classified {idx + 1}/{len(sections_pdf)} sections...")

print(f"✓ Classification complete: {len(results)} sections")

# COMMAND ----------

# MAGIC %md
# MAGIC # Document-Level Deduplication

# COMMAND ----------

# Clean up duplicate I/E within same document (keep highest confidence)

df_final = pd.DataFrame(results)

dedup_categories = ["INCLUSION_CRITERIA", "EXCLUSION_CRITERIA"]

for doc_id in df_final["document_id"].unique():
    doc_rows = df_final[df_final["document_id"] == doc_id]
    
    for category in dedup_categories:
        cat_mask = doc_rows["classification"] == category
        cat_rows = doc_rows[cat_mask]
        
        if len(cat_rows) > 1:
            # Keep highest confidence, demote others
            keep_idx = cat_rows["classification_confidence"].idxmax()
            drop_idx = cat_rows.index[cat_rows.index != keep_idx]
            df_final.loc[drop_idx, "classification"] = "OTHER_SECTION"
            df_final.loc[drop_idx, "category_domain"] = "OTHER"
            df_final.loc[drop_idx, "category_level1"] = "OTHER"

print(f"✓ Deduplication complete")

# COMMAND ----------

# MAGIC %md
# MAGIC # Write Results

# COMMAND ----------

results_df = spark.createDataFrame(df_final)

results_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(config.full_table(config.silver_classifications))

print(f"✓ Written to {config.full_table(config.silver_classifications)}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Classification Summary

# COMMAND ----------

print("=" * 60)
print("CLASSIFICATION SUMMARY")
print("=" * 60)

# Status distribution
print("\n📊 STATUS DISTRIBUTION")
status_counts = spark.sql(f"""
    SELECT classification_status, COUNT(*) as count,
           ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as pct
    FROM {config.full_table(config.silver_classifications)}
    GROUP BY classification_status
    ORDER BY count DESC
""")
display(status_counts)

# By domain
print("\n📊 BY DOMAIN")
domain_counts = spark.sql(f"""
    SELECT category_domain, COUNT(*) as count,
           ROUND(AVG(classification_confidence), 3) as avg_conf
    FROM {config.full_table(config.silver_classifications)}
    WHERE classification_status = 'SUCCESS'
    GROUP BY category_domain
    ORDER BY count DESC
""")
display(domain_counts)

# Top classifications
print("\n📊 TOP 15 CLASSIFICATIONS")
top_classes = spark.sql(f"""
    SELECT classification, category_domain, COUNT(*) as count,
           ROUND(AVG(classification_confidence), 3) as avg_conf
    FROM {config.full_table(config.silver_classifications)}
    WHERE classification_status = 'SUCCESS'
    GROUP BY classification, category_domain
    ORDER BY count DESC
    LIMIT 15
""")
display(top_classes)

# COMMAND ----------

# MAGIC %md
# MAGIC # Validation: Eligibility Sections

# COMMAND ----------

print("\n✓ INCLUSION CRITERIA")
inclusion = spark.sql(f"""
    SELECT document_id, section_title, classification_confidence
    FROM {config.full_table(config.silver_classifications)}
    WHERE classification = 'INCLUSION_CRITERIA'
    ORDER BY classification_confidence DESC
    LIMIT 10
""")
display(inclusion)

print("\n✓ EXCLUSION CRITERIA")
exclusion = spark.sql(f"""
    SELECT document_id, section_title, classification_confidence
    FROM {config.full_table(config.silver_classifications)}
    WHERE classification = 'EXCLUSION_CRITERIA'
    ORDER BY classification_confidence DESC
    LIMIT 10
""")
display(exclusion)

# COMMAND ----------

# MAGIC %md
# MAGIC # Finalize Document Status

# COMMAND ----------

completion_sql = f"""
MERGE INTO {config.full_table("document_registry")} AS d
USING (
    SELECT sec.document_id
    FROM {config.full_table(config.silver_sections)} sec
    INNER JOIN {config.full_table(config.silver_classifications)} cls
        ON sec.section_id = cls.section_id
    GROUP BY sec.document_id
    HAVING COUNT(*) = (
        SELECT COUNT(*)
        FROM {config.full_table(config.silver_sections)} s2
        WHERE s2.document_id = sec.document_id
    )
) AS done
ON d.document_id = done.document_id
WHEN MATCHED THEN
  UPDATE SET d.processing_status = 'DOCUMENT_PREPROCESSING_COMPLETE'
"""

spark.sql(completion_sql)
print("✓ Document registry updated")

# COMMAND ----------

print("=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)
print(f"Output: {config.full_table(config.silver_classifications)}")
print(f"Categories: {len(CATEGORY_DEFINITIONS)}")
print(f"Method: {config.classification_method}")
print(f"Version: {config.classifier_version}")
