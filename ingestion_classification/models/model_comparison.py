# Databricks notebook source
# MODEL COMPARISON: SapBERT vs Dual-Model (SapBERT + PubMedBERT)
#
# PURPOSE: Validate model approach using test protocols with standard clinical taxonomy
#
# OUTPUT: Comprehensive comparison report including:
#   - Overall Accuracy (with ground truth from extracted headings)
#   - Accuracy by Case Type (Short/Long/Ambiguous)
#   - Speed Comparison
#
# DATA SOURCE: Clinical trial protocols from ClinicalTrials.gov test dataset
# CATEGORIES: Initial 87 standard categories from biomedical document taxonomy
# PyMuPDF have been used instead of Dockling because it is faster
# COMMAND ----------

# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple
import time
import pandas as pd
import fitz  # PyMuPDF 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load both models
print("\nLoading SapBERT...")
sap_tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
sap_model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(device)
sap_model.eval()

print("Loading PubMedBERT...")
pub_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
pub_model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract").to(device)
pub_model.eval()

print("Both models loaded")

# COMMAND ----------

# MAGIC %md
# MAGIC # Embedding Functions

# COMMAND ----------

def embed_sapbert(texts: List[str]) -> np.ndarray:
    with torch.no_grad():
        inputs = sap_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        outputs = sap_model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy()

def embed_pubmed(texts: List[str]) -> np.ndarray:
    with torch.no_grad():
        inputs = pub_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        outputs = pub_model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy()

def embed_dual(texts: List[str], sap_weight: float = 0.7) -> np.ndarray:
    sap_emb = embed_sapbert(texts)
    pub_emb = embed_pubmed(texts)
    fused = sap_weight * sap_emb + (1 - sap_weight) * pub_emb
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    return fused / norms

# COMMAND ----------

# MAGIC %md
# MAGIC # Standard Category Definitions

# COMMAND ----------

# 87 initial standard categories from biomedical document taxonomy
# Please note i'm increasing the categories so the category number might not match
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
    "HBA1C": ["hba1c", "hemoglobin a1c", "glycated hemoglobin", "a1c"],
    "HBA1C_TARGET": ["hba1c target", "a1c goal", "glycemic control target"],
    "CREATININE": ["creatinine", "serum creatinine", "scr"],
    "CREATININE_CLEARANCE": ["creatinine clearance", "crcl", "cockcroft gault"],
    "EGFR": ["egfr", "estimated glomerular filtration rate", "gfr"],
    "EGFR_THRESHOLD": ["egfr threshold", "egfr cutoff", "egfr limit"],
    "BUN": ["bun", "blood urea nitrogen", "urea"],
    "ALBUMIN": ["albumin", "serum albumin"],
    "BILIRUBIN_TOTAL": ["total bilirubin", "tbil"],
    "BILIRUBIN_DIRECT": ["direct bilirubin", "conjugated bilirubin", "dbil"],
    "BILIRUBIN_INDIRECT": ["indirect bilirubin", "unconjugated bilirubin"],
    "ALP": ["alkaline phosphatase", "alp", "alk phos"],
    "GGT": ["ggt", "gamma-glutamyl transferase", "gamma gt"],
    "INR": ["inr", "international normalized ratio", "prothrombin time"],
    "PTT": ["ptt", "aptt", "activated partial thromboplastin time"],
    "TSH": ["tsh", "thyroid stimulating hormone", "thyrotropin"],
    "FREE_T4": ["free t4", "ft4", "free thyroxine"],
    "FREE_T3": ["free t3", "ft3", "free triiodothyronine"],
    "PSA": ["psa", "prostate specific antigen"],
    "URINALYSIS": ["urinalysis", "ua", "urine analysis"],
    "PROTEIN_URINE": ["proteinuria", "urine protein", "protein in urine"],
    "MICROALBUMIN_URINE": ["microalbuminuria", "urine albumin", "albumin creatinine ratio"],
    
    # === INFECTIOUS DISEASE ===
    "HIV": ["hiv", "human immunodeficiency virus", "hiv test"],
    "HEPATITIS_B": ["hepatitis b", "hbv", "hbsag", "hepatitis b surface antigen"],
    "HEPATITIS_C": ["hepatitis c", "hcv", "hepatitis c antibody"],
    "TB": ["tuberculosis", "tb", "ppd", "quantiferon"],
    "COVID": ["covid", "covid-19", "sars-cov-2", "coronavirus"],
    "VACCINATION": ["vaccination", "immunization", "vaccine"],
    "VACCINATION_COVID": ["covid vaccine", "covid-19 vaccination"],
    "VACCINATION_INFLUENZA": ["flu vaccine", "influenza vaccine"],
    "VACCINATION_LIVE": ["live vaccine", "live attenuated vaccine"],
    
    # === CARDIAC ===
    "ECG": ["ecg", "ekg", "electrocardiogram"],
    "QTC_INTERVAL": ["qtc", "qtc interval", "corrected qt"],
    "QTC_PROLONGATION": ["qtc prolongation", "prolonged qtc", "long qt"],
    "EJECTION_FRACTION": ["ejection fraction", "lvef", "ef"],
    "TROPONIN": ["troponin", "cardiac troponin", "troponin i", "troponin t"],
    "BNP": ["bnp", "brain natriuretic peptide", "nt-probnp"],
    "ECHOCARDIOGRAM": ["echocardiogram", "echo", "transthoracic echo", "tte"],
    "STRESS_TEST": ["stress test", "exercise stress test", "cardiac stress"],
    "HOLTER_MONITOR": ["holter monitor", "ambulatory ecg", "24-hour ecg"],
    "CARDIAC_CATHETERIZATION": ["cardiac catheterization", "cardiac cath", "angiography"],
    
    # === IMAGING ===
    "CHEST_XRAY": ["chest x-ray", "chest radiograph", "cxr"],
    "CT_SCAN": ["ct scan", "computed tomography", "ct imaging"],
    "MRI": ["mri", "magnetic resonance imaging"],
    "PET_SCAN": ["pet scan", "positron emission tomography"],
    "ULTRASOUND": ["ultrasound", "sonography", "us"],
    "MAMMOGRAM": ["mammogram", "mammography", "breast imaging"],
    "DEXA_SCAN": ["dexa scan", "bone density", "dxa"],
    
    # === ONCOLOGY ===
    "TUMOR_SIZE": ["tumor size", "lesion size", "mass size"],
    "TUMOR_STAGE": ["tumor stage", "cancer stage", "tnm stage"],
    "TUMOR_GRADE": ["tumor grade", "histologic grade"],
    "TUMOR_MARKER": ["tumor marker", "cancer marker", "tumor antigen"],
    "CEA": ["cea", "carcinoembryonic antigen"],
    "CA125": ["ca-125", "ca 125", "cancer antigen 125"],
    "CA199": ["ca 19-9", "ca 199", "cancer antigen 19-9"],
    "AFP": ["afp", "alpha-fetoprotein"],
    "METASTASIS": ["metastasis", "metastatic disease", "distant spread"],
    "RECURRENCE": ["recurrence", "disease recurrence", "tumor recurrence"],
    "PERFORMANCE_STATUS": ["performance status", "ecog", "karnofsky"],
    
    # === PROCEDURES ===
    "BIOPSY": ["biopsy", "tissue sample", "histopathology"],
    "SURGERY": ["surgery", "surgical procedure", "operation"],
    "CHEMOTHERAPY": ["chemotherapy", "chemo", "cytotoxic therapy"],
    "RADIATION": ["radiation therapy", "radiotherapy", "radiation treatment"],
    "IMMUNOTHERAPY": ["immunotherapy", "immune checkpoint inhibitor"],
    "DIALYSIS": ["dialysis", "hemodialysis", "peritoneal dialysis"],
    "TRANSFUSION": ["transfusion", "blood transfusion"],
    
    # === PSYCHIATRIC ===
    "DEPRESSION": ["depression", "major depressive disorder", "mdd"],
    "ANXIETY": ["anxiety", "anxiety disorder", "gad"],
    "BIPOLAR": ["bipolar", "bipolar disorder", "manic depression"],
    "SCHIZOPHRENIA": ["schizophrenia", "psychotic disorder"],
    "PTSD": ["ptsd", "post-traumatic stress disorder"],
    "SUICIDAL_IDEATION": ["suicidal ideation", "suicidal thoughts", "suicide risk"],
    
    # === NEUROLOGICAL ===
    "SEIZURE": ["seizure", "epilepsy", "convulsion"],
    "STROKE": ["stroke", "cerebrovascular accident", "cva"],
    "DEMENTIA": ["dementia", "alzheimer", "cognitive impairment"],
    "PARKINSONS": ["parkinson", "parkinsons disease"],
    "NEUROPATHY": ["neuropathy", "peripheral neuropathy", "nerve damage"],
    
    # === STUDY-SPECIFIC ===
    "WASHOUT_PERIOD": ["washout period", "washout duration"],
    "PROHIBITED_MEDICATION": ["prohibited medication", "excluded medication", "forbidden drug"],
    "CONCOMITANT_MEDICATION": ["concomitant medication", "concurrent medication"],
    "PRIOR_THERAPY": ["prior therapy", "previous treatment", "prior treatment"],
    "DEVICE_IMPLANT": ["implanted device", "medical device", "implant"],
    "ALLERGY": ["allergy", "allergic reaction", "hypersensitivity"],
    "ADVERSE_EVENT": ["adverse event", "ae", "adverse reaction"],
}

print(f"Loaded {len(CATEGORY_DEFINITIONS)} category definitions")

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Test Data

# COMMAND ----------

# NOTE: In production, this would load from your clinical trial protocol dataset
# For this demonstration, we'll use synthetic test cases

def load_test_data() -> List[Dict]:
    """
    Load test cases from protocol documents.
    In production, this reads from your document extraction pipeline.
    """
    # Placeholder - replace with actual data loading
    test_cases = [
        {
            "document_id": "PROTOCOL_001",
            "text": "Subjects must have a BMI between 18.5 and 30 kg/m²",
            "ground_truth": "BMI_RANGE",
            "type": "short"
        },
        # Add more test cases from your dataset
    ]
    return test_cases

# Load test data
test_cases = load_test_data()
print(f"Loaded {len(test_cases)} test cases")

# COMMAND ----------

# MAGIC %md
# MAGIC # Precompute Category Embeddings

# COMMAND ----------

# Embed all category definitions
category_texts = []
category_labels = []

for cat, phrases in CATEGORY_DEFINITIONS.items():
    for phrase in phrases:
        category_texts.append(phrase)
        category_labels.append(cat)

print(f"Embedding {len(category_texts)} category phrases...")

# Get embeddings for all three approaches
sap_cat_embs = embed_sapbert(category_texts)
pub_cat_embs = embed_pubmed(category_texts)
dual_cat_embs = embed_dual(category_texts)

print("Category embeddings computed")

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Comparison

# COMMAND ----------

def predict_category(text_embedding: np.ndarray, cat_embeddings: np.ndarray) -> Tuple[str, float]:
    """Return predicted category and confidence score."""
    similarities = text_embedding @ cat_embeddings.T
    best_idx = similarities.argmax()
    return category_labels[best_idx], similarities[best_idx]

results = []

for tc in test_cases:
    text = tc["text"]
    ground_truth = tc["ground_truth"]
    
    # Get embeddings for test text
    sap_emb = embed_sapbert([text])[0]
    pub_emb = embed_pubmed([text])[0]
    dual_emb = embed_dual([text])[0]
    
    # Predict with each approach
    sap_pred, sap_conf = predict_category(sap_emb, sap_cat_embs)
    pub_pred, pub_conf = predict_category(pub_emb, pub_cat_embs)
    dual_pred, dual_conf = predict_category(dual_emb, dual_cat_embs)
    
    results.append({
        "document_id": tc["document_id"],
        "text": text,
        "type": tc["type"],
        "ground_truth": ground_truth,
        "sap_pred": sap_pred,
        "sap_conf": float(sap_conf),
        "sap_correct": sap_pred == ground_truth,
        "pub_pred": pub_pred,
        "pub_conf": float(pub_conf),
        "pub_correct": pub_pred == ground_truth,
        "dual_pred": dual_pred,
        "dual_conf": float(dual_conf),
        "dual_correct": dual_pred == ground_truth,
    })

df = pd.DataFrame(results)

# COMMAND ----------

# MAGIC %md
# MAGIC # Overall Accuracy

# COMMAND ----------

# Calculate accuracy
sap_acc = df['sap_correct'].mean() * 100
pub_acc = df['pub_correct'].mean() * 100
dual_acc = df['dual_correct'].mean() * 100

print(f"""
OVERALL ACCURACY (from {len(df)} test cases across {df['document_id'].nunique()} protocols):

  {'Model':<20} {'Accuracy':>12} {'Difference vs SapBERT':>25}
  {'-'*60}
  {'SapBERT Only':<20} {sap_acc:>11.1f}% {'—':>25}
  {'PubMedBERT Only':<20} {pub_acc:>11.1f}% {f'{pub_acc - sap_acc:+.1f}%':>25}
  {'Dual (70/30)':<20} {dual_acc:>11.1f}% {f'{dual_acc - sap_acc:+.1f}%':>25}
""")

# By case type
print(f"\nACCURACY BY CASE TYPE:")
print(f"\n  {'Case Type':<15} {'Count':>8} {'SapBERT':>12} {'PubMedBERT':>12} {'Dual':>12}")
print(f"  {'-'*60}")

for case_type in ["short", "long", "ambiguous"]:
    subset = df[df['type'] == case_type]
    if len(subset) > 0:
        sap_type_acc = subset['sap_correct'].mean() * 100
        pub_type_acc = subset['pub_correct'].mean() * 100
        dual_type_acc = subset['dual_correct'].mean() * 100
        print(f"  {case_type.title():<15} {len(subset):>8} {sap_type_acc:>11.1f}% {pub_type_acc:>11.1f}% {dual_type_acc:>11.1f}%")

# Average confidence (correct predictions only)
print(f"\nAVERAGE CONFIDENCE (correct predictions only):")
sap_correct_conf = df[df['sap_correct']]['sap_conf'].mean() if df['sap_correct'].sum() > 0 else 0
pub_correct_conf = df[df['pub_correct']]['pub_conf'].mean() if df['pub_correct'].sum() > 0 else 0
dual_correct_conf = df[df['dual_correct']]['dual_conf'].mean() if df['dual_correct'].sum() > 0 else 0
print(f"  SapBERT:    {sap_correct_conf:.3f}")
print(f"  PubMedBERT: {pub_correct_conf:.3f}")
print(f"  Dual:       {dual_correct_conf:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Speed Comparison

# COMMAND ----------

# Take sample for speed test
sample_texts = [tc["text"] for tc in test_cases[:50]]

# Benchmark SapBERT
start = time.time()
for _ in range(3):
    _ = embed_sapbert(sample_texts)
sap_time = (time.time() - start) / 3

# Benchmark Dual
start = time.time()
for _ in range(3):
    _ = embed_dual(sample_texts)
dual_time = (time.time() - start) / 3

print(f"""
SPEED COMPARISON ({len(sample_texts)} texts, avg of 3 runs):

  {'Metric':<25} {'SapBERT Only':>15} {'Dual Model':>15}
  {'-'*60}
  {'Inference Time':<25} {f'{sap_time*1000:.1f} ms':>15} {f'{dual_time*1000:.1f} ms':>15}
  {'Overhead':<25} {'—':>15} {f'+{(dual_time/sap_time - 1)*100:.0f}% slower':>15}
  {'Models to Load':<25} {'1':>15} {'2':>15}
  {'Memory Usage':<25} {'~1.5 GB':>15} {'~3.0 GB':>15}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion

# COMMAND ----------

print("RECOMMENDATION")

delta_dual_vs_sap = dual_acc - sap_acc

if delta_dual_vs_sap >= 5:
    print(f"""
DUAL MODEL JUSTIFIED

The dual approach shows +{delta_dual_vs_sap:.1f}% accuracy improvement over SapBERT alone.

Justification:
- SapBERT excels at short clinical terms (UMLS-aligned)
- PubMedBERT adds value for longer narrative sections  
- 70/30 fusion captures benefits of both
- Worth the ~2x inference time for {delta_dual_vs_sap:.1f}% accuracy gain

Recommend: Keep dual-model in production pipeline.
""")
elif delta_dual_vs_sap >= 2:
    print(f"""
MARGINAL BENEFIT

The dual approach shows +{delta_dual_vs_sap:.1f}% accuracy improvement.

This is a marginal gain that requires evaluation of accuracy vs complexity trade-off.

Options:
1. Keep dual if accuracy is critical
2. Use SapBERT-only for simplicity
3. Use dual only for ambiguous/long sections (hybrid approach)

Recommend: Evaluate based on production requirements.
""")
else:
    print(f"""
DUAL MODEL NOT JUSTIFIED

The dual approach shows only +{delta_dual_vs_sap:.1f}% accuracy difference.

SapBERT alone is sufficient for this classification task.

KEY FINDING:
The dual-model approach shows only ±{abs(delta_dual_vs_sap):.1f}% accuracy difference compared 
to SapBERT alone. This marginal difference does not justify the added complexity 
and inference overhead.

Recommend: Use SapBERT-only.
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Export Data for Report Generation

# COMMAND ----------

# Save results for report generation
report_data = {
    "test_count": len(df),
    "protocol_count": df['document_id'].nunique(),
    "category_count": len(CATEGORY_DEFINITIONS),
    "sap_accuracy": round(sap_acc, 1),
    "pub_accuracy": round(pub_acc, 1),
    "dual_accuracy": round(dual_acc, 1),
    "delta_dual_vs_sap": round(delta_dual_vs_sap, 1),
    "sap_time_ms": round(sap_time * 1000, 1),
    "dual_time_ms": round(dual_time * 1000, 1),
    "speed_overhead_pct": round((dual_time/sap_time - 1) * 100, 0),
    "by_type": {},
}

for case_type in ["short", "long", "ambiguous"]:
    subset = df[df['type'] == case_type]
    if len(subset) > 0:
        report_data["by_type"][case_type] = {
            "count": len(subset),
            "sap_acc": round(subset['sap_correct'].mean() * 100, 1),
            "pub_acc": round(subset['pub_correct'].mean() * 100, 1),
            "dual_acc": round(subset['dual_correct'].mean() * 100, 1),
        }

print("Report Data for PDF Generation:")
print(report_data)

# COMMAND ----------

# MAGIC %md
# MAGIC # Save Results to Delta Table

# COMMAND ----------

results_df = spark.createDataFrame(df)
results_df.write.format("delta").option("mergeSchema", "true").mode("overwrite").saveAsTable(
    "clinical_demo.nlp_analysis.model_comparison_results"
)
print("Results saved to clinical_demo.nlp_analysis.model_comparison_results")

# COMMAND ----------

print("TEST COMPLETE")
print(f"Test cases:     {len(df)}")
print(f"Protocols:      {df['document_id'].nunique()}")
print(f"Categories:     {len(CATEGORY_DEFINITIONS)}")
print(f"SapBERT:        {sap_acc:.1f}%")
print(f"PubMedBERT:     {pub_acc:.1f}%")
print(f"Dual:           {dual_acc:.1f}%")
print(f"Delta:          {delta_dual_vs_sap:+.1f}%")
