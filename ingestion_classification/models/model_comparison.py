# Databricks notebook source
# MODEL COMPARISON: SapBERT vs Dual-Model (SapBERT + PubMedBERT)
#
# PURPOSE: Validate model approach using 91 test protocols with the standard taxonomy
#
# OUTPUT: Same format as Model_Comparison_Report.pdf
#   - Overall Accuracy (with ground truth from extracted headings)
#   - Accuracy by Case Type (Short/Long/Ambiguous)
#   - Speed Comparison
#
# DATA SOURCE: the team's 91 protocol IDs from the test dataset
# CATEGORIES: 110 standard categories from Domain_and_Insight_Requirements_Oct_2025.xlsx

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

# 110 standard categories from Domain_and_Insight_Requirements_Oct_2025.xlsx
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
    
    # === DOCUMENT STRUCTURE ===
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

print(f"Loaded {len(CATEGORY_DEFINITIONS)} categories from standard team requirements")

# COMMAND ----------

# MAGIC %md
# MAGIC # Build Category Prototypes

# COMMAND ----------

def build_prototypes(embed_fn):
    """Build category prototypes using given embedding function."""
    labels = []
    prototypes = []
    for category, synonyms in CATEGORY_DEFINITIONS.items():
        embs = embed_fn(synonyms)
        proto = embs.mean(axis=0)
        proto = proto / np.linalg.norm(proto)
        labels.append(category)
        prototypes.append(proto)
    return labels, np.stack(prototypes)

print("Building prototypes for each model...")
labels_sap, protos_sap = build_prototypes(embed_sapbert)
labels_pub, protos_pub = build_prototypes(embed_pubmed)
labels_dual, protos_dual = build_prototypes(embed_dual)
print(f"Built prototypes for {len(labels_sap)} categories")

# COMMAND ----------

# MAGIC %md
# MAGIC # Load the team's 91 Protocol Documents

# COMMAND ----------

# 91 test protocols
TEST_PROTOCOL_IDS = [
    "PRO00031720", "PRO00031767", "PRO00031814", "PRO00032231", "PRO00032574", 
    "PRO00032782", "PRO00032795", "PRO00033629", "PRO00040027", "PRO00041923", 
    "PRO00042322", "PRO00042489", "PRO00042642", "PRO00043043", "PRO00043392", 
    "PRO00043470", "PRO00044090", "PRO00044744", "PRO00045171", "PRO00045520", 
    "PRO00045742", "PRO00045831", "PRO00045868", "PRO00045959", "PRO00047228", 
    "PRO00047313", "PRO00047500", "PRO00048064", "PRO00048461", "PRO00048540", 
    "PRO00048623", "PRO00049546", "PRO00050275", "PRO00050388", "PRO00051023", 
    "PRO00051089", "PRO00052298", "PRO00054008", "PRO00054752", "PRO00055683", 
    "PRO00056909", "PRO00057334", "PRO00057410", "PRO00057520", "PRO00058173", 
    "PRO00059175", "PRO00059536", "PRO00060414", "PRO00061037", "PRO00061963", 
    "PRO00063092", "PRO00063375", "PRO00064017", "PRO00064446", "PRO00064555", 
    "PRO00065263", "PRO00065371", "PRO00066019", "PRO00066459", "PRO00066518", 
    "PRO00067239", "PRO00067295", "PRO00067390", "PRO00067485", "PRO00068546", 
    "PRO00068874", "PRO00068993", "PRO00069703", "PRO00071169", "PRO00071865", 
    "PRO00071983", "PRO00072221", "PRO00072233", "PRO00072915", "PRO00073042", 
    "PRO00073846", "PRO00074091", "PRO00074461", "PRO00074527", "PRO00074629", 
    "PRO00075000", "PRO00075429", "PRO00077885", "PRO00078287", "PRO00078536", 
    "PRO00078669", "PRO00079960", "PRO00081136", "PRO00081211", "PRO00081261", 
    "PRO00081532"
]

print(f"Loading the team's {len(TEST_PROTOCOL_IDS)} protocol IDs...")

# Get PDF paths from test storage
protocol_list = "'" + "','".join(TEST_PROTOCOL_IDS) + "'"

docs_df = spark.sql(f"""
    SELECT 
        protocol_id,
        path,
        title
    FROM dev_clinical.doc_test.documents
    WHERE protocol_id IN ({protocol_list})
      AND path IS NOT NULL
""")

doc_count = docs_df.count()
print(f"Found {doc_count} documents in the test dataset")

# COMMAND ----------

# MAGIC %md
# MAGIC # Extract Sections with Type Classification (Short/Long/Ambiguous)

# COMMAND ----------

docs_pdf = docs_df.toPandas()

print(f"\nExtracting sections from {len(docs_pdf)} PDFs...")

test_cases = []  # Will hold (text, expected_category, case_type)

for idx, row in docs_pdf.iterrows():
    try:
        pdf_path = row['path']
        if pdf_path.startswith("dbfs:/Volumes/"):
            pdf_path = "/" + pdf_path.replace("dbfs:/", "")
        elif pdf_path.startswith("dbfs:"):
            pdf_path = pdf_path.replace("dbfs:", "/dbfs")
        
        doc = fitz.open(pdf_path)
        
        for page_num in range(min(15, len(doc))):  # First 15 pages
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        text = "".join([span["text"] for span in line["spans"]]).strip()
                        
                        if not text or len(text) < 3:
                            continue
                        
                        word_count = len(text.split())
                        avg_font_size = sum([span.get("size", 12) for span in line["spans"]]) / len(line["spans"])
                        is_bold = any(["bold" in span.get("font", "").lower() for span in line["spans"]])
                        
                        # Determine case type based on characteristics
                        if word_count <= 5 and (is_bold or text.isupper() or text.istitle() or avg_font_size >= 11):
                            case_type = "short"
                        elif word_count >= 15:
                            case_type = "long"
                        elif 5 < word_count < 15:
                            case_type = "ambiguous"
                        else:
                            continue
                        
                        # Try to determine expected category by matching against known patterns
                        text_lower = text.lower()
                        expected_category = None
                        
                        for category, synonyms in CATEGORY_DEFINITIONS.items():
                            for syn in synonyms:
                                if syn.lower() in text_lower:
                                    expected_category = category
                                    break
                            if expected_category:
                                break
                        
                        # Only include if we can determine expected category (for accuracy measurement)
                        if expected_category:
                            test_cases.append({
                                "text": text,
                                "expected": expected_category,
                                "type": case_type,
                                "document_id": row['protocol_id'],
                                "page_num": page_num + 1,
                                "word_count": word_count
                            })
        
        doc.close()
        
        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx + 1}/{len(docs_pdf)} documents, {len(test_cases)} test cases found...")
            
    except Exception as e:
        print(f"  Error processing {row['protocol_id']}: {str(e)[:50]}")
        continue

print(f"\nExtracted {len(test_cases)} test cases with ground truth labels")

# Summarize by type
test_df = pd.DataFrame(test_cases)
print(f"\nTest cases by type:")
print(test_df['type'].value_counts())

# COMMAND ----------

# MAGIC %md
# MAGIC # Run Classification Comparison

# COMMAND ----------

def classify(text: str, embed_fn, prototypes: np.ndarray, labels: List[str]) -> Tuple[str, float]:
    emb = embed_fn([text])[0]
    scores = prototypes @ emb
    best_idx = np.argmax(scores)
    return labels[best_idx], float(scores[best_idx])

print(f"Classifying {len(test_cases)} test cases with 3 models...")
start_time = time.time()

results = []

for idx, tc in enumerate(test_cases):
    text = tc["text"]
    expected = tc["expected"]
    case_type = tc["type"]
    
    # SapBERT only
    sap_label, sap_conf = classify(text, embed_sapbert, protos_sap, labels_sap)
    sap_correct = sap_label == expected
    
    # PubMedBERT only
    pub_label, pub_conf = classify(text, embed_pubmed, protos_pub, labels_pub)
    pub_correct = pub_label == expected
    
    # Dual (70/30)
    dual_label, dual_conf = classify(text, embed_dual, protos_dual, labels_dual)
    dual_correct = dual_label == expected
    
    results.append({
        "text": text[:60] + "..." if len(text) > 60 else text,
        "expected": expected,
        "type": case_type,
        "document_id": tc["document_id"],
        "sap_label": sap_label,
        "sap_conf": round(sap_conf, 3),
        "sap_correct": sap_correct,
        "pub_label": pub_label,
        "pub_conf": round(pub_conf, 3),
        "pub_correct": pub_correct,
        "dual_label": dual_label,
        "dual_conf": round(dual_conf, 3),
        "dual_correct": dual_correct,
    })
    
    if (idx + 1) % 500 == 0:
        print(f"  Processed {idx + 1}/{len(test_cases)}...")

elapsed = time.time() - start_time
df = pd.DataFrame(results)
print(f"\nClassification complete in {elapsed:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC # Results Analysis

# COMMAND ----------

print("DETAILED RESULTS (Sample)")

for _, row in df.head(20).iterrows():
    print(f"\n[{row['type'].upper()}] {row['text']}")
    print(f"  Expected: {row['expected']}")
    print(f"  SapBERT:   {row['sap_label']:30} conf={row['sap_conf']:.3f} {'Yes' if row['sap_correct'] else 'No'}")
    print(f"  PubMedBERT:{row['pub_label']:30} conf={row['pub_conf']:.3f} {'Yes' if row['pub_correct'] else 'No'}")
    print(f"  Dual:      {row['dual_label']:30} conf={row['dual_conf']:.3f} {'Yes' if row['dual_correct'] else 'No'}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Summary Statistics

# COMMAND ----------

print("ACCURACY SUMMARY")

# Overall accuracy
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
- SapBERT excels at short biomedical terms (UMLS-aligned)
- PubMedBERT adds value for longer narrative sections  
- 70/30 fusion captures benefits of both
- Worth the ~2x inference time for {delta_dual_vs_sap:.1f}% accuracy gain

Recommend: Keep dual-model in production pipeline.
""")
elif delta_dual_vs_sap >= 2:
    print(f"""
MARGINAL BENEFIT

The dual approach shows +{delta_dual_vs_sap:.1f}% accuracy improvement.

This is a marginal gain that may not justify the added complexity.

Options:
1. Keep dual if accuracy is critical
2. Use SapBERT-only for simplicity (the team's preference)
3. Use dual only for ambiguous/long sections (hybrid approach)

Recommend: Discuss trade-off with the team.
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
# MAGIC # Export Data for PDF Report Generation

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
    "dev_clinical.doc_test.model_comparison_results"
)
print("Results saved to dev_clinical.doc_test.model_comparison_results")

# COMMAND ----------

print("TEST COMPLETE")
print(f"Test cases:     {len(df)}")
print(f"Protocols:      {df['document_id'].nunique()}")
print(f"Categories:     {len(CATEGORY_DEFINITIONS)}")
print(f"SapBERT:        {sap_acc:.1f}%")
print(f"PubMedBERT:     {pub_acc:.1f}%")
print(f"Dual:           {dual_acc:.1f}%")
print(f"Delta:          {delta_dual_vs_sap:+.1f}%")