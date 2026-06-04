import streamlit as st

st.set_page_config(page_title="About", page_icon="📊", layout="wide")

st.title("📊 About This Project")

st.markdown("""
## Healthcare Claims Intelligence Suite

Built by **Darshan Balaji** as a portfolio project demonstrating end-to-end 
AI development in the healthcare claims domain.

---

## Data Sources

**CMS Medicare Part B Physician & Other Practitioners Dataset (2024)**
- 500,000 rows sampled from 3.25GB public dataset
- Source: data.cms.gov
- Used for: EDA, CPT category distributions, ICD classifier training text

**Synthetic Claims Dataset (50,000 records)**
- Generated using Python Faker library
- Denial labels assigned using realistic probability rules based on known 
  payer patterns (ICD-CPT mismatch, late filing, upcoding signals)
- Used for: denial predictor and anomaly detector training

---

## Models

| Model | Algorithm | Primary Metric |
|-------|-----------|---------------|
| Denial Predictor | Logistic Regression (threshold=0.4) | Recall = 0.683 |
| Anomaly Detector | Isolation Forest (fresh fit per batch) | Precision@20 = 17/20 |
| ICD Classifier | Naive Bayes + TF-IDF | Top-3 Accuracy = 0.9999* |

*Near-perfect accuracy reflects data leakage in the CMS dataset — identical 
procedure descriptions appear across many providers. Performance on novel 
clinical language will be lower.

---

## Known Limitations

- **Synthetic denial labels** — real claim-level denial data is HIPAA protected 
  and not publicly available. In production, CARC codes from payer remittance 
  advice would replace synthetic labels.
- **Recall gap** — denial predictor achieves 0.683 recall vs 0.80 target. 
  Gap is attributed to limited signal in synthetic data, not model architecture.
- **Specialty bias** — Psychiatry, Pain Management, and Oncology are predicted 
  as denied for nearly all claims regardless of other features. A production 
  system would require fairness-aware resampling.
- **ICD classifier leakage** — near-perfect test accuracy reflects duplicate 
  descriptions in CMS data, not true generalization to clinical language.

---

## What I Would Do Differently

1. Use real claim-level data with actual denial outcomes from payer remittance 
   advice (CARC codes) instead of synthetic labels
2. Deduplicate CMS procedure descriptions before train-test split to get an 
   honest evaluation of the ICD classifier
3. Add fairness-aware resampling by specialty to address the bias identified 
   in high-denial specialties
4. Implement model monitoring to detect distribution shift when the app is 
   used with real claims data

---

## Technical Stack

- **Python 3.10** — core language
- **scikit-learn** — ML models and evaluation
- **pandas / numpy** — data processing
- **Streamlit** — web application framework
- **Anthropic Claude API** — AI explanations
- **CMS Medicare data** — real-world billing patterns

---

## AI Development Lifecycle (AIDLC) Phases

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Problem Definition | PROBLEM_STATEMENT.md |
| 2 | Data Collection & EDA | 01_eda.ipynb |
| 3 | Data Preparation | 02_data_prep.ipynb |
| 4 | Model Development | 03_modeling.ipynb |
| 5 | Evaluation & Fairness | 04_evaluation.ipynb |
| 6 | Deployment | This Streamlit app |
""")