import streamlit as st

st.set_page_config(page_title="Home", page_icon="🏥", layout="wide")

st.title("🏥 Healthcare Claims Intelligence Suite")
st.markdown("### Built by Darshan Balaji")

st.markdown("---")

st.markdown("""
## Why Claims Denial Prediction Matters

Imagine you have just been diagnosed with diabetes. Your doctor orders a 
blood test. Three weeks later you receive a letter: *claim denied.*

In the US alone, over **$262 billion** is lost annually in administrative 
overhead tied to claim denials. A wrong denial has real consequences — 
providers lose revenue when appeal windows close, and patients face delays 
in care while disputes are resolved.

## What This Tool Does

This application gives medical billers and revenue cycle managers three 
AI-powered modules to identify problems before a claim ever reaches the payer:

| Module | What it does |
|--------|-------------|
| 🔍 Denial Predictor | Enter a claim and get a denial risk score with explanation |
| ⚠️ Anomaly Detector | Upload a batch of claims and flag statistical outliers |
| 🏷️ ICD Classifier | Type a diagnosis description and get predicted CPT categories |

## Key Findings from the Data

- Claims filed after 60 days show **nearly double** the denial rate of on-time claims
- Psychiatry, Pain Management, and Oncology show denial rates **8-10% above average**
- Extreme billing (>3x CPT median) is the strongest individual predictor of denial
- Mental health claims show different coding patterns that affect model performance

## Fairness Note

This model was evaluated for fairness across ICD chapters, provider 
specialties, and patient age groups. A known limitation is that high-denial 
specialties (Psychiatry, Pain Management, Oncology) are predicted as denied 
for nearly all claims regardless of other features — a bias that would need 
addressing in a production system.

## Data Sources

- **CMS Medicare Part B Physician & Other Practitioners dataset (2024)** — 
  real billing patterns used to inform synthetic data generation
- **Synthetic claims dataset (50,000 records)** — generated using realistic 
  denial probability rules based on known payer patterns

*This tool was trained on synthetic data and is intended as a portfolio 
demonstration. Predictions should not be used for real claims decisions.*
""")