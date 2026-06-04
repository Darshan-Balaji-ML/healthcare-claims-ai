import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import anthropic

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.model_loader import load_models

st.set_page_config(page_title="Denial Predictor", page_icon="🔍", layout="wide")

lr, iso, scaler, nb, vectorizer = load_models()

st.title("🔍 Claim Denial Predictor")
st.markdown("Enter claim details below to assess denial risk before submission.")
st.warning("⚠️ This tool was trained on synthetic data and is for portfolio demonstration only.")

col1, col2 = st.columns(2)

with col1:
    specialty = st.selectbox("Provider Specialty", [
        'Cardiology', 'Dermatology', 'Emergency Medicine',
        'Family Practice', 'Gastroenterology', 'Internal Medicine',
        'Neurology', 'Oncology', 'Ophthalmology', 'Orthopedic Surgery',
        'Pain Management', 'Physical Therapy', 'Psychiatry',
        'Radiology', 'Urology'
    ])
    icd_chapter = st.selectbox("ICD-10 Chapter", [
        'Chapter II - Neoplasms',
        'Chapter IV - Endocrine/Metabolic',
        'Chapter V - Mental Health',
        'Chapter VI - Nervous System',
        'Chapter IX - Circulatory',
        'Chapter X - Respiratory',
        'Chapter XI - Digestive',
        'Chapter XII - Skin',
        'Chapter XIII - Musculoskeletal',
        'Chapter XIV - Genitourinary',
        'Chapter XIX - Injury',
        'Chapter XXI - Preventive'
    ])
    icd_cpt_mismatch = st.checkbox("ICD-CPT Mismatch Flagged")

with col2:
    patient_age = st.slider("Patient Age", 18, 90, 65)
    billed_amount = st.number_input("Billed Amount ($)", 
                                     min_value=0.0, value=200.0, step=10.0)
    days_to_submission = st.slider("Days to Submission", 0, 90, 15)

if st.button("Predict Denial Risk"):
    # Build features
    cpt_medians = {
        '99213': 150, '99214': 220, '99232': 180, '93000': 90,
        '71046': 140, '27447': 8000, '80053': 60, '90658': 40,
        '99285': 600, '20610': 120, '97110': 80, '99223': 350,
        '43239': 1200, '70553': 1800, '99091': 55
    }
    avg_median = np.mean(list(cpt_medians.values()))
    billed_to_median_ratio = billed_amount / avg_median
    high_cost_outlier = int(billed_to_median_ratio > 3)
    duplicate_flag = 0

    specialty_denial_rates = {
        'Cardiology': 0.24, 'Dermatology': 0.25,
        'Emergency Medicine': 0.23, 'Family Practice': 0.23,
        'Gastroenterology': 0.25, 'Internal Medicine': 0.25,
        'Neurology': 0.25, 'Oncology': 0.35,
        'Ophthalmology': 0.24, 'Orthopedic Surgery': 0.23,
        'Pain Management': 0.34, 'Physical Therapy': 0.24,
        'Psychiatry': 0.35, 'Radiology': 0.26, 'Urology': 0.24
    }
    provider_denial_rate = specialty_denial_rates.get(specialty, 0.25)

    age_groups = {'<30': 0, '30-50': 0, '50-65': 0, '65-80': 0, '80+': 0}
    if patient_age < 30: age_groups['<30'] = 1
    elif patient_age < 50: age_groups['30-50'] = 1
    elif patient_age < 65: age_groups['50-65'] = 1
    elif patient_age < 80: age_groups['65-80'] = 1
    else: age_groups['80+'] = 1

    filing_buckets = {'0-7': 0, '8-30': 0, '31-60': 0, '61+': 0}
    if days_to_submission <= 7: filing_buckets['0-7'] = 1
    elif days_to_submission <= 30: filing_buckets['8-30'] = 1
    elif days_to_submission <= 60: filing_buckets['31-60'] = 1
    else: filing_buckets['61+'] = 1

    specialties_list = [
        'Cardiology', 'Dermatology', 'Emergency Medicine',
        'Family Practice', 'Gastroenterology', 'Internal Medicine',
        'Neurology', 'Oncology', 'Ophthalmology', 'Orthopedic Surgery',
        'Pain Management', 'Physical Therapy', 'Psychiatry',
        'Radiology', 'Urology'
    ]
    specialty_encoded = {f'provider_specialty_{s}': int(s == specialty)
                         for s in specialties_list}

    gender_encoded = {'patient_gender_F': 0, 'patient_gender_M': 1}

    icd_chapters = [
        'Chapter II - Neoplasms', 'Chapter IV - Endocrine/Metabolic',
        'Chapter IX - Circulatory', 'Chapter V - Mental Health',
        'Chapter VI - Nervous System', 'Chapter X - Respiratory',
        'Chapter XI - Digestive', 'Chapter XII - Skin',
        'Chapter XIII - Musculoskeletal', 'Chapter XIV - Genitourinary',
        'Chapter XIX - Injury', 'Chapter XXI - Preventive'
    ]
    icd_encoded = {f'icd_chapter_{c}': int(c == icd_chapter)
                   for c in icd_chapters}

    input_dict = {
        'billed_to_median_ratio': billed_to_median_ratio,
        'duplicate_flag': duplicate_flag,
        'provider_denial_rate': provider_denial_rate,
        'icd_cpt_mismatch': int(icd_cpt_mismatch),
        'high_cost_outlier': high_cost_outlier,
        **specialty_encoded,
        **gender_encoded,
        **{f'filing_delay_bucket_{k}': v for k, v in filing_buckets.items()},
        **{f'patient_age_group_{k}': v for k, v in age_groups.items()},
        **icd_encoded
    }

    input_df = pd.DataFrame([input_dict])
    X_test_cols = lr.feature_names_in_
    for col in X_test_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[X_test_cols]

    prob = lr.predict_proba(input_df)[0][1]

    st.markdown("---")
    if prob >= 0.6:
        st.error(f"🔴 High Denial Risk: {prob:.0%}")
    elif prob >= 0.4:
        st.warning(f"🟡 Medium Denial Risk: {prob:.0%}")
    else:
        st.success(f"🟢 Low Denial Risk: {prob:.0%}")

    # Top risk factors
    coefficients = lr.coef_[0]
    input_values = input_df.values[0]
    contributions = coefficients * input_values
    contrib_df = pd.DataFrame({
        'Feature': X_test_cols,
        'Contribution': contributions
    }).sort_values('Contribution', ascending=False)

    top_risks = contrib_df[contrib_df['Contribution'] > 0].head(5)
    if not top_risks.empty:
        st.markdown("**Top Risk Factors:**")
        for _, row in top_risks.iterrows():
            st.markdown(f"- `{row['Feature']}`")

    # Claude explanation
    client = anthropic.Anthropic()
    with st.spinner("Generating AI explanation..."):
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": f"""You are a healthcare billing expert. A claim has been 
                flagged by an AI denial predictor with the following details:
                - Denial probability: {prob:.0%}
                - Provider specialty: {specialty}
                - Days to submission: {days_to_submission}
                - Billed to median ratio: {billed_to_median_ratio:.2f}
                - ICD-CPT mismatch: {icd_cpt_mismatch}
                - ICD chapter: {icd_chapter}
                
                In 2-3 sentences, explain why this claim is at risk and what 
                a medical biller should do to address it. Be specific and practical."""
            }]
        )
    st.markdown("**AI Explanation:**")
    st.info(message.content[0].text)