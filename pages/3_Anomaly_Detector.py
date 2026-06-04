import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(page_title="Anomaly Detector", page_icon="⚠️", layout="wide")

st.title("⚠️ Anomaly Detector")
st.markdown("Upload a CSV of claims to flag statistically unusual billing patterns.")
st.warning("⚠️ This tool was trained on synthetic data and is for portfolio demonstration only.")

st.markdown("### Required CSV columns:")
st.code("billed_amount, days_to_submission, provider_specialty, cpt_code, patient_age, service_date, is_denied")

uploaded_file = st.file_uploader("Upload claims CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown(f"**{len(df)} claims loaded**")
    st.dataframe(df.head())

    try:
        # Recreate engineered features
        cpt_medians = df.groupby('cpt_code')['billed_amount'].median()
        df['billed_to_median_ratio'] = df.apply(
            lambda row: row['billed_amount'] / cpt_medians.get(
                row['cpt_code'], row['billed_amount']), axis=1)

        percentile_95 = df.groupby('cpt_code')['billed_amount'].transform(
            lambda x: x.quantile(0.95))
        df['high_cost_outlier'] = (
            df['billed_amount'] > percentile_95).astype(int)

        specialty_denial_rates = {
            'Cardiology': 0.24, 'Dermatology': 0.25,
            'Emergency Medicine': 0.23, 'Family Practice': 0.23,
            'Gastroenterology': 0.25, 'Internal Medicine': 0.25,
            'Neurology': 0.25, 'Oncology': 0.35,
            'Ophthalmology': 0.24, 'Orthopedic Surgery': 0.23,
            'Pain Management': 0.34, 'Physical Therapy': 0.24,
            'Psychiatry': 0.35, 'Radiology': 0.26, 'Urology': 0.24
        }
        df['provider_denial_rate'] = df['provider_specialty'].map(
            specialty_denial_rates).fillna(0.25)

        df['duplicate_flag'] = df.duplicated(
            subset=['patient_age', 'cpt_code', 'service_date'],
            keep=False).astype(int)

        df['filing_delay_bucket_61+'] = (
            df['days_to_submission'] > 60).astype(int)

        anomaly_features = [
            'billed_to_median_ratio',
            'high_cost_outlier',
            'provider_denial_rate',
            'duplicate_flag',
            'filing_delay_bucket_61+'
        ]

        X_anomaly = df[anomaly_features]

        # Fresh scaler and model fit on uploaded batch
        fresh_scaler = StandardScaler()
        X_scaled = fresh_scaler.fit_transform(X_anomaly)

        fresh_iso = IsolationForest(contamination=0.05, random_state=42)
        fresh_iso.fit(X_scaled)

        df['anomaly_score'] = fresh_iso.decision_function(X_scaled)
        df['is_anomaly'] = fresh_iso.predict(X_scaled)
        df['anomaly_label'] = df['is_anomaly'].map(
            {1: '✅ Normal', -1: '🚨 Anomaly'})

        # Summary metrics
        n_anomalies = (df['is_anomaly'] == -1).sum()
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Claims", len(df))
        col2.metric("Anomalies Flagged", n_anomalies)
        col3.metric("Anomaly Rate", f"{n_anomalies/len(df):.1%}")

        st.info("""
        **Fresh retraining applied** — the anomaly detector was fit on this 
        uploaded batch, so anomalies are flagged relative to *this dataset's* 
        own patterns rather than the synthetic training data. This makes the 
        results meaningful regardless of the data source.
        """)

        # Top anomalies
        st.markdown("### 🚨 Top 10 Most Anomalous Claims")
        top_anomalies = df.nsmallest(10, 'anomaly_score')[
            ['provider_specialty', 'cpt_code', 'billed_amount',
             'days_to_submission', 'billed_to_median_ratio',
             'duplicate_flag', 'anomaly_score', 'anomaly_label']
        ]
        st.dataframe(top_anomalies, use_container_width=True)

        # Download
        st.markdown("### Download Full Results")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv,
            file_name="anomaly_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
        st.markdown("Make sure your CSV has the required columns listed above.")