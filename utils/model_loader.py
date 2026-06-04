import joblib
import os
import streamlit as st

@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.dirname(__file__))
    
    lr = joblib.load(os.path.join(base, 'models/denial_predictor_lr.pkl'))
    iso = joblib.load(os.path.join(base, 'models/anomaly_detector.pkl'))
    scaler = joblib.load(os.path.join(base, 'models/anomaly_scaler.pkl'))
    nb = joblib.load(os.path.join(base, 'models/icd_classifier.pkl'))
    vectorizer = joblib.load(os.path.join(base, 'models/tfidf_vectorizer.pkl'))
    
    return lr, iso, scaler, nb, vectorizer
