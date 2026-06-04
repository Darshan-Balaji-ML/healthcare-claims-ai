import streamlit as st
import sys
import os
import anthropic

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.model_loader import load_models

st.set_page_config(page_title="ICD Classifier", page_icon="🏷️", layout="wide")

lr, iso, scaler, nb, vectorizer = load_models()

st.title("🏷️ ICD Code Classifier")
st.markdown("Enter a plain-English diagnosis description to get predicted CPT categories.")
st.warning("⚠️ This tool was trained on CMS procedure descriptions and is for portfolio demonstration only.")

st.markdown("""
### How to use
Type a diagnosis or procedure description in plain English. The model will predict 
the most likely CPT category and provide an AI explanation of the coding.
""")

description = st.text_area(
    "Diagnosis / Procedure Description",
    placeholder="e.g. patient presents with chest tightness and shortness of breath",
    height=100
)

if st.button("Classify"):
    if description.strip() == "":
        st.error("Please enter a description.")
    else:
        # Clean and vectorize
        import re
        from nltk.corpus import stopwords
        import nltk
        nltk.download('stopwords', quiet=True)
        stop_words = set(stopwords.words('english'))

        def clean_text(text):
            text = str(text).lower()
            text = ''.join([c for c in text if c.isalpha() or c == ' '])
            text = ' '.join([w for w in text.split() if w not in stop_words])
            return text

        cleaned = clean_text(description)
        X_tfidf = vectorizer.transform([cleaned])
        
        # Get top 3 predictions
        probs = nb.predict_proba(X_tfidf)[0]
        top3_indices = probs.argsort()[-3:][::-1]
        top3_categories = nb.classes_[top3_indices]
        top3_probs = probs[top3_indices]

        st.markdown("---")
        st.markdown("### Predicted CPT Categories")

        for i, (cat, prob) in enumerate(zip(top3_categories, top3_probs)):
            if i == 0:
                st.success(f"**#{i+1} {cat}** — {prob:.1%} confidence")
            else:
                st.info(f"**#{i+1} {cat}** — {prob:.1%} confidence")

        # Claude explanation
        client = anthropic.Anthropic()
        with st.spinner("Generating AI explanation..."):
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{
                    "role": "user",
                    "content": f"""You are a medical coding expert. A clinician entered 
                    this description: "{description}"
                    
                    The AI classifier predicted these CPT categories:
                    1. {top3_categories[0]} ({top3_probs[0]:.1%} confidence)
                    2. {top3_categories[1]} ({top3_probs[1]:.1%} confidence)
                    3. {top3_categories[2]} ({top3_probs[2]:.1%} confidence)
                    
                    In 2-3 sentences, explain what CPT category this description 
                    most likely belongs to and why. Be specific about what procedure 
                    or service category best fits this description."""
                }]
            )
        st.markdown("**AI Explanation:**")
        st.info(message.content[0].text)

        st.markdown("---")
        st.caption("""
        **Note:** This classifier was trained on CMS procedure descriptions and 
        achieves near-perfect accuracy due to data leakage in the training set. 
        Performance on novel clinical language may be lower. See the About page 
        for full methodology details.
        """)