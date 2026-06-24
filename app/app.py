import streamlit as st
import joblib
import os
import numpy as np

from preprocess import clean_text

# Page Configuration
st.set_page_config(
    page_title="Phishing Email Detection",
    page_icon="🛡️",
    layout="wide"
)

# Load Model Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rf_model_path = os.path.join(BASE_DIR, "model", "rf_model.pkl")
tfidf_path = os.path.join(BASE_DIR, "model", "tfidf.pkl")


@st.cache_resource
def load_model():
    rf_model = joblib.load(rf_model_path)
    tfidf = joblib.load(tfidf_path)
    return rf_model, tfidf


rf_model, tfidf = load_model()

# Title
st.title("🛡️ Phishing Email Detection System")

st.divider()

# Email Input
email_text = st.text_area(
    label="📧 Enter Email Content:",
    height=280,
    placeholder="""
Example:

Dear Customer,

Your account has been suspended.

Please verify your account immediately by clicking the link below:

http://fake-bank.com

Thank you.
"""
)

# Analyze Button
if st.button("🔍 Analyze Email", use_container_width=True):

    if not email_text.strip():
        st.warning("⚠️ Please enter an email.")
    else:

        # Preprocessing
        cleaned_text = clean_text(email_text)

        # Vectorization
        vector = tfidf.transform([cleaned_text])

        # Prediction
        prediction = rf_model.predict(vector)[0]
        probability = rf_model.predict_proba(vector)[0]

        legitimate_prob = probability[0] * 100
        phishing_prob = probability[1] * 100

        if prediction == 1:
            result_label = "Phishing Email"
            risk_level = "High Risk"
        else:
            result_label = "Legitimate Email"
            risk_level = "Low Risk"

        st.divider()

        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Legitimate Probability",
                f"{legitimate_prob:.2f}%"
            )

        with col2:
            st.metric(
                "Phishing Probability",
                f"{phishing_prob:.2f}%"
            )

        st.caption("Phishing Probability")
        st.progress(int(phishing_prob))

        if prediction == 1:
            st.error("⚠️ Phishing Email Detected")

            st.warning(
                "This email contains characteristics commonly associated with phishing attacks. "
                "Please be cautious before clicking links or sharing sensitive information."
            )
        else:
            st.success("✅ Legitimate Email")

            st.info(
                "This email was classified as legitimate based on the textual content analyzed by the model."
            )

        st.write(f"**Risk Level:** {risk_level}")

        # Top TF-IDF Keywords
        st.subheader("🔑 Important Keywords")

        feature_names = tfidf.get_feature_names_out()

        scores = vector.toarray()[0]

        top_indices = np.argsort(scores)[::-1]

        top_words = [
            feature_names[i]
            for i in top_indices
            if scores[i] > 0
        ][:10]

        if top_words:
            st.write(", ".join(top_words))
        else:
            st.write("No significant keywords detected.")

        # View Preprocessed Text
        with st.expander("🔎 View Preprocessed Text"):
            st.write(cleaned_text)

