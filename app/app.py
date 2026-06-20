import streamlit as st
import joblib
import os
from preprocess import clean_text

# Cấu hình trang

st.set_page_config(page_title="Phishing Email Detection", page_icon="🛡️", layout="wide")


# Tải mô hình và TF-IDF
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rf_model_path = os.path.join(BASE_DIR, "model", "rf_model.pkl")
tfidf_path = os.path.join(BASE_DIR, "model", "tfidf.pkl")

rf_model = joblib.load(rf_model_path)
tfidf = joblib.load(tfidf_path)


# Sidebar

with st.sidebar:

    st.header("📋 Project Information")

    st.markdown("""
    **Đề tài:**

    Nghiên cứu phát hiện Phishing Email sử dụng  Machine Learning

    ---

    **Dataset**
    - CEAS_08

    **Machine Learning**
    - Random Forest

    **Text Representation**
    - TF-IDF

    **Programming Language**
    - Python

    **Framework**
    - Streamlit
    """)


# Tiêu đề
st.title("🛡️ Phishing Email Detection System")


st.divider()

# Nhập email

email_text = st.text_area(
    label="📧 Nhập nội dung Email cần kiểm tra:",
    height=250,
    placeholder="""
Ví dụ:

Dear customer,

Your account has been suspended.

Please verify your account immediately by clicking the link below:

http://fake-bank.com

Thank you.
""",
)
# Nút dự đoán
if st.button("🔍 Analyze Email", use_container_width=True):

    if email_text.strip() == "":
        st.warning("⚠️ Vui lòng nhập nội dung Email.")

    else:
        cleaned_text = clean_text(email_text)

        vector = tfidf.transform([cleaned_text])

        prediction = rf_model.predict(vector)[0]

        probability = rf_model.predict_proba(vector)

        phishing_prob = probability[0][1] * 100

        st.divider()

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error(f"⚠️ Phishing Email Detected")
            st.write(f"Confidence: {phishing_prob:.2f}%")

        else:
            st.success("✅ Legitimate Email")
            st.write(f"Confidence: {100 - phishing_prob:.2f}%")

st.divider()

# Chân
st.caption("Nguyen Thi Minh Hang - Mac Quynh Mai - Le Thi Minh Tam")
