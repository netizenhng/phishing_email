import streamlit as st
import joblib
import os
from preprocess import clean_text

# CẤU HÌNH TRANG


st.set_page_config(page_title="Phishing Email Detection", page_icon="🛡️", layout="wide")


# TẢI MÔ HÌNH VÀ TF-IDF


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rf_model_path = os.path.join(BASE_DIR, "model", "rf_model.pkl")
tfidf_path = os.path.join(BASE_DIR, "model", "tfidf.pkl")

rf_model = joblib.load(rf_model_path)
tfidf = joblib.load(tfidf_path)


# SIDEBAR


with st.sidebar:
    st.header("📋 Project Information")

    st.markdown("""
    **Đề tài:**

    Nghiên cứu phát hiện Phishing Email sử dụng Machine Learning

    ---

    **Dataset**
    - CEAS_08
    - Nazario
    - Nigerian_Fraud
    - SpamAssassin

    **Machine Learning**
    - Random Forest

    **Text Representation**
    - TF-IDF

    **Programming Language**
    - Python

    **Framework**
    - Streamlit
    """)

    st.divider()

    st.success("✅ Model Loaded")


# TIÊU ĐỀ


st.title("🛡️ Phishing Email Detection System based on Text Content Analysis")

st.markdown("""
    Hệ thống hỗ trợ phân loại Email dựa trên nội dung văn bản, sử dụng kỹ thuật
    **TF-IDF** để biểu diễn văn bản và mô hình **Random Forest** để dự đoán.
    """)

st.divider()


# Ô NHẬP EMAIL


email_text = st.text_area(
    label="📧 Nhập nội dung Email cần kiểm tra:",
    height=280,
    placeholder="""
Ví dụ:

Dear customer,

Your account has been suspended.

Please verify your account immediately by clicking the link below:

http://fake-bank.com

Thank you.
""",
)


# NÚT PHÂN TÍCH


if st.button("🔍 Analyze Email", use_container_width=True):
    if email_text.strip() == "":
        st.warning("⚠️ Vui lòng nhập nội dung Email.")
    else:
        cleaned_text = clean_text(email_text)

        vector = tfidf.transform([cleaned_text])

        prediction = rf_model.predict(vector)[0]

        probability = rf_model.predict_proba(vector)[0]

        legitimate_prob = probability[0] * 100
        phishing_prob = probability[1] * 100

        st.divider()

        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Legitimate Probability", value=f"{legitimate_prob:.2f}%")

        with col2:
            st.metric(label="Phishing Probability", value=f"{phishing_prob:.2f}%")

        st.progress(int(phishing_prob))

        if prediction == 1:
            st.error("⚠️ Phishing Email Detected")
            risk_level = "High Risk"
            st.warning(
                "Email này có nhiều dấu hiệu giống Email lừa đảo. Người dùng cần thận trọng trước khi nhấn vào liên kết hoặc cung cấp thông tin cá nhân."
            )
        else:
            st.success("✅ Legitimate Email")
            risk_level = "Low Risk"
            st.info(
                "Email này được mô hình phân loại là Email hợp lệ dựa trên nội dung văn bản đã phân tích."
            )

        st.markdown(f"**Risk Level:** {risk_level}")

        with st.expander("🔎 View Preprocessed Text"):
            st.write(cleaned_text)

        with st.expander("⚙️ How the system works"):
            st.markdown("""
            Quy trình xử lý của hệ thống:

            1. Nhận nội dung Email từ người dùng.
            2. Tiền xử lý văn bản: chuyển chữ thường, loại bỏ URL, ký tự đặc biệt và stopwords.
            3. Chuyển văn bản đã làm sạch thành vector số bằng TF-IDF.
            4. Đưa vector vào mô hình Random Forest.
            5. Hiển thị kết quả phân loại và xác suất dự đoán.
            """)

st.divider()


# FOOTER


st.caption("Nguyen Thi Minh Hang - Mac Quynh Mai - Le Thi Minh Tam")
