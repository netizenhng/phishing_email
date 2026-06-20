import pandas as pd
import re
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 1. Đọc dữ liệu
df = pd.read_csv("phishing_emails.csv")

# 2. Kiểm tra cột dữ liệu
print(df.columns)

# Giả sử dữ liệu có hai cột:
# - text: nội dung email
# - label: nhãn (1 = phishing, 0 = legitimate)

# 3. Làm sạch văn bản
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)                 # Xóa HTML
    text = re.sub(r'http\S+|www\S+', ' URL ', text)   # Thay URL bằng token URL
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)        # Thay địa chỉ email bằng token EMAIL
    text = re.sub(r'\d+', ' NUMBER ', text)           # Thay số bằng token NUMBER
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df["clean_text"] = df["text"].apply(preprocess_text)

# 4. Tách dữ liệu đầu vào và nhãn
X = df["clean_text"]
y = df["label"]

# 5. Chia tập dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Khởi tạo TF-IDF
tfidf = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

# 7. Biến đổi văn bản thành vector số
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 8. Khởi tạo mô hình Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'
)

# 9. Huấn luyện mô hình
model.fit(X_train_tfidf, y_train)

# 10. Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test_tfidf)

# 11. Đánh giá mô hình
acc = accuracy_score(y_test, y_pred)
pre = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Precision:", pre)
print("Recall:", rec)
print("F1-score:", f1)
print("Confusion Matrix:\n", cm)
print(classification_report(y_test, y_pred))

# 12. Lưu mô hình và bộ TF-IDF để dùng lại
joblib.dump(model, "phishing_rf_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")