import re
import nltk

from nltk.corpus import stopwords

# Chỉ tải lần đầu
try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))


def clean_text(text):

    # Chuyển về string
    text = str(text)

    # Chuyển chữ thường
    text = text.lower()

    # Xóa ký tự đặc biệt
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tách từ
    words = text.split()

    # Loại bỏ stopwords
    words = [word for word in words if word not in stop_words]

    # Ghép lại
    return " ".join(words)
