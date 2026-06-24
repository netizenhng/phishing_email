import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))
    
try:
    nltk.data.find("corpora/wordnet") 
except LookupError: 
    nltk.download("wordnet") 
    
try: 
    nltk.data.find("corpora/omw-1.4") 
except LookupError: 
    nltk.download("omw-1.4") 
    
lemmatizer = WordNetLemmatizer()

def clean_text(text): 
    text = str(text).lower() 
    text = re.sub(r"http\S+|www\S+", " URLTOKEN ", text) 
    text = re.sub(r"\S+@\S+", " EMAILTOKEN ", text) 
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text) 
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split() 
    words = [ lemmatizer.lemmatize(word) for word in words if word not in stop_words ]
    return " ".join(words)
