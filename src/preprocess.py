# Load IMDb dataset, clean text by removing punctuation,
# converting to lowercase, removing stopwords,
# and return cleaned reviews and labels

import pandas as pd
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    return " ".join(words)

def load_data(path):
    df = pd.read_csv(path)
    df['review'] = df['review'].apply(clean_text)
    return df['review'], df['sentiment']
