import pandas as pd
import numpy as np
import string
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Load the dataset
data_path = 'data/spam.csv'  # Relative path assuming you're running from root directory
data = pd.read_csv(data_path, encoding='latin-1')[['v1', 'v2']]
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

# Clean messages
data['cleaned'] = data['message'].apply(clean_text)

# Feature extraction
X = data['cleaned']
y = data['label']
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ✅ Create models folder if not exists
os.makedirs('app/models', exist_ok=True)

# Save model and vectorizer
joblib.dump(model, 'app/models/spam_model.pkl')
joblib.dump(vectorizer, 'app/models/vectorizer.pkl')
