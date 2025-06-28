from flask import Flask, request, render_template
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)

model = joblib.load('app/models/spam_model.pkl')
vectorizer = joblib.load('app/models/vectorizer.pkl')

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

@app.route("/", methods=["GET", "POST"])
def index():
    result = ""
    if request.method == "POST":
        message = request.form["message"]
        cleaned = clean_text(message)
        vect_msg = vectorizer.transform([cleaned])
        pred = model.predict(vect_msg)[0]
        result = "Spam" if pred == 1 else "Not Spam"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
