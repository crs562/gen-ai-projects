from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import numpy as np
import pickle

app = Flask(__name__)
model = load_model("sentiment_model.h5")
max_len = 500

# Load word index
word_index = {k: (v + 3) for k, v in pickle.load(open("word_index.pkl", "rb")).items()}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    review = request.form["review"]
    words = review.lower().split()
    encoded = [word_index.get(w, 2) for w in words]  # 2 is OOV token
    padded = sequence.pad_sequences([encoded], maxlen=max_len)
    prediction = model.predict(padded)[0][0]
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    return render_template("index.html", sentiment=sentiment, confidence=f"{prediction:.2f}")

if __name__ == "__main__":
    app.run(debug=True)