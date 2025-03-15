from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load Model & Vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    
    sentiment = "Positive" if prediction[0] == 1 else "Negative"
    return jsonify({"text": text, "sentiment": sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
