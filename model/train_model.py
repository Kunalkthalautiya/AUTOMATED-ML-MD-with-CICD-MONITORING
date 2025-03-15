import joblib
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Sample dataset (Sentiment Analysis)
data = {"text": ["I love this!", "This is bad", "Awesome!", "Not great"],
        "label": [1, 0, 1, 0]}  # 1: Positive, 0: Negative

df = pd.DataFrame(data)

# Convert text to feature vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train a simple classifier
model = MultinomialNB()
model.fit(X, y)

# Save model & vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model trained and saved!")
