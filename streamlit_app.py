import streamlit as st
import pickle
import os

st.title("🧠 AI Mental Health Sentiment Analyzer")

# Use absolute paths
model_path = r"D:\talk-sense\models\sentiment_model.pkl"
vectorizer_path = r"D:\talk-sense\models\tfidf_vectorizer.pkl"

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

user_input = st.text_area("Enter a sentence to analyze:")
if user_input:
    features = vectorizer.transform([user_input])
    prediction = model.predict(features)[0]
    st.write("### 🩵 Predicted Emotion/Sentiment:")
    st.success(prediction)
