import streamlit as st
from transformers import pipeline
import pandas as pd
import altair as alt

# -----------------------------
# Load zero-shot classifier with custom message
# -----------------------------
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Placeholder for loading message
loading_placeholder = st.empty()  # empty container

# Show custom loading message
loading_placeholder.info("⚡ Loading AI model, please wait...")

# Load the classifier
classifier = load_classifier()

# Clear the message once loaded
loading_placeholder.empty()

# -----------------------------
# Candidate emotion labels
# -----------------------------
labels = [
    "happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral",
    "excited", "celebration", "grateful", "love", "motivated", "hopeful",
    "proud", "relieved", "peaceful", "optimistic", "content", "joyful", "cheerful",
    "anxious", "lonely", "tired", "frustrated", "guilty", "embarrassed",
    "jealous", "overwhelmed", "disappointed", "pessimistic", "stressed", "sadness",
    "confused", "curious", "nostalgic", "thoughtful", "shocked",
    "skeptical", "doubtful", "inspired", "amused", "indifferent",
    "empathetic", "forgiving", "resentful", "caring", "friendly", "hostile",
    "bored", "calm", "relaxed", "overjoyed", "confident", "helpless"
]

# -----------------------------
# Analyze emotion
# -----------------------------
def analyze_emotion(text: str):
    if not text or not isinstance(text, str) or text.strip() == "":
        return "neutral", 0.0, []
    
    result = classifier(text, candidate_labels=labels)
    return result['labels'][0], round(result['scores'][0], 3), result

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="AI Mental Health Sentiment Analyzer", layout="centered")
st.title("🧠 AI Mental Health Sentiment Analyzer")

user_input = st.text_area("Enter a sentence to analyze:")

if user_input:
    label, score, full_result = analyze_emotion(user_input)
    
    st.write("### 🩵 Predicted Emotion/Sentiment:")
    st.success(f"{label} ({score*100:.1f}% confidence)")
    
    st.write("### 🔹 Emotion Scores:")
    df_scores = pd.DataFrame({
        "Emotion": full_result['labels'],
        "Score": full_result['scores']
    })
    
    chart = alt.Chart(df_scores).mark_bar(color="#4CAF50").encode(
        x=alt.X('Score', title='Confidence'),
        y=alt.Y('Emotion', sort='-x', title='Emotion')
    ).properties(height=600)
    
    st.altair_chart(chart, use_container_width=True)
