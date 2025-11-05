import streamlit as st
from transformers import pipeline
import pandas as pd
import altair as alt

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AI Mental Health Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# Gradient background & custom styles
# -----------------------------
st.markdown("""
<style>
/* Page background gradient */
body {
    background: linear-gradient(135deg, #FFDEE9 0%, #B5FFFC 100%);
}

/* Header styling with pastel gradient frame */
.header-frame {
    background: linear-gradient(135deg, #FFC0CB, #87CEEB);  /* baby pink to sky blue */
    padding: 15px 30px;
    border-radius: 20px;
    text-align: center;
    display: inline-block;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
}

.header-frame h1 {
    color: #6B5B95;
    font-size: 36px;
    margin: 0;
    font-family: 'Bookman Old Style', serif;
}

/* Label for text area */
.stTextArea label {
    font-size: 24px;  /* bigger font for the label */
    font-weight: bold;
    font-family: 'Times New Roman', serif;
    color: #6B5B95;
}

/* Text area customization */
.stTextArea textarea {
    height: 120px;  
    width: 100%;
    font-size: 20px;
    font-family: 'Times New Roman', serif;
    padding: 20px;
    border-radius: 20px;
    border: 2px solid #6B5B95;
    box-shadow: 3px 3px 15px rgba(0,0,0,0.1);
    background-color: #fff0f5;
    color: #333;
}

/* Placeholder text color */
.stTextArea textarea::placeholder {
    color: black !important;   
    font-size: 18px;
    font-family: 'Times New Roman', serif;
}

/* Sidebar styling */
.stSidebar .sidebar-content {
    background-color: #F7F3E3;
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Custom header
# -----------------------------
st.markdown("""
<div class="header-frame">
    <h1>🧠 AI Mental Health Sentiment Analyzer</h1>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Load zero-shot classifier
# -----------------------------
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

with st.spinner("⚡ Loading AI model, please wait..."):
    classifier = load_classifier()

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
# Emojis for top emotions
# -----------------------------
emoji_dict = {
    "happy": "😄", "sad": "😢", "angry": "😡", "fearful": "😱", 
    "disgusted": "🤢", "surprised": "😲", "neutral": "😐",
    "excited": "🤩", "celebration": "🎉", "grateful": "🙏", "love": "❤️",
    "motivated": "💪", "hopeful": "🌈", "proud": "🏆", "relieved": "😌",
    "peaceful": "☮️", "optimistic": "🌞", "content": "😊", "joyful": "🥳",
    "cheerful": "😁", "anxious": "😰", "lonely": "😔", "tired": "😴"
}

# -----------------------------
# Emotion analysis function
# -----------------------------
def analyze_emotion(text: str):
    if not text or not isinstance(text, str) or text.strip() == "":
        return "neutral", 0.0, []
    result = classifier(text, candidate_labels=labels)
    return result['labels'][0], round(result['scores'][0], 3), result

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("ℹ️ How to use this app")
st.sidebar.write("""
1. Enter any sentence in the text box below.
2. Click outside the box or press Enter.
3. See the predicted emotion and confidence scores.
4. Top 3 emotions are shown for better insight.
""")

show_top3 = st.sidebar.checkbox("Show Top 3 Emotions", value=True)
show_chart = st.sidebar.checkbox("Show Emotion Chart", value=True)
show_emoji = st.sidebar.checkbox("Show Emoji", value=True)

if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Centered input box
# -----------------------------
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    user_input = st.text_area(
        "Enter a sentence to analyze:",  # this label now larger (24px)
        key="user_input",
        height=120,
        placeholder="Type your text here (Press Ctrl+Enter)",
    )

# -----------------------------
# Analyze input
# -----------------------------
if user_input:
    label, score, full_result = analyze_emotion(user_input)
    st.session_state.history.append((user_input, label, score))

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("Predicted Emotion")
        emoji = emoji_dict.get(label.lower(), "🧠") if show_emoji else ""
        st.success(f"{emoji} {label} ({score*100:.1f}% confidence)")

        if show_top3:
            st.write("### 🔹 Top 3 Emotions:")
            for i in range(3):
                color = ["#FFD1DC", "#AEC6CF", "#77DD77"][i]  # pastel shades
                st.markdown(
                    f"<span style='background-color:{color}; padding:5px 10px; border-radius:5px;'>{full_result['labels'][i]} ({full_result['scores'][i]*100:.1f}%)</span>",
                    unsafe_allow_html=True
                )

        st.subheader("Confidence Level")
        st.progress(int(score*100))

    with col_right:
        if show_chart:
            st.subheader("Emotion Confidence Scores")
            df_scores = pd.DataFrame({
                "Emotion": full_result['labels'],
                "Score": full_result['scores']
            })
            pastel_colors = [
                "#AEC6CF", "#FFB347", "#FFD1DC", "#77DD77", "#CBAACB",
                "#F5CBA7", "#B5EAD7", "#FF6961", "#CB99C9", "#FFB6C1"
            ]
            chart = alt.Chart(df_scores).mark_bar().encode(
                x=alt.X('Score', title='Confidence'),
                y=alt.Y('Emotion', sort='-x', title='Emotion'),
                color=alt.Color('Score', scale=alt.Scale(range=pastel_colors)),
                tooltip=['Emotion', alt.Tooltip('Score', format='.2f')]
            ).properties(height=400)

            st.altair_chart(chart, use_container_width=True)

# -----------------------------
# Show history
# -----------------------------
if st.session_state.history:
    st.subheader("📜 Previous Predictions")
    for text, lbl, scr in reversed(st.session_state.history[-5:]):
        st.write(f"💬 {text} → {lbl} ({scr*100:.1f}%)")

