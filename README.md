# 🧠 Talk-Sense: Real-Time Speech Emotion & Sentiment Analyzer

## 📘 Project Summary
**Talk-Sense** is a data science and machine learning project built to analyze text (such as social media posts or chat messages) and detect emotions like **sadness, stress, or positivity**.  
It uses **Natural Language Processing (NLP)** and **Machine Learning algorithms** to preprocess input text, extract key features, classify emotion/sentiment, and present the results through an interactive dashboard.

---

## ✨ Key Features
- 🧹 **Cleans and preprocesses** user text input  
- 🧠 **Predicts emotional sentiment** using ML models *(Logistic Regression + TF-IDF features)*  
- 📊 **Dashboard interface** built using **Streamlit**  
- 🧩 **Beginner-friendly structure** and modular codebase  

---

## ⚙️ How to Install and Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/talk-sense.git
cd talk-sense
```

### 2️⃣ (Recommended) Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3️⃣  Install Dependencies
```bash
pip install -r requirements.txt
```
### 4️⃣ Download and Prepare Dataset

* Put your data file (CSV) inside data/raw/.

### 5️⃣ Preprocess the Data
```bash
python src/features/preprocessing.py
```

### 6️⃣ Perform Feature Engineering
```bash
python src/features/feature_engineering.py
```

### 7️⃣ Train the Model
```bash
python src/models/train_model.py
```

### 8️⃣ Evaluate the Model
```bash
cd src/app
streamlit run streamlit_app.py
```

### 9️⃣ Launch the Streamlit Dashboard
```bash
cd src/app
streamlit run streamlit_app.py
```

## 💡 Example Usages
### ▶️ Console Usage

To classify individual text inputs:
```python
from src.features.preprocessing import clean_text


sample = "I am feeling extremely happy today!"
print(clean_text(sample))
```

## 🌐 Streamlit Web App

1. Enter any sentence or paragraph into the input box.

2. Click Analyze.

3. View the predicted emotion/sentiment (e.g., Positive, Sad, Stressed).

4. See dashboard charts for sentiment trends (if implemented).

## 🧩 Project Structure
```css
talk-sense/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── app/
│   │   └── streamlit_app.py
│   ├── features/
│   │   ├── preprocessing.py
│   │   └── feature_engineering.py
│   └── models/
│       ├── train_model.py
│       └── evaluate_model.py
│
├── tests/
│   └── test_preprocessing.py
│
├── requirements.txt
├── README.md
└── .gitignore

```
## 🧠 Future Improvements

* Add audio-based emotion detection using speech recognition + deep learning

* Deploy the app on Streamlit Cloud / Hugging Face Spaces

* Integrate real-time chat sentiment tracking