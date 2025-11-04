import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

print("Current working directory:", os.getcwd())
print("File exists:", os.path.exists(r"D:\talk-sense\data\processed\cleaned_data.csv"))

df = pd.read_csv(r"D:\talk-sense\data\processed\cleaned_data.csv")

# 🧹 Clean up missing or empty text values
df = df.dropna(subset=['clean_text'])
df = df[df['clean_text'].str.strip() != '']

X = df['clean_text']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# Save processed features
with open(r"D:\talk-sense\data\processed\X_train.pkl", 'wb') as f:
    pickle.dump(X_train_vectors, f)
with open(r"D:\talk-sense\data\processed\X_test.pkl", 'wb') as f:
    pickle.dump(X_test_vectors, f)

y_train.to_csv(r"D:\talk-sense\data\processed\y_train.csv", index=False)
y_test.to_csv(r"D:\talk-sense\data\processed\y_test.csv", index=False)

with open(r"D:\talk-sense\models\tfidf_vectorizer.pkl", 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Feature engineering complete! Files saved successfully.")
