# D:\talk-sense\src\features\preprocessing.py
import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

print("Current working directory:", os.getcwd())
RAW_PATH = r"D:\talk-sense\data\raw\talk_sense_tweets.csv"
PROCESSED_PATH = r"D:\talk-sense\data\processed\cleaned_data.csv"
print("File exists:", os.path.exists(RAW_PATH))

# Ensure stopwords available
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""                        # handle NaN / non-string
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    text = ' '.join(w for w in text.split() if w and w not in STOPWORDS)
    return text.strip()

def choose_text_column(df):
    """
    If dataframe has a column name containing 'text' return it.
    Otherwise pick the text-like column (object dtype with largest median length).
    """
    # 1) direct name match
    for col in df.columns:
        if 'text' in str(col).lower():
            return col

    # 2) choose object/string column with largest median length
    obj_cols = [c for c in df.columns if df[c].dtype == 'object' or df[c].dtype == 'string']
    if obj_cols:
        def median_len(col):
            return df[col].astype(str).str.len().median()
        likely = max(obj_cols, key=median_len)
        return likely

    # 3) fallback: last column
    return df.columns[-1]

if __name__ == "__main__":
    # Try reading with header; if it looks like the first row is data, re-read without header
    df_try = pd.read_csv(RAW_PATH, dtype=str, encoding='utf-8', nrows=5)  # only read a sample
    print("Sample read (first 5 rows):")
    print(df_try.head())
    print("Columns (initial read):", list(df_try.columns))

    # Heuristic: if column names look like actual sentence text (i.e. long strings), the file had no header
    header_looks_like_data = any(len(str(col)) > 50 or ' ' in str(col) for col in df_try.columns)
    if header_looks_like_data:
        print("Detected that CSV likely has NO header. Re-reading with header=None.")
        df = pd.read_csv(RAW_PATH, header=None, dtype=str, encoding='utf-8')
        print("Columns after header=None read:", list(df.columns))
        # If you know the exact structure (4 columns id,topic,sentiment,text) you can set them here:
        if df.shape[1] == 4:
            df.columns = ['id', 'topic', 'sentiment', 'text']
            print("Assigned columns: ['id','topic','sentiment','text']")
        else:
            # keep numeric column names 0..N-1
            print(f"File has {df.shape[1]} columns (no header).")
    else:
        # file had header: use full read
        df = pd.read_csv(RAW_PATH, dtype=str, encoding='utf-8')
        print("Used header from file. Columns:", list(df.columns))

    # pick which column to treat as text
    text_col = choose_text_column(df)
    print("Selected text column:", text_col)

    # preview a few entries from the chosen text column
    print("Preview of chosen text column (first 5):")
    print(df[text_col].astype(str).head())

    # create cleaned column
    df['clean_text'] = df[text_col].apply(clean_text)

    # ensure output folder exists
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False, encoding='utf-8')
    print("✅ Cleaned data saved to:", PROCESSED_PATH)
