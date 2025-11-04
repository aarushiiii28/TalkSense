

import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# =========================
# Configuration
# =========================
MODEL_NAME = 'distilbert-base-uncased'
DATA_PATH = os.path.join("data", "processed", "cleaned_data.csv")
OUTPUT_DIR = os.path.join("data", "processed")

# =========================
# Load Model and Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()  # put model in evaluation mode

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =========================
# Embedding Function
# =========================
def get_embedding(text):
    """Return CLS token embedding for a given text."""
    if not isinstance(text, str) or text.strip() == "":
        return np.zeros(model.config.hidden_size)
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
    
    return cls_embedding.squeeze().detach().cpu().numpy()

# =========================
# Main Script
# =========================
if __name__ == "__main__":
    print("🔹 Loading cleaned dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Clean data
    df = df.dropna(subset=["clean_text"])
    df = df[df["clean_text"].str.strip() != ""]

    print(f"✅ Loaded {len(df)} samples.")
    
    embeddings = []
    print("🔸 Generating embeddings (this may take a few minutes)...")
    for text in tqdm(df["clean_text"].tolist()):
        emb = get_embedding(text)
        embeddings.append(emb)
    
    # Stack all embeddings into a numpy array
    X = np.stack(embeddings)
    y = df["sentiment"].values

    # =========================
    # Save Processed Features
    # =========================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "X_embeddings.npy"), X)
    pd.DataFrame(y, columns=["sentiment"]).to_csv(
        os.path.join(OUTPUT_DIR, "y_labels.csv"), index=False
    )

    print("✅ Feature engineering complete!")
    print(f"Saved: X_embeddings.npy and y_labels.csv in {OUTPUT_DIR}")
