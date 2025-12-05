"""
scripts/train_relevancy_model.py

Purpose:
 - Train MTIE (Meril Tender Intelligence Engine) Relevancy Classifier
 - Input: scrapped_tender_with_label.csv (complete unified dataset)
 - Output: relevancy_model.pkl + vectorizer.pkl + training_report.txt

The model uses:
  - Items text (primary)
  - Falls back to description/title if Items is empty
  - Falls back to combining all available fields if needed
  - TF-IDF vectorizer + Logistic Regression classifier
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


ROOT = Path(".")
PROC = ROOT / "data" / "processed"

DATA_FILE = PROC / "scrapped_tender_with_label.csv"
MODEL_OUT = PROC / "relevancy_model.pkl"
VECT_OUT = PROC / "vectorizer.pkl"
REPORT_OUT = PROC / "training_report.txt"


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------
def clean_text(txt):
    if pd.isna(txt):
        return ""
    txt = str(txt).lower()
    txt = re.sub(r"[^a-z0-9\s/-]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


# ---------------------------------------------------------
# MAIN TRAINING PIPELINE
# ---------------------------------------------------------
def train_model():

    print("\n============================")
    print(" TRAINING MTIE RELEVANCY MODEL")
    print("============================\n")

    # -------------------------------
    # Load dataset
    # -------------------------------
    print("[1/6] Loading dataset...")

    df = pd.read_csv(DATA_FILE, dtype=str).fillna("")

    print(f"[OK] Loaded {len(df)} rows\n")

    # Required field
    if "label_relevant" not in df.columns:
        raise KeyError("Dataset missing 'label_relevant' column")

    # -------------------------------
    # SAFE TEXT SELECTION LOGIC
    # -------------------------------
    print("[1b] Building text_for_model (safe fallback system)...")

    def choose_text(r):
        # 1 — Items is BEST
        if "Items" in r and str(r["Items"]).strip():
            return r["Items"]

        # 2 — Description
        if "description" in r and str(r["description"]).strip():
            return r["description"]

        # 3 — Title
        if "title" in r and str(r["title"]).strip():
            return r["title"]

        # 4 — Merge multiple fields if available
        parts = []
        for col in [
            "Items", "description", "title",
            "Department", "buyer_location",
            "quantity", "estimated_value"
        ]:
            if col in r and str(r[col]).strip():
                parts.append(str(r[col]))

        if parts:
            return " ".join(parts)

        # 5 — FINAL fallback
        return ""

    df["text_for_model"] = df.apply(choose_text, axis=1)
    df["clean_text"] = df["text_for_model"].apply(clean_text)
    df["y"] = df["label_relevant"].astype(int)

    X = df["clean_text"]
    y = df["y"]

    print("[OK] Text cleaned and prepared\n")


    # -------------------------------
    # Split
    # -------------------------------
    print("[2/6] Splitting train/test...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"[OK] Train: {len(X_train)} rows, Test: {len(X_test)} rows\n")


    # -------------------------------
    # TF-IDF Vectorizer
    # -------------------------------
    print("[3/6] Training TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        max_features=60000,
        min_df=2
    )

    X_train_vec = vectorizer.fit_transform(X_train)

    print("[OK] Vectorizer trained\n")


    # -------------------------------
    # Train Logistic Regression
    # -------------------------------
    print("[4/6] Training Logistic Regression classifier...")

    model = LogisticRegression(
        max_iter=600,
        class_weight="balanced"
    )

    model.fit(X_train_vec, y_train)

    print("[OK] Model training done\n")


    # -------------------------------
    # Evaluate
    # -------------------------------
    print("[5/6] Evaluating model...")

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm, "\n")

    # Save evaluation summary
    with open(REPORT_OUT, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))

    print(f"[SAVED] Training report → {REPORT_OUT}\n")


    # -------------------------------
    # Save Model + Vectorizer
    # -------------------------------
    print("[6/6] Saving model + vectorizer...")

    joblib.dump(model, MODEL_OUT)
    joblib.dump(vectorizer, VECT_OUT)

    print(f"[SAVED] Model → {MODEL_OUT}")
    print(f"[SAVED] Vectorizer → {VECT_OUT}")
    print("\n[ DONE ] MTIE Relevancy Model training completed!\n")


# ---------------------------------------------------------
if __name__ == "__main__":
    train_model()
