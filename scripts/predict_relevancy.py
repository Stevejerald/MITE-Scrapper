"""
scripts/predict_relevancy.py

Purpose:
 - Load MTIE Relevancy Model
 - Predict relevance for:
      (A) single tender text (Items/description)
      (B) OR entire CSV file
 - Output predictions + probability score

Usage:
  python scripts/predict_relevancy.py --text "surgical suture absorbable..."
  python scripts/predict_relevancy.py --file data/raw/new_tenders.csv
"""

import pandas as pd
import argparse
import joblib
import re
import string
from pathlib import Path

ROOT = Path(".")
PROC = ROOT / "data" / "processed"

MODEL_FILE = PROC / "relevancy_model.pkl"
VECT_FILE = PROC / "vectorizer.pkl"
DEFAULT_OUTPUT = PROC / "predicted_tenders.csv"


# -------------------------------------------------------------
# CLEAN TEXT
# -------------------------------------------------------------
def clean_text(txt):
    if txt is None or pd.isna(txt):
        return ""
    txt = str(txt).lower()
    txt = re.sub(r"[^a-z0-9\s/-]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


# -------------------------------------------------------------
# LOAD MODEL + VECTORIZER
# -------------------------------------------------------------
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_FILE}")

    if not VECT_FILE.exists():
        raise FileNotFoundError(f"Missing vectorizer: {VECT_FILE}")

    print("[OK] Loaded relevancy model + vectorizer")

    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECT_FILE)
    return model, vectorizer


# -------------------------------------------------------------
# PREDICT SINGLE TEXT
# -------------------------------------------------------------
def predict_single(text):
    model, vectorizer = load_model()

    clean = clean_text(text)
    vec = vectorizer.transform([clean])

    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0][1]  # probability of relevant=1

    print("\n====================")
    print(" SINGLE TENDER RESULT")
    print("====================")
    print(f"Input Text: {text}")
    print(f"Prediction: {'RELEVANT' if pred==1 else 'NOT RELEVANT'}")
    print(f"Probability: {proba:.4f}")
    print("====================\n")


# -------------------------------------------------------------
# PREDICT CSV FILE
# -------------------------------------------------------------
def predict_csv(file_path, output_path=DEFAULT_OUTPUT):
    model, vectorizer = load_model()

    if not Path(file_path).exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(file_path, dtype=str).fillna("")

    print(f"\n[INFO] Loaded CSV: {file_path} ({len(df)} rows)")

    # Pick the best text column available
    text_column = None
    for col in ["Items", "description", "title", "full_text"]:
        if col in df.columns:
            text_column = col
            break

    if text_column is None:
        raise KeyError("CSV must contain at least one of: Items, description, title, full_text")

    print(f"[INFO] Using text column: {text_column}")

    df["clean_text"] = df[text_column].apply(clean_text)
    X_vec = vectorizer.transform(df["clean_text"])

    df["predicted_label"] = model.predict(X_vec)
    df["probability"] = model.predict_proba(X_vec)[:, 1]

    df.to_csv(output_path, index=False)

    print(f"[SAVED] Predictions saved → {output_path}")
    print("\nPreview:")
    print(df[["tender_number", text_column, "predicted_label", "probability"]].head().to_string(index=False))


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTIE Relevancy Predictor")
    parser.add_argument("--text", type=str, help="Single tender text to classify")
    parser.add_argument("--file", type=str, help="CSV file for bulk prediction")
    parser.add_argument("--out", type=str, help="Output CSV file")

    args = parser.parse_args()

    if args.text:
        predict_single(args.text)

    elif args.file:
        out_file = args.out if args.out else DEFAULT_OUTPUT
        predict_csv(args.file, out_file)

    else:
        print("Usage:")
        print("  python scripts/predict_relevancy_model.py --text \"suture absorbable...\"")
        print("  python scripts/predict_relevancy_model.py --file new_tenders.csv --out output.csv")
