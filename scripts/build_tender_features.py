"""
scripts/build_tender_features.py
MTIE – Feature Engineering Script

Creates:
 - clean_text column
 - product keyword matches
 - numeric + boolean features
 - final tender_features.csv ready for ML

Input:
  data/processed/merged_tenders.csv
  data/processed/product_master.csv

Output:
  data/processed/tender_features.csv
"""

import pandas as pd
import numpy as np
import re
import string
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(".")
PROC = ROOT / "data" / "processed"
OUT = PROC / "tender_features.csv"


# -------------------------------------------------------------
# CLEAN TEXT FUNCTION
# -------------------------------------------------------------
def clean_text(text: str) -> str:
    if pd.isna(text) or text is None:
        return ""

    text = str(text).lower()
    text = text.replace("\n", " ").replace("\r", " ")

    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# -------------------------------------------------------------
# MATCH PRODUCT KEYWORDS
# -------------------------------------------------------------
def build_product_keyword_list():
    """
    Loads product_master.csv and builds keyword list
    """
    prod_path = PROC / "product_master.csv"
    if not prod_path.exists():
        print("[ERROR] product_master.csv not found.")
        return []

    df = pd.read_csv(prod_path, dtype=str).fillna("")

    # product_keywords: "stent catheter guidewire"
    # product_aliases: "stent;stent balloon;..."
    keywords = []

    for _, row in df.iterrows():
        # keyword list from product_keywords (space-separated)
        kws = str(row.get("product_keywords", "")).split()
        keywords.extend(kws)

        # aliases (semi-colon separated)
        aliases = str(row.get("product_aliases", "")).split(";")
        for a in aliases:
            words = a.lower().strip().split()
            keywords.extend(words)

    # remove duplicates
    keywords = list(set([k.strip() for k in keywords if len(k.strip()) > 2]))

    print(f"[INFO] Loaded {len(keywords)} unique product keywords.")
    return keywords


# -------------------------------------------------------------
# KEYWORD MATCH COUNT
# -------------------------------------------------------------
def count_product_keywords(text: str, keyword_list):
    count = 0
    for kw in keyword_list:
        if kw in text:
            count += 1
    return count


# -------------------------------------------------------------
# STOPWORD REMOVAL
# -------------------------------------------------------------
def remove_stopwords(text: str) -> str:
    tokens = text.split()
    cleaned = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return " ".join(cleaned)


# -------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------
def build_tender_features():
    merged_path = PROC / "merged_tenders.csv"
    if not merged_path.exists():
        print("[ERROR] merged_tenders.csv not found. Run merge_tenders.py first.")
        return

    df = pd.read_csv(merged_path, dtype=str).fillna("")

    # ---------------------------------------------------------
    # BUILD CLEAN TEXT COLUMN
    # ---------------------------------------------------------
    print("[INFO] Generating clean_text...")

    df["clean_text"] = (
        df["title"].astype(str) + " "
        + df["description"].astype(str) + " "
        + df["full_text"].astype(str)
    ).apply(clean_text)

    # Remove stopwords
    df["clean_text"] = df["clean_text"].apply(remove_stopwords)

    # ---------------------------------------------------------
    # PRODUCT KEYWORD MATCHES
    # ---------------------------------------------------------
    print("[INFO] Matching product keywords...")
    keyword_list = build_product_keyword_list()

    df["product_kw_count"] = df["clean_text"].apply(
        lambda x: count_product_keywords(x, keyword_list)
    )

    # ---------------------------------------------------------
    # NUMERIC FEATURES
    # ---------------------------------------------------------
    print("[INFO] Creating numeric and boolean features...")

    df["text_length"] = df["clean_text"].apply(lambda x: len(x))
    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
    df["has_quantity"] = df["quantity"].apply(lambda x: 1 if str(x).strip() not in ["", "0"] else 0)
    df["has_uom"] = df["uom"].apply(lambda x: 1 if str(x).strip() not in ["", "0"] else 0)
    df["has_estimated_value"] = df["estimated_value"].apply(lambda x: 1 if str(x).strip() not in ["", "0"] else 0)
    df["has_hospital_words"] = df["clean_text"].apply(
        lambda x: 1 if any(w in x for w in ["hospital", "medical", "college", "aiims", "govt"]) else 0
    )

    # ---------------------------------------------------------
    # FINAL LABEL (RELEVANT FOR MTIE TRAINING)
    # all tender24x7 files are relevant = 1
    # gem contracts = 1
    # later: add negative samples from random GeM scraping
    # ---------------------------------------------------------
    df["label"] = df["source_file"].apply(
        lambda x: 1 if x in ["fresh", "live", "archive", "GemContracts"] else 0
    )

    # ---------------------------------------------------------
    # SAVE OUTPUT
    # ---------------------------------------------------------
    df.to_csv(OUT, index=False)
    print(f"[OK] Tender features saved to: {OUT}")
    print(f"[INFO] Total rows: {len(df)}")
    print(df.head(10).to_string())


# -------------------------------------------------------------
# EXECUTE
# -------------------------------------------------------------
if __name__ == "__main__":
    build_tender_features()
