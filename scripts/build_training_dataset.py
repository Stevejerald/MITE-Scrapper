"""
scripts/build_training_dataset.py

Purpose:
 - Create a CLEAN training dataset with:
     label = 1 (relevant) → from merged_tenders_with_items.csv
     label = 0 (irrelevant) → selected from problem_tender_features.csv

Logic for negative samples:
 - product_kw_count == 0
 - meril_can_supply == 0
 - historical_participation == 0
 - composite_score_norm < 0.15
 - contains NON-medical keywords

Outputs:
 - data/processed/training_dataset.csv
"""

from pathlib import Path
import pandas as pd
import re

ROOT = Path(".")
PROC = ROOT / "data" / "processed"

MERGED_WITH_ITEMS = PROC / "merged_tenders_with_items.csv"
PROBLEM_FEATURES = PROC / "problem_tender_features.csv"
OUT = PROC / "training_dataset.csv"

# ---------------------------------------
# NON-MEDICAL KEYWORDS LIST
# ---------------------------------------
NON_MEDICAL_KEYWORDS = [
    "computer", "printer", "ups", "server", "projector", "router", "switch",
    "mouse", "keyboard", "laptop", "desktop",
    "chair", "table", "furniture", "window", "door", "painting", "civil",
    "electrical", "wiring", "lights", "fan", "air conditioner", "ac",
    "vehicle", "bus", "car", "diesel", "petrol",
    "books", "uniform", "cloth", "shoe", "stationery", "pen",
    "construction", "road", "cement", "plumbing"
]

def contains_non_medical(text):
    text = text.lower()
    return any(kw in text for kw in NON_MEDICAL_KEYWORDS)

# ---------------------------------------
# MAIN FUNCTION
# ---------------------------------------
def build_training_dataset():

    print("\n============================")
    print("BUILDING TRAINING DATASET...")
    print("============================\n")

    # ---------------------------
    # Load positive samples
    # ---------------------------
    print("[1/5] Loading positive samples (Meril tenders)...")

    if not MERGED_WITH_ITEMS.exists():
        raise FileNotFoundError(f"Missing file: {MERGED_WITH_ITEMS}")

    pos = pd.read_csv(MERGED_WITH_ITEMS, dtype=str).fillna("")
    pos["label"] = 1

    print(f"[OK] Loaded {len(pos)} positive samples")

    # ---------------------------
    # Load problem feature data
    # ---------------------------
    print("\n[2/5] Loading scraped-based features for negatives...")

    if not PROBLEM_FEATURES.exists():
        raise FileNotFoundError(f"Missing file: {PROBLEM_FEATURES}")

    pf = pd.read_csv(PROBLEM_FEATURES, dtype=str)
    pf = pf.fillna("")

    # Convert numeric fields
    for col in ["product_kw_count", "meril_can_supply", "historical_participation", "composite_score_norm"]:
        pf[col] = pd.to_numeric(pf[col], errors="coerce").fillna(0)

    print(f"[OK] Loaded {len(pf)} scraped tenders")

    # ---------------------------
    # Select NEGATIVE samples
    # ---------------------------
    print("\n[3/5] Selecting negative samples...")

    neg = pf[
        (pf["product_kw_count"] == 0) &
        (pf["meril_can_supply"] == 0) &
        (pf["historical_participation"] == 0) &
        (pf["composite_score_norm"] < 0.15)
    ].copy()

    # Non-medical keyword filter
    neg["non_medical_flag"] = neg["clean_items"].apply(contains_non_medical)
    neg = neg[neg["non_medical_flag"] == True]

    neg["label"] = 0

    print(f"[OK] Selected {len(neg)} negative samples")

    # Optional: Cap negatives to balance dataset (optional)
    neg = neg.head(len(pos))  # balanced dataset
    print(f"[INFO] Balanced negatives to → {len(neg)} rows")

    # ---------------------------
    # Combine positive + negative
    # ---------------------------
    print("\n[4/5] Combining datasets...")

    final_cols = [
        "tender_number",
        "clean_items",
        "department",
        "quantity",
        "start_date",
        "end_date",
        "label"
    ]

    pos_final = pos.rename(columns={"description": "clean_items"})[final_cols]
    neg_final = neg[final_cols]

    df = pd.concat([pos_final, neg_final], ignore_index=True)

    print(f"[OK] Combined dataset → {len(df)} rows")

    # ---------------------------
    # Save dataset
    # ---------------------------
    print("\n[5/5] Saving dataset...")

    PROC.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)

    print(f"[SAVED] {OUT}")
    print("\n[PREVIEW]")
    print(df.sample(10).to_string(index=False))

    print("\n[DONE] Training dataset built successfully.\n")


if __name__ == "__main__":
    build_training_dataset()
