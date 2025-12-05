"""
scripts/merge_scraped_details.py

Purpose:
 - Join merged_tenders_unique.csv with data_scrapped.csv
 - Match on tender_number = Bid Number
 - Add: Items, Department, Quantity, Detail URL, Start Date, End Date
 - Output final merged file for AI model training

Input:
 - data/processed/merged_tenders_unique.csv
 - data/raw/data_scrapped.csv

Output:
 - data/processed/merged_tenders_with_items.csv
"""

from pathlib import Path
import pandas as pd

ROOT = Path(".")
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

MERGED = PROC / "merged_tenders_unique.csv"
SCRAPED = RAW / "data_scrapped.csv"
OUTPUT = PROC / "merged_tenders_with_items.csv"

def normalize(s):
    if pd.isna(s):
        return ""
    return str(s).strip().lower()

def run_merge():

    print("\n==============================")
    print(" MERGING SCRAPED DETAILS")
    print("==============================\n")

    # ----------------------
    # Load merged tenders
    # ----------------------
    print("[1/4] Loading merged_tenders_unique.csv...")
    df_merged = pd.read_csv(MERGED, dtype=str).fillna("")
    df_merged["tender_number_norm"] = df_merged["tender_number"].apply(normalize)
    print(f"[OK] Loaded {len(df_merged)} rows")

    # ----------------------
    # Load scraped dataset
    # ----------------------
    print("\n[2/4] Loading data_scrapped.csv...")
    df_scraped = pd.read_csv(SCRAPED, dtype=str).fillna("")
    df_scraped["tender_number_norm"] = df_scraped["Bid Number"].apply(normalize)
    print(f"[OK] Loaded {len(df_scraped)} rows")

    # ----------------------
    # Merge based on tender_number_norm
    # ----------------------
    print("\n[3/4] Joining on tender_number_norm...")

    df_joined = df_merged.merge(
        df_scraped[
            ["tender_number_norm", "Items", "Quantity", "Department",
             "Detail URL", "Start Date", "End Date"]
        ],
        on="tender_number_norm",
        how="left"
    )

    print(f"[OK] Joined → {len(df_joined)} rows")

    # ----------------------
    # Save output
    # ----------------------
    print("\n[4/4] Saving final merged file...")
    df_joined.to_csv(OUTPUT, index=False)
    print(f"[SAVED] {OUTPUT}")
    print("\n[DONE] merge_scraped_details.py completed.\n")

if __name__ == "__main__":
    run_merge()
