"""
scripts/merge_tenders.py
Corrected version with:
 - Fallback Tender Numbers for blank fields
 - Unique + duplicate output files
 - Full MTIE schema
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
import chardet
import csv

ROOT = Path(".")
RAW_TENDER_DIR = ROOT / "data" / "raw" / "Tender24x7"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FRESH = RAW_TENDER_DIR / "fresh.csv"
LIVE = RAW_TENDER_DIR / "live.csv"
ARCHIVE = RAW_TENDER_DIR / "archive.csv"
GEM = RAW_TENDER_DIR / "GemContracts.csv"

OUT_ALL = PROCESSED_DIR / "merged_tenders_with_duplicates.csv"
OUT_UNIQUE = PROCESSED_DIR / "merged_tenders_unique.csv"
OUT_MAIN = PROCESSED_DIR / "merged_tenders.csv"

# --------------------------------------------------------
# HELPERS
# --------------------------------------------------------

def detect_encoding(path):
    with open(path, "rb") as f:
        sample = f.read(50000)
    res = chardet.detect(sample)
    return res.get("encoding") or "utf-8"

def clean_text(x):
    if pd.isna(x) or x is None:
        return ""
    s = str(x)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def read_table(path: Path):
    if not path.exists():
        print(f"[WARN] {path} not found.")
        return pd.DataFrame()

    encodings_to_try = [
        "utf-8",
        "ISO-8859-1",
        "latin1",
        "windows-1252",
        "utf-16",
        "utf-16le",
        "utf-16be"
    ]

    for enc in encodings_to_try:
        try:
            if path.suffix.lower() in [".xls", ".xlsx"]:
                df = pd.read_excel(path, dtype=str)
                print(f"[OK] Loaded Excel: {path}")
                return df

            df = pd.read_csv(path, dtype=str, encoding=enc, engine="python", sep=",", on_bad_lines="skip")
            print(f"[OK] Loaded CSV: {path} using encoding {enc}")
            df.columns = [c.strip() for c in df.columns]
            return df

        except Exception as e:
            print(f"[WARN] Failed with encoding {enc}: {e}")
            continue

    print(f"[ERROR] Could not read file: {path}")
    return pd.DataFrame()


# --------------------------------------------------------
# TENDER MAPPERS
# --------------------------------------------------------

def map_t24x7_row(row, source, row_idx):
    tn = row.get("Tender Number") or ""
    tn = tn.strip()

    # If Tender Number is blank → generate fallback
    if tn == "":
        tn = f"AUTO_{source.upper()}_{row_idx:06d}"

    bid_id = row.get("Tender Id") or ""
    site = row.get("Site Location") or ""
    end_date = row.get("Tender Due Date") or ""
    quantity = row.get("Quantity") or ""
    uom = row.get("Uom") or ""
    remarks = row.get("Remarks") or ""
    estimated_cost = row.get("Estimated Cost") or ""
    bidder_company = row.get("Bidder Company") or ""
    bidder_price = row.get("Bidder Price") or ""

    title = " ".join([
        clean_text(row.get("Model") or ""),
        clean_text(row.get("Lob") or ""),
        clean_text(row.get("Segment") or "")
    ]).strip()

    short_desc = clean_text(row.get("Meril Ref No") or "")

    full_text = " ".join([
        clean_text(row.get("Model") or ""),
        clean_text(row.get("Meril Ref No") or ""),
        clean_text(row.get("Remarks") or ""),
        clean_text(row.get("HO Name") or ""),
        clean_text(row.get("Site Location") or ""),
        clean_text(row.get("Sales Cat") or ""),
        clean_text(row.get("Lob") or ""),
        clean_text(row.get("Segment") or "")
    ]).strip()

    return {
        "source_file": source,
        "bid_id": bid_id,
        "tender_number": tn,
        "contract_no": "",
        "department_name": clean_text(row.get("HO Name") or row.get("Zonal Head Name") or ""),
        "buyer_name": clean_text(row.get("Distributor Master") or ""),
        "buyer_location": clean_text(site),
        "title": clean_text(title),
        "description": short_desc,
        "quantity": quantity,
        "uom": uom,
        "estimated_value": estimated_cost,
        "start_date": "",
        "end_date": end_date,
        "bidder_company": bidder_company,
        "bidder_price": bidder_price,
        "full_text": full_text,
        "raw_row": row.to_dict(),
        "label_relevant": 1
    }

def map_gem_row(row, row_idx):
    bid_no = row.get("Bid No.") or row.get("Bid No") or ""
    bid_no = str(bid_no).strip()

    # Fallback Tender Number for GEM if blank
    if bid_no == "":
        bid_no = f"AUTO_GEM_{row_idx:06d}"

    contract_no = row.get("Contract No.") or ""
    date = row.get("Date") or ""
    dept = row.get("Department Name") or ""
    dept_loc = row.get("Department Location") or ""
    category = row.get("Category") or ""
    brand = row.get("Brand Name") or ""
    company = row.get("Company Name") or ""
    ordered_qty = row.get("Ordered Qty") or ""
    total_price = row.get("Total Price") or ""
    remarks = row.get("Remarks") or ""

    title = clean_text(category) or clean_text(brand)
    full_text = " ".join([
        clean_text(category),
        clean_text(brand),
        clean_text(company),
        clean_text(remarks),
        clean_text(dept)
    ])

    return {
        "source_file": "GemContracts",
        "bid_id": bid_no,
        "tender_number": bid_no,
        "contract_no": contract_no,
        "department_name": clean_text(dept),
        "buyer_name": clean_text(dept),
        "buyer_location": clean_text(dept_loc),
        "title": clean_text(title),
        "description": clean_text(remarks),
        "quantity": ordered_qty,
        "uom": "",
        "estimated_value": total_price,
        "start_date": date,
        "end_date": date,
        "bidder_company": company,
        "bidder_price": "",
        "full_text": full_text,
        "raw_row": row.to_dict(),
        "label_relevant": 1
    }

# --------------------------------------------------------
# LOADING + MERGING
# --------------------------------------------------------

def df_from_tender24(files):
    rows = []
    for f in files:
        df = read_table(f)
        if df.empty:
            continue
        df = df.fillna("")

        for idx, r in df.iterrows():
            mapped = map_t24x7_row(r, f.stem, idx + 1)
            rows.append(mapped)
    return pd.DataFrame(rows)

def df_from_gem(path):
    df = read_table(path)
    if df.empty:
        return pd.DataFrame()
    df = df.fillna("")
    
    rows = []
    for idx, r in df.iterrows():
        mapped = map_gem_row(r, idx + 1)
        rows.append(mapped)
    return pd.DataFrame(rows)

def unify_and_save():
    t24 = df_from_tender24([FRESH, LIVE, ARCHIVE])
    gem = df_from_gem(GEM)

    merged = pd.concat([t24, gem], ignore_index=True, sort=False)

    # Save WITH duplicates
    merged.to_csv(OUT_ALL, index=False)
    print(f"[OK] ALL tenders written (with duplicates): {OUT_ALL} ({len(merged)} rows)")

    # Dedup: tender_number_norm only
    merged["tender_number_norm"] = merged["tender_number"].str.lower().str.strip()
    unique = merged.drop_duplicates(subset=["tender_number_norm"], keep="first")

    unique.to_csv(OUT_UNIQUE, index=False)
    unique.to_csv(OUT_MAIN, index=False)

    print(f"[OK] Unique tenders written: {OUT_UNIQUE} ({len(unique)} rows)")
    print(f"[INFO] Preview of unique tenders:\n{unique.head(10).to_string(index=False)}")

if __name__ == "__main__":
    unify_and_save()
