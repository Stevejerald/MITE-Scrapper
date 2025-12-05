"""
scripts/build_product_master.py
Updated: Auto-detect encoding for CSVs (handles Windows-1252, ISO-8859-1)
"""

import re
import chardet   # NEW dependency
from pathlib import Path
import pandas as pd
import numpy as np
import csv

ROOT = Path(".")
RAW_DIR = ROOT / "data" / "raw" / "ProductFiles"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ENDO_FILE = RAW_DIR / "EndoProducts.csv"
INSTR_FILE = RAW_DIR / "instrument_table.csv"
REAGENT_FILE = RAW_DIR / "reagent.csv"
DIAGNO_FILE = RAW_DIR / "DiagnoProducts.csv"

OUT_FILE = PROCESSED_DIR / "product_master.csv"


# -----------------------------------------
# Utility: detect encoding of CSV files
# -----------------------------------------
def detect_encoding(path):
    with open(path, "rb") as f:
        raw = f.read(50000)  # sample
    result = chardet.detect(raw)
    return result["encoding"] or "utf-8"


# -----------------------------------------
def clean_text(s):
    if pd.isna(s) or s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def normalize_token(t):
    t = clean_text(t).lower()
    t = re.sub(r'[^a-z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def extract_aliases_from_name(name):
    name = clean_text(name)
    if not name:
        return []
    tmp = re.sub(r'[\/\-\(\)\,\|]+', ' ', name)
    tokens = [t for t in re.split(r'\s+', tmp) if t]
    tokens = [normalize_token(t) for t in tokens if len(t) > 1 and not t.isdigit()]
    aliases = set()
    aliases.add(normalize_token(name))
    for i in range(len(tokens)):
        aliases.add(tokens[i])
        if i+1 < len(tokens):
            aliases.add(tokens[i] + " " + tokens[i+1])
        if i+2 < len(tokens):
            aliases.add(tokens[i] + " " + tokens[i+1] + " " + tokens[i+2])
    aliases = [a for a in aliases if 3 <= len(a) <= 100]
    aliases = sorted(set(aliases), key=lambda x: (-len(x), x))
    return aliases


# -----------------------------------------
# Map functions for each file
# -----------------------------------------
def map_endoproducts_row(row):
    sku = clean_text(row.get("SKU Code") or "")
    short_spec = clean_text(row.get("Short Specification") or "")
    detailed = clean_text(row.get("Detailed Specification") or "")
    brand = clean_text(row.get("Brand") or "")
    sub_brand = clean_text(row.get("Sub Brand") or "")
    lob = clean_text(row.get("LOB") or "")
    portfolio = clean_text(row.get("Portfolio") or "")
    speciality = clean_text(row.get("Speciality") or "")
    section = clean_text(row.get("Section") or "")

    product_name = short_spec or detailed or (brand + " " + sku).strip() or sku
    product_keywords = " ".join([
        product_name,
        detailed,
        brand,
        sub_brand,
        lob,
        portfolio,
        speciality,
        section
    ]).strip()

    aliases = extract_aliases_from_name(product_name)
    if brand:
        aliases.append(normalize_token(brand))
    if sub_brand:
        aliases.append(normalize_token(sub_brand))

    product_category = " | ".join([p for p in [lob, portfolio, speciality, section] if p])

    return {
        "product_id": sku if sku else np.nan,
        "product_name": product_name,
        "product_keywords": product_keywords,
        "product_aliases": ";".join(dict.fromkeys(aliases)),
        "product_category": product_category,
        "source_file": "EndoProducts.csv"
    }

def map_instrument_row(row):
    product_code = clean_text(row.get("Product Code") or "")
    name = clean_text(row.get("Name of Instrument") or "")
    category = clean_text(row.get("Category") or "")
    segment = clean_text(row.get("Segment") or "")

    product_name = name or product_code
    product_keywords = " ".join([product_name, category, segment]).strip()

    aliases = extract_aliases_from_name(product_name)
    if category:
        aliases.append(normalize_token(category))

    product_category = " | ".join([p for p in [category, segment] if p])

    return {
        "product_id": product_code if product_code else np.nan,
        "product_name": product_name,
        "product_keywords": product_keywords,
        "product_aliases": ";".join(dict.fromkeys(aliases)),
        "product_category": product_category,
        "source_file": "instrument_table.csv"
    }

def map_reagent_row(row):
    product_code = clean_text(row.get("Product Code") or "")
    mat_desc = clean_text(row.get("Material Description") or "")
    category = clean_text(row.get("Category") or "")
    pack_size = clean_text(row.get("Pack Size") or "")

    product_name = mat_desc or product_code
    product_keywords = " ".join([product_name, category, pack_size]).strip()

    aliases = extract_aliases_from_name(product_name)
    if category:
        aliases.append(normalize_token(category))

    product_category = category

    return {
        "product_id": product_code if product_code else np.nan,
        "product_name": product_name,
        "product_keywords": product_keywords,
        "product_aliases": ";".join(dict.fromkeys(aliases)),
        "product_category": product_category,
        "source_file": "reagent.csv"
    }

# -----------------------------------------
def read_with_autodetect(path):
    """Attempts to read CSV with detected encoding."""
    try:
        encoding = detect_encoding(path)
        return pd.read_csv(path, dtype=str, encoding=encoding)
    except Exception as e:
        print(f"[ERROR] Could not read {path}: {e}")
        return pd.DataFrame()


# -----------------------------------------
def build_product_master():

    rows = []

    # EndoProducts.csv
    if ENDO_FILE.exists():
        df = read_with_autodetect(ENDO_FILE)
        df.columns = [c.strip() for c in df.columns]
        for _, r in df.fillna("").iterrows():
            rows.append(map_endoproducts_row(r))
    else:
        print(f"[WARN] {ENDO_FILE} not found.")

    # instrument_table.csv
    if INSTR_FILE.exists():
        df = read_with_autodetect(INSTR_FILE)
        df.columns = [c.strip() for c in df.columns]
        for _, r in df.fillna("").iterrows():
            rows.append(map_instrument_row(r))
    else:
        print(f"[WARN] {INSTR_FILE} not found.")

    # reagent.csv
    if REAGENT_FILE.exists():
        df = read_with_autodetect(REAGENT_FILE)
        df.columns = [c.strip() for c in df.columns]
        for _, r in df.fillna("").iterrows():
            rows.append(map_reagent_row(r))
    else:
        print(f"[WARN] {REAGENT_FILE} not found.")

    if not rows:
        print("[ERROR] No products processed!")
        return

    pm = pd.DataFrame(rows)

    # Deduplicate
    if pm["product_id"].notna().any():
        pm = pm.drop_duplicates(subset=["product_id"], keep="first")
    else:
        pm = pm.drop_duplicates(subset=["product_name"], keep="first")

    # Fill missing IDs
    for idx in pm[pm["product_id"].isna()].index:
        pm.at[idx, "product_id"] = f"GENPROD-{idx:06d}"

    pm.to_csv(OUT_FILE, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"[OK] product_master written to {OUT_FILE} ({len(pm)} products).")
    print(pm.head(10).to_string(index=False))


if __name__ == "__main__":
    build_product_master()
