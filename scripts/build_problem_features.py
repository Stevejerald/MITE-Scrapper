"""
scripts/build_problem_features.py

Version 4 - Async + Hybrid Concurrency (recommended)

Highlights:
 - Uses asyncio as orchestrator
 - Uses ProcessPoolExecutor for CPU-bound per-row work (product matching + fuzzy)
 - Uses ThreadPoolExecutor for CSV write tasks
 - Sends rows in batches to workers for better performance
 - Shows live progress and ETA
 - Retries failed batches sequentially at the end and writes an error-log CSV if any failures

Inputs:
 - data/raw/data_scrapped.csv (Bid Number, Items, Quantity, Department, Detail URL, Start Date, End Date)
 - data/processed/product_master.csv
 - data/processed/merged_tenders_with_items.csv

Outputs:
 - data/processed/problem_tender_features.csv
 - data/processed/problem_tender_mapping.csv
 - data/processed/problem_tender_errors.csv (if failures)

Notes:
 - Tune BATCH_SIZE and WORKER_COUNT depending on available CPU & memory.
 - On Windows: multiprocessing requires top-level picklable functions (we abide by that).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import re
import string
import csv
import os
import time
import asyncio
from difflib import SequenceMatcher
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

ROOT = Path(".")
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

SCRAPED = RAW / "data_scrapped.csv"
PRODUCT_MASTER = PROC / "product_master.csv"
MERGED_UNIQUE = PROC / "merged_tenders_with_items.csv"

OUT_FEATURES = PROC / "problem_tender_features.csv"
OUT_MAPPING = PROC / "problem_tender_mapping.csv"
OUT_ERRORS = PROC / "problem_tender_errors.csv"

# Tunables
BATCH_SIZE = int(os.environ.get("MTIE_BATCH_SIZE", 512))   # rows per batch sent to a worker
WORKER_COUNT = int(os.environ.get("MTIE_WORKERS",  max(1, (os.cpu_count() or 2) - 1)))
FUZZY_THRESHOLD = float(os.environ.get("MTIE_FUZZY_THRESHOLD", 0.85))

# -------------------------
# Utility functions (must be top-level for multiprocessing pickle)
# -------------------------
def clean_text(s):
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.lower()
    keep = "/-"
    punct = "".join([c for c in string.punctuation if c not in keep])
    s = re.sub(r"[{}]".format(re.escape(punct)), " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def token_set(text):
    tokens = [t for t in re.split(r"\s+", text) if len(t) > 2]
    return set(tokens)

def simple_token_overlap_score(a_tokens, b_tokens):
    if not a_tokens or not b_tokens:
        return 0.0
    inter = a_tokens.intersection(b_tokens)
    union = a_tokens.union(b_tokens)
    return len(inter) / len(union) if union else 0.0

def ratio_similarity(a, b):
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

# Worker helper: receives a list of rows (dictionaries) and processes them
# Returns two lists: feature_rows, mapping_rows and optionally errors list
def process_batch(batch_rows, products, merged_rows_list, merged_set, fuzzy_threshold):
    feature_rows = []
    mapping_rows = []
    errors = []

    for idx_local, row in batch_rows:
        try:
            tn_raw = str(row.get("tender_number", "")).strip()
            tn = tn_raw if tn_raw else f"AUTO_SCRAPED_{idx_local+1:06d}"

            items = row.get("Items", "")
            qty = row.get("Quantity", "")
            dept = row.get("Department", "")
            url = row.get("Detail URL", "")
            start = row.get("Start Date", "")
            end = row.get("End Date", "")

            clean_items = clean_text(items)
            t_tokens = token_set(clean_items)

            # product matching
            scored = []
            text_clean_name = clean_items
            for p in products:
                s_token = simple_token_overlap_score(t_tokens, p["tokens"])
                s_name = ratio_similarity(clean_text(p["product_name"]), text_clean_name)
                score = (s_token * 0.8) + (s_name * 0.2)
                if score > 0:
                    scored.append((score, p))

            if scored:
                scored.sort(key=lambda x: x[0], reverse=True)
                top = scored[:5]
                best_matches = [{"product_id": p["product_id"], "product_name": p["product_name"], "score": float(score), "category": p.get("category","")} for score, p in top]
            else:
                best_matches = []

            product_kw_count = sum(1 for m in best_matches if m["score"] >= 0.05)
            meril_can_supply = 1 if product_kw_count > 0 else 0
            historical_participation = 1 if str(tn).strip().lower() in merged_set else 0
            top_products = ";".join([f'{m["product_id"]}||{m["product_name"]}||{m["score"]:.3f}' for m in best_matches])

            feat = {
                "tender_number": tn,
                "detail_url": url,
                "raw_items": items,
                "clean_items": clean_items,
                "product_kw_count": product_kw_count,
                "meril_can_supply": meril_can_supply,
                "top_products": top_products,
                "quantity": qty,
                "department": dept,
                "start_date": start,
                "end_date": end,
                "text_length": len(clean_items),
                "word_count": len(clean_items.split()),
                "has_quantity": 1 if str(qty).strip() not in ["", "0"] else 0,
                "historical_participation": historical_participation
            }

            # mapping (exact or fuzzy)
            key = str(tn).strip().lower()
            found = 0
            matched_tn = ""
            src_file = ""
            lbl = 0

            if key != "":
                if key in merged_set:
                    # exact match - find first row
                    for mr in merged_rows_list:
                        if str(mr.get("tender_number","")).strip().lower() == key:
                            found = 1
                            matched_tn = mr.get("tender_number","")
                            src_file = mr.get("source_file","")
                            lbl = int(mr.get("label_relevant", 0)) if mr.get("label_relevant", "") != "" else 0
                            break
                else:
                    # fuzzy compare (iterate merged list)
                    best_score = 0.0
                    best_row = None
                    for mr in merged_rows_list:
                        m_tn = str(mr.get("tender_number","")).strip().lower()
                        if not m_tn:
                            continue
                        score = ratio_similarity(key, m_tn)
                        if score > best_score:
                            best_score = score
                            best_row = mr
                    if best_score >= fuzzy_threshold and best_row is not None:
                        found = 1
                        matched_tn = best_row.get("tender_number","")
                        src_file = best_row.get("source_file","")
                        lbl = int(best_row.get("label_relevant", 0)) if best_row.get("label_relevant", "") != "" else 0

            map_row = {
                "scraped_tender_number": tn,
                "original_bid_number": tn_raw,
                "found_in_merged": int(found),
                "matched_tender_number": matched_tn,
                "source_file": src_file,
                "label_relevant": int(lbl)
            }

            feature_rows.append(feat)
            mapping_rows.append(map_row)

        except Exception as e:
            # record error for this idx_local, include exception text
            errors.append({"idx": idx_local, "exception": repr(e), "row": row})

    return feature_rows, mapping_rows, errors

# -------------------------
# Loader helpers
# -------------------------
def load_scraped():
    if not SCRAPED.exists():
        raise FileNotFoundError(f"Missing scraped file: {SCRAPED}")
    df = pd.read_csv(SCRAPED, dtype=str).fillna("")
    df.columns = [c.strip() for c in df.columns]
    if "Bid Number" not in df.columns or "Items" not in df.columns:
        raise KeyError("data_scrapped.csv must contain 'Bid Number' and 'Items' columns")
    df = df.rename(columns={"Bid Number": "tender_number"})
    return df

def load_products():
    if not PRODUCT_MASTER.exists():
        raise FileNotFoundError(f"Missing product master: {PRODUCT_MASTER}")
    pm = pd.read_csv(PRODUCT_MASTER, dtype=str).fillna("")
    pm.columns = [c.strip() for c in pm.columns]
    products = []
    for _, r in pm.iterrows():
        pid = r.get("product_id","")
        pname = r.get("product_name","")
        pk = str(r.get("product_keywords",""))
        pa = str(r.get("product_aliases",""))
        combined = " ".join([pname, pk, pa])
        products.append({
            "product_id": pid,
            "product_name": pname,
            "tokens": token_set(clean_text(combined)),
            "category": r.get("product_category","")
        })
    return products

def load_merged():
    if not MERGED_UNIQUE.exists():
        print(f"[WARN] {MERGED_UNIQUE} not found; historical participation disabled")
        return [], set()
    mh = pd.read_csv(MERGED_UNIQUE, dtype=str).fillna("")
    mh.columns = [c.strip() for c in mh.columns]
    if "tender_number" not in mh.columns:
        raise KeyError("merged_tenders_with_items.csv must contain 'tender_number' column")
    mh["tender_number_norm"] = mh["tender_number"].astype(str).str.lower().str.strip()
    merged_rows_list = mh.to_dict(orient="records")
    merged_set = set(mh["tender_number_norm"].tolist())
    return merged_rows_list, merged_set

# -------------------------
# Async orchestrator
# -------------------------
async def build_problem_features_async(batch_size=BATCH_SIZE, workers=WORKER_COUNT, fuzzy_threshold=FUZZY_THRESHOLD):
    print("\n============================")
    print("BUILDING PROBLEM FEATURES (ASYNC HYBRID)...")
    print("============================\n")

    # load inputs synchronously (quick)
    print("[STEP 1/7] Loading scraped tenders...")
    df = load_scraped()
    total_rows = len(df)
    print(f"[OK] Loaded scraped → {total_rows} rows")

    print("\n[STEP 2/7] Loading product master...")
    products = load_products()
    print(f"[OK] Loaded products → {len(products)}")

    print("\n[STEP 3/7] Loading merged tenders (history)...")
    merged_rows_list, merged_set = load_merged()
    print(f"[OK] Merged history rows → {len(merged_rows_list)}")

    # prepare batches: convert dataframe to list-of-dicts with global index
    rows = list(df.to_dict(orient="records"))
    indexed = list(enumerate(rows))   # (global_idx, row_dict)

    # create executors
    loop = asyncio.get_running_loop()
    proc_executor = ProcessPoolExecutor(max_workers=workers)
    thread_executor = ThreadPoolExecutor(max_workers=2)

    # function wrapper that calls process_batch in executor
    process_in_executor = partial(
        process_batch,
        products=products,
        merged_rows_list=merged_rows_list,
        merged_set=merged_set,
        fuzzy_threshold=fuzzy_threshold
    )

    # schedule batches (send BATCH_SIZE rows per batch)
    batches = []
    for i in range(0, len(indexed), batch_size):
        batch = indexed[i:i+batch_size]
        batches.append(batch)

    total_batches = len(batches)
    print(f"\n[STEP 4/7] Processing {len(indexed)} rows in {total_batches} batches (batch_size={batch_size}) using {workers} worker processes")

    features_accum = []
    mapping_accum = []
    errors_accum = []

    start_time = time.time()
    completed_batches = 0

    # create async tasks for each batch using run_in_executor
    tasks = []
    for b in batches:
        # run process_batch(b, products, merged_rows_list, merged_set, fuzzy_threshold) in process pool
        tasks.append(loop.run_in_executor(proc_executor, process_in_executor, b))

    # iterate tasks as they complete using asyncio.as_completed
    for coro in asyncio.as_completed(tasks):
        try:
            feature_rows, mapping_rows, errors = await coro
            features_accum.extend(feature_rows)
            mapping_accum.extend(mapping_rows)
            if errors:
                errors_accum.extend(errors)
        except Exception as e:
            # catastrophic batch failure; log and continue
            errors_accum.append({"idx_batch_exception": repr(e)})
        completed_batches += 1

        # progress print
        elapsed = time.time() - start_time
        processed_rows = min(len(features_accum), total_rows)
        pct = (processed_rows / total_rows) * 100 if total_rows else 100
        avg_per_batch = elapsed / max(1, completed_batches)
        eta = avg_per_batch * (total_batches - completed_batches)
        print(f"   → Batches done {completed_batches}/{total_batches} | rows processed ~{processed_rows}/{total_rows} ({pct:.2f}%) | elapsed {elapsed:.1f}s | ETA {eta:.1f}s")

    # shutdown executors
    proc_executor.shutdown(wait=True)
    thread_executor.shutdown(wait=True)

    print("\n[OK] Batch processing complete")
    print(f"[INFO] Feature rows generated: {len(features_accum)}")
    print(f"[INFO] Mapping rows generated: {len(mapping_accum)}")
    print(f"[INFO] Errors recorded: {len(errors_accum)}")

    # optional retry for failed rows: serialize errors to CSV for inspection
    if errors_accum:
        print(f"[WARN] Writing {len(errors_accum)} errors to {OUT_ERRORS}")
        PROC.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(errors_accum).to_csv(OUT_ERRORS, index=False, quoting=csv.QUOTE_MINIMAL)

    # Build DataFrames and composite score
    out_df = pd.DataFrame(features_accum)
    if out_df.empty:
        print("[WARN] No features produced. Exiting.")
        return

    out_df["composite_score"] = out_df["product_kw_count"].astype(float) * 0.8 + out_df["historical_participation"].astype(float) * 1.0
    max_comp = out_df["composite_score"].max() if not out_df.empty else 0
    out_df["composite_score_norm"] = out_df["composite_score"].apply(lambda x: x / max_comp if max_comp > 0 else 0.0)

    mapping_df = pd.DataFrame(mapping_accum)

    # write CSVs using thread executor to avoid blocking
    def write_csvs():
        PROC.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(OUT_FEATURES, index=False, quoting=csv.QUOTE_MINIMAL)
        mapping_df.to_csv(OUT_MAPPING, index=False, quoting=csv.QUOTE_MINIMAL)

    await loop.run_in_executor(thread_executor, write_csvs)

    elapsed_total = time.time() - start_time
    print(f"\n[STEP 7/7] Saved features ({len(out_df)}) and mapping ({len(mapping_df)}) — elapsed {elapsed_total:.1f}s")
    print(f"[SAVED] {OUT_FEATURES}")
    print(f"[SAVED] {OUT_MAPPING}")

# -------------------------
# Entrypoint
# -------------------------
def build_problem_features():
    asyncio.run(build_problem_features_async())

if __name__ == "__main__":
    build_problem_features()