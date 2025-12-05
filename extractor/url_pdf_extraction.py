#!/usr/bin/env python3
"""
url_pdf_extraction.py

Pipeline with optional status JSON updates (status_path).
"""
import os
import sys
import re
import csv
import json
import time
import argparse
import logging
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import List, Tuple, Set
from pathlib import Path

import requests
from tqdm import tqdm

# extraction libs (import-level; ensure installed in same venv)
import pdfplumber
import pytesseract
from PyPDF2 import PdfReader

# ---------- Defaults ----------
DEFAULT_CSV = "filtered_main.csv"
PDF_FOLDER_DEFAULT = "PDF"
OUTPUT_FOLDER_DEFAULT = "OUTPUT"
FULLDATA_FOLDER = "FULLDATA"
DOWNLOAD_MAP = "download_map.csv"
EXTRACT_MAP = "extract_map.csv"
FULLDATA_FILENAME = "data.json"

# filename sanitize
FNAME_SAFE_RE = re.compile(r'[^A-Za-z0-9\-\._]')

# sanitization & behavior
OCR_THRESHOLD_CHARS = 40
REMOVE_CID_TOKENS = True
CID_RE = re.compile(r'\(cid:\d+\)')
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]+')

# ---------------- utility helpers ----------------
def make_safe_filename(s: str, max_len=180):
    s = (s or "").strip()
    s = FNAME_SAFE_RE.sub('_', s)
    return s[:max_len].rstrip('_')

def filename_from_url(url: str, attempt_idx: int = 0) -> str:
    parsed = urllib.parse.urlparse(url)
    base = os.path.basename(parsed.path) or parsed.netloc
    if base == '':
        base = parsed.netloc
    name = urllib.parse.unquote(base)
    if not os.path.splitext(name)[1]:
        name = name + ".pdf"
    name = make_safe_filename(name)
    if attempt_idx:
        name = f"{os.path.splitext(name)[0]}_{attempt_idx}{os.path.splitext(name)[1]}"
    return name

# ---------------- status helper ----------------
def write_status_file(status_path: str, updates: dict):
    if not status_path:
        return
    try:
        # safe read
        s = {}
        if os.path.exists(status_path):
            try:
                with open(status_path, "r", encoding="utf-8") as f:
                    s = json.load(f)
            except Exception:
                s = {}
        s.update(updates)
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)
    except Exception:
        # best-effort; don't raise
        pass

# ---------------- downloader ----------------
def download_one(url: str, dest_folder: str, session: requests.Session, timeout=30, max_retries=2, resume=True) -> Tuple[str, str, bool, str]:
    """
    Download URL -> dest_folder; returns (url, saved_path_or_None, ok_bool, msg)
    """
    try:
        url = (url or "").strip()
        if not url:
            return (url, None, False, "empty_url")

        fname = None
        # try HEAD for content-disposition
        try:
            head = session.head(url, allow_redirects=True, timeout=timeout)
            cd = head.headers.get('content-disposition')
            if cd:
                m = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^;"\']+)', cd, flags=re.I)
                if m:
                    fname = urllib.parse.unquote(m.group(1))
        except Exception:
            pass

        if not fname:
            fname = filename_from_url(url)

        os.makedirs(dest_folder, exist_ok=True)
        dest_path = os.path.join(dest_folder, fname)

        # resume skip
        if resume and os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            return (url, dest_path, True, "skipped_exists")

        # avoid collision with zero-length files
        idx = 0
        while os.path.exists(dest_path) and os.path.getsize(dest_path) == 0:
            idx += 1
            dest_path = os.path.join(dest_folder, filename_from_url(url, idx))

        last_err = None
        for attempt in range(max_retries + 1):
            try:
                with session.get(url, stream=True, timeout=timeout) as r:
                    r.raise_for_status()
                    cd = r.headers.get('content-disposition')
                    if cd:
                        m = re.search(r'filename\*?=(?:UTF-8\'\')?["\']?([^;"\']+)', cd, flags=re.I)
                        if m:
                            fname_cd = urllib.parse.unquote(m.group(1))
                            fname_cd = make_safe_filename(fname_cd)
                            dest_path = os.path.join(dest_folder, fname_cd)
                            k = 0
                            while os.path.exists(dest_path):
                                k += 1
                                dest_path = os.path.join(dest_folder, f"{os.path.splitext(fname_cd)[0]}_{k}{os.path.splitext(fname_cd)[1]}")
                    with open(dest_path, "wb") as out_f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                out_f.write(chunk)
                return (url, dest_path, True, "downloaded")
            except Exception as e:
                last_err = str(e)
                time.sleep(0.2)
                continue
        return (url, None, False, f"failed_after_retries:{last_err}")
    except Exception as e:
        return (url, None, False, str(e))

def download_all(urls: List[str], dest_folder: str, max_workers=8, timeout=30, resume=True, status_path: str = None) -> List[Tuple[str,str,bool,str]]:
    """
    Downloads urls concurrently. Writes incremental progress to status_path (if provided).
    """
    os.makedirs(dest_folder, exist_ok=True)
    results = []
    session = requests.Session()
    session.headers.update({"User-Agent": "pdf-downloader-bot/1.0"})
    total = len(urls)
    completed = 0
    write_status_file(status_path, {"download_total": total, "download_done": 0, "stage": "downloading", "message": "Downloading PDFs..."})
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(download_one, url, dest_folder, session, timeout, 2, resume): url for url in urls}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading PDFs"):
            url = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = (url, None, False, str(e))
            results.append(res)
            completed += 1
            # update status file incrementally
            write_status_file(status_path, {"download_done": completed, "download_total": total, "stage": "downloading", "message": f"Downloaded {completed}/{total} PDFs"})
    write_status_file(status_path, {"download_done": completed, "download_total": total, "stage": "downloaded", "message": "Download stage finished"})
    return results

def write_download_map(map_path: str, records):
    with open(map_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url", "saved_path", "ok", "message"])
        for r in records:
            w.writerow([r[0], r[1] or "", "1" if r[2] else "0", r[3]])

def load_urls_from_csv(csv_path: str, preferred_cols=None) -> Tuple[List[str], str]:
    if preferred_cols is None:
        preferred_cols = ["Details URL", "Detail URL", "Detail_URL", "Details_URL"]
    urls = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        chosen_col = None
        for pc in preferred_cols:
            if pc in fields:
                chosen_col = pc
                break
        if not chosen_col:
            for fld in fields:
                if fld and 'detail' in fld.lower():
                    chosen_col = fld
                    break
        if not chosen_col:
            raise ValueError(f"Could not find a 'Details' URL column. CSV columns: {fields}")
        for row in reader:
            urls.append(row.get(chosen_col, ""))
    return urls, chosen_col

# ---------------- sanitization / text helpers ----------------
def strip_devanagari_keep_english(text: str) -> str:
    if not text:
        return ""
    parts = re.findall(r'[^\u0900-\u097F]+', text)
    if not parts:
        return ""
    joined = "".join(parts)
    joined = re.sub(r'\n\s*\n+', '\n\n', joined)
    joined = re.sub(r'[ \t\f\r\v]+', ' ', joined)
    joined = re.sub(r' *\n *', '\n', joined)
    return joined.strip(" \t\n\r")

def remove_cid_tokens(text: str) -> str:
    if not text:
        return text
    return CID_RE.sub('', text) if REMOVE_CID_TOKENS else text

def sanitize_text(text: str) -> str:
    if not text:
        return ""
    s = strip_devanagari_keep_english(text)
    s = remove_cid_tokens(s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# ---------------- OCR helper ----------------
def ocr_page_image(page, resolution=200):
    try:
        pil = page.to_image(resolution=resolution).original
        return pytesseract.image_to_string(pil)
    except Exception:
        return ""

# ---------------- table/word extraction ----------------
def clean_table_cells(table):
    cleaned = []
    for row in table:
        cleaned_row = []
        for cell in row:
            if cell is None:
                cleaned_row.append("")
            else:
                c = sanitize_text(str(cell))
                cleaned_row.append(c)
        cleaned.append(cleaned_row)
    return cleaned

def extract_words_with_bbox(page):
    words = []
    try:
        words_raw = page.extract_words() or []
        for w in words_raw:
            txt_raw = w.get("text", "") or ""
            cleaned = sanitize_text(txt_raw)
            words.append({
                "text": cleaned,
                "x0": float(w.get("x0", 0)),
                "x1": float(w.get("x1", 0)),
                "top": float(w.get("top", 0)),
                "bottom": float(w.get("bottom", 0)),
                "doctop": float(w.get("doctop", 0))
            })
    except Exception:
        pass
    return words

def extract_tables(page):
    out_tables = []
    try:
        raw_tables = page.extract_tables() or []
        for t in raw_tables:
            out_tables.append(clean_table_cells(t))
    except Exception:
        pass
    return out_tables

# ---------------- page struct ----------------
def page_to_struct(page, page_no:int, use_ocr_if_needed:bool):
    text = page.extract_text() or ""
    used_ocr = False
    if (not text or len(text.strip()) < OCR_THRESHOLD_CHARS) and use_ocr_if_needed:
        ocr_text = ocr_page_image(page)
        if ocr_text and len(ocr_text.strip()) >= OCR_THRESHOLD_CHARS:
            text = ocr_text
            used_ocr = True
    cleaned_text = sanitize_text(text)
    lines = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()]
    tables = extract_tables(page)
    words = extract_words_with_bbox(page)
    return {
        "page_number": page_no,
        "used_ocr": used_ocr,
        "cleaned_text": cleaned_text,
        "lines": lines,
        "tables": tables,
        "words": words  # will be stripped before disk write
    }

# ---------------- deep annotation scanning using PyPDF2 ----------------
def _ensure_iterable(raw):
    if not raw:
        return []
    try:
        if hasattr(raw, "get_object"):
            raw = raw.get_object()
    except Exception:
        pass
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return [raw]

def _collect_uris_and_attachments_from_obj(obj, seen: Set[int], pdf_reader: PdfReader):
    found = []
    oid = id(obj)
    if oid in seen:
        return found
    seen.add(oid)
    try:
        if hasattr(obj, "get_object"):
            obj = obj.get_object()
    except Exception:
        pass
    try:
        if isinstance(obj, dict):
            for k, v in obj.items():
                kn = str(k)
                if kn == "/URI":
                    try:
                        found.append({"uri": str(v), "attachment_filename": None, "source": "dict:/URI"})
                    except Exception:
                        pass
                if kn == "/F":
                    try:
                        found.append({"uri": str(v), "attachment_filename": None, "source": "dict:/F"})
                    except Exception:
                        pass
                if kn == "/EF" or kn == "/FS":
                    try:
                        if isinstance(v, dict):
                            if "/F" in v:
                                try:
                                    name = v["/F"]
                                    if hasattr(name, "get_object"):
                                        name = name.get_object()
                                    if isinstance(name, dict) and "/F" in name:
                                        fn = name["/F"]
                                        found.append({"uri": None, "attachment_filename": str(fn), "source": "dict:/EF:/F"})
                                    else:
                                        found.append({"uri": None, "attachment_filename": str(name), "source": "dict:/EF"})
                                except Exception:
                                    pass
                        else:
                            found.append({"uri": None, "attachment_filename": str(v), "source": "dict:/EF_raw"})
                    except Exception:
                        pass
                try:
                    found.extend(_collect_uris_and_attachments_from_obj(v, seen, pdf_reader))
                except Exception:
                    pass
        elif isinstance(obj, (list, tuple)):
            for el in obj:
                try:
                    found.extend(_collect_uris_and_attachments_from_obj(el, seen, pdf_reader))
                except Exception:
                    pass
    except Exception:
        pass
    return found

def get_page_annotations_deep(pdf_reader: PdfReader, page_index: int):
    results = []
    try:
        page = pdf_reader.pages[page_index]
    except Exception:
        return results

    def norm_rect(r):
        try:
            r = list(r)
            x0 = min(float(r[0]), float(r[2])); x1 = max(float(r[0]), float(r[2]))
            y0 = min(float(r[1]), float(r[3])); y1 = max(float(r[1]), float(r[3]))
            return [x0, y0, x1, y1]
        except Exception:
            return None

    raw_annots = None
    try:
        raw_annots = page.get("/Annots") or page.get("/Annots", None)
    except Exception:
        try:
            raw_annots = page.get("/Annots")
        except Exception:
            raw_annots = None

    annot_items = _ensure_iterable(raw_annots)
    for a in annot_items:
        try:
            try:
                a_obj = a.get_object()
            except Exception:
                a_obj = a
            rect = None
            try:
                if "/Rect" in a_obj:
                    rect = norm_rect(a_obj["/Rect"])
            except Exception:
                rect = None

            found = _collect_uris_and_attachments_from_obj(a_obj, set(), pdf_reader)
            if found:
                for f in found:
                    results.append({"uri": f.get("uri"), "attachment_filename": f.get("attachment_filename"), "rect": rect, "source": f.get("source")})
            else:
                try:
                    a_dict = a_obj.get("/A") if "/A" in a_obj else None
                    if a_dict:
                        try:
                            a_dict = a_dict.get_object() if hasattr(a_dict, "get_object") else a_dict
                        except Exception:
                            pass
                        if isinstance(a_dict, dict) and "/URI" in a_dict:
                            results.append({"uri": str(a_dict["/URI"]), "attachment_filename": None, "rect": rect, "source": "annot:/A:/URI"})
                except Exception:
                    pass
        except Exception:
            continue

    try:
        page_obj = page.get_object() if hasattr(page, "get_object") else page
    except Exception:
        page_obj = page
    for key in ("/AA", "/OpenAction"):
        try:
            if key in page_obj:
                found = _collect_uris_and_attachments_from_obj(page_obj[key], set(), pdf_reader)
                for f in found:
                    results.append({"uri": f.get("uri"), "attachment_filename": f.get("attachment_filename"), "rect": None, "source": f.get("source")})
        except Exception:
            pass

    try:
        catalog = pdf_reader.trailer["/Root"]
        if "/Names" in catalog:
            names = catalog["/Names"]
            try:
                names = names.get_object() if hasattr(names, "get_object") else names
            except Exception:
                pass
            if isinstance(names, dict) and "/EmbeddedFiles" in names:
                ef = names["/EmbeddedFiles"]
                try:
                    ef = ef.get_object() if hasattr(ef, "get_object") else ef
                except Exception:
                    pass
                if isinstance(ef, dict) and "/Names" in ef:
                    arr = ef["/Names"]
                    try:
                        arr = arr.get_object() if hasattr(arr, "get_object") else arr
                    except Exception:
                        pass
                    if isinstance(arr, (list, tuple)):
                        for i in range(0, len(arr), 2):
                            try:
                                name = arr[i]
                                fs = arr[i+1]
                                fname = str(name)
                                try:
                                    fs_obj = fs.get_object() if hasattr(fs, "get_object") else fs
                                    if isinstance(fs_obj, dict) and "/F" in fs_obj:
                                        fname = str(fs_obj["/F"])
                                except Exception:
                                    pass
                                results.append({"uri": None, "attachment_filename": fname, "rect": None, "source": "catalog:/Names/EmbeddedFiles"})
                            except Exception:
                                pass
    except Exception:
        pass

    try:
        found = _collect_uris_and_attachments_from_obj(page_obj, set(), pdf_reader)
        for f in found:
            results.append({"uri": f.get("uri"), "attachment_filename": f.get("attachment_filename"), "rect": None, "source": f.get("source")})
    except Exception:
        pass

    return results

# ---------------- map rect -> label ----------------
def map_link_to_label(link_rect, words, lines):
    if not link_rect:
        return None
    x0, y0, x1, y1 = link_rect
    if words:
        label_words = []
        pad_x = (x1 - x0) * 0.1 + 2.0
        pad_y = (y1 - y0) * 0.2 + 2.0
        ex0, ey0, ex1, ey1 = x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y
        for w in words:
            cx = (w["x0"] + w["x1"]) / 2.0
            cy = (w["top"] + w["bottom"]) / 2.0
            if ex0 <= cx <= ex1 and ey0 <= cy <= ey1:
                label_words.append((w["x0"], w["text"]))
        if label_words:
            label_words.sort(key=lambda t: t[0])
            return " ".join(t[1] for t in label_words).strip()

    if lines:
        for ln in lines:
            if ln and len(ln) <= 120:
                return ln.strip()
    return None

# ---------------- optional save embedded attachments (kept as-is) ----------------
def _save_embedded_files_from_catalog(pdf_reader: PdfReader, out_dir: str):
    try:
        os.makedirs(out_dir, exist_ok=True)
        catalog = pdf_reader.trailer["/Root"]
        if "/Names" in catalog:
            names = catalog["/Names"]
            try:
                names = names.get_object() if hasattr(names, "get_object") else names
            except Exception:
                pass
            if isinstance(names, dict) and "/EmbeddedFiles" in names:
                ef = names["/EmbeddedFiles"]
                try:
                    ef = ef.get_object() if hasattr(ef, "get_object") else ef
                except Exception:
                    pass
                if isinstance(ef, dict) and "/Names" in ef:
                    arr = ef["/Names"]
                    try:
                        arr = arr.get_object() if hasattr(arr, "get_object") else arr
                    except Exception:
                        pass
                    if isinstance(arr, (list, tuple)):
                        for i in range(0, len(arr), 2):
                            try:
                                name = arr[i]
                                fs = arr[i+1]
                                fname = str(name)
                                fs_obj = fs.get_object() if hasattr(fs, "get_object") else fs
                                if isinstance(fs_obj, dict) and "/EF" in fs_obj:
                                    ef_dict = fs_obj["/EF"]
                                    file_stream = ef_dict.get("/F") or ef_dict.get("/f")
                                    try:
                                        file_stream = file_stream.get_object()
                                    except Exception:
                                        pass
                                    try:
                                        data = file_stream.get_data()
                                        out_path = os.path.join(out_dir, fname)
                                        with open(out_path, "wb") as outf:
                                            outf.write(data)
                                    except Exception:
                                        pass
                            except Exception:
                                pass
    except Exception:
        pass

# ---------------- single-PDF processor (deep) ----------------
def process_single_pdf_file_deep(pdf_path: str, output_path: str, use_ocr_if_needed: bool):
    try:
        pdf_reader = PdfReader(pdf_path)
        result = {"source_file": os.path.basename(pdf_path), "num_pages": 0, "pages": [], "links": []}

        # with pdfplumber to extract text/tables/words (words used internally)
        with pdfplumber.open(pdf_path) as pdf:
            result["num_pages"] = len(pdf.pages)
            for i, page in enumerate(pdf.pages, start=1):
                struct = page_to_struct(page, i, use_ocr_if_needed)
                ann_results = get_page_annotations_deep(pdf_reader, i-1)
                for a in ann_results:
                    uri = a.get("uri")
                    rect = a.get("rect")
                    attachment = a.get("attachment_filename")
                    source = a.get("source")
                    label = map_link_to_label(rect, struct.get("words", []), struct.get("lines", [])) if rect else None
                    result["links"].append({
                        "page_number": i,
                        "uri": uri,
                        "attachment_filename": attachment,
                        "rect": rect,
                        "label": label,
                        "source": source
                    })
                result["pages"].append(struct)

        combined = "\n\n".join(p["cleaned_text"] for p in result["pages"] if p["cleaned_text"])
        result["combined_cleaned_text"] = combined

        # Remove heavy/internal-only 'words' arrays before writing JSONs (user requested)
        for p in result["pages"]:
            if "words" in p:
                p.pop("words", None)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return (pdf_path, output_path, True, "OK")
    except Exception as e:
        return (pdf_path, output_path, False, str(e))

# ---------------- batch extraction ----------------
def batch_extract(pdf_folder: str, output_folder: str, max_workers: int, use_ocr_if_needed: bool, status_path: str = None) -> List[Tuple[str,str,bool,str]]:
    os.makedirs(output_folder, exist_ok=True)
    pdfs = [f for f in sorted(os.listdir(pdf_folder)) if f.lower().endswith(".pdf")]
    if not pdfs:
        logging.info("No PDF files found inside %s", pdf_folder)
        write_status_file(status_path, {"stage": "extracting", "extract_total": 0, "extract_done": 0, "message": "No PDFs to extract"})
        return []

    tasks = [(os.path.join(pdf_folder, p), os.path.join(output_folder, os.path.splitext(p)[0] + ".json")) for p in pdfs]
    results = []
    max_workers = max(1, max_workers or 1)

    total = len(tasks)
    done = 0
    write_status_file(status_path, {"stage": "extracting", "extract_total": total, "extract_done": 0, "message": "Starting extraction..."})

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        future_map = {exe.submit(process_single_pdf_file_deep, src, out, use_ocr_if_needed): (src, out) for src, out in tasks}
        iterator = as_completed(future_map)
        iterator = tqdm(iterator, total=len(future_map), desc="Processing PDFs")
        for fut in iterator:
            src, out = future_map[fut]
            try:
                pdfp, outp, ok, msg = fut.result()
                if not ok:
                    logging.error("Failed %s -> %s : %s", pdfp, outp, msg)
                else:
                    logging.debug("Done %s -> %s", pdfp, outp)
                results.append((pdfp, outp, ok, msg))
            except Exception as e:
                logging.exception("Processing failed for %s: %s", src, e)
                results.append((src, out, False, str(e)))
            done += 1
            # update status
            write_status_file(status_path, {"extract_done": done, "extract_total": total, "stage": "extracting", "message": f"Extracted {done}/{total} PDFs"})
    write_status_file(status_path, {"extract_done": done, "extract_total": total, "stage": "extracted", "message": "Extraction completed"})
    return results

# ---------------- merge OUTPUT JSONs into FULLDATA/data.json ----------------
def build_fulldata_json(output_folder: str, fulldata_folder: str, fulldata_filename: str = FULLDATA_FILENAME) -> Tuple[str,bool,str]:
    os.makedirs(fulldata_folder, exist_ok=True)
    files = [f for f in sorted(os.listdir(output_folder)) if f.lower().endswith(".json")]
    if not files:
        return ("", False, "no_jsons_found")

    merged = {}
    skipped = []
    for fname in files:
        fp = os.path.join(output_folder, fname)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                parsed = json.load(f)
            merged[fname] = parsed
        except Exception as e:
            logging.warning("Skipping unreadable JSON %s : %s", fp, e)
            skipped.append((fname, str(e)))
            continue

    out_path = os.path.join(fulldata_folder, fulldata_filename)
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
        msg = f"merged {len(merged)} files"
        if skipped:
            msg += f", skipped {len(skipped)}"
        return (out_path, True, msg)
    except Exception as e:
        return ("", False, str(e))

# ---------------- run_pipeline (importable) ----------------
def run_pipeline(
    csv_path: str,
    pdf_folder: str = PDF_FOLDER_DEFAULT,
    out_folder: str = OUTPUT_FOLDER_DEFAULT,
    fulldata_folder: str = FULLDATA_FOLDER,
    download_workers: int = 6,
    download_timeout: int = 30,
    extract_workers: int = None,
    no_ocr: bool = False,
    skip_download: bool = False,
    download_only: bool = False,
    log_level: str = "info",
    status_path: str = None  # optional path to JSON status file updated live
):
    import shutil

    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(message)s")

    use_ocr_if_needed = not no_ocr
    os.makedirs(pdf_folder, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)
    os.makedirs(fulldata_folder, exist_ok=True)

    # ---------------------- DOWNLOAD STAGE ----------------------
    if not skip_download:
        logging.info("Loading URLs from CSV: %s", csv_path)
        urls, used_col = load_urls_from_csv(csv_path, preferred_cols=["Details URL", "Detail URL", "Detail_URL"])
        logging.info("Using URL column: %s ; Found %d URLs", used_col, len(urls))

        urls = [u.strip() for u in urls if u and u.strip()]
        seen = set()
        unique_urls = []

        for u in urls:
            if u not in seen:
                seen.add(u)
                unique_urls.append(u)

        total_urls = len(unique_urls)
        write_status_file(status_path, {"download_total": total_urls, "download_done": 0, "stage": "downloading", "message": f"Starting downloads ({total_urls})"})

        logging.info("Starting download of %d unique URLs into %s", total_urls, pdf_folder)

        dl_results = download_all(unique_urls, pdf_folder,
                                  max_workers=download_workers,
                                  timeout=download_timeout,
                                  resume=True,
                                  status_path=status_path)

        ok_count = sum(1 for r in dl_results if r[2])
        failed = [r for r in dl_results if not r[2]]

        logging.info("Download complete. Success=%d Failed=%d", ok_count, len(failed))

        write_download_map(os.path.join(os.getcwd(), DOWNLOAD_MAP), dl_results)
        write_status_file(status_path, {"download_done": len(dl_results), "download_total": total_urls, "stage": "downloaded", "message": f"Download finished. Success={ok_count} Fail={len(failed)}"})

        if failed:
            logging.warning("Some downloads failed. See download_map.csv")
            # record failures into status
            write_status_file(status_path, {"errors": [f for f in failed[:50]]})  # keep sample

    if download_only:
        logging.info("Download-only flag set — stopping early.")
        return {"status": "download_only", "download_map": os.path.join(os.getcwd(), DOWNLOAD_MAP)}

    # ---------------------- EXTRACTION STAGE ----------------------
    logging.info("Extracting PDF files: %s -> %s", pdf_folder, out_folder)

    extract_workers_final = extract_workers if extract_workers is not None else max(1, cpu_count() - 1)
    # if OCR enabled, be conservative
    if use_ocr_if_needed:
        extract_workers_final = max(1, min(extract_workers_final, max(1, cpu_count() // 2)))

    write_status_file(status_path, {"extract_total": 0, "extract_done": 0, "stage": "extracting", "message": "Preparing extraction..."})

    results = batch_extract(pdf_folder, out_folder, extract_workers_final, use_ocr_if_needed, status_path=status_path)

    success_count = sum(1 for r in results if r[2])
    fail_count = len(results) - success_count

    logging.info("Extraction finished. Total=%d Success=%d Failed=%d",
                 len(results), success_count, fail_count)

    # Write extract map
    try:
        with open(EXTRACT_MAP, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["pdf_path", "json_path", "ok", "message"])
            for r in results:
                w.writerow([r[0], r[1] or "", "1" if r[2] else "0", r[3]])
    except Exception:
        logging.error("Could not write extract_map.csv")

    # ---------------------- BUILD FULLDATA ----------------------
    out_path, ok, msg = build_fulldata_json(out_folder, fulldata_folder, FULLDATA_FILENAME)

    if ok:
        logging.info("FULLDATA created: %s (%s)", out_path, msg)
        write_status_file(status_path, {"stage": "building_full", "message": "Building FULLDATA...", "full_data_path": out_path})
    else:
        logging.warning("Could not generate FULLDATA: %s", msg)
        write_status_file(status_path, {"stage": "error", "message": f"Could not generate FULLDATA: {msg}"})

    # ---------------------- DELETE PDF FOLDER ----------------------
    try:
        shutil.rmtree(pdf_folder, ignore_errors=True)
        logging.info("Deleted PDF folder after extraction: %s", pdf_folder)
        write_status_file(status_path, {"message": "Deleted PDF folder", "stage": "cleanup"})
    except Exception as e:
        logging.error("Could not delete PDF folder: %s", e)
        write_status_file(status_path, {"message": f"Could not delete PDF folder: {e}", "stage": "cleanup"})

    # final status
    write_status_file(status_path, {"stage": "done", "message": "Pipeline finished", "extraction_total": len(results), "extraction_success": success_count, "extraction_failed": fail_count, "full_data_path": out_path if ok else None})

    return {
        "status": "done",
        "full_data_path": out_path if ok else None,
        "extraction_total": len(results),
        "extraction_success": success_count,
        "extraction_failed": fail_count
    }


# ---------------- CLI compatibility ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Download PDFs from CSV and extract to JSONs (deep). Produces FULLDATA/data.json (dict by filename).")
    p.add_argument("--csv", default=DEFAULT_CSV, help="CSV file containing URLs (default filtered_main.csv)")
    p.add_argument("--pdf-folder", default=PDF_FOLDER_DEFAULT, help="Folder to save PDFs")
    p.add_argument("--out-folder", default=OUTPUT_FOLDER_DEFAULT, help="Folder where JSON outputs will be saved")
    p.add_argument("--download-workers", type=int, default=6, help="Number of download threads")
    p.add_argument("--download-timeout", type=int, default=30, help="Timeout seconds per download")
    p.add_argument("--extract-workers", type=int, default=max(1, cpu_count()-1), help="Number of parallel extractor worker processes")
    p.add_argument("--no-ocr", action="store_true", help="Disable OCR (faster for text PDFs)")
    p.add_argument("--skip-download", action="store_true", help="Skip downloading and only run extraction on existing PDF folder")
    p.add_argument("--download-only", action="store_true", help="Only download, do not run extraction")
    p.add_argument("--log", default="info", choices=["debug","info","warning","error"], help="Logging level")
    p.add_argument("--status-path", default=None, help="Optional path to status.json to write progress")
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")
    run_pipeline(
        csv_path=args.csv,
        pdf_folder=args.pdf_folder,
        out_folder=args.out_folder,
        fulldata_folder=FULLDATA_FOLDER,
        download_workers=args.download_workers,
        download_timeout=args.download_timeout,
        extract_workers=args.extract_workers,
        no_ocr=args.no_ocr,
        skip_download=args.skip_download,
        download_only=args.download_only,
        log_level=args.log,
        status_path=args.status_path
    )

if __name__ == "__main__":
    main()