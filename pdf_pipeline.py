import os
import time
import re
import mysql.connector
import subprocess
from playwright.sync_api import sync_playwright
import requests
import sys
import shutil

# --------------------------------------------------------
# DIRECTORY STRUCTURE
# --------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_DIR = os.path.join(BASE_DIR, "PDF")
JSON_DIR = os.path.join(BASE_DIR, "OUTPUT")
EXTRACTOR_DIR = os.path.join(BASE_DIR, "extractor")
EXTRACTOR_PATH = os.path.join(EXTRACTOR_DIR, "url_pdf_extraction.py")

TEMP_PDF_DIR = "TEMP_PDF"
TEMP_OUT_DIR = "TEMP_OUT"


os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)
os.makedirs(EXTRACTOR_DIR, exist_ok=True) 


# --------------------------------------------------------
# FILENAME SAFETY
# --------------------------------------------------------

def safe_name(text):
    return re.sub(r'[^A-Za-z0-9_-]', '_', text)


# --------------------------------------------------------
# DATABASE
# --------------------------------------------------------

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "tender_automation_with_ai"
}

def db_connect():
    return mysql.connector.connect(**DB_CONFIG)

def fetch_pending_tenders(limit=50):
    conn = db_connect()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT bid_number, detail_url
        FROM gem_tenders
        WHERE bid_number NOT IN (SELECT bid_number FROM gem_tender_docs)
        LIMIT %s
    """, (limit,))

    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def save_doc_record(bid_number, detail_url, pdf_url, pdf_path, json_path):
    conn = db_connect()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO gem_tender_docs (bid_number, detail_url, pdf_url, pdf_path, json_path)
        VALUES (%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            pdf_url = VALUES(pdf_url),
            pdf_path = VALUES(pdf_path),
            json_path = VALUES(json_path)
    """, (bid_number, detail_url, pdf_url, pdf_path, json_path))

    conn.commit()
    cursor.close()
    conn.close()


# --------------------------------------------------------
# GET PDF URL
# --------------------------------------------------------

def extract_pdf_url(detail_url):

    if detail_url.lower().endswith(".pdf"):
        return detail_url

    if "showbidDocument" in detail_url:
        return detail_url

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        try:
            page.goto(detail_url, timeout=0, wait_until="domcontentloaded")
        except:
            browser.close()
            return detail_url

        links = page.query_selector_all("a")
        for link in links:
            href = link.get_attribute("href")
            if href and ".pdf" in href.lower():
                if href.startswith("/"):
                    href = "https://bidplus.gem.gov.in" + href
                browser.close()
                return href

        browser.close()
        return None


# --------------------------------------------------------
# DOWNLOAD PDF
# --------------------------------------------------------

def download_pdf(pdf_url, bid_number):
    safe_bid = safe_name(bid_number)
    pdf_path = os.path.join(PDF_DIR, f"{safe_bid}.pdf")

    # ⛔ Skip download if PDF already exists
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 1000:
        print("⚠️ PDF already exists locally — skipping download")
        return pdf_path

    r = requests.get(pdf_url, timeout=30)
    if r.status_code == 200:
        with open(pdf_path, "wb") as f:
            f.write(r.content)
        return pdf_path

    return None



# --------------------------------------------------------
# RUN EXTRACTOR FOR ONLY ONE PDF
# --------------------------------------------------------

def extract_json_from_pdf(pdf_path, bid_number):
    safe_bid = safe_name(bid_number)
    json_output_path = os.path.join(JSON_DIR, f"{safe_bid}.json")

    PY = sys.executable

    # -------------------------------------
    # 1️⃣ Skip if JSON already exists
    # -------------------------------------
    if os.path.exists(json_output_path) and os.path.getsize(json_output_path) > 500:
        print("⚠️ JSON already exists — skipping extraction")
        return json_output_path

    # -------------------------------------
    # 2️⃣ Reset temp folders
    # -------------------------------------
    if os.path.exists(TEMP_PDF_DIR):
        shutil.rmtree(TEMP_PDF_DIR)
    if os.path.exists(TEMP_OUT_DIR):
        shutil.rmtree(TEMP_OUT_DIR)

    os.makedirs(TEMP_PDF_DIR, exist_ok=True)
    os.makedirs(TEMP_OUT_DIR, exist_ok=True)

    # -------------------------------------
    # 3️⃣ Copy PDF to TEMP_PDF_DIR
    # -------------------------------------
    temp_pdf_name = os.path.basename(pdf_path)
    temp_pdf_path = os.path.join(TEMP_PDF_DIR, temp_pdf_name)
    shutil.copy(pdf_path, temp_pdf_path)

    # -------------------------------------
    # 4️⃣ Run extractor
    # -------------------------------------
    cmd = [
        PY,
        EXTRACTOR_PATH,
        "--skip-download",
        "--pdf-folder", TEMP_PDF_DIR,
        "--out-folder", TEMP_OUT_DIR,
        "--extract-workers", "1"
    ]

    subprocess.run(cmd)

    # -------------------------------------
    # 5️⃣ Extractor output may name JSON after original filename
    # e.g. GEM_2025_B_6663523.json
    # -------------------------------------

    temp_json_expected = os.path.join(TEMP_OUT_DIR, f"{safe_bid}.json")

    # Try to move expected JSON output
    if os.path.exists(temp_json_expected):
        shutil.move(temp_json_expected, json_output_path)
        return json_output_path

    # -------------------------------------
    # 6️⃣ If extractor used PDF's basename instead, handle that
    # -------------------------------------
    alt_json_name = os.path.splitext(temp_pdf_name)[0] + ".json"
    alt_json_path = os.path.join(TEMP_OUT_DIR, alt_json_name)

    if os.path.exists(alt_json_path):
        shutil.move(alt_json_path, json_output_path)
        return json_output_path

    # -------------------------------------
    # 7️⃣ No JSON produced
    # -------------------------------------
    return None


# --------------------------------------------------------
# MAIN WORKFLOW
# --------------------------------------------------------

def process_tenders():

    tenders = fetch_pending_tenders()
    if not tenders:
        print("No pending tenders.")
        return

    for row in tenders:
        bid = row["bid_number"]
        detail_url = row["detail_url"]

        print(f"\n🔍 Processing {bid}")

        pdf_url = extract_pdf_url(detail_url)
        if not pdf_url:
            print("❌ No PDF URL in tender page.")
            continue

        print("📄 PDF URL:", pdf_url)

        pdf_path = download_pdf(pdf_url, bid)
        
        if not pdf_path:
            print("❌ Failed to download PDF.")
            continue

        print("📥 PDF saved:", pdf_path)

        json_path = extract_json_from_pdf(pdf_path, bid)
        if not json_path:
            print("❌ JSON extraction failed.")
            continue

        print("📤 JSON saved:", json_path)

        save_doc_record(bid, detail_url, pdf_url, pdf_path, json_path)
        print(f"✔ DB record saved for {bid}")


# --------------------------------------------------------
# RUN LOOP
# --------------------------------------------------------

if __name__ == "__main__":
    while True:
        process_tenders()
        print("\n⏳ Waiting 2 minutes...\n")
        time.sleep(120)
