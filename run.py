# run_realtime.py
import asyncio
import json
import os
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import mysql.connector
import pandas as pd
from playwright.async_api import async_playwright

# ---------------------------
# CONFIG
# ---------------------------
BASE_URL = "https://bidplus.gem.gov.in"
SAVE_EVERY = 5
QUEUE_MAXSIZE = 20000
BATCH_SIZE = 500
BATCH_TIMEOUT = 5.0

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "tender_automation_with_ai",
    "autocommit": False,
}

CSV_SNAPSHOT_EVERY = 600
LOG_FILE = "realtime_scraper.log"

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger("realtime")

# ---------------------------
# AI MODEL LOADING (ML classifier)
# ---------------------------
import joblib

MODEL_FILE = "data/processed/relevancy_model.pkl"
VECT_FILE = "data/processed/vectorizer.pkl"

# load model and vectorizer (ensure these files exist)
try:
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECT_FILE)
    logger.info("Loaded ML relevancy model and vectorizer.")
except Exception:
    logger.exception("Failed to load ML model or vectorizer. Exiting.")
    raise

def clean_text(txt):
    if txt is None:
        return ""
    txt = str(txt).lower()
    txt = re.sub(r"[^a-z0-9\s/-]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def predict_relevance(text):
    clean = clean_text(text)
    vec = vectorizer.transform([clean])
    pred = int(model.predict(vec)[0])
    proba = float(model.predict_proba(vec)[0][1])
    return pred, proba

# ---------------------------
# KEYWORD MATCHING MODEL (MATCHER) - load same as FastAPI
# ---------------------------
from app.matching.datastore import KeywordStore
from app.matching.matcher import Matcher

# Build data dir relative to this file so it matches FastAPI path
BACKEND_DIR = os.path.join(os.path.dirname(__file__), "app")
DATA_DIR = os.path.join(BACKEND_DIR, "data")

DIAGNOSTIC_CSV = os.path.join(DATA_DIR, "keywords_diagnostic.csv")
ENDO_CSV = os.path.join(DATA_DIR, "keywords_endo.csv")

STORE = KeywordStore()
# Load CSVs only if present to avoid crashes; fail-fast if your environment requires them
if os.path.exists(DIAGNOSTIC_CSV):
    STORE.load_csv(DIAGNOSTIC_CSV, category="Diagnostic")
    logger.info(f"Loaded Diagnostic keywords: {DIAGNOSTIC_CSV}")
else:
    logger.warning(f"Diagnostic CSV not found at {DIAGNOSTIC_CSV}")

if os.path.exists(ENDO_CSV):
    STORE.load_csv(ENDO_CSV, category="Endo")
    logger.info(f"Loaded Endo keywords: {ENDO_CSV}")
else:
    logger.warning(f"Endo CSV not found at {ENDO_CSV}")

MATCHER = Matcher(STORE)
logger.info("Matcher initialized with loaded KeywordStore.")

# ---------------------------
# SQL (upsert) - extended to include matching columns
# ---------------------------
# Columns inserted:
# (page_no, bid_number, detail_url, items, quantity, department, start_date, end_date,
#  relevance, relevance_score, match_count, match_relevency, matches, matches_status)
UPSERT_SQL = """
INSERT INTO gem_tenders
  (page_no, bid_number, detail_url, items, quantity, department, start_date, end_date,
   relevance, relevance_score, match_count, match_relevency, matches, matches_status)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
ON DUPLICATE KEY UPDATE
  page_no = VALUES(page_no),
  detail_url = VALUES(detail_url),
  items = VALUES(items),
  quantity = VALUES(quantity),
  department = VALUES(department),
  start_date = VALUES(start_date),
  end_date = VALUES(end_date),
  relevance = VALUES(relevance),
  relevance_score = VALUES(relevance_score),
  match_count = VALUES(match_count),
  match_relevency = VALUES(match_relevency),
  matches = VALUES(matches),
  matches_status = VALUES(matches_status)
;
"""

# ---------------------------
# GLOBALS
# ---------------------------
SHUTDOWN = False

# ---------------------------
# DB FUNCTIONS
# ---------------------------
def db_connect():
    return mysql.connector.connect(**DB_CONFIG)

def db_execute_many(rows):
    """
    rows: list of tuples matching the UPSERT values order.
    """
    if not rows:
        return 0
    conn = db_connect()
    cur = conn.cursor()
    try:
        cur.executemany(UPSERT_SQL, rows)
        conn.commit()
        return len(rows)
    finally:
        cur.close()
        conn.close()

# ---------------------------
# SCRAPER UTILITIES
# ---------------------------
async def apply_sorting(page):
    logger.info("Applying sorting: Bid Start Date -> Latest")
    dropdown_btn = await page.query_selector("#currentSort")
    if dropdown_btn:
        await dropdown_btn.click()
        await asyncio.sleep(0.5)
    sort_option = await page.query_selector("#Bid-Start-Date-Latest")
    if sort_option:
        await sort_option.click()
        await asyncio.sleep(1.5)
    else:
        logger.warning("Sorting option not found.")

async def extract_total_counts(page):
    await page.goto(f"{BASE_URL}/all-bids", timeout=0, wait_until="networkidle")
    await asyncio.sleep(1.5)
    await apply_sorting(page)

    total_records = 0
    total_pages = 1

    records_el = await page.query_selector("span.pos-bottom")
    if records_el:
        txt = await records_el.inner_text()
        m = re.search(r"of\s+(\d+)\s+records", txt)
        if m:
            total_records = int(m.group(1))

    last_page_el = await page.query_selector("#light-pagination a.page-link:nth-last-child(2)")
    if last_page_el:
        t = (await last_page_el.inner_text()).strip()
        if t.isdigit():
            total_pages = int(t)

    return total_records, total_pages

async def scrape_single_page_to_rows(page, page_no):
    """
    Scrape visible cards on `page` and return a list of tuples matching UPSERT_SQL order.
    """
    for _ in range(3):
        await page.mouse.wheel(0, 2000)
        await asyncio.sleep(0.2)

    cards = await page.query_selector_all("div.card")
    rows = []

    for c in cards:
        try:
            bid_link = await c.query_selector(".block_header a.bid_no_hover")
            bid_no = (await bid_link.inner_text()).strip() if bid_link else ""
            if not bid_no:
                continue

            detail_url = BASE_URL + "/" + (await bid_link.get_attribute("href")).lstrip("/")

            item_el = await c.query_selector(".card-body .col-md-4 .row:nth-child(1) a")
            items = (await item_el.inner_text()).strip() if item_el else ""

            qty_el = await c.query_selector(".card-body .col-md-4 .row:nth-child(2)")
            quantity = (await qty_el.inner_text()).replace("Quantity:", "").strip() if qty_el else ""

            dept_el = await c.query_selector(".card-body .col-md-5 .row:nth-child(2)")
            department = (await dept_el.inner_text()).strip() if dept_el else ""

            start_el = await c.query_selector("span.start_date")
            start_date = (await start_el.inner_text()).strip() if start_el else ""

            end_el = await c.query_selector("span.end_date")
            end_date = (await end_el.inner_text()).strip() if end_el else ""

            # ---- AI relevance prediction (ML model) ----
            pred, score = predict_relevance(items)

            # ---- MATCHER keyword relevancy (same behavior as API) ----
            try:
                match_result = MATCHER.analyze(items, category_filter="all")
            except Exception:
                logger.exception("Matcher analyze failed for items; falling back to no-matches.")
                match_result = {}

            match_count = match_result.get("matched_count", len(match_result.get("matches", [])))
            match_relevency = match_result.get("score_pct", 0)  # already 0-100 in API behavior
            matches_list = match_result.get("matches", [])
            # ensure JSON serializable
            try:
                matches_json = json.dumps(matches_list, ensure_ascii=False)
            except Exception:
                # fallback: store stringified minimal info
                safe_matches = [{"phrase": m.get("phrase")} for m in matches_list if isinstance(m, dict)]
                matches_json = json.dumps(safe_matches, ensure_ascii=False)

            matches_status = "Yes" if match_count > 0 else "No"

            # Build the tuple in the same order as UPSERT_SQL VALUES
            rows.append((
                page_no,
                bid_no,
                detail_url,
                items,
                quantity,
                department,
                start_date,
                end_date,
                pred,
                score,
                match_count,
                match_relevency,
                matches_json,
                matches_status,
            ))
        except Exception:
            # isolate problem to one card and continue
            logger.exception("Error scraping a card — skipping it.")
            continue

    return rows

# ---------------------------
# SCRAPER WORKER
# ---------------------------
async def scraper_worker(queue: asyncio.Queue, interval_seconds=60):
    global SHUTDOWN
    logger.info("Scraper starting...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="chrome", headless=True,
                                          args=["--disable-blink-features=AutomationControlled"])
        context = await browser.new_context()
        page = await context.new_page()

        while not SHUTDOWN:
            try:
                total_records, total_pages = await extract_total_counts(page)
                logger.info(f"Found {total_records} records across {total_pages} pages.")

                page_no = 1
                rows = await scrape_single_page_to_rows(page, page_no)

                for r in rows:
                    try:
                        queue.put_nowait(r)
                    except asyncio.QueueFull:
                        await queue.put(r)

                while page_no < total_pages and not SHUTDOWN:
                    next_btn = await page.query_selector("#light-pagination a.next")
                    if not next_btn:
                        break
                    await next_btn.click()
                    await asyncio.sleep(1.2)

                    page_no += 1
                    rows = await scrape_single_page_to_rows(page, page_no)

                    for r in rows:
                        try:
                            queue.put_nowait(r)
                        except asyncio.QueueFull:
                            await queue.put(r)

                await page.goto(f"{BASE_URL}/all-bids", timeout=0, wait_until="networkidle")
                await asyncio.sleep(0.5)

                logger.info(f"Scraper sleeping for {interval_seconds}s.")
                for _ in range(int(interval_seconds)):
                    if SHUTDOWN:
                        break
                    await asyncio.sleep(1)

            except Exception:
                logger.exception("Scraper error — retrying in 10s.")
                await asyncio.sleep(10)

        logger.info("Scraper shutting down...")
        await browser.close()

# ---------------------------
# DB CONSUMER
# ---------------------------
async def db_consumer(queue: asyncio.Queue, executor: ThreadPoolExecutor):
    global SHUTDOWN
    logger.info("DB consumer starting...")
    buffer = []
    last_flush = time.time()

    async def flush_buffer():
        nonlocal buffer, last_flush
        if not buffer:
            last_flush = time.time()
            return
        rows_to_commit = buffer
        buffer = []

        try:
            processed = await asyncio.get_event_loop().run_in_executor(
                executor, db_execute_many, rows_to_commit
            )
            logger.info(f"DB: upserted {processed} rows.")
        except Exception:
            logger.exception("DB upsert failed; batch dropped.")

        last_flush = time.time()

    last_snapshot = time.time()
    csv_rows_for_snapshot = []

    while not (SHUTDOWN and queue.empty()):
        try:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                item = None

            if item:
                buffer.append(item)
                csv_rows_for_snapshot.append(item)
                queue.task_done()

            if len(buffer) >= BATCH_SIZE:
                await flush_buffer()

            if buffer and (time.time() - last_flush) >= BATCH_TIMEOUT:
                await flush_buffer()

            if (time.time() - last_snapshot) >= CSV_SNAPSHOT_EVERY and csv_rows_for_snapshot:
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                df = pd.DataFrame([{
                    "page_no": r[0],
                    "bid_number": r[1],
                    "detail_url": r[2],
                    "items": r[3],
                    "quantity": r[4],
                    "department": r[5],
                    "start_date": r[6],
                    "end_date": r[7],
                    "relevance": r[8],
                    "relevance_score": r[9],
                    "match_count": r[10],
                    "match_relevency": r[11],
                    "matches": r[12],
                    "matches_status": r[13],
                } for r in csv_rows_for_snapshot])

                df.to_csv(f"snapshot_{ts}.csv", index=False)
                logger.info(f"Snapshot saved: snapshot_{ts}.csv")
                csv_rows_for_snapshot = []
                last_snapshot = time.time()

        except Exception:
            logger.exception("DB consumer error.")
            await asyncio.sleep(1)

    await flush_buffer()
    logger.info("DB consumer shutting down.")

# ---------------------------
# SIGNALS
# ---------------------------
def handle_signal():
    global SHUTDOWN
    logger.info("Received stop signal — shutting down...")
    SHUTDOWN = True

# ---------------------------
# MAIN
# ---------------------------
async def main():
    queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    executor = ThreadPoolExecutor(max_workers=4)

    scraper_task = asyncio.create_task(scraper_worker(queue, interval_seconds=300))
    consumer_task = asyncio.create_task(db_consumer(queue, executor))

    await asyncio.gather(scraper_task, consumer_task)

if __name__ == "__main__":
    try:
        # handle SIGTERM gracefully if available
        for sig in ("SIGINT", "SIGTERM"):
            try:
                asyncio.get_event_loop().add_signal_handler(getattr(signal, sig), handle_signal)
            except Exception:
                # add_signal_handler may not be implemented on Windows event loop
                pass

        asyncio.run(main())
    except KeyboardInterrupt:
        handle_signal()
        time.sleep(1)
        logger.info("Shutdown requested via KeyboardInterrupt")
