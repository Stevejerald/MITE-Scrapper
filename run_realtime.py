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
SAVE_EVERY = 5  # still used if you want CSV batches (optional)
QUEUE_MAXSIZE = 20000  # capacity of in-memory queue
BATCH_SIZE = 500       # bulk upsert batch for DB
BATCH_TIMEOUT = 5.0    # seconds: flush if no new rows in this time
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "tender_automation_with_ai",
    "autocommit": False,
}
CSV_SNAPSHOT_EVERY = 600  # seconds to create periodic CSV snapshot (optional)
LOG_FILE = "realtime_scraper.log"

# ---------------------------
# LOGGING (simple)
# ---------------------------
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
# SQL (upsert)
# ---------------------------
UPSERT_SQL = """
INSERT INTO gem_tenders
  (page_no, bid_number, detail_url, items, quantity, department, start_date, end_date)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
ON DUPLICATE KEY UPDATE
  page_no = VALUES(page_no),
  detail_url = VALUES(detail_url),
  items = VALUES(items),
  quantity = VALUES(quantity),
  department = VALUES(department),
  start_date = VALUES(start_date),
  end_date = VALUES(end_date)
;
"""

# ---------------------------
# GLOBALS
# ---------------------------
SHUTDOWN = False


# ---------------------------
# DB EXECUTOR FUNCTIONS (blocking; run in threadpool)
# ---------------------------
def db_connect():
    return mysql.connector.connect(**DB_CONFIG)


def db_execute_many(rows):
    """Blocking: execute many upserts in one executemany call."""
    if not rows:
        return 0
    conn = db_connect()
    cur = conn.cursor()
    try:
        cur.executemany(UPSERT_SQL, rows)
        conn.commit()
        processed = len(rows)
        return processed
    finally:
        cur.close()
        conn.close()


# ---------------------------
# Playwright scraper utilities
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
        logger.warning("Sorting option not found; continuing.")


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
    """Scrape visible cards and return list of param-tuples for DB upsert."""
    # Scroll to load
    for _ in range(3):
        await page.mouse.wheel(0, 2000)
        await asyncio.sleep(0.2)

    cards = await page.query_selector_all("div.card")
    rows = []
    for c in cards:
        bid_link = await c.query_selector(".block_header a.bid_no_hover")
        bid_no = (await bid_link.inner_text()).strip() if bid_link else ""
        if not bid_no:
            continue
        detail_url = BASE_URL + "/" + (await bid_link.get_attribute("href")).lstrip("/") if bid_link else ""
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
        rows.append((
            page_no,
            bid_no,
            detail_url,
            items,
            quantity,
            department,
            start_date,
            end_date,
        ))
    return rows


# ---------------------------
# Scraper coroutine
# ---------------------------
async def scraper_worker(queue: asyncio.Queue, interval_seconds=60):
    """
    Continuously scrape pages and push row tuples into the queue.
    interval_seconds: how long to wait after a full pass before starting again.
    """
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
                # Start from page 1 and walk through; you can optimize to only scrape recent pages
                page_no = 1
                # First page is already loaded
                rows = await scrape_single_page_to_rows(page, page_no)
                for r in rows:
                    # await queue.put(r)  # will block if full
                    try:
                        queue.put_nowait(r)
                    except asyncio.QueueFull:
                        # Backpressure: prefer dropping or waiting; here we wait a bit then put
                        logger.warning("Queue full — waiting to enqueue")
                        await queue.put(r)

                # Crawl subsequent pages
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
                            logger.warning("Queue full — waiting to enqueue")
                            await queue.put(r)

                # Optionally return to first page for next run
                await page.goto(f"{BASE_URL}/all-bids", timeout=0, wait_until="networkidle")
                await asyncio.sleep(0.5)

                # Wait interval before next full pass
                logger.info(f"Scraper sleeping for {interval_seconds}s before next pass.")
                for _ in range(int(interval_seconds)):
                    if SHUTDOWN:
                        break
                    await asyncio.sleep(1)

            except Exception as e:
                logger.exception("Scraper encountered an exception; sleeping 10s then retrying.")
                await asyncio.sleep(10)

        logger.info("Scraper shutting down...")
        await browser.close()


# ---------------------------
# Consumer / DB upsert coroutine
# ---------------------------
async def db_consumer(queue: asyncio.Queue, executor: ThreadPoolExecutor):
    """
    Consume rows from queue and batch upsert into DB using blocking db_execute_many run in executor.
    """
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
        # run blocking DB call in executor
        try:
            processed = await asyncio.get_event_loop().run_in_executor(executor, db_execute_many, rows_to_commit)
            logger.info(f"DB: upserted {processed} rows.")
        except Exception:
            logger.exception("DB upsert failed — will drop batch to avoid reprocessing.")
        last_flush = time.time()

    # Periodic CSV snapshot optional (simple)
    last_snapshot = time.time()
    csv_rows_for_snapshot = []

    while not (SHUTDOWN and queue.empty()):
        try:
            # Wait up to small timeout for a row
            try:
                item = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                item = None

            if item is not None:
                buffer.append(item)
                csv_rows_for_snapshot.append(item)
                queue.task_done()

            # Flush by size
            if len(buffer) >= BATCH_SIZE:
                logger.info("Buffer reached BATCH_SIZE — flushing to DB.")
                await flush_buffer()

            # Flush by timeout
            if buffer and (time.time() - last_flush) >= BATCH_TIMEOUT:
                logger.info("BATCH_TIMEOUT reached — flushing to DB.")
                await flush_buffer()

            # Periodic CSV snapshot
            if (time.time() - last_snapshot) >= CSV_SNAPSHOT_EVERY and csv_rows_for_snapshot:
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                csv_fn = f"snapshot_{ts}.csv"
                df = pd.DataFrame([{
                    "page_no": r[0],
                    "bid_number": r[1],
                    "detail_url": r[2],
                    "items": r[3],
                    "quantity": r[4],
                    "department": r[5],
                    "start_date": r[6],
                    "end_date": r[7],
                } for r in csv_rows_for_snapshot])
                df.to_csv(csv_fn, index=False)
                logger.info(f"Snapshot written: {csv_fn} ({len(csv_rows_for_snapshot)} rows)")
                csv_rows_for_snapshot = []
                last_snapshot = time.time()

        except Exception:
            logger.exception("Error in DB consumer loop — continuing.")
            await asyncio.sleep(1)

    # final flush on shutdown
    logger.info("DB consumer final flush before shutdown.")
    await flush_buffer()
    logger.info("DB consumer shutting down.")


# ---------------------------
# SIGNALS & RUNNER
# ---------------------------
def handle_signal():
    global SHUTDOWN
    logger.info("Received stop signal — shutting down gracefully...")
    SHUTDOWN = True


async def main():
    queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    executor = ThreadPoolExecutor(max_workers=4)

    # Run scraper and consumer concurrently
    # Scraper runs full passes every interval (set in scraper_worker arguments)
    scraper_task = asyncio.create_task(scraper_worker(queue, interval_seconds=300))  # run full pass every 300s (5min)
    consumer_task = asyncio.create_task(db_consumer(queue, executor))

    # Wait until both tasks finish (they will run until SHUTDOWN)
    await asyncio.gather(scraper_task, consumer_task)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Windows CTRL+C handling
        handle_signal()
        time.sleep(1)
        logger.info("Shutdown requested via KeyboardInterrupt")
