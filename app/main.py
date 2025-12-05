"""
main.py

FastAPI application exposing the /api/analyze endpoint.
Loads keyword CSVs at startup and serves analysis requests.

Save as:
product_matcher/backend/app/main.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os

from app.matching.datastore import KeywordStore
from app.matching.matcher import Matcher

# --------
# Config
# --------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DIAGNOSTIC_CSV = os.path.join(DATA_DIR, "keywords_diagnostic.csv")
ENDO_CSV = os.path.join(DATA_DIR, "keywords_endo.csv")

# --------
# FastAPI app
# --------
app = FastAPI(title="Product Matching API")

# Allow CORS from local dev (adjust origins for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # change to frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------
# Request model
# --------
class AnalyzeRequest(BaseModel):
    text: str
    category: Optional[str] = "all"   # "all", "diagnostic", "endo"

# --------
# Startup: load keywords and instantiate matcher
# --------
STORE = KeywordStore()
try:
    # load diagnostic and endo CSVs if present
    if os.path.exists(DIAGNOSTIC_CSV):
        STORE.load_csv(DIAGNOSTIC_CSV, category="Diagnostic")
    else:
        app.logger = getattr(app, "logger", None)
        print(f"[warning] Diagnostic CSV not found at {DIAGNOSTIC_CSV}")

    if os.path.exists(ENDO_CSV):
        STORE.load_csv(ENDO_CSV, category="Endo")
    else:
        print(f"[warning] Endo CSV not found at {ENDO_CSV}")
except Exception as e:
    # fail-fast if CSV reading errors occur
    raise RuntimeError(f"Failed to load keyword CSVs: {e}")

MATCHER = Matcher(STORE)


# --------
# Endpoints
# --------
@app.get("/api/health")
async def health():
    return {"status": "ok", "keywords_loaded": STORE.size()}


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="text must be a non-empty string")
    # normalize category param
    cat = (req.category or "all").strip().lower()
    if cat not in ("all", "diagnostic", "endo"):
        raise HTTPException(status_code=400, detail="category must be one of: all, diagnostic, endo")

    result = MATCHER.analyze(req.text, category_filter=cat)
    # safety: trim large match lists
    if "matches" in result and isinstance(result["matches"], list):
        result["matches"] = result["matches"][:200]
    return result
