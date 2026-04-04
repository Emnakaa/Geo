"""Shared configuration for the GEO pipeline.

All secrets are read from environment variables; no hard-coded keys.
Copy .env.example → .env and fill in your keys.
"""

import os
import sys
from dotenv import load_dotenv

# Force UTF-8 stdout/stderr on Windows (avoids cp1252 UnicodeEncodeError
# when print statements contain em-dashes, arrows, or other non-ASCII chars).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

load_dotenv()  # loads .env if present, no-op otherwise

# ── API Keys ──────────────────────────────────────────────────────────────────
GROQ_API_KEY    = os.environ.get("GROQ_API_KEY", "")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
SERPAPI_KEY     = os.environ.get("SERPAPI_KEY", "")
APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN", "")

# ── Model names ───────────────────────────────────────────────────────────────
MODEL_INTENT          = "llama-3.3-70b-versatile"  # Agent 0 intent discovery
MODEL_ANALYST         = "llama-3.3-70b-versatile"  # Agent 1 heavy analysis
MODEL_ANALYST2        = "llama-3.3-70b-versatile"
MODEL_FALLBACK_HEAVY  = "qwen/qwen3-32b"

QUERY_MODELS = [
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
]

MODEL_EXTRACTOR  = "mistral-small-latest"       # Mistral extraction
MODEL_EXTRACTOR2 = "llama-3.3-70b-versatile"
MODEL_EXTRACTOR3 = "llama-3.1-8b-instant"
GROQ_MODEL       = "llama-3.3-70b-versatile"   # Agent 2 default

# ── Fallback chain (tried in order on 429 rate-limit errors) ──────────────────
# TPM (per-minute) limits: short wait then retry same model.
# TPD (per-day) limits: skip model entirely, try next in chain.
GROQ_FALLBACK_CHAIN = [
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "llama-3.1-8b-instant",
    "llama3-8b-8192",          # older alias, very high limits
]

# ── Pipeline defaults ─────────────────────────────────────────────────────────
N_RUNS     = 2
OUTPUT_DIR = "geo_output"

# ── Derived paths (created at pipeline startup, not import time) ──────────────
RAW_OUTPUT_PATH          = f"{OUTPUT_DIR}/raw_responses.csv"
EXTRACTED_ENTITIES_PATH  = f"{OUTPUT_DIR}/extracted_entities.csv"
CLEAN_ENTITIES_PATH      = f"{OUTPUT_DIR}/clean_entities.csv"
FEATURES_PATH            = f"{OUTPUT_DIR}/entity_features.csv"
FEATURES_GLOBAL_PATH     = f"{OUTPUT_DIR}/entity_features_global.csv"

# Agent 2
AGENT2_OUTPUT_DIR  = "agent2_output"
CHECKPOINT_FILE    = f"{AGENT2_OUTPUT_DIR}/web_features.csv"
VALIDATION_LOG     = f"{AGENT2_OUTPUT_DIR}/validation_log.csv"
AUDIT_LOG          = f"{AGENT2_OUTPUT_DIR}/audit_log.json"
