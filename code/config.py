"""
Central configuration. All paths resolve relative to the repo root so the code
runs the same from any working directory. No secrets are stored here; any
credentials are read from environment variables only (none are required for the
default no-cost local backend).
"""
from __future__ import annotations

import os

# repo_root/code/config.py -> repo_root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(REPO_ROOT, "dataset")


def _load_dotenv(path: str) -> None:
    """Minimal, dependency-free .env loader.

    Reads KEY=VALUE lines from repo_root/.env and sets them in os.environ
    *without* overriding variables already present in the real environment
    (so an exported var still wins). Secrets therefore live only in .env,
    never in code. Quotes and inline blank lines/comments are handled.
    """
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except OSError:
        pass


_load_dotenv(os.path.join(REPO_ROOT, ".env"))

CLAIMS_CSV = os.path.join(DATASET_DIR, "claims.csv")
SAMPLE_CLAIMS_CSV = os.path.join(DATASET_DIR, "sample_claims.csv")
USER_HISTORY_CSV = os.path.join(DATASET_DIR, "user_history.csv")
EVIDENCE_REQ_CSV = os.path.join(DATASET_DIR, "evidence_requirements.csv")

OUTPUT_CSV = os.path.join(REPO_ROOT, "output.csv")
CACHE_DIR = os.path.join(REPO_ROOT, ".cache")

# Backend selection: "gemini" (default, hosted free tier) or "heuristic"
# (no model). If the gemini backend is requested but no key/connectivity is
# available, run.py falls back to heuristic so a valid output.csv is always
# produced.
BACKEND = os.environ.get("EVIDENCE_BACKEND", "gemini").strip().lower()

# --- Google Gemini (read from env / .env only; never hardcode) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_ENDPOINT = os.environ.get(
    "GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com/v1beta"
)
GEMINI_TIMEOUT = int(os.environ.get("GEMINI_TIMEOUT", "120"))
# Client-side throttle to stay safely UNDER the free-tier RPM (requests/minute).
# Observed 429s at 8 RPM on the free tier, so default to 5 (12s spacing).
GEMINI_RPM = int(os.environ.get("GEMINI_RPM", "5"))
# Few retries, and 429-quota fails fast (see gemini_backend) so we never burn
# the daily request quota by retrying an exhausted key.
GEMINI_MAX_RETRIES = int(os.environ.get("GEMINI_MAX_RETRIES", "3"))
# Thinking budget (TOKENS, not requests — quota-safe). A small budget helps the
# model reason about subtle claim-vs-image contradictions. maxOutputTokens must
# exceed thinking + JSON so the response is never truncated.
GEMINI_THINKING_BUDGET = int(os.environ.get("GEMINI_THINKING_BUDGET", "512"))
GEMINI_MAX_OUTPUT_TOKENS = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "2048"))

# Determinism / cost controls
TEMPERATURE = 0.0
USE_CACHE = os.environ.get("EVIDENCE_USE_CACHE", "1") != "0"
MAX_IMAGES_PER_CLAIM = int(os.environ.get("EVIDENCE_MAX_IMAGES", "3"))
