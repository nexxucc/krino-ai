#!/usr/bin/env python3
"""
Entry point: read dataset/claims.csv, run the evidence-review pipeline, and write
output.csv with the 14 required columns in the required order.

Usage:
    python code/main.py                      # uses dataset/claims.csv -> output.csv
    python code/main.py --claims PATH --out PATH
    EVIDENCE_BACKEND=heuristic python code/main.py   # force model-free run

No secrets required. Default backend is the local Ollama model; if it is not
available the pipeline falls back to the deterministic heuristic backend so a
valid output.csv is always produced.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

# allow running both as `python code/main.py` and `python -m code.main`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config  # noqa: E402
from data_loaders import load_claims  # noqa: E402
from run import process_claims  # noqa: E402
from schema import OUTPUT_COLUMNS  # noqa: E402


def write_output(rows, out_path: str) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in OUTPUT_COLUMNS})


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-Modal Evidence Review")
    ap.add_argument("--claims", default=config.CLAIMS_CSV, help="input claims CSV")
    ap.add_argument("--out", default=config.OUTPUT_CSV, help="output CSV path")
    ap.add_argument("--quiet", action="store_true", help="suppress progress logs")
    args = ap.parse_args()

    claims = load_claims(args.claims)
    if not args.quiet:
        print(f"[main] loaded {len(claims)} claims from {args.claims}", file=sys.stderr)

    rows = process_claims(claims, config.DATASET_DIR, verbose=not args.quiet)
    write_output(rows, args.out)
    print(f"[main] wrote {len(rows)} rows -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
