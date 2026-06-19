#!/usr/bin/env python3
"""
Strategy comparison (satisfies README "compare >= 2 strategies").

Runs THREE strategies on dataset/sample_claims.csv and prints a side-by-side
metric table:

  1. heuristic        - model-free baseline (conversation parse + rules + history)
  2. gemini-hybrid    - Gemini observation + deterministic rule overrides (primary)
  3. gemini-direct    - Gemini's own verdict, trusted as-is (minimal rules)

Strategies 2 and 3 reuse the SAME cached Gemini observations, so the comparison
costs ZERO extra API calls (only postprocess differs).

Usage:
    python code/evaluation/compare_strategies.py
"""
from __future__ import annotations

import csv
import json
import os
import sys

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

import importlib.util  # noqa: E402

import config  # noqa: E402
from postprocess import PostprocessConfig  # noqa: E402
from run import build_backend, process_claims  # noqa: E402
from vlm.heuristic_backend import HeuristicBackend  # noqa: E402

# reuse the scoring helpers from evaluation/main.py (load by path to avoid the
# name collision with code/main.py, which is also "main").
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("eval_main", os.path.join(_EVAL_DIR, "main.py"))
_eval_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_main)
evaluate = _eval_main.evaluate
load_sample = _eval_main.load_sample

REPORT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "strategy_comparison.json")


def _flat(report: dict) -> dict:
    out = {}
    for k, v in report["exact_accuracy"].items():
        out[f"acc:{k}"] = v
    out["f1:claim_status"] = report["macro_f1"]["claim_status"]
    out["f1:issue_type"] = report["macro_f1"]["issue_type"]
    out["jac:risk_flags"] = report["soft"]["risk_flags_jaccard"]
    out["jac:support_ids"] = report["soft"]["supporting_image_ids_jaccard"]
    return out


def main():
    claims, gold = load_sample(config.SAMPLE_CLAIMS_CSV)
    results = {}

    # 1. heuristic (its own model-free observations)
    print("[compare] running heuristic baseline ...", file=sys.stderr)
    pred = process_claims(claims, config.DATASET_DIR, backend=HeuristicBackend(),
                          verbose=False, pp_config=PostprocessConfig(decision_mode="hybrid"))
    results["heuristic"] = _flat(evaluate(gold, pred))

    # shared Gemini backend (observations are cached, so no extra API calls)
    gem = build_backend(verbose=False)
    # 2. gemini-hybrid (primary)
    print("[compare] running gemini-hybrid ...", file=sys.stderr)
    pred = process_claims(claims, config.DATASET_DIR, backend=gem,
                          verbose=False, pp_config=PostprocessConfig(decision_mode="hybrid"))
    results["gemini-hybrid"] = _flat(evaluate(gold, pred))

    # 3. gemini-direct (trust the model verdict)
    print("[compare] running gemini-direct ...", file=sys.stderr)
    pred = process_claims(claims, config.DATASET_DIR, backend=gem,
                          verbose=False, pp_config=PostprocessConfig(decision_mode="direct"))
    results["gemini-direct"] = _flat(evaluate(gold, pred))

    metrics = list(next(iter(results.values())).keys())
    strategies = list(results.keys())
    width = max(len(m) for m in metrics) + 2
    print("\n==== STRATEGY COMPARISON (sample_claims.csv, n=20) ====")
    header = "metric".ljust(width) + "".join(s.ljust(16) for s in strategies)
    print(header)
    print("-" * len(header))
    for m in metrics:
        row = m.ljust(width) + "".join(f"{results[s][m]:<16.4f}" for s in strategies)
        print(row)

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[compare] wrote {REPORT_JSON}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
