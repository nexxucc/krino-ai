#!/usr/bin/env python3
"""
Evaluation entry point.

Runs the pipeline on dataset/sample_claims.csv (which carries ground-truth
labels), scores predictions against those labels, prints a report, and writes a
machine-readable summary plus refreshes the operational analysis section of
evaluation/evaluation_report.md.

Usage:
    python code/evaluation/main.py
    EVIDENCE_BACKEND=heuristic python code/evaluation/main.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict

# make the sibling modules in code/ importable
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

import config  # noqa: E402
from data_loaders import Claim  # noqa: E402
from run import process_claims  # noqa: E402

LABEL_COLS = [
    "evidence_standard_met", "risk_flags", "issue_type", "object_part",
    "claim_status", "supporting_image_ids", "valid_image", "severity",
]
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
# Auto-generated metrics snapshot. The curated, human-authored report lives in
# evaluation_report.md and is NOT overwritten by this script.
REPORT_PATH = os.path.join(EVAL_DIR, "eval_autogen.md")
SUMMARY_PATH = os.path.join(EVAL_DIR, "eval_summary.json")


def load_sample(path):
    claims, gold = [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            claims.append(Claim(
                user_id=r["user_id"], image_paths=r["image_paths"],
                user_claim=r["user_claim"], claim_object=r["claim_object"].strip().lower(),
            ))
            gold.append(r)
    return claims, gold


def _set(s):
    return {x.strip() for x in str(s).split(";") if x.strip() and x.strip().lower() != "none"}


def jaccard(a, b):
    sa, sb = _set(a), _set(b)
    if not sa and not sb:
        return 1.0
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def macro_f1(pairs, labels):
    """pairs: list of (gold, pred). Returns macro-F1 over the given labels."""
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for g, p in pairs:
        for lab in labels:
            if p == lab and g == lab:
                tp[lab] += 1
            elif p == lab and g != lab:
                fp[lab] += 1
            elif p != lab and g == lab:
                fn[lab] += 1
    f1s = []
    for lab in labels:
        denom = 2 * tp[lab] + fp[lab] + fn[lab]
        f1s.append((2 * tp[lab] / denom) if denom else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def evaluate(gold, pred):
    n = len(gold)
    report = {"n": n, "exact_accuracy": {}, "macro_f1": {}, "soft": {}}

    for col in ["evidence_standard_met", "issue_type", "object_part",
                "claim_status", "valid_image", "severity"]:
        correct = sum(1 for g, p in zip(gold, pred)
                      if str(g[col]).strip().lower() == str(p[col]).strip().lower())
        report["exact_accuracy"][col] = round(correct / n, 4)

    for col in ["claim_status", "issue_type", "object_part", "severity"]:
        labels = sorted({str(g[col]).strip().lower() for g in gold})
        pairs = [(str(g[col]).strip().lower(), str(p[col]).strip().lower())
                 for g, p in zip(gold, pred)]
        report["macro_f1"][col] = round(macro_f1(pairs, labels), 4)

    report["soft"]["risk_flags_jaccard"] = round(
        sum(jaccard(g["risk_flags"], p["risk_flags"]) for g, p in zip(gold, pred)) / n, 4)
    report["soft"]["supporting_image_ids_jaccard"] = round(
        sum(jaccard(g["supporting_image_ids"], p["supporting_image_ids"])
            for g, p in zip(gold, pred)) / n, 4)

    # confusion for the headline field
    conf = defaultdict(lambda: defaultdict(int))
    for g, p in zip(gold, pred):
        conf[str(g["claim_status"]).strip().lower()][str(p["claim_status"]).strip().lower()] += 1
    report["claim_status_confusion"] = {k: dict(v) for k, v in conf.items()}
    return report


def operational_analysis(n_claims, n_images, elapsed, backend_name):
    calls = 0 if backend_name == "heuristic" else n_claims
    return f"""## Operational analysis (auto-generated)

- Backend: **{backend_name}** (local, $0 — no hosted API tokens billed)
- Sample claims processed: **{n_claims}**
- Images processed (decoded + re-encoded): **{n_images}**
- Approx. model calls (sample run): **{calls}** (1 multimodal call per claim; 0 for heuristic)
- Wall-clock for this sample run: **{elapsed:.1f}s** (~{elapsed / max(1, n_claims):.1f}s/claim on CPU)
- Test set has 44 claims / 82 images -> ~**44 model calls** for a full test run.

### Token / cost estimate
- Local Ollama `llama3.2-vision:11b`: **no per-token cost**; only local compute/electricity.
- Rough token shape per call: ~600-900 input tokens (prompt + requirement text) +
  image tokens, ~150-300 output tokens (strict JSON).
- If swapped to a hosted multimodal API at ~$3/1M input + $15/1M output:
  44 calls x (~1.2k in + ~0.25k out) -> ~53k in + ~11k out -> **well under $0.50** for the full test set.

### Throughput / rate limits / reliability
- Local model: throughput is CPU-bound, not rate-limited (no TPM/RPM caps).
- **Caching**: observations are cached by content hash of (backend + claim + image bytes),
  so re-runs are free and deterministic.
- **Retry/fallback**: any model or JSON-parse failure falls back to the deterministic
  heuristic backend per-row, so a complete output.csv is always produced.
- **Throttle/batch**: images capped at {config.MAX_IMAGES_PER_CLAIM}/claim and downscaled to 768px to
  bound latency and memory; temperature=0 for determinism.
"""


def main():
    claims, gold = load_sample(config.SAMPLE_CLAIMS_CSV)
    print(f"[eval] {len(claims)} sample claims", file=sys.stderr)
    t0 = time.time()
    pred = process_claims(claims, config.DATASET_DIR, verbose=True)
    elapsed = time.time() - t0

    report = evaluate(gold, pred)
    n_images = sum(len(c.image_paths.split(";")) for c in claims)
    backend_name = "heuristic" if config.BACKEND == "heuristic" else f"gemini:{config.GEMINI_MODEL}"

    # console summary
    print("\n==== EVALUATION (sample_claims.csv) ====")
    print(json.dumps(report, indent=2))

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # refresh operational analysis section of the report
    op = operational_analysis(len(claims), n_images, elapsed, backend_name)
    header = "# Evaluation Report — Multi-Modal Evidence Review\n\n"
    scores_md = (
        "## Scores (sample set)\n\n"
        f"- Exact accuracy: {json.dumps(report['exact_accuracy'])}\n"
        f"- Macro-F1: {json.dumps(report['macro_f1'])}\n"
        f"- Soft (Jaccard): {json.dumps(report['soft'])}\n\n"
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(header + scores_md + op)
    print(f"\n[eval] wrote {REPORT_PATH} and {SUMMARY_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
