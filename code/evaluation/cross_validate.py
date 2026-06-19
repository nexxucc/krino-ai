#!/usr/bin/env python3
"""
Cross-validation harness (overfitting referee).

We do not train a model, so CV here selects the RULE/PROMPT configuration that
generalizes, instead of the one that maxes the full-20 score. Configuration =
(symmetric_override, allow_high_severity, verifier on/off); decision_mode is
fixed to the comparison winner ("hybrid").

For each fold we pick the config with the best claim_status on the TRAINING
folds, then score it on the held-out fold. The aggregated held-out score is an
honest, overfitting-resistant estimate. All variant predictions are precomputed
from CACHED observations, so this whole harness makes ZERO API calls (assuming
the observe + verifier passes were already run once on the sample).

Usage:
    python code/evaluation/cross_validate.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE_DIR)

import importlib.util  # noqa: E402

import config  # noqa: E402
from postprocess import PostprocessConfig  # noqa: E402
from run import build_backend, process_claims  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "eval_main", os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py"))
_eval_main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_main)
load_sample = _eval_main.load_sample
macro_f1 = _eval_main.macro_f1

OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cv_results.json")


def _status(rows, idx):
    return [str(rows[i]["claim_status"]).strip().lower() for i in idx]


def _gold_status(gold, idx):
    return [str(gold[i]["claim_status"]).strip().lower() for i in idx]


def claim_status_score(pred, gold, idx):
    """Return (accuracy, macro_f1) over the given indices."""
    if not idx:
        return 0.0, 0.0
    gs = _gold_status(gold, idx)
    ps = _status(pred, idx)
    acc = sum(1 for g, p in zip(gs, ps) if g == p) / len(idx)
    labels = sorted(set(gs))
    f1 = macro_f1(list(zip(gs, ps)), labels)
    return acc, f1


def stratified_folds(gold, k):
    """Deterministic stratified k-fold index lists (round-robin within class)."""
    by_class = defaultdict(list)
    for i, g in enumerate(gold):
        by_class[str(g["claim_status"]).strip().lower()].append(i)
    folds = [[] for _ in range(k)]
    for _, idxs in sorted(by_class.items()):
        for j, i in enumerate(idxs):
            folds[j % k].append(i)
    return folds


def select_and_eval(variants, gold, folds):
    """Per fold: pick the config best on the train indices, score on held-out."""
    n = len(gold)
    all_idx = set(range(n))
    held_correct = 0
    chosen = []
    per_fold = []
    for f in folds:
        test_idx = f
        train_idx = sorted(all_idx - set(test_idx))
        # choose config by train claim_status (acc, then f1)
        best_name, best_key = None, (-1, -1)
        for name, pred in variants.items():
            acc, f1 = claim_status_score(pred, gold, train_idx)
            if (acc, f1) > best_key:
                best_key, best_name = (acc, f1), name
        # score chosen config on held-out fold
        gs = _gold_status(gold, test_idx)
        ps = _status(variants[best_name], test_idx)
        c = sum(1 for g, p in zip(gs, ps) if g == p)
        held_correct += c
        chosen.append(best_name)
        per_fold.append({"test_n": len(test_idx), "chosen": best_name,
                         "held_acc": round(c / len(test_idx), 3)})
    return held_correct / n, chosen, per_fold


def main():
    claims, gold = load_sample(config.SAMPLE_CLAIMS_CSV)
    n = len(claims)
    gem = build_backend(verbose=False)

    # Precompute predictions for every config variant (cached observations).
    variants = {}
    for verifier in (False, True):
        for so in (True, False):
            for hi in (False, True):
                cfg = PostprocessConfig(symmetric_override=so, allow_high_severity=hi)
                name = f"v{int(verifier)}-{cfg.tag()}"
                variants[name] = process_claims(
                    claims, config.DATASET_DIR, backend=gem, verbose=False,
                    pp_config=cfg, verifier=verifier)

    # Full-data claim_status per variant (for reference / report)
    full = {name: claim_status_score(pred, gold, list(range(n)))
            for name, pred in variants.items()}

    # LOOCV and 5-fold stratified selection
    loo_folds = [[i] for i in range(n)]
    k5_folds = stratified_folds(gold, 5)
    loo_acc, loo_chosen, _ = select_and_eval(variants, gold, loo_folds)
    k5_acc, k5_chosen, k5_detail = select_and_eval(variants, gold, k5_folds)

    # Most-frequently chosen config across LOOCV folds = recommended config
    freq = defaultdict(int)
    for c in loo_chosen:
        freq[c] += 1
    recommended = max(freq.items(), key=lambda kv: (kv[1], full[kv[0]]))[0]

    print("\n==== CROSS-VALIDATION (claim_status, n=20) ====")
    print(f"{'config':<22}{'full_acc':<10}{'full_f1':<10}")
    print("-" * 42)
    for name in sorted(variants, key=lambda x: full[x], reverse=True):
        a, f = full[name]
        print(f"{name:<22}{a:<10.3f}{f:<10.3f}")
    print(f"\nLOOCV held-out acc (per-fold config selection): {loo_acc:.3f}")
    print(f"5-fold stratified held-out acc:                 {k5_acc:.3f}")
    print(f"Most-selected config across LOOCV folds: {recommended} "
          f"(chosen {freq[recommended]}/{n})")
    print(f"  -> full-data: acc={full[recommended][0]:.3f}, f1={full[recommended][1]:.3f}")

    results = {
        "n": n,
        "full_data": {k: {"acc": round(v[0], 4), "f1": round(v[1], 4)} for k, v in full.items()},
        "loocv_heldout_acc": round(loo_acc, 4),
        "fivefold_heldout_acc": round(k5_acc, 4),
        "recommended_config": recommended,
        "loocv_selection_freq": dict(freq),
        "fivefold_detail": k5_detail,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[cv] wrote {OUT_JSON}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
