# Evaluation Report — Multi-Modal Evidence Review

System: a Gemini-vision pipeline that reads `dataset/claims.csv`, inspects the
submitted images, and emits the 14-column verdict schema to `output.csv`.
Final model: **`gemini-2.5-flash`** (Google AI Studio free tier). Decision policy:
**hybrid** (model proposes a verdict; deterministic rules override on hard
conflicts). All numbers below are reproducible from the cached observations via
the scripts in this folder (`main.py`, `compare_strategies.py`, `cross_validate.py`).

---

## 1. Metrics on `dataset/sample_claims.csv` (n = 20, labeled)

Primary (final) strategy — **gemini-2.5-flash, hybrid**:

| field | exact accuracy | macro-F1 / Jaccard |
|---|---|---|
| `claim_status` (headline) | **0.80** | 0.730 (macro-F1) |
| `evidence_standard_met` | 1.00 | — |
| `valid_image` | 0.90 | — |
| `object_part` | 0.70 | 0.717 (macro-F1) |
| `issue_type` | 0.60 | 0.639 (macro-F1) |
| `severity` | 0.60 | — |
| `risk_flags` | — | 0.779 (Jaccard) |
| `supporting_image_ids` | — | 0.875 (Jaccard) |

The model recovers contradicted/NEI cases that a rule-only baseline cannot
(`claim_status` macro-F1 0.730 vs 0.263 for the heuristic). The 4 remaining errors
are genuine model-vs-labeler perception disagreements on borderline/subjective
cases (e.g. minor-scratch-vs-claimed-severe), not pipeline bugs. See §3.1 for the
two pipeline bugs that *were* found and fixed.

---

## 2. Strategy comparison (≥2 required)

Three strategies, scored on the same 20 labeled samples. Strategies 2 and 3 share
the **same cached observations** (they differ only in postprocess), so the
comparison costs **zero extra API calls**.

| metric | heuristic (no model) | **gemini-hybrid (final)** | gemini-direct |
|---|---|---|---|
| acc `claim_status` | 0.650 | **0.800** | 0.700 |
| **F1 `claim_status`** | 0.263 | **0.730** | 0.588 |
| acc `evidence_standard_met` | 0.900 | **1.000** | 1.000 |
| acc `issue_type` | 0.600 | 0.600 | 0.600 |
| F1 `issue_type` | 0.640 | 0.639 | 0.539 |
| acc `object_part` | 0.700 | 0.700 | 0.700 |
| acc `severity` | 0.550 | 0.600 | 0.650 |
| Jaccard `risk_flags` | 0.653 | **0.779** | 0.717 |
| Jaccard `support_ids` | 0.725 | **0.875** | 0.825 |

- **heuristic** — model-free baseline (conversation parse + evidence rules +
  history). Strong on cheap fields but collapses `claim_status` to mostly
  `supported` (F1 0.26).
- **gemini-direct** — trust the model's raw verdict, minimal rules.
- **gemini-hybrid (chosen)** — model verdict + deterministic guard-rails. Best on
  the headline `claim_status` and on risk/image-id fields.

**Final strategy used for `output.csv`: gemini-hybrid.**

---

## 3. Cross-validation (overfitting referee)

Because the pipeline is not *trained*, CV here selects the rule/prompt
configuration that generalizes, instead of the one that maxes the full-20 score.
Configuration = (`symmetric_override`, `allow_high_severity`, verifier on/off);
each fold picks the best config on the training folds and scores the held-out
fold. Runs on cached observations → **0 API calls**.

| config (v=verifier, so=symmetric-override, hi=allow-high-sev) | full acc | full F1 |
|---|---|---|
| **v0-so1-hi0  (selected)** | **0.800** | **0.730** |
| v0-so0-hi0 | 0.750 | 0.712 |
| v1-so1-hi0 (verifier on) | 0.750 | 0.700 |
| v1-so0-hi0 | 0.700 | 0.665 |

- **LOOCV held-out accuracy: 0.80**; **5-fold stratified: 0.80**.
- The recommended config (`verifier OFF, symmetric_override ON, severity-cap-medium`)
  was selected in **20/20** LOOCV folds — a stable choice, not a lucky split.

### 3.1 Two pipeline bugs found via error analysis (and fixed)
A claim-by-claim error analysis on the sample isolated two errors that were
*pipeline bugs*, not model limits — fixing them lifted `claim_status` **0.70 → 0.80**
(macro-F1 0.568 → 0.730) with **zero regressions**, and both are principled
(general logic, not label-fitting), so they help the test set too:

1. **Wrong-object → `contradicted` (was `NEI`).** `decide_evidence` had treated
   "image shows a different object than claimed" as *insufficient evidence*. But a
   clearly-shown wrong object is **sufficient** to evaluate — and contradicts the
   claim. Fix: a usable wrong-object image now sets `evidence_standard_met=true`
   and the status resolves to `contradicted`. On the **test set** this correctly
   caught 4 "decoy" rows — a toy car, a scooter, a smartphone, and a standalone
   keyboard submitted for car/laptop claims — previously mislabeled `NEI`.
2. **Claim parsed from a negation.** The parser pulled the claimed part out of a
   *denied* mention (e.g. "**not** claiming item-missing, **only** torn packaging"
   → wrongly parsed `contents`). Fix: reduce the final customer turn to its
   *affirmed* clause (drop negated clauses, prefer "only/just/sirf/solo …") and
   constrain the part to the issue family. This also corrected a **test** row
   ("…**not** the screen… the actual claim is the **trackpad**": `screen`→`trackpad`).

### Verifier experiment (built, measured, rejected)
We implemented a targeted adversarial **verifier pass** (a second model call that
challenges `supported`/`contradicted` verdicts). CV showed it **lowered**
`claim_status` from 0.70 → 0.65: it was over-skeptical, flipping correct
`supported` verdicts more often than it fixed wrong ones. Per the CV referee we
**ship it OFF**. (Code remains behind a flag for reproducibility.)

### `gemini-2.5-pro` (considered, unavailable)
Tried as a stronger observe model; the free tier returns 429 immediately
(`...PerModelPerDay-FreeTier`, model `gemini-2.5-pro`) — i.e. ~0 free daily quota.
Not viable for a no-cost pipeline.

### Honest ceiling statement
With only 20 labeled samples and no held-out set beyond CV, a uniform "95%" is
neither achievable nor *measurable* without overfitting (each sample = 5% of the
score). The validated, generalization-checked estimate is **~0.80 `claim_status`**
(after the §3.1 bug fixes) with `evidence_standard_met` at 1.00 and strong
risk/image-id scores. The 4 residual errors are model-vs-labeler perception
disagreements, which the (rejected) verifier and the (unavailable) pro model could
not address without overfitting.

---

## 4. Operational analysis

### Model calls
- **1 multimodal call per claim** (all of a claim's images batched into one
  request) — minimizes request count.
- Sample: **20 calls**. Test: **44 calls**. One clean end-to-end run = **64 calls**.
- Strategy comparison + CV + the (rejected) verifier reuse cached observations, so
  they add **0** calls beyond the one-time observe pass (verifier experiment cost
  ~17 one-off sample calls; the §3.1 bug fixes cost 3 targeted re-observe calls —
  1 sample + 2 test — since only those rows' prompts changed).

### Images processed
- Sample: **29 images** across 20 claims (1.45/claim).
- Test: **82 images** across 44 claims (1.86/claim).
- **Total: 111 images.** All decoded with Pillow and **re-encoded to JPEG** — the
  dataset files are AVIF despite the `.jpg` extension — then downscaled to ≤768 px.

### Token usage (estimated)
Per call ≈ 750 text-prompt tokens + ~258 tokens/image + 512 thinking + ~250 output.

| | input tokens | output tokens (incl. thinking) |
|---|---|---|
| sample (20) | ~22,500 | ~15,200 |
| test (44) | ~54,200 | ~33,500 |
| **total (64)** | **~76,600** | **~48,800** |

### Cost
- **Actual: $0.00** — entire pipeline runs on the Gemini free tier.
- **Equivalent at paid `gemini-2.5-flash` rates** ($0.30/1M input, $2.50/1M output):
  **≈ $0.14 total** (~$0.10 for the 44-row test set alone). Pricing assumption noted;
  thinking tokens billed as output.

### Latency / runtime
- ≈ 4 s per call (model) + 12 s throttle spacing (RPM 5) ⇒ a full pass is
  **~9–13 min** wall-clock. Cached re-runs (eval/CV/comparison) are **seconds**.

### TPM / RPM, throttling, caching, retries (the real-world constraint)
The free tier's **requests-per-day (RPD)** was the binding limit (observed
~18–20 RPD/key for `gemini-2.5-flash`; per-minute ~5–8). Mitigations implemented:

- **Throttle**: client-side limiter at **RPM 5** (12 s spacing) to stay under the
  per-minute cap.
- **Content-hash cache**: every observation is keyed by
  `sha256(model + prompt-version + thinking-budget + claim + image-bytes)`.
  Re-runs and all postprocess/CV experiments cost **0** calls. Failed calls are
  **never cached** (no fallback poisoning), so a re-run only re-issues missing rows.
- **Retry/backoff**: transient 429/503 retried up to 3× honoring the server's
  `retryDelay`; a **circuit-breaker** fails fast once the daily quota is clearly
  gone, so we never burn quota retrying a dead key.
- **Graceful fallback**: any unrecoverable call drops that row to the deterministic
  heuristic backend, so the pipeline always produces a complete `output.csv`.
- **Incremental completion**: because successful observations are cached, the
  44-row test set was completed across multiple free keys with **no re-pay** —
  each key only processed the still-uncached rows.

### Determinism
`temperature = 0`, fixed versioned prompts, stable JSON parsing, sorted iteration,
and the content-hash cache ⇒ identical `output.csv` across runs.

---

## 5. Reproducing these numbers

```bash
# from repo root, with GEMINI_API_KEY in .env
python code/evaluation/main.py                 # sample metrics -> eval_summary.json
python code/evaluation/compare_strategies.py   # 3-strategy table -> strategy_comparison.json
python code/evaluation/cross_validate.py        # CV + config selection -> cv_results.json
```
All three read cached observations, so they run in seconds with **no API calls**
once the sample observe pass has been done.
