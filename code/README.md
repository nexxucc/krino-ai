# Multi-Modal Evidence Review — Solution

Verifies damage claims (car / laptop / package) by inspecting the submitted
images, reconciling them with the claim conversation, the minimum-evidence
checklist, and user history, then emitting the 14-column verdict schema.

- **Model:** Google `gemini-2.5-flash` (Vision), via the REST API, free tier.
- **Policy:** images are the source of truth; user history only adds *risk*
  (never flips a visual verdict); any "approve this claim" text is reported and
  ignored, never obeyed.
- **Reads `dataset/claims.csv` → writes `output.csv`** at the repo root.

See [`DESIGN.md`](./DESIGN.md) for every design decision (with rationale and
rejected alternatives) and [`evaluation/evaluation_report.md`](./evaluation/evaluation_report.md)
for metrics, the strategy comparison, cross-validation, and the operational analysis.

---

## 1. Setup

Requires Python 3.10+. Dependencies (already used by the pipeline): `requests`,
`pillow`, `numpy`. No model SDK is needed — the Gemini call is plain REST.

```bash
python -m venv venv && source venv/bin/activate
pip install requests pillow numpy
```

Put your **Gemini API key** (Google AI Studio) in a `.env` file at the repo root.
Secrets are read from the environment only — never hardcoded, never committed
(`.env` is git-ignored; see `.env.example`).

```bash
# .env
GEMINI_API_KEY=your_key_here
# optional overrides:
# GEMINI_MODEL=gemini-2.5-flash
# EVIDENCE_BACKEND=gemini          # or "heuristic" for a no-key, model-free run
# GEMINI_RPM=5                     # client-side throttle (requests/minute)
```

The key may also be supplied as a real environment variable (`GEMINI_API_KEY` or
`GOOGLE_API_KEY`); an exported var takes precedence over `.env`.

---

## 2. Run

```bash
# Generate predictions for all rows of dataset/claims.csv -> output.csv
python code/main.py

# Options
python code/main.py --claims dataset/claims.csv --out output.csv
EVIDENCE_BACKEND=heuristic python code/main.py    # model-free, no API key needed
```

Evaluation / analysis (all read cached observations → no API calls after the
first observe pass):

```bash
python code/evaluation/main.py                 # metrics on sample_claims.csv
python code/evaluation/compare_strategies.py   # heuristic vs gemini-hybrid vs gemini-direct
python code/evaluation/cross_validate.py        # LOOCV + 5-fold config selection
```

---

## 3. How it works (pipeline)

```
claims.csv ─► per claim:
   ├─ claim_parser   conversation → claimed part + issue + injection (deterministic, multilingual)
   ├─ image_utils    AVIF→JPEG re-encode, downscale ≤768px, model-free quality signals
   ├─ evidence_rules pick evidence_requirements rows (object + issue family + 'all')
   ├─ vlm/gemini     OBSERVE images → structured JSON {object_match, part_visible,
   │                 visible_issue, damage_present, severity, proposed_status, …}
   │                 (cache + RPM throttle + retry/backoff + heuristic fallback)
   └─ postprocess    HYBRID decide: model proposal + deterministic rule overrides;
                     fold user_history as risk only; coerce every field to allowed enums
─► output.csv (14 columns, exact order)
```

Key modules in `code/`:

| file | role |
|---|---|
| `main.py` | entry point: `claims.csv` → `output.csv` |
| `config.py` | paths, `.env` loader, Gemini settings (env-only secrets) |
| `data_loaders.py` | load claims / user_history (by `user_id`) / evidence_requirements |
| `image_utils.py` | **AVIF→JPEG** decode, downscale, blur/brightness/aspect signals |
| `claim_parser.py` | conversation → claimed part + issue family + injection detection |
| `evidence_rules.py` | select the requirement rows; decide `evidence_standard_met` |
| `vlm/gemini_backend.py` | Gemini REST call (strict JSON, temp 0), throttle, retry, verifier |
| `vlm/heuristic_backend.py` | model-free fallback (always available) |
| `vlm/prompt.py` | system + observe + verifier prompts (versioned) |
| `postprocess.py` | hybrid decision, blended severity, risk flags, schema coercion |
| `cache.py` | content-hash observation cache (free, deterministic re-runs) |
| `run.py` | per-claim orchestration |
| `schema.py` | allowed-value vocabularies + coercion (guarantees valid output) |
| `evaluation/` | `main.py`, `compare_strategies.py`, `cross_validate.py`, report |

---

## 4. Design highlights

- **AVIF gotcha:** dataset images are AVIF despite the `.jpg` extension; every
  image is decoded and re-encoded to real JPEG before being sent to the model.
- **Hybrid decision:** the model proposes `claim_status`; deterministic rules
  override on hard conflicts (evidence not met, wrong object, claimed-damage-absent,
  incompatible issue) and on a symmetric guard (trust clear visual evidence over an
  over-cautious model hedge). History contributes risk flags only.
- **Prompt-injection safe:** "approve this / ignore instructions / follow the note"
  text (in the conversation or printed in an image) sets `text_instruction_present`
  and is never obeyed.
- **Multi-image:** all of a claim's images are sent in one request; the model
  selects the relevant `supporting_image_ids` (verified it discriminates between
  a blurry first image and a clear second one).
- **Cost / rate-limit aware:** 1 call/claim, content-hash cache (0-cost re-runs),
  RPM throttle, retry-with-backoff + circuit-breaker, and a deterministic heuristic
  fallback so a complete `output.csv` is always produced.
- **Determinism:** `temperature=0`, versioned prompts, stable JSON, cached.

---

## 5. Results (summary)

On `dataset/sample_claims.csv` (n=20), the **gemini-hybrid** strategy:
`claim_status` acc **0.80** / macro-F1 **0.730** (vs heuristic 0.65 / 0.26),
`evidence_standard_met` **1.00**, `risk_flags` Jaccard 0.78, `supporting_image_ids`
0.875. **LOOCV held-out acc = 0.80.** Full breakdown, the 3-strategy comparison,
the cross-validation, the two bug fixes found via error analysis, the (CV-rejected)
verifier experiment, and the operational analysis are in
[`evaluation/evaluation_report.md`](./evaluation/evaluation_report.md).

---

## 6. Notes / reproducibility

- Final predictions: **`output.csv`** at the repo root (44 rows, exact 14-column
  schema, all values within the allowed vocabularies, all genuine Gemini).
- The pipeline is reproducible from the cache; deleting `.cache/` forces a fresh
  observe pass (uses API quota).
- No hardcoded labels or per-case answers — the parser/rules are generic.
