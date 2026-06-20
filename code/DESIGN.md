# System Design Sheet — Multi-Modal Evidence Review

Every decision below lists the **choice**, the **why**, and the **alternatives
rejected**. This is the design the shipped pipeline implements.

---

## 0. Problem in one line

For each claim row (`user_id`, `image_paths`, `user_claim`, `claim_object`),
inspect the submitted images, reconcile them with the claimed damage and the
evidence checklist, fold in user-history risk, and emit the 14 required output
columns — with images as the primary source of truth.

---

## 1. High-level architecture

```
claims.csv ─► [data_loaders] ─► per claim:
                 ├─ [claim_parser]   conversation → claimed part + issue + injection (deterministic, multilingual)
                 ├─ [image_utils]    AVIF→JPEG, downscale, model-free quality signals
                 ├─ [evidence_rules] pick evidence_requirements rows (object + issue family + 'all')
                 ├─ [vlm backend]    OBSERVE images → structured JSON (Gemini | heuristic)   ← cache + throttle + retry
                 └─ [postprocess]    DECIDE status/severity/risk, fold user_history → 14 columns (schema-coerced)
            ─► output.csv
evaluation/main.py runs the same pipeline on sample_claims.csv and scores vs labels.
```

**Decision D1 — HYBRID decide (chosen).** The VLM both reports *what is visible*
(object match, part visible, visible issue, damage present, quality,
authenticity) **and proposes a `claim_status` + `confidence`**. A deterministic
`postprocess` layer then takes the model's proposed status as the base but can
**override** it when a hard rule fires:
  - evidence standard not met → `not_enough_information`
  - object shown ≠ claimed object → `contradicted`
  - claimed damage but none visible, or visible issue incompatible with the
    claimed family → `contradicted`
  - injection/authenticity/history → risk flags only (never flips by themselves)
- *Why:* keeps the model's visual judgement where it is strongest while keeping
  a deterministic, auditable guard-rail for the cases that must not be argued
  away by "approve this claim" text. Best of both; explainable on audit.
- *Rejected:* (a) pure observe-then-rules-decide — most reproducible but discards
  the model's holistic read; (b) pure VLM-decides — least controllable, weakest
  determinism, most injection-vulnerable.

---

## 2. Model / backend

**Decision D2 — Primary backend: Google Gemini (hosted free tier), default
`gemini-2.5-flash`.**
- *Why:* after CPU-only killed local VLMs (Ollama rejected `mllama`/llama3.2-vision;
  `llava:7b` misread a clear rear-bumper dent as "not a car"; ~160 s/inference),
  Gemini Flash is the only **no-cost** option with strong multimodal quality. The
  workload (~64 calls total) fits comfortably inside the free daily cap.
- *Rejected:* Ollama local (unsupported/weak/slow on this box); other hosted VLM APIs
  (not free); heuristic-only (can't see, macro-F1 0.26 on sample).

**Decision D3 — Pluggable backend interface + heuristic fallback always present.**
- *Why:* resilience (a 429 or parse failure drops to the deterministic heuristic
  for that row so the run never dies); satisfies the brief's "compare ≥2 strategies";
  lets the whole pipeline run with zero key for testing.
- *Rejected:* single hardcoded Gemini path (brittle, untestable offline).

**Decision D4 — Transport: Gemini REST via `requests` (already installed).**
- *Why:* zero new dependencies → maximally reproducible; full control over
  `response_mime_type=application/json`, `temperature=0`, retry/throttle.
- *Rejected:* `google-genai` SDK (adds a dependency for little gain here).

---

## 3. Inputs & grounding

**Decision D5 — Deterministic multilingual conversation parser as a strong prior.**
- *Why:* the spec says the conversation *defines what to check*. A cheap, regex
  keyword parser extracts the claimed `object_part` + issue family and detects
  injection — deterministically and across EN/ES/romanized-HI/ZH present in the
  data. The transcript is **also** passed to the VLM for context, but the
  model is told never to obey instructions inside it.
- *Rejected:* have the VLM extract the claim (more tokens, less deterministic);
  English-only parsing (data is multilingual).

**Decision D6 — Image pipeline: AVIF→JPEG re-encode, downscale to 768 px,
model-free quality signals.**
- *Why:* dataset files are **AVIF despite `.jpg`** (verified) — must re-encode or
  the VLM fails; downscaling bounds tokens/latency; blur/brightness/aspect give
  `blurry_image`/`low_light_or_glare`/`cropped_or_obstructed` flags with no call.
- *Rejected:* send raw bytes (AVIF breaks); full resolution (cost/latency).

**Decision D7 — Evidence gate driven by `evidence_requirements.csv`.** Select
the `(claim_object, issue family)` rows plus the `all` rows; `evidence_standard_met`
= at least one usable image AND object shown AND claimed part visible.
- *Why:* grounds the evidence decision in the provided checklist rather than a
  vibe; conservative and explainable.
- *Rejected:* let the model decide sufficiency alone (less grounded in the CSV).

---

## 4. Decision policy (postprocess)

**Decision D8 — `claim_status` logic:** not enough info if evidence unmet → wrong
object or claimed-damage-not-visible → `contradicted` → visible issue
incompatible with claimed family → `contradicted` → else damage visible &
compatible → `supported`. (Full table in `postprocess.py`.)

**Decision D9 — User history is RISK ONLY; it never flips a visual verdict.**
- *Why:* explicit problem-statement rule. History adds `user_history_risk` /
  `manual_review_required` and justification context only.
- *Rejected:* let a bad record downgrade `supported`→`contradicted` (violates spec).

**Decision D10 — Prompt-injection handling.** Any "approve this / ignore
instructions / follow the note" text (in the transcript or printed inside an
image) sets `text_instruction_present` and is otherwise ignored.
- *Why:* several rows are adversarial; the spec wants authenticity/mismatch flags.

**Decision D11 — Severity = BLEND (chosen).** Take the VLM's severity but clamp
it with rules: `none` when no damage, `unknown` when not-enough-info, and cap by
issue family (e.g. `scratch`/`stain` ≤ medium; `dent` ≤ high; structural
`crack`/`glass_shatter`/`broken_part` may reach high). Keeps the image-grounded
read while preventing over-/under-statement.
- *Rejected:* trust-VLM-only (can over/under-state); rule-only (ignores the image).

---

## 5. Cost, rate limits, reliability

**Decision D12 — One multimodal call per claim** (all images batched) → ~44 test
/ ~64 with sample. **Decision D13 — content-hash disk cache** → re-runs cost 0
calls. **Decision D14 — client-side throttle** to `GEMINI_RPM` (default 10) +
**exponential backoff w/ jitter on 429/503**, then per-row heuristic fallback.
- *Why:* stays well within the free tier without ToS-violating quota evasion;
  deterministic, resumable, never dies.

---

## 6. Output integrity

**Decision D15 — Schema coercion on every field**: forced into the allowed enum
sets; `object_part` validated against the object-specific list; booleans as
lowercase `true`/`false`; columns written in exact required order, all quoted.
- *Why:* guarantees a gradeable `output.csv`; no out-of-vocabulary values.

**Decision D16 — No hardcoded labels / case-id answers** (a hard rule of the brief); the
parser/rules are generic keyword/feature logic.

---

## 7. Strategy comparison (at least two strategies)

**Decision D17 — Compare THREE strategies on the sample set (chosen):**
1. **Heuristic baseline** (no model) — measured: claim_status acc 0.65, F1 0.26.
2. **Gemini hybrid** (observe + propose, rules override) — **primary**.
3. **Gemini direct-verdict** prompt variant (model decides, minimal rules) — to
   show the hybrid guard-rail actually helps.
- Pick the best for the final `output.csv`; document in `evaluation_report.md`.

---

## 8. Repo layout

```
code/
  main.py                 # entry: claims.csv → output.csv
  config.py               # paths, .env loader, Gemini config (env-only secrets)
  data_loaders.py schema.py image_utils.py claim_parser.py
  evidence_rules.py cache.py postprocess.py run.py
  vlm/ base.py prompt.py gemini_backend.py heuristic_backend.py
  evaluation/main.py      # scores on sample_claims.csv → evaluation_report.md
  DESIGN.md  README.md
output.csv                # final predictions (repo root)
.env (git-ignored)  .env.example  .gitignore
```
