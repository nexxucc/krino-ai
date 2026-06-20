# Krino

**Krino** is a multi-modal evidence-review pipeline for damage claims. Given a
short support conversation and one or more submitted photos, it inspects the
images, reconciles them with what the customer actually claimed, folds in the
account's prior history, and returns a structured verdict:

> Is the claim **supported**, **contradicted**, or is there **not enough
> information** to decide?

It works across three object types — **cars**, **laptops**, and **packages** —
and treats the images as the source of truth. A vision-language model
(`gemini-2.5-flash`) reports *what is actually visible*; a deterministic rule
layer turns that observation into the final verdict. That split keeps the
decision explainable and auditable, and it means the system is never talked into
approving a claim by instructions planted in the conversation or printed inside a
photo.

> Built for the **HackerRank Orchestrate** challenge (June 2026) — the dataset
> and the output schema come from there. Everything under [`code/`](./code/) is
> the solution.

---

## What it does

For each claim it reads the conversation, the image set, the user's history, and
a minimum-evidence checklist, and emits a structured row covering:

- **`evidence_standard_met`** — is the image set even sufficient to judge the
  claim? (plus a short reason)
- **`claim_status`** — `supported` / `contradicted` / `not_enough_information`,
  with a justification grounded in the visible evidence.
- **`object_part`, `issue_type`, `severity`** — what part is affected, what kind
  of damage, how bad.
- **`risk_flags`** — history risk, manual-review triggers, prompt-injection
  attempts, image-quality problems, claim/image mismatches.
- **`supporting_image_ids`, `valid_image`** — which photos back the decision.

The exact column order and allowed values are in
[`problem_statement.md`](./problem_statement.md).

---

## How it works

```
claims.csv ─► per claim:
   ├─ parse conversation   → claimed part + issue + injection (deterministic, multilingual)
   ├─ prepare images       → AVIF→JPEG re-encode, downscale, model-free quality signals
   ├─ select requirements  → the relevant rows of the evidence checklist
   ├─ observe (VLM)        → structured JSON: object match, part visible, damage,
   │                         severity, proposed status …   (cached + throttled + retried)
   └─ decide (rules)       → hybrid verdict + severity + risk flags, coerced to the schema
─► output.csv
```

The model **proposes**; deterministic rules **dispose**. The rule layer overrides
the model on hard conflicts (evidence not met, wrong object shown, claimed damage
absent, incompatible issue) and trusts clear visual evidence over an over-cautious
model hedge. User history only ever adds *risk* — it never flips a visual verdict.

See [`code/DESIGN.md`](./code/DESIGN.md) for every design decision with its
rationale and the alternatives that were rejected.

---

## Setup

Python 3.10+. Three dependencies (`requests`, `pillow`, `numpy`):

```bash
python -m venv venv && source venv/bin/activate
pip install -r code/requirements.txt
```

The vision model needs a Google AI Studio (Gemini) API key, read from the
environment only — never hardcoded, never committed:

```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY=...
```

---

## Run

```bash
# Generate predictions for every row of dataset/claims.csv -> output.csv
python code/main.py

# Model-free baseline (no key needed): rules + conversation parse only
EVIDENCE_BACKEND=heuristic python code/main.py
```

Full setup and options are documented in [`code/README.md`](./code/README.md).

---

## Evaluation

The pipeline ships with its own evaluation suite (metrics on the labeled sample
set, a multi-strategy comparison, and cross-validation to guard against
overfitting). All of it reads cached observations, so it re-runs in seconds with
no API calls:

```bash
python code/evaluation/main.py                # metrics on the labeled sample
python code/evaluation/compare_strategies.py  # heuristic vs hybrid vs direct
python code/evaluation/cross_validate.py      # LOOCV + k-fold config selection
```

On the labeled sample the hybrid strategy reaches **0.80** `claim_status`
accuracy (macro-F1 **0.73**) with `evidence_standard_met` at **1.00** — well
above the model-free baseline (0.65 / 0.26). The full breakdown, the strategy
comparison, and the operational analysis (model calls, tokens, cost, runtime,
rate limits) are in
[`code/evaluation/evaluation_report.md`](./code/evaluation/evaluation_report.md).

---

## Layout

```text
.
├── README.md                 # you are here
├── problem_statement.md      # task framing + full I/O schema
├── output.csv                # final predictions
├── code/                     # the solution
│   ├── main.py               # entry point: claims.csv -> output.csv
│   ├── claim_parser.py       # conversation -> claimed part + issue (multilingual)
│   ├── image_utils.py        # AVIF->JPEG, downscale, quality signals
│   ├── evidence_rules.py     # evidence-checklist selection + sufficiency gate
│   ├── postprocess.py        # hybrid decision, severity, risk flags, schema coercion
│   ├── vlm/                  # Gemini REST backend + heuristic fallback + prompts
│   ├── evaluation/           # metrics, strategy comparison, cross-validation, report
│   └── ...                   # config, data loaders, cache, schema, orchestration
└── dataset/                  # claims, history, evidence requirements, images
```

---

## Notes

- **Deterministic.** `temperature=0`, versioned prompts, stable JSON, and a
  content-hash cache make `output.csv` reproducible across runs.
- **No hardcoded answers.** The parser and rules are generic feature logic; there
  are no per-case labels.
- **Secrets via environment only.** `.env` is git-ignored; only `.env.example`
  (a placeholder) is tracked.
- **Resilient.** Any failed model call falls back to a deterministic heuristic
  for that row, so a complete `output.csv` is always produced.
