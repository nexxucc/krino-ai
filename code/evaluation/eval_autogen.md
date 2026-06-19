# Evaluation Report — Multi-Modal Evidence Review

## Scores (sample set)

- Exact accuracy: {"evidence_standard_met": 1.0, "issue_type": 0.6, "object_part": 0.7, "claim_status": 0.8, "valid_image": 0.9, "severity": 0.6}
- Macro-F1: {"claim_status": 0.7296, "issue_type": 0.639, "object_part": 0.7167, "severity": 0.32}
- Soft (Jaccard): {"risk_flags_jaccard": 0.7792, "supporting_image_ids_jaccard": 0.875}

## Operational analysis (auto-generated)

- Backend: **gemini:gemini-2.5-flash** (local, $0 — no hosted API tokens billed)
- Sample claims processed: **20**
- Images processed (decoded + re-encoded): **29**
- Approx. model calls (sample run): **20** (1 multimodal call per claim; 0 for heuristic)
- Wall-clock for this sample run: **1.9s** (~0.1s/claim on CPU)
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
- **Throttle/batch**: images capped at 3/claim and downscaled to 768px to
  bound latency and memory; temperature=0 for determinism.
