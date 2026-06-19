"""
Per-claim orchestration shared by main.py (test) and evaluation/main.py (sample).

Flow per row:
  parse conversation -> load+normalize images -> select evidence rules ->
  VLM observe (cache-backed, with heuristic fallback) -> evidence decision ->
  postprocess into the 14-column row.
"""
from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional

import config
from cache import ObservationCache, make_key
from claim_parser import parse_claim
from data_loaders import (
    UserHistory,
    load_evidence_requirements,
    load_user_history,
)
from evidence_rules import decide_evidence, requirements_text, select_requirements
from image_utils import load_image_set
from postprocess import build_output_row
from vlm.base import VLMBackend, VLMObservation
from vlm.heuristic_backend import HeuristicBackend


def build_backend(verbose: bool = True) -> VLMBackend:
    """Pick the configured backend, falling back to heuristic if unavailable."""
    if config.BACKEND == "heuristic":
        if verbose:
            print("[backend] heuristic (model-free) selected", file=sys.stderr)
        return HeuristicBackend()
    # default: gemini (hosted free tier)
    try:
        from vlm.gemini_backend import GeminiBackend

        be = GeminiBackend(
            api_key=config.GEMINI_API_KEY,
            model=config.GEMINI_MODEL,
            endpoint=config.GEMINI_ENDPOINT,
            timeout=config.GEMINI_TIMEOUT,
            temperature=config.TEMPERATURE,
            rpm=config.GEMINI_RPM,
            max_retries=config.GEMINI_MAX_RETRIES,
            max_images=config.MAX_IMAGES_PER_CLAIM,
            thinking_budget=config.GEMINI_THINKING_BUDGET,
            max_output_tokens=config.GEMINI_MAX_OUTPUT_TOKENS,
        )
        if be.available():
            if verbose:
                print(f"[backend] {be.name} ready", file=sys.stderr)
            return be
        if verbose:
            print(
                "[backend] GEMINI_API_KEY not set; falling back to heuristic",
                file=sys.stderr,
            )
    except Exception as exc:  # import or init error
        if verbose:
            print(f"[backend] gemini init failed ({exc}); using heuristic", file=sys.stderr)
    return HeuristicBackend()


def _observe(
    backend: VLMBackend,
    fallback: HeuristicBackend,
    cache: ObservationCache,
    image_set,
    parsed,
    claim_object,
    user_claim,
    req_text,
) -> VLMObservation:
    key = make_key(backend.name, claim_object, user_claim, image_set)
    cached = cache.get(key)
    if cached is not None:
        return cached
    try:
        obs = backend.analyze(image_set, parsed, claim_object, user_claim, req_text)
        # Only cache a genuine result from the primary backend. A transient
        # failure (e.g. a 429) must NOT poison the cache with fallback output,
        # or later runs would serve heuristic results under the model's key.
        cache.put(key, obs)
        return obs
    except Exception as exc:  # any backend failure -> heuristic, keep pipeline alive
        print(f"  ! backend error ({exc}); using heuristic for this row", file=sys.stderr)
        return fallback.analyze(image_set, parsed, claim_object, user_claim, req_text)


def _verify(backend, cache, image_set, parsed, claim_object, user_claim, obs):
    """Run the adversarial challenge pass on a verdict, with its own cache entry
    so a challenge is never re-paid. Targeted: only supported/contradicted
    verdicts (a clear NEI gains nothing from a challenge). No-op if the backend
    has no challenge() (e.g. heuristic)."""
    if not hasattr(backend, "challenge"):
        return obs
    if obs.proposed_status not in ("supported", "contradicted"):
        return obs
    vkey = make_key(backend.name + ":verify", claim_object, user_claim, image_set)
    cached = cache.get(vkey)
    if cached is not None:
        return cached
    try:
        vobs = backend.challenge(image_set, parsed, claim_object, user_claim, obs)
        cache.put(vkey, vobs)
        return vobs
    except Exception as exc:
        print(f"  ! verifier error ({exc}); keeping first verdict", file=sys.stderr)
        return obs


def process_claims(
    claims,
    dataset_dir: str,
    backend: Optional[VLMBackend] = None,
    verbose: bool = True,
    pp_config=None,
    verifier: bool = False,
) -> List[Dict[str, str]]:
    from postprocess import PostprocessConfig
    pp_config = pp_config or PostprocessConfig()
    backend = backend or build_backend(verbose=verbose)
    fallback = HeuristicBackend()
    cache = ObservationCache(config.CACHE_DIR, enabled=config.USE_CACHE)
    history = load_user_history(config.USER_HISTORY_CSV)
    requirements = load_evidence_requirements(config.EVIDENCE_REQ_CSV)

    rows: List[Dict[str, str]] = []
    t0 = time.time()
    for i, claim in enumerate(claims, 1):
        parsed = parse_claim(claim.user_claim, claim.claim_object)
        image_set = load_image_set(claim.image_paths, dataset_dir)
        reqs = select_requirements(requirements, claim.claim_object, parsed)
        req_text = requirements_text(reqs)

        obs = _observe(
            backend, fallback, cache, image_set, parsed,
            claim.claim_object, claim.user_claim, req_text,
        )
        if verifier:
            obs = _verify(
                backend, cache, image_set, parsed,
                claim.claim_object, claim.user_claim, obs,
            )

        evidence = decide_evidence(
            parsed=parsed,
            vlm_part_visible=obs.part_visible,
            vlm_object_match=obs.object_match,
            any_usable_image=image_set.any_usable(),
            matched_ids=[r.requirement_id for r in reqs],
        )
        hist = history.get(claim.user_id)
        rows.append(build_output_row(claim, parsed, image_set, obs, evidence, hist,
                                     cfg=pp_config))
        if verbose:
            print(
                f"  [{i}/{len(claims)}] {claim.user_id} {claim.claim_object} "
                f"-> {rows[-1]['claim_status']}",
                file=sys.stderr,
            )
    if verbose:
        print(f"[done] {len(claims)} claims in {time.time() - t0:.1f}s", file=sys.stderr)
    return rows
