"""
Canonical output schema, allowed-value vocabularies, and coercion helpers.

Everything the system emits must pass through `coerce_*` so that no value can
ever fall outside the closed sets defined in problem_statement.md. The evaluator
compares against `sample_claims.csv`, so the column order here is authoritative
and must not change.
"""
from __future__ import annotations

from typing import List

# ---------------------------------------------------------------------------
# Output column order (exact, per problem_statement.md "Required output")
# ---------------------------------------------------------------------------
OUTPUT_COLUMNS: List[str] = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

INPUT_COLUMNS: List[str] = ["user_id", "image_paths", "user_claim", "claim_object"]

# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------
CLAIM_OBJECTS = {"car", "laptop", "package"}

CLAIM_STATUS = {"supported", "contradicted", "not_enough_information"}

ISSUE_TYPES = {
    "dent", "scratch", "crack", "glass_shatter", "broken_part", "missing_part",
    "torn_packaging", "crushed_packaging", "water_damage", "stain", "none", "unknown",
}

OBJECT_PARTS = {
    "car": {
        "front_bumper", "rear_bumper", "door", "hood", "windshield", "side_mirror",
        "headlight", "taillight", "fender", "quarter_panel", "body", "unknown",
    },
    "laptop": {
        "screen", "keyboard", "trackpad", "hinge", "lid", "corner", "port",
        "base", "body", "unknown",
    },
    "package": {
        "box", "package_corner", "package_side", "seal", "label", "contents",
        "item", "unknown",
    },
}

RISK_FLAGS = {
    "none", "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
    "wrong_angle", "wrong_object", "wrong_object_part", "damage_not_visible",
    "claim_mismatch", "possible_manipulation", "non_original_image",
    "text_instruction_present", "user_history_risk", "manual_review_required",
}

SEVERITY = {"none", "low", "medium", "high", "unknown"}

# Issue types that are physically meaningful per object family. Used only as a
# soft sanity hint (not a hard filter) when coercing model output.
OBJECT_ISSUE_HINTS = {
    "car": {"dent", "scratch", "crack", "glass_shatter", "broken_part",
            "missing_part", "none", "unknown"},
    "laptop": {"crack", "scratch", "dent", "broken_part", "missing_part",
               "water_damage", "stain", "glass_shatter", "none", "unknown"},
    "package": {"torn_packaging", "crushed_packaging", "water_damage", "stain",
                "missing_part", "broken_part", "none", "unknown"},
}


# ---------------------------------------------------------------------------
# Coercion helpers — every one returns a guaranteed-valid value.
# ---------------------------------------------------------------------------
def _norm(value) -> str:
    return str(value).strip().lower().replace(" ", "_") if value is not None else ""


def coerce_enum(value, allowed: set, default: str) -> str:
    v = _norm(value)
    if v in allowed:
        return v
    # tolerate a few common synonyms
    synonyms = {
        "shattered_glass": "glass_shatter",
        "shatter": "glass_shatter",
        "broken": "broken_part",
        "missing": "missing_part",
        "torn": "torn_packaging",
        "crushed": "crushed_packaging",
        "water": "water_damage",
        "insufficient": "not_enough_information",
        "not_enough_info": "not_enough_information",
        "inconclusive": "not_enough_information",
        "support": "supported",
        "contradict": "contradicted",
    }
    if v in synonyms and synonyms[v] in allowed:
        return synonyms[v]
    return default


def coerce_object_part(value, claim_object: str) -> str:
    allowed = OBJECT_PARTS.get(_norm(claim_object), set())
    if not allowed:
        return "unknown"
    return coerce_enum(value, allowed, "unknown")


def coerce_claim_status(value) -> str:
    return coerce_enum(value, CLAIM_STATUS, "not_enough_information")


def coerce_issue_type(value) -> str:
    return coerce_enum(value, ISSUE_TYPES, "unknown")


def coerce_severity(value) -> str:
    return coerce_enum(value, SEVERITY, "unknown")


def coerce_bool(value) -> str:
    """Return lowercase 'true'/'false' strings, matching the sample CSV."""
    if isinstance(value, bool):
        return "true" if value else "false"
    v = _norm(value)
    if v in {"true", "yes", "1", "t"}:
        return "true"
    if v in {"false", "no", "0", "f"}:
        return "false"
    return "false"


def coerce_risk_flags(values) -> str:
    """Accept list/str; return a clean semicolon-joined, deduped flag string."""
    if values is None:
        return "none"
    if isinstance(values, str):
        parts = [p for chunk in values.split(";") for p in [chunk.strip()]]
    else:
        parts = list(values)
    seen = []
    for p in parts:
        f = _norm(p)
        if f and f != "none" and f in RISK_FLAGS and f not in seen:
            seen.append(f)
    return ";".join(seen) if seen else "none"


def coerce_image_ids(values) -> str:
    """Semicolon-joined image ids, or 'none'."""
    if values is None:
        return "none"
    if isinstance(values, str):
        parts = values.split(";")
    else:
        parts = list(values)
    out = []
    for p in parts:
        s = str(p).strip()
        if s and s.lower() != "none" and s not in out:
            out.append(s)
    return ";".join(out) if out else "none"
