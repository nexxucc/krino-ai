"""
Fuse the visual observation, the deterministic claim parse, the evidence
decision, and user history into the final 14-column output row.

This module is the ONE place the core policy lives:
  * Images are the source of truth for issue_type / object_part / claim_status.
  * User history NEVER flips a clear visual verdict; it only adds risk_flags
    (user_history_risk, manual_review_required) and justification context.
  * Any instruction-like text (conversation or in-image) is reported via
    text_instruction_present and otherwise ignored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from claim_parser import ParsedClaim
from data_loaders import UserHistory
from evidence_rules import EvidenceDecision
from image_utils import ImageSet
from vlm.base import VLMObservation
import schema


@dataclass(frozen=True)
class PostprocessConfig:
    """Tunable rule knobs, selected via cross-validation (never hand-fit to the
    20 labels). decision_mode is fixed to the comparison winner by default."""
    decision_mode: str = "hybrid"       # "hybrid" | "direct"
    symmetric_override: bool = True      # trust clear visual evidence over a model hedge
    allow_high_severity: bool = False    # let severe structural breakage reach 'high'

    def tag(self) -> str:
        return f"{self.decision_mode}-so{int(self.symmetric_override)}-hi{int(self.allow_high_severity)}"


def _decide_status(
    obs: VLMObservation,
    parsed: ParsedClaim,
    evidence: EvidenceDecision,
    cfg: "PostprocessConfig" = PostprocessConfig(),
) -> str:
    """Decide claim_status.

    mode="hybrid"  (D1, default): start from the model's proposed status, then
                   apply deterministic hard-rule overrides + the symmetric guard.
    mode="direct"  (comparison strategy): trust the model's proposed status as-is
                   (only fall back to NEI when there is no usable evidence at all).
    """
    mode = cfg.decision_mode
    proposed = schema.coerce_claim_status(obs.proposed_status)

    if mode == "direct":
        # Pure VLM verdict; the only floor is "no usable image -> can't decide".
        if not obs.object_match and not obs.part_visible:
            return "not_enough_information"
        return proposed

    # --- hard guard-rails (override the model) ---
    # 1. Evidence standard not met -> cannot evaluate.
    if not evidence.met:
        return "not_enough_information"
    # 2. Object shown is not the claimed object.
    if not obs.object_match:
        return "contradicted"
    claimed_damage = parsed.issue_type not in ("none", "unknown")
    # 3. User claims damage but none is visible on the (visible) part.
    if claimed_damage and not obs.damage_present:
        return "contradicted"
    # 4. Visible issue conflicts with the claimed issue family.
    if (
        claimed_damage
        and obs.damage_present
        and obs.visible_issue_type not in ("none", "unknown")
        and not _issue_compatible(parsed.issue_type, obs.visible_issue_type)
    ):
        return "contradicted"

    # --- no override fired: trust the model's holistic verdict ---
    # (this is where the model catches "minor vs. severe" style contradictions
    #  that the rules above cannot see). Guard a couple of incoherent combos.
    if proposed == "supported" and not obs.damage_present and claimed_damage:
        return "contradicted"
    # Symmetric guard-rail: if the images clearly show the claimed damage on the
    # visible claimed part (compatible issue), trust that over an over-cautious
    # model hedge of contradicted/NEI.
    if (
        cfg.symmetric_override
        and proposed in ("contradicted", "not_enough_information")
        and obs.object_match
        and obs.part_visible
        and obs.damage_present
        and claimed_damage
        and _issue_compatible(parsed.issue_type, obs.visible_issue_type)
    ):
        return "supported"
    return proposed


# issue families that should be treated as compatible (severity wording varies)
_COMPATIBLE = [
    {"dent", "crushed_packaging"},
    {"crack", "glass_shatter", "broken_part"},
    {"scratch", "stain"},
    {"torn_packaging", "broken_part"},
    {"water_damage", "stain"},
    {"missing_part", "broken_part"},
]


def _issue_compatible(claimed: str, visible: str) -> bool:
    if claimed == visible:
        return True
    for grp in _COMPATIBLE:
        if claimed in grp and visible in grp:
            return True
    return False


def _build_risk_flags(
    obs: VLMObservation,
    parsed: ParsedClaim,
    evidence: EvidenceDecision,
    status: str,
    history: Optional[UserHistory],
    claim_object: str,
) -> list:
    flags = []
    flags.extend(obs.quality_flags)
    if not obs.object_match:
        flags.append("wrong_object")
    # damage claimed but not visible (in NEI or evidence-met cases alike)
    if not obs.damage_present and parsed.issue_type not in ("none", "unknown"):
        flags.append("damage_not_visible")
    if status == "contradicted":
        flags.append("claim_mismatch")
    if obs.possible_manipulation:
        flags.append("possible_manipulation")
    if obs.non_original_image:
        flags.append("non_original_image")
    if obs.text_instruction_in_image or parsed.injection_detected:
        flags.append("text_instruction_present")
    # part visible but a DIFFERENT valid part shown than claimed. Compare the
    # COERCED part (the model's visible_object_part is free text and would never
    # string-match the normalized claimed part otherwise -> false positives).
    vis_part = schema.coerce_object_part(obs.visible_object_part, claim_object)
    if (
        evidence.met
        and vis_part != "unknown"
        and parsed.claimed_part not in ("unknown", "")
        and vis_part != parsed.claimed_part
    ):
        flags.append("wrong_object_part")

    # ---- user history: RISK ONLY, never flips the verdict ----
    if history is not None:
        if "user_history_risk" in history.history_flags:
            flags.append("user_history_risk")
        if "manual_review_required" in history.history_flags:
            flags.append("manual_review_required")
        # derive risk from a poor track record even if not explicitly flagged
        if history.past_claim_count >= 3 and history.rejection_rate >= 0.4:
            flags.append("user_history_risk")
    # user-history risk and contradictions/authenticity concerns warrant review
    if (
        status == "contradicted"
        or obs.possible_manipulation
        or obs.non_original_image
        or "user_history_risk" in flags
    ):
        flags.append("manual_review_required")
    return flags


# severity ordering for the BLEND clamp (D11); 'unknown' handled separately
_SEV_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}
# Per-issue-family caps. The model is systematically "hot" (reads high where the
# labels say low/medium), so caps are conservative: cosmetic surface issues cap
# at low, most damage caps at medium, and only severe structural breakage may
# reach high (and only when the model itself is confident).
_SEV_CAP = {
    "scratch": "low", "stain": "low",
    "dent": "medium", "crack": "medium", "water_damage": "medium",
    "missing_part": "medium", "torn_packaging": "medium", "crushed_packaging": "medium",
    "glass_shatter": "medium", "broken_part": "medium",
}
# 'high' is reserved for catastrophic structural breakage and is very rarely the
# right label from a single photo; only emit it for structural issues when the
# model is near-certain.
_HIGH_ALLOWED = {"glass_shatter", "broken_part"}


def _severity(obs: VLMObservation, status: str, issue_type: str,
              cfg: "PostprocessConfig" = PostprocessConfig()) -> str:
    """BLEND (D11): take the VLM severity, then clamp by issue family.

    With cfg.allow_high_severity, severe structural breakage may reach 'high'
    (when the model is confident); otherwise everything caps at 'medium'.
    """
    if status == "not_enough_information":
        return "unknown"
    if not obs.damage_present:
        return "none"
    sev = obs.severity if obs.severity in _SEV_ORDER else "medium"
    cap = _SEV_CAP.get(issue_type, "medium")
    if cfg.allow_high_severity and issue_type in _HIGH_ALLOWED:
        cap = "high"
    if _SEV_ORDER.get(sev, 2) > _SEV_ORDER[cap]:
        sev = cap
    # only emit 'high' for structural issues with a confident model
    if sev == "high" and not (issue_type in _HIGH_ALLOWED and obs.confidence >= 0.75):
        sev = "medium"
    return sev


def _supporting_ids(obs: VLMObservation, image_set: ImageSet, status: str, evidence) -> list:
    if status == "not_enough_information" or not evidence.met:
        return []
    if obs.supporting_image_ids:
        return obs.supporting_image_ids
    usable = image_set.usable
    return [usable[0].image_id] if usable else []


def build_output_row(
    claim,
    parsed: ParsedClaim,
    image_set: ImageSet,
    obs: VLMObservation,
    evidence: EvidenceDecision,
    history: Optional[UserHistory],
    cfg: "PostprocessConfig" = PostprocessConfig(),
) -> Dict[str, str]:
    status = _decide_status(obs, parsed, evidence, cfg=cfg)

    # object_part: prefer the claimed part; fall back to what the model saw
    object_part = parsed.claimed_part
    if object_part in ("unknown", "") and obs.visible_object_part:
        object_part = obs.visible_object_part

    # issue_type:
    #  - supported  -> the validated CLAIMED issue (labels track the claim, not
    #                  the model's "most severe visible" read).
    #  - contradicted -> what is actually VISIBLE (or none if no damage).
    #  - NEI       -> unknown.
    if status == "not_enough_information":
        issue_type = "unknown"
    elif status == "supported":
        if parsed.issue_type not in ("none", "unknown"):
            issue_type = parsed.issue_type
        elif obs.damage_present and obs.visible_issue_type not in ("none", "unknown"):
            issue_type = obs.visible_issue_type
        else:
            issue_type = "none"
    else:  # contradicted
        if obs.visible_issue_type != "unknown":
            issue_type = obs.visible_issue_type
        elif parsed.issue_type not in ("none", "unknown"):
            issue_type = parsed.issue_type
        else:
            issue_type = "unknown"

    risk = _build_risk_flags(obs, parsed, evidence, status, history, claim.claim_object)
    severity = _severity(obs, status, issue_type, cfg)
    support_ids = _supporting_ids(obs, image_set, status, evidence)
    valid_image = image_set.any_usable() and obs.object_match and not obs.non_original_image

    # justification, grounded and mentioning image ids when available
    just = obs.notes.strip() if obs.notes else evidence.reason
    if support_ids:
        just = f"{just} (see {', '.join(support_ids)})."
    if history is not None and ("user_history_risk" in risk or "manual_review_required" in risk):
        just = f"{just} User history adds review risk."

    return {
        "user_id": claim.user_id,
        "image_paths": claim.image_paths,
        "user_claim": claim.user_claim,
        "claim_object": claim.claim_object,
        "evidence_standard_met": schema.coerce_bool(evidence.met),
        "evidence_standard_met_reason": evidence.reason,
        "risk_flags": schema.coerce_risk_flags(risk),
        "issue_type": schema.coerce_issue_type(issue_type),
        "object_part": schema.coerce_object_part(object_part, claim.claim_object),
        "claim_status": schema.coerce_claim_status(status),
        "claim_status_justification": just.strip(),
        "supporting_image_ids": schema.coerce_image_ids(support_ids),
        "valid_image": schema.coerce_bool(valid_image),
        "severity": schema.coerce_severity(severity),
    }
