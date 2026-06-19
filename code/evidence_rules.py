"""
Map a claim to the relevant rows of evidence_requirements.csv and decide whether
the submitted image set meets the minimum evidence standard.

The selection key is (claim_object, issue_family). The "all" rules always apply.
The decision is intentionally conservative: evidence is "met" only when at least
one usable image plausibly shows the claimed part/condition, per the VLM's
observation plus model-free quality signals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from claim_parser import ParsedClaim
from data_loaders import EvidenceRequirement


@dataclass
class EvidenceDecision:
    met: bool
    reason: str
    matched_requirements: List[str]


def select_requirements(
    requirements: List[EvidenceRequirement],
    claim_object: str,
    parsed: ParsedClaim,
) -> List[EvidenceRequirement]:
    """Return the applicable requirement rows: object-specific family match + 'all'."""
    selected: List[EvidenceRequirement] = []
    family = parsed.issue_family
    for req in requirements:
        if req.claim_object == "all":
            selected.append(req)
        elif req.claim_object == claim_object:
            # match by issue family when we have one; otherwise include the
            # object's general rows so the prompt still has context.
            if family and family.lower() in req.applies_to.lower():
                selected.append(req)
            elif family is None:
                selected.append(req)
    # de-dup preserving order
    seen, out = set(), []
    for r in selected:
        if r.requirement_id not in seen:
            seen.add(r.requirement_id)
            out.append(r)
    return out


def requirements_text(reqs: List[EvidenceRequirement]) -> str:
    return "\n".join(f"- ({r.requirement_id}) {r.minimum_image_evidence}" for r in reqs)


def decide_evidence(
    parsed: ParsedClaim,
    vlm_part_visible: bool,
    vlm_object_match: bool,
    any_usable_image: bool,
    matched_ids: List[str],
) -> EvidenceDecision:
    """
    Evidence standard is met when:
      - there is at least one usable image, AND
      - the claimed object is the one shown (object match), AND
      - the claimed part/area is actually visible to inspect.
    """
    if not any_usable_image:
        return EvidenceDecision(
            False, "No usable image was available to evaluate the claim.", matched_ids
        )
    if not vlm_object_match:
        # A clearly-shown WRONG object is itself sufficient to evaluate the
        # claim -> it contradicts it. So the evidence standard IS met; the
        # status layer turns this into 'contradicted' (not NEI).
        return EvidenceDecision(
            True,
            "The images clearly show a different object than claimed, which is enough to evaluate (and contradict) the claim.",
            matched_ids,
        )
    if not vlm_part_visible:
        return EvidenceDecision(
            False,
            f"The claimed area ({parsed.claimed_part}) is not visible clearly enough in the images to evaluate the claim.",
            matched_ids,
        )
    return EvidenceDecision(
        True,
        f"The claimed {parsed.claimed_part} is visible in the image set and can be inspected for the reported issue.",
        matched_ids,
    )
