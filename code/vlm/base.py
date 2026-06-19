"""
Backend interface + the shared observation schema every backend returns.

A backend's single job is to OBSERVE the images and return structured facts:
what object/part is visible, what damage (if any) is visible, per-image quality
issues, severity, and which image ids are most relevant. It does NOT decide the
final claim_status or fold in user history — that lives in postprocess.py, so the
"images are the source of truth, history only adds risk" rule is enforced in one
place regardless of backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from claim_parser import ParsedClaim
from image_utils import ImageSet


@dataclass
class VLMObservation:
    object_match: bool = True          # do the images show the claimed object type?
    part_visible: bool = False         # is the claimed part visible to inspect?
    visible_issue_type: str = "unknown"
    visible_object_part: str = "unknown"
    damage_present: bool = False       # is any damage actually visible?
    severity: str = "unknown"
    # --- hybrid decision (D1): the model's own proposal, reconciled in postprocess
    proposed_status: str = "not_enough_information"  # supported|contradicted|not_enough_information
    confidence: float = 0.0            # model's self-reported confidence 0..1
    supporting_image_ids: List[str] = field(default_factory=list)
    # per-image quality / authenticity observations
    quality_flags: List[str] = field(default_factory=list)
    possible_manipulation: bool = False
    non_original_image: bool = False
    text_instruction_in_image: bool = False
    # free-text grounding used in justifications
    notes: str = ""
    raw: Dict = field(default_factory=dict)


class VLMBackend:
    name = "base"

    def analyze(
        self,
        image_set: ImageSet,
        parsed: ParsedClaim,
        claim_object: str,
        user_claim: str,
        requirements_text: str,
    ) -> VLMObservation:
        raise NotImplementedError

    def available(self) -> bool:
        return True
