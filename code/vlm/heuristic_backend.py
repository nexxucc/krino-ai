"""
Model-free backend.

Produces a schema-valid observation purely from the deterministic conversation
parse plus the cheap image quality signals computed in image_utils. It cannot
truly "see" damage, so it takes the claimed part/issue at face value when a
usable, reasonable-quality image of the right shape exists, and downgrades to
"not visible" when the only images are blurry/dark/extreme-aspect.

This guarantees the pipeline and evaluator always run with zero downloads and is
the automatic fallback when the local model is unavailable. It is deterministic.
"""
from __future__ import annotations

from claim_parser import ParsedClaim
from image_utils import ImageSet
from vlm.base import VLMBackend, VLMObservation


class HeuristicBackend(VLMBackend):
    name = "heuristic"

    def analyze(self, image_set, parsed, claim_object, user_claim, requirements_text):
        usable = image_set.usable
        obs = VLMObservation()
        obs.visible_object_part = parsed.claimed_part
        obs.visible_issue_type = parsed.issue_type

        if not usable:
            obs.object_match = False
            obs.part_visible = False
            obs.damage_present = False
            obs.severity = "unknown"
            obs.proposed_status = "not_enough_information"
            obs.confidence = 0.3
            obs.notes = "No usable image could be decoded for review."
            return obs

        # quality flags aggregated across images
        quality = set()
        clear_images = []
        for im in usable:
            if im.is_blurry:
                quality.add("blurry_image")
            if im.is_low_light:
                quality.add("low_light_or_glare")
            if im.is_extreme_aspect:
                quality.add("cropped_or_obstructed")
            if not (im.is_blurry or im.is_low_light or im.is_extreme_aspect):
                clear_images.append(im)
        obs.quality_flags = sorted(quality)

        # Without real vision we assume the claimed object is shown.
        obs.object_match = True
        # The part is considered visible if at least one image is reasonable.
        obs.part_visible = len(clear_images) > 0 or len(usable) > 0
        # Pick supporting images: prefer clear ones, else the sharpest.
        ranked = sorted(usable, key=lambda im: im.blur_score, reverse=True)
        chosen = clear_images or ranked[:1]
        obs.supporting_image_ids = [im.image_id for im in chosen[:2]]

        # Take the claimed issue at face value (no real damage detection).
        if obs.visible_issue_type in ("none", "unknown"):
            obs.damage_present = False
            obs.severity = "unknown"
        else:
            obs.damage_present = True
            obs.severity = "medium"  # neutral default; cannot gauge magnitude

        # Heuristic proposal: it cannot see contradictions, so it trusts a
        # well-formed claim with a usable image (low confidence so the rule
        # layer dominates on conflicts).
        if obs.part_visible and obs.damage_present:
            obs.proposed_status = "supported"
        elif not obs.part_visible:
            obs.proposed_status = "not_enough_information"
        else:
            obs.proposed_status = "supported"
        obs.confidence = 0.4

        # Authenticity / injection signals come from text, not pixels, here.
        obs.text_instruction_in_image = parsed.injection_detected
        obs.notes = (
            f"Heuristic review: claimed {parsed.claimed_part} "
            f"({parsed.issue_type}); {len(usable)} usable image(s)."
        )
        return obs
