"""
CSV loaders for claims, user history, and evidence requirements.

Uses the stdlib `csv` module (robust to the quoted, pipe-containing claim text)
rather than pandas, so the core pipeline has no heavy import on the hot path.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Claim:
    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str


@dataclass
class UserHistory:
    user_id: str
    past_claim_count: int = 0
    accept_claim: int = 0
    manual_review_claim: int = 0
    rejected_claim: int = 0
    last_90_days_claim_count: int = 0
    history_flags: List[str] = field(default_factory=list)
    history_summary: str = ""

    @property
    def rejection_rate(self) -> float:
        return self.rejected_claim / self.past_claim_count if self.past_claim_count else 0.0


@dataclass
class EvidenceRequirement:
    requirement_id: str
    claim_object: str
    applies_to: str
    minimum_image_evidence: str


def _to_int(v) -> int:
    try:
        return int(str(v).strip())
    except (ValueError, TypeError):
        return 0


def load_claims(path: str) -> List[Claim]:
    with open(path, newline="", encoding="utf-8") as f:
        return [
            Claim(
                user_id=r["user_id"],
                image_paths=r["image_paths"],
                user_claim=r["user_claim"],
                claim_object=r["claim_object"].strip().lower(),
            )
            for r in csv.DictReader(f)
        ]


def load_user_history(path: str) -> Dict[str, UserHistory]:
    out: Dict[str, UserHistory] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            flags = [x.strip() for x in (r.get("history_flags") or "").split(";")
                     if x.strip() and x.strip().lower() != "none"]
            out[r["user_id"]] = UserHistory(
                user_id=r["user_id"],
                past_claim_count=_to_int(r.get("past_claim_count")),
                accept_claim=_to_int(r.get("accept_claim")),
                manual_review_claim=_to_int(r.get("manual_review_claim")),
                rejected_claim=_to_int(r.get("rejected_claim")),
                last_90_days_claim_count=_to_int(r.get("last_90_days_claim_count")),
                history_flags=flags,
                history_summary=(r.get("history_summary") or "").strip(),
            )
    return out


def load_evidence_requirements(path: str) -> List[EvidenceRequirement]:
    with open(path, newline="", encoding="utf-8") as f:
        return [
            EvidenceRequirement(
                requirement_id=r["requirement_id"],
                claim_object=r["claim_object"].strip().lower(),
                applies_to=r["applies_to"].strip(),
                minimum_image_evidence=r["minimum_image_evidence"].strip(),
            )
            for r in csv.DictReader(f)
        ]


def get_history(history: Dict[str, UserHistory], user_id: str) -> Optional[UserHistory]:
    return history.get(user_id)
