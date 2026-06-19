"""
Deterministic parsing of the claim conversation.

Extracts, without any model call:
  - the claimed object_part (object-scoped)
  - the claimed issue_type / issue family
  - whether the conversation contains injection / "obey me" instructions
  - whether the user themselves flags multiple parts

The dataset mixes English, Spanish, romanized Hindi, and romanized Chinese, so
keyword tables include common cross-lingual variants. This parser is a strong
prior that the VLM result is reconciled against; it is never the sole authority
on what the *image* shows.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

# ---------------------------------------------------------------------------
# Object-part keyword tables  (keyword -> canonical part)
# ---------------------------------------------------------------------------
PART_KEYWORDS = {
    "car": [
        (r"front\s*bumper|parachoques\s*delantero|front\s*bump", "front_bumper"),
        (r"rear\s*bumper|back\s*bumper|parachoques\s*trasero|parachoques\s*de\s*atras|rear\s*bump", "rear_bumper"),
        (r"windshield|wind\s*screen|front\s*glass|windscreen", "windshield"),
        (r"side\s*mirror|wing\s*mirror|mirror|mirror|espejo", "side_mirror"),
        (r"head\s*light|headlamp|front\s*light", "headlight"),
        (r"tail\s*light|back\s*light|rear\s*light", "taillight"),
        (r"\bhood\b|bonnet|capo", "hood"),
        (r"\bdoor\b|puerta|darwaza", "door"),
        (r"fender", "fender"),
        (r"quarter\s*panel", "quarter_panel"),
        (r"body\s*panel|car\s*body|\bbody\b|panel", "body"),
    ],
    "laptop": [
        (r"screen|display|pantalla|monitor", "screen"),
        (r"key\s*board|keys?|keycap|teclas|teclado", "keyboard"),
        (r"track\s*pad|touch\s*pad|trackpad", "trackpad"),
        (r"hinge|bisagra", "hinge"),
        (r"\blid\b|tapa", "lid"),
        (r"corner|esquina|palm\s*rest|palm-rest", "corner"),
        (r"\bport\b|usb|charging\s*port", "port"),
        (r"\bbase\b|bottom", "base"),
        (r"body|outer\s*body|chassis|cuerpo", "body"),
    ],
    "package": [
        (r"corner|package\s*corner|box\s*corner|esquina", "package_corner"),
        (r"seal|tape|flap|sello", "seal"),
        (r"label|shipping\s*label|etiqueta", "label"),
        (r"contents?|item\s*inside|product\s*inside|inner\s*item|missing", "contents"),
        (r"\bitem\b|product", "item"),
        (r"\bside\b|package\s*side", "package_side"),
        (r"\bbox\b|carton|cardboard|parcel|packaging|package|paquete", "box"),
    ],
}

# ---------------------------------------------------------------------------
# Issue keyword tables (keyword -> canonical issue_type)
# ---------------------------------------------------------------------------
ISSUE_KEYWORDS = [
    (r"glass\s*shatter|shattered\s*glass|shatter", "glass_shatter"),
    (r"crack|cracked|grieta|toot|crack", "crack"),
    (r"\bdent|dented|dab\s*gaya|hail\s*dent|deform", "dent"),
    (r"scratch|scrape|scuff|rasgu|mark\b", "scratch"),
    (r"torn|tear|phati|phata|open(ed)?\s*pack|torn[-\s]*open", "torn_packaging"),
    (r"crush|crushed|dab\s*gaya|caved", "crushed_packaging"),
    (r"water\s*damage|wet|liquid|water[-\s]*damaged|moja", "water_damage"),
    (r"stain|oily|oil\s*stain|mancha|dirty\s*mark", "stain"),
    (r"missing\s*part|missing\s*key|missing\s*content|missing\s*item|missing|faltan|falta", "missing_part"),
    (r"broke|broken|toot\s*gaya|roto|broke\s*off", "broken_part"),
]

# Map a raw issue_type to the evidence_requirements "applies_to" family.
ISSUE_TO_FAMILY = {
    "car": {
        "dent": "dent or scratch", "scratch": "dent or scratch",
        "crack": "crack, broken, or missing part",
        "glass_shatter": "crack, broken, or missing part",
        "broken_part": "crack, broken, or missing part",
        "missing_part": "crack, broken, or missing part",
    },
    "laptop": {
        "crack": "screen, keyboard, or trackpad", "scratch": "screen, keyboard, or trackpad",
        "stain": "screen, keyboard, or trackpad", "water_damage": "screen, keyboard, or trackpad",
        "missing_part": "screen, keyboard, or trackpad", "broken_part": "hinge, lid, corner, body, or port",
        "dent": "hinge, lid, corner, body, or port",
    },
    "package": {
        "crushed_packaging": "crushed, torn, or seal damage",
        "torn_packaging": "crushed, torn, or seal damage",
        "broken_part": "contents or inner item", "missing_part": "contents or inner item",
        "water_damage": "water, stain, or label damage", "stain": "water, stain, or label damage",
    },
}

# Phrases that try to make the system obey text instead of evaluating images.
INJECTION_PATTERNS = [
    r"ignore (all|any|previous) instructions?",
    r"approve (the )?claim",
    r"approve (this|it) (immediately|quickly)",
    r"skip manual review",
    r"mark (this|the) (row|claim) (as )?support",
    r"the note (says|is enough)",
    r"follow (it|the note)",
    r"accept this quickly",
    r"approve kar dena|approve kar do",
    r"mark .* supported",
    r"should be approved",
]

# Phrases hinting the user may be pressuring/escalating (not injection, but risk).
PRESSURE_PATTERNS = [
    r"escalate publicly", r"reopening tickets", r"keep reopening", r"i will escalate",
    r"tired of repeat reviews",
]


@dataclass
class ParsedClaim:
    claimed_part: str = "unknown"
    candidate_parts: List[str] = field(default_factory=list)
    issue_type: str = "unknown"
    candidate_issues: List[str] = field(default_factory=list)
    issue_family: Optional[str] = None
    multi_part: bool = False
    injection_detected: bool = False
    pressure_detected: bool = False
    identity_dependent: bool = False  # claim hinges on vehicle color/side/identity


def _last_customer_text(conversation: str) -> str:
    """The final claim usually lands in the last customer/cliente turn."""
    segments = re.split(r"\|", conversation)
    cust = [s for s in segments
            if re.search(r"(customer|cliente|client)\s*:", s, re.I)]
    return cust[-1] if cust else conversation


# Cues used to isolate the AFFIRMED claim from negated/incidental mentions.
_NEG_CUES = re.compile(r"\b(not|no|nahi|nahin|mat|nunca|cannot|can't|won't|don't|ignore)\b|n't", re.I)
_FOCUS_CUES = re.compile(r"\b(only|just|sirf|solo|actual claim|the claim is)\b", re.I)


def _affirmed_text(last: str) -> str:
    """Reduce the last customer turn to its AFFIRMED content: drop negated
    clauses, and prefer an 'only/just/sirf/solo ...' scoping clause when present.
    Prevents picking a part/issue out of a denial (e.g. "not claiming item
    missing, only torn packaging"). Falls back to the full text if nothing
    affirmative remains."""
    clauses = re.split(r"[,;.]| but | lekin | pero ", last)
    affirmed = [c for c in clauses if c.strip() and not _NEG_CUES.search(c)]
    focus = [c for c in affirmed if _FOCUS_CUES.search(c)]
    text = " ".join(focus) if focus else " ".join(affirmed)
    return text.strip() or last


def parse_claim(conversation: str, claim_object: str) -> ParsedClaim:
    text = conversation.lower()
    last = _last_customer_text(conversation).lower()
    last_aff = _affirmed_text(last)
    pc = ParsedClaim()

    # --- parts: collect all matches; prefer one found in the final customer turn
    part_rules = PART_KEYWORDS.get(claim_object, [])
    found_parts: List[str] = []
    for pattern, part in part_rules:
        if re.search(pattern, text) and part not in found_parts:
            found_parts.append(part)
    pc.candidate_parts = found_parts
    last_parts = [part for pattern, part in part_rules if re.search(pattern, last_aff)]
    if last_parts:
        pc.claimed_part = last_parts[0]
    elif found_parts:
        pc.claimed_part = found_parts[0]
    pc.multi_part = len(found_parts) > 1 and bool(
        re.search(r"\b(both|two|and the|plus|first.*second|dono|ambos)\b", text)
    )

    # --- issues
    found_issues: List[str] = []
    for pattern, issue in ISSUE_KEYWORDS:
        if re.search(pattern, text) and issue not in found_issues:
            found_issues.append(issue)
    pc.candidate_issues = found_issues
    last_issues = [issue for pattern, issue in ISSUE_KEYWORDS if re.search(pattern, last_aff)]
    if last_issues:
        pc.issue_type = last_issues[0]
    elif found_issues:
        pc.issue_type = found_issues[0]

    pc.issue_family = ISSUE_TO_FAMILY.get(claim_object, {}).get(pc.issue_type)

    # --- injection / pressure / identity
    pc.injection_detected = any(re.search(p, text) for p in INJECTION_PATTERNS)
    pc.pressure_detected = any(re.search(p, text) for p in PRESSURE_PATTERNS)
    pc.identity_dependent = bool(
        re.search(r"\b(blue|black|red|white|silver|my\s+\w+\s+car)\b.*car|left\s*side|right\s*side", text)
    )
    return pc
