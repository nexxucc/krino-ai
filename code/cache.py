"""
On-disk JSON cache for VLM observations.

Keyed by a content hash of (model/backend name + claim object + claim text +
the bytes of every image). Identical inputs therefore reuse the prior result,
making re-runs free and deterministic. Disable with EVIDENCE_USE_CACHE=0.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from typing import Optional

from image_utils import ImageSet
from vlm.base import VLMObservation


def make_key(backend_name: str, claim_object: str, user_claim: str, image_set: ImageSet) -> str:
    h = hashlib.sha256()
    h.update(backend_name.encode())
    h.update(b"\x00")
    h.update(claim_object.encode())
    h.update(b"\x00")
    h.update(user_claim.encode("utf-8"))
    for im in image_set.images:
        h.update(b"\x00")
        h.update(im.image_id.encode())
        if im.jpeg_bytes:
            h.update(hashlib.sha256(im.jpeg_bytes).digest())
        else:
            h.update(b"missing")
    return h.hexdigest()


class ObservationCache:
    def __init__(self, cache_dir: str, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        if enabled:
            os.makedirs(cache_dir, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.json")

    def get(self, key: str) -> Optional[VLMObservation]:
        if not self.enabled:
            return None
        p = self._path(key)
        if not os.path.exists(p):
            return None
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            return VLMObservation(**data)
        except (json.JSONDecodeError, TypeError):
            return None

    def put(self, key: str, obs: VLMObservation) -> None:
        if not self.enabled:
            return
        with open(self._path(key), "w", encoding="utf-8") as f:
            json.dump(asdict(obs), f)
