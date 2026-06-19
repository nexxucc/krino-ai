"""
Image loading and normalization.

IMPORTANT: the dataset files end in `.jpg` but are actually AVIF-encoded
(verified with PIL: `Image.open(...).format == 'AVIF'`). Sending the raw bytes
to a VLM with an `image/jpeg` media type would ship AVIF payloads and fail or
silently degrade. So every image is decoded with Pillow and *re-encoded* to a
real JPEG before it leaves this module.

We also compute lightweight quality signals (blur, brightness, aspect) that the
heuristic backend and risk layer can use without any model call.
"""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from PIL import Image, ImageFilter

# Cap the long edge to keep VLM payloads small and inference fast on CPU.
MAX_EDGE = 768
JPEG_QUALITY = 85


def image_id_from_path(path: str) -> str:
    """`images/test/case_001/img_1.jpg` -> `img_1` (filename without extension)."""
    return os.path.splitext(os.path.basename(path.strip()))[0]


def split_image_paths(image_paths: str) -> List[str]:
    return [p.strip() for p in str(image_paths).split(";") if p.strip()]


@dataclass
class LoadedImage:
    image_id: str
    rel_path: str
    abs_path: str
    ok: bool = False
    width: int = 0
    height: int = 0
    jpeg_bytes: Optional[bytes] = None
    # quick, model-free quality signals
    blur_score: float = 0.0          # lower = blurrier (variance of Laplacian-ish)
    brightness: float = 0.0          # mean luma 0-255
    is_blurry: bool = False
    is_low_light: bool = False
    is_extreme_aspect: bool = False
    error: str = ""

    def b64(self) -> str:
        return base64.b64encode(self.jpeg_bytes).decode("ascii") if self.jpeg_bytes else ""


def _quality_signals(img: Image.Image) -> dict:
    """Cheap heuristics: blur via edge variance, brightness via mean luma."""
    gray = img.convert("L")
    # downsize for speed
    small = gray.resize((min(256, gray.width), min(256, gray.height)))
    arr = np.asarray(small, dtype=np.float32)
    brightness = float(arr.mean())
    # variance of a high-pass (edge) image approximates focus/sharpness
    edges = np.asarray(small.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    blur_score = float(edges.var())
    w, h = img.size
    aspect = max(w, h) / max(1, min(w, h))
    return {
        "brightness": brightness,
        "blur_score": blur_score,
        "is_blurry": blur_score < 80.0,        # tuned conservative threshold
        "is_low_light": brightness < 55.0,
        "is_extreme_aspect": aspect > 3.0,
    }


def load_image(rel_path: str, dataset_dir: str) -> LoadedImage:
    """Resolve, decode (AVIF-safe), re-encode to JPEG, and score quality."""
    abs_path = os.path.join(dataset_dir, rel_path)
    li = LoadedImage(
        image_id=image_id_from_path(rel_path),
        rel_path=rel_path,
        abs_path=abs_path,
    )
    if not os.path.exists(abs_path):
        li.error = "file_not_found"
        return li
    try:
        with Image.open(abs_path) as im:
            im = im.convert("RGB")
            # downscale preserving aspect
            im.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
            li.width, li.height = im.size
            sig = _quality_signals(im)
            li.brightness = sig["brightness"]
            li.blur_score = sig["blur_score"]
            li.is_blurry = sig["is_blurry"]
            li.is_low_light = sig["is_low_light"]
            li.is_extreme_aspect = sig["is_extreme_aspect"]
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=JPEG_QUALITY)
            li.jpeg_bytes = buf.getvalue()
            li.ok = True
    except Exception as exc:  # corrupt / undecodable
        li.error = f"decode_error:{type(exc).__name__}"
    return li


@dataclass
class ImageSet:
    images: List[LoadedImage] = field(default_factory=list)

    @property
    def usable(self) -> List[LoadedImage]:
        return [im for im in self.images if im.ok]

    @property
    def ids(self) -> List[str]:
        return [im.image_id for im in self.images]

    def any_usable(self) -> bool:
        return len(self.usable) > 0


def load_image_set(image_paths: str, dataset_dir: str) -> ImageSet:
    return ImageSet(images=[load_image(p, dataset_dir) for p in split_image_paths(image_paths)])
