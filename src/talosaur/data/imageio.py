"""Image loading/resizing helpers used by fetchers and the curation pipeline (Pillow)."""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np


def resize_short_side(img, short_side: int | None, max_long_side: int | None = None):
    """Downscale a PIL image so its short side is ``short_side`` (never upscales)."""
    from PIL import Image

    w, h = img.size
    scale = 1.0
    if short_side and min(w, h) > short_side:
        scale = short_side / min(w, h)
    if max_long_side and max(w, h) * scale > max_long_side:
        scale = max_long_side / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
    return img


def save_jpeg(img, path: str | os.PathLike, quality: int = 90) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if img.mode != "RGB":
        img = img.convert("RGB")
    tmp = path.with_suffix(path.suffix + ".tmp")
    img.save(tmp, format="JPEG", quality=quality, optimize=False)
    os.replace(tmp, path)
    return path


def decode_resize_save(
    data: bytes, out_path: str | os.PathLike, short_side: int, max_long_side: int | None = None, quality: int = 90
) -> tuple[int, int, int, int]:
    """Decode encoded image bytes, downscale, save as JPEG.

    Returns ``(orig_w, orig_h, new_w, new_h)`` so callers can rescale box coordinates.
    """
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        ow, oh = im.size
        im = im.convert("RGB")
        im = resize_short_side(im, short_side, max_long_side)
        save_jpeg(im, out_path, quality)
        return ow, oh, im.size[0], im.size[1]


def load_rgb(path: str | os.PathLike, max_side: int | None = None) -> np.ndarray:
    """Load as uint8 RGB. For JPEGs, ``draft`` lets libjpeg decode at 1/2..1/8 scale (fast)."""
    from PIL import Image

    from talosaur.data.h5store import open_image

    with Image.open(open_image(path)) as im:
        if max_side and im.format == "JPEG":
            w, h = im.size
            im.draft("RGB", (max(1, w * max_side // max(w, h)), max(1, h * max_side // max(w, h))))
        im = im.convert("RGB")
        if max_side and max(im.size) > max_side:
            s = max_side / max(im.size)
            im = im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s))), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def image_size(path: str | os.PathLike) -> tuple[int, int]:
    from PIL import Image

    from talosaur.data.h5store import open_image

    with Image.open(open_image(path)) as im:
        return im.size
