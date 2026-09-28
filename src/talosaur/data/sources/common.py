"""Helpers shared by the source ingesters."""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from talosaur.data.imageio import resize_short_side, save_jpeg
from talosaur.data.licenses import train_commercial_ok
from talosaur.data.schema import Record
from talosaur.data.sources.base import SourceInfo

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts", ".mxf", ".h264", ".ts"}

_FRAME_SUFFIX = re.compile(r"([_\-. ]?(frame|frm|f|img|image)?[_\-. ]?\d{1,7})$", re.IGNORECASE)


def video_id_from_frame_name(name: str) -> str:
    """Strip a trailing frame counter: ``EX1708_VID_..._ROVHD_frame_001678`` -> ``EX1708_VID_..._ROVHD``."""
    stem = Path(name).stem
    base = _FRAME_SUFFIX.sub("", stem)
    return base or stem


def ingest_image(
    src: str | Path,
    root: Path,
    source: str,
    rel: str,
    short_side: int,
    quality: int = 90,
) -> tuple[str, int, int, int, int]:
    """Copy an image into ``images/<source>/<rel>.jpg`` resized; returns (path, ow, oh, w, h)."""
    from PIL import Image, ImageOps

    out_rel = f"images/{source}/{rel}"
    if not out_rel.lower().endswith(".jpg"):
        out_rel = str(Path(out_rel).with_suffix(".jpg"))
    out = root / out_rel
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        ow, oh = im.size
        if out.exists():
            from talosaur.data.imageio import image_size

            w, h = image_size(out)
            return out_rel, ow, oh, w, h
        im = resize_short_side(im.convert("RGB"), short_side)
        save_jpeg(im, out, quality)
        return out_rel, ow, oh, im.size[0], im.size[1]


def ingest_mask(src: str | Path, root: Path, source: str, rel: str, size: tuple[int, int]) -> str:
    """Store a binary mask (nearest-neighbour resized to the stored image size) as PNG."""
    from PIL import Image

    out_rel = str(Path(f"masks/{source}/{rel}").with_suffix(".png"))
    out = root / out_rel
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as m:
            arr = np.asarray(m.convert("L")) > 0
        Image.fromarray(arr.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST).save(out)
    return out_rel


def make_record(
    info: SourceInfo,
    image_id: str,
    path: str,
    group_id: str,
    width: int,
    height: int,
    license_id: str | None = None,
    **kw,
) -> Record:
    lic = license_id or info.license_id
    labeled = bool(kw.get("frame_label", -1) != -1 or kw.get("boxes") or kw.get("mask_path"))
    return Record(
        image_id=image_id,
        path=path,
        source=info.name,
        group_id=group_id,
        license=lic,
        train_commercial_ok=train_commercial_ok(lic, info.ml_training_clause),
        width=width,
        height=height,
        attribution=info.attribution,
        labeled=labeled,
        **kw,
    )


def xywh_to_norm_xyxy(b: Iterable[float], w: float, h: float) -> list[float]:
    x, y, bw, bh = (float(v) for v in b)
    x0, y0 = max(0.0, x / w), max(0.0, y / h)
    x1, y1 = min(1.0, (x + bw) / w), min(1.0, (y + bh) / h)
    return [round(x0, 6), round(y0, 6), round(x1, 6), round(y1, 6)]


def find_files(directory: Path, exts: set[str]) -> list[Path]:
    return sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in exts and p.is_file())
