"""Turning boxes / masks / points into targets on a ViT patch grid (numpy only)."""

from __future__ import annotations

import numpy as np


def boxes_to_coverage(boxes, grid_hw: tuple[int, int], supersample: int = 16) -> np.ndarray:
    """Fraction of each patch covered by the *union* of normalised xyxy boxes -> (h, w) float32.

    Rasterises on a ``supersample``-times finer grid using pixel *centres* (unbiased; error is at
    most one sub-pixel per box edge), then area-averages back to the patch grid.
    """
    gh, gw = grid_hw
    H, W = gh * supersample, gw * supersample
    yc = (np.arange(H) + 0.5) / H
    xc = (np.arange(W) + 0.5) / W
    canvas = np.zeros((H, W), dtype=bool)
    for x0, y0, x1, y1 in boxes:
        rows = (yc >= y0) & (yc < y1)
        cols = (xc >= x0) & (xc < x1)
        canvas |= rows[:, None] & cols[None, :]
    return canvas.reshape(gh, supersample, gw, supersample).mean(axis=(1, 3)).astype(np.float32)


def mask_to_coverage(mask: np.ndarray, grid_hw: tuple[int, int]) -> np.ndarray:
    """Area-average a binary (H, W) mask onto the patch grid -> (h, w) float32 in [0, 1]."""
    from talosaur.data.synthetic import resize_bilinear

    gh, gw = grid_hw
    m = np.asarray(mask, dtype=np.float32)
    k = 8
    return np.clip(resize_bilinear(m, gh * k, gw * k).reshape(gh, k, gw, k).mean(axis=(1, 3)), 0, 1)


def patch_targets(coverage: np.ndarray, pos_thr: float = 0.3, neg_thr: float = 0.0) -> np.ndarray:
    """1 where coverage >= pos_thr, 0 where coverage <= neg_thr, -1 (ignore) in between."""
    t = np.full(coverage.shape, -1, dtype=np.int8)
    t[coverage >= pos_thr] = 1
    t[coverage <= neg_thr] = 0
    return t


def box_centroid_and_size(boxes, animal=None) -> tuple[float, float, float] | None:
    """Area-weighted centroid (cx, cy) and sqrt(total area) of the animal boxes, normalised."""
    bs = [b for b, a in zip(boxes, animal if animal is not None else [True] * len(boxes)) if a]
    if not bs:
        return None
    b = np.asarray(bs, dtype=np.float64)
    area = np.clip(b[:, 2] - b[:, 0], 0, 1) * np.clip(b[:, 3] - b[:, 1], 0, 1)
    if area.sum() <= 0:
        return None
    cx = float(((b[:, 0] + b[:, 2]) / 2 * area).sum() / area.sum())
    cy = float(((b[:, 1] + b[:, 3]) / 2 * area).sum() / area.sum())
    return cx, cy, float(np.sqrt(min(area.sum(), 1.0)))


def sample_presence_crops(
    boxes,
    is_animal,
    rng: np.random.Generator,
    n: int = 4,
    scale: tuple[float, float] = (0.35, 0.7),
    pos_frac: float = 0.5,
    aspect: float = 1.0,
) -> list[tuple[list[float], int]]:
    """Crop-level presence labels for datasets with boxes but no empty frames (FathomNet).

    Positive: an animal box has >= ``pos_frac`` of its area inside the crop. Negative: the crop
    overlaps no box at all (of any concept). Anything else is ambiguous and skipped.
    Returns [(crop_xyxy_normalised, label)].
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    animal = np.asarray(is_animal if is_animal is not None else [True] * len(b), dtype=bool)
    out = []
    for _ in range(n * 6):
        if len(out) >= n:
            break
        s = rng.uniform(*scale)
        cw, ch = s * np.sqrt(aspect), s / np.sqrt(aspect)
        if cw > 1 or ch > 1:
            continue
        x0, y0 = rng.uniform(0, 1 - cw), rng.uniform(0, 1 - ch)
        crop = np.array([x0, y0, x0 + cw, y0 + ch])
        if len(b) == 0:
            out.append((crop.tolist(), 0))
            continue
        ix0 = np.maximum(b[:, 0], crop[0])
        iy0 = np.maximum(b[:, 1], crop[1])
        ix1 = np.minimum(b[:, 2], crop[2])
        iy1 = np.minimum(b[:, 3], crop[3])
        inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
        area = np.clip(b[:, 2] - b[:, 0], 1e-9, None) * np.clip(b[:, 3] - b[:, 1], 1e-9, None)
        frac = inter / area
        if (animal & (frac >= pos_frac)).any():
            out.append((crop.tolist(), 1))
        elif (inter <= 0).all():
            out.append((crop.tolist(), 0))
    return out
