"""Heatmap -> target: connected animal blobs on the patch grid, sub-patch centroid, apparent size.

Pure numpy; grids are tiny (e.g. 13x7), so a simple union-find labelling is plenty fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Target:
    found: bool
    cx: float = 0.5  # normalised image x (0 = left edge, 1 = right edge)
    cy: float = 0.5  # normalised image y (0 = top, 1 = bottom)
    size: float = 0.0  # sqrt(area fraction of the blob)
    mass: float = 0.0  # sum of probabilities in the blob
    peak: float = 0.0  # max probability in the blob
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # normalised x0, y0, x1, y1
    n_blobs: int = 0
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)  # the blob's grid cells

    def as_dict(self) -> dict:
        return {
            "found": self.found,
            "cx": self.cx,
            "cy": self.cy,
            "size": self.size,
            "mass": self.mass,
            "peak": self.peak,
            "n_blobs": self.n_blobs,
        }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def smooth3x3(p: np.ndarray) -> np.ndarray:
    """Light [1 2 1] smoothing (edge-replicated) - suppresses single-patch speckle."""
    k = np.array([0.25, 0.5, 0.25])
    q = np.pad(p, 1, mode="edge")
    q = k[0] * q[:-2] + k[1] * q[1:-1] + k[2] * q[2:]
    return k[0] * q[:, :-2] + k[1] * q[:, 1:-1] + k[2] * q[:, 2:]


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected component labels (1..n) for a small boolean grid."""
    h, w = mask.shape
    parent = list(range(h * w))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    idx = np.arange(h * w).reshape(h, w)
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and mask[yy, xx]:
                a, b = find(idx[y, x]), find(idx[yy, xx])
                if a != b:
                    parent[max(a, b)] = min(a, b)
    labels = np.zeros((h, w), dtype=np.int32)
    roots: dict[int, int] = {}
    for y, x in zip(ys, xs):
        r = find(idx[y, x])
        labels[y, x] = roots.setdefault(r, len(roots) + 1)
    return labels, len(roots)


def find_blobs(
    prob: np.ndarray,
    thr: float = 0.5,
    min_mass: float = 0.6,
    gamma: float = 2.0,
    smooth: bool = False,
    thr_low: float | None = None,
) -> tuple[list[Target], float, int]:
    """All animal blobs in the heatmap (each with its grid ``mask``), the heatmap's peak, and the
    number of connected regions before the mass / seed checks.

    Hysteresis: a blob is the connected region with probability >= ``thr_low`` (default ``thr``)
    and must contain a cell >= ``thr``, so a small, distant animal covering one or two patches is
    still found, while its extent (and so its apparent size) is not cut short. ``min_mass`` (sum of
    probabilities in the blob) rejects single weak cells (marine snow, backscatter); persistence
    over time is left to the tracker and state machine. ``smooth`` applies a 3x3 blur first; it
    suppresses speckle but also halves the peak of a one-patch target, so it is off by default."""
    p = smooth3x3(prob) if smooth else prob
    h, w = p.shape
    lo = thr if thr_low is None else min(thr_low, thr)
    labels, n = label_components(p >= lo)
    ys, xs = np.mgrid[0:h, 0:w]
    blobs = []
    for k in range(1, n + 1):
        m = labels == k
        wgt = p[m] ** gamma
        mass = float(p[m].sum())
        if mass < min_mass or p[m].max() < thr:
            continue
        cx = float(((xs[m] + 0.5) / w * wgt).sum() / wgt.sum())
        cy = float(((ys[m] + 0.5) / h * wgt).sum() / wgt.sum())
        bbox = (xs[m].min() / w, ys[m].min() / h, (xs[m].max() + 1) / w, (ys[m].max() + 1) / h)
        size = float(np.sqrt(m.sum() / (h * w)))
        blobs.append(Target(True, cx, cy, size, mass, float(p[m].max()), bbox, n, m))
    return blobs, float(p.max()), n


def order_blobs(
    blobs: list[Target], prev_xy: tuple[float, float] | None = None, gate: float = 0.35
) -> list[Target]:
    """Preference order: the blob closest to the previous track position (within ``gate``, in
    normalised image units) first, then the rest by mass."""
    rest = sorted(blobs, key=lambda b: -b.mass)
    if prev_xy is None or not blobs:
        return rest
    near = min(blobs, key=lambda b: np.hypot(b.cx - prev_xy[0], b.cy - prev_xy[1]))
    if np.hypot(near.cx - prev_xy[0], near.cy - prev_xy[1]) > gate:
        return rest
    return [near] + [b for b in rest if b is not near]


def find_target(
    prob: np.ndarray,
    thr: float = 0.5,
    prev_xy: tuple[float, float] | None = None,
    gate: float = 0.35,
    min_mass: float = 0.6,
    gamma: float = 2.0,
    smooth: bool = False,
    thr_low: float | None = None,
) -> Target:
    """The preferred blob (see :func:`find_blobs` and :func:`order_blobs`), or ``found=False``."""
    blobs, peak, n = find_blobs(prob, thr, min_mass, gamma, smooth, thr_low)
    if not blobs:
        return Target(False, peak=peak, n_blobs=n)
    return order_blobs(blobs, prev_xy, gate)[0]
