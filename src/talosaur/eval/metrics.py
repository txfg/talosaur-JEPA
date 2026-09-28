"""Classification and steering metrics (numpy only, no sklearn dependency)."""

from __future__ import annotations

import numpy as np

# Camera Module 3 Wide, in air (Raspberry Pi docs / retailer specs): 102 x 67 degrees.
DEFAULT_HFOV_DEG = 102.0
DEFAULT_VFOV_DEG = 67.0


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with ties."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = _rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="mergesort")
    yt = y[order]
    tp = np.cumsum(yt)
    precision = tp / np.arange(1, len(yt) + 1)
    return float((precision * yt).sum() / yt.sum())


def tpr_at_fpr(y: np.ndarray, score: np.ndarray, fpr: float = 0.05) -> float:
    """True-positive rate at the threshold where the false-positive rate is <= ``fpr``."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(score, dtype=np.float64)
    if y.sum() == 0 or (~y).sum() == 0:
        return float("nan")
    neg = np.sort(s[~y])[::-1]
    k = int(np.floor(fpr * len(neg)))
    thr = neg[k] if k < len(neg) else -np.inf
    return float((s[y] > thr).mean())


def balanced_accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y).astype(bool)
    p = np.asarray(pred).astype(bool)
    tpr = (p & y).sum() / max(1, y.sum())
    tnr = (~p & ~y).sum() / max(1, (~y).sum())
    return float(0.5 * (tpr + tnr))


_IOU_THRESHOLDS = tuple(np.linspace(0.05, 0.95, 19).tolist())


def best_iou(prob: np.ndarray, target: np.ndarray, thresholds=_IOU_THRESHOLDS) -> tuple[float, float]:
    """Best IoU over thresholds between probability maps and binary targets (flattened)."""
    p = np.asarray(prob).reshape(-1)
    t = np.asarray(target).reshape(-1).astype(bool)
    best, thr_best = 0.0, 0.5
    for thr in thresholds:
        m = p >= thr
        inter = (m & t).sum()
        union = (m | t).sum()
        iou = inter / union if union else 0.0
        if iou > best:
            best, thr_best = float(iou), float(thr)
    return best, thr_best


def soft_centroid(
    prob: np.ndarray, gamma: float = 2.0, floor: float = 0.1
) -> tuple[float, float, float] | None:
    """Probability-weighted centroid of an (h, w) heatmap in normalised image coords, plus
    sqrt(area fraction) of cells above 0.5. ``gamma`` sharpens, ``floor`` ignores weak cells."""
    p = np.asarray(prob, dtype=np.float64)
    h, w = p.shape
    wgt = np.clip(p - floor, 0, None) ** gamma
    if wgt.sum() <= 1e-12:
        return None
    ys, xs = np.mgrid[0:h, 0:w]
    cx = float(((xs + 0.5) / w * wgt).sum() / wgt.sum())
    cy = float(((ys + 0.5) / h * wgt).sum() / wgt.sum())
    size = float(np.sqrt((p >= 0.5).mean()))
    return cx, cy, size


def angular_error_deg(
    c_pred, c_true, hfov: float = DEFAULT_HFOV_DEG, vfov: float = DEFAULT_VFOV_DEG
) -> float:
    """Approximate bearing/elevation error in degrees for normalised image coordinates, as if the
    frame were seen through the Camera Module 3 Wide (linear angle mapping)."""
    dx = (c_pred[0] - c_true[0]) * hfov
    dy = (c_pred[1] - c_true[1]) * vfov
    return float(np.hypot(dx, dy))


def bootstrap_ci(
    values_fn, n: int, reps: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI of ``values_fn(indices)`` over ``n`` items."""
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        v = values_fn(idx)
        if v == v:  # skip NaN resamples (e.g. one class missing)
            stats.append(v)
    if not stats:
        return float("nan"), float("nan")
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))
