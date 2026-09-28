"""Per-image quality / condition metrics and the dark / murky / clear buckets.

All metrics are computed on a small copy (long side ~256 px) and most are normalised by
mean luminance, so "murky" is not confused with "dark". The thresholds in
:class:`BucketThresholds` are starting points: calibrate them on ~300 hand-labelled images
from your own data with ``scripts/data/label_frames.py``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

_EPS = 1e-6


# --------------------------------------------------------------------------- colour helpers


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """sRGB in [0, 1] -> CIELAB (D65). Returns L in [0, 100], a*, b*."""
    lin = _srgb_to_linear(x)
    m = np.array(
        [[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]],
        dtype=np.float32,
    )
    xyz = lin @ m.T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    d = 6.0 / 29.0
    f = np.where(xyz > d**3, np.cbrt(xyz), xyz / (3 * d * d) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return L, a, b


# --------------------------------------------------------------------------- metrics


def compute_metrics(rgb: np.ndarray) -> dict[str, float]:
    """Condition metrics for one uint8 RGB image (ideally pre-resized to ~256 px long side)."""
    from scipy import ndimage as ndi

    x = rgb.astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    lum_mean = float(Y.mean())
    lum_p95 = float(np.percentile(Y, 95))
    contrast = float(Y.std())
    denom = lum_mean + 0.05
    norm_contrast = contrast / denom

    # Underwater dark-channel ratio (G, B only). Reported for the dataset statistics only: in
    # blue/green water it mostly tracks the water hue, so it is not part of the clarity score.
    win = max(3, int(round(min(Y.shape) / 28)) | 1)
    dark = ndi.minimum_filter(np.minimum(g, b), size=win)
    bright = ndi.maximum_filter(np.maximum(g, b), size=win)
    haze = float(dark.mean() / (bright.mean() + 1e-3))

    L, A, B = rgb_to_lab(x)
    am, bm = float(A.mean()), float(B.mean())
    cast_mag = math.hypot(am, bm)
    cast_hue = math.degrees(math.atan2(bm, am))
    red_ratio = float(r.mean() / (0.5 * (g.mean() + b.mean()) + 1e-3))

    lap = ndi.laplace(Y)
    sharpness = float(np.log10(lap.var() / denom**2 + 1e-9))

    # Immerkaer (1996) fast noise sigma estimate
    kern = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = ndi.convolve(Y, kern, mode="reflect")
    h, w = Y.shape
    noise = float(math.sqrt(math.pi / 2.0) * np.abs(conv[1:-1, 1:-1]).sum() / (6.0 * max(w - 2, 1) * max(h - 2, 1)))

    # UCIQE (Yang & Sowmya 2015) with chroma scaled by 1/100 and L by 1/100.
    chroma = np.hypot(A, B) / 100.0
    con_l = float((np.percentile(L, 99) - np.percentile(L, 1)) / 100.0)
    mx = x.max(axis=-1)
    sat = (mx - x.min(axis=-1)) / (mx + _EPS)
    uciqe = float(0.4680 * chroma.std() + 0.2745 * con_l + 0.2576 * sat.mean())

    # Structure measures on a ~64 px, median-filtered copy: area-averaging suppresses sensor
    # noise (dark frames) and the median removes marine snow, so only real content counts.
    f = max(1, int(round(min(Y.shape) / 64)))
    hh, ww = (Y.shape[0] // f) * f, (Y.shape[1] // f) * f
    Ys = Y[:hh, :ww].reshape(hh // f, f, ww // f, f).mean(axis=(1, 3))
    Ym = ndi.median_filter(Ys, size=3) / denom
    gy, gx = np.gradient(Ym)
    gmag = np.hypot(gx, gy)
    grad_energy = float(gmag.mean())
    edge_density = float((gmag > 0.1).mean())

    R, G, Bc = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    rg, yb = R - G, 0.5 * (R + G) - Bc
    colorfulness = float(math.hypot(rg.std(), yb.std()) + 0.3 * math.hypot(rg.mean(), yb.mean()))

    m = {
        "lum_mean": lum_mean,
        "lum_p95": lum_p95,
        "contrast": contrast,
        "norm_contrast": norm_contrast,
        "haze": haze,
        "cast_mag": cast_mag,
        "cast_hue": cast_hue,
        "red_ratio": red_ratio,
        "sharpness": sharpness,
        "noise": noise,
        "uciqe": uciqe,
        "grad_energy": grad_energy,
        "edge_density": edge_density,
        "colorfulness": colorfulness,
    }
    m["clarity"] = clarity_score(m)
    return m


def _unit(v: float, lo: float, hi: float) -> float:
    return float(min(1.0, max(0.0, (v - lo) / (hi - lo))))


def clarity_score(m: dict[str, float]) -> float:
    """0 (murky) .. 1 (clear), brightness-normalised. Fixed reference ranges keep the score
    comparable across datasets; the bucket thresholds are what gets calibrated.

    ``haze`` (the underwater dark-channel ratio) is reported but not used: in blue/green water
    it mostly tracks the water hue rather than turbidity.
    """
    c1 = _unit(m["norm_contrast"], 0.03, 0.30)
    c2 = _unit(m["sharpness"], -4.0, -1.0)
    c3 = _unit(m["edge_density"], 0.0, 0.05)
    c4 = _unit(m["uciqe"], 0.18, 0.40)
    return 0.40 * c1 + 0.30 * c2 + 0.15 * c3 + 0.15 * c4


# --------------------------------------------------------------------------- buckets


@dataclass
class BucketThresholds:
    dark_lum: float = 0.10  # mean luminance below => "dark"
    dim_lum: float = 0.25  # below => "dim", else "bright"
    murky_clarity: float = 0.35  # clarity below => "murky"
    clear_clarity: float = 0.55  # clarity at/above => "clear", in between => "moderate"
    empty_grad: float = 0.012  # grad_energy below AND ...
    empty_edges: float = 0.01  # ... edge_density below => candidate "empty open water"

    def to_dict(self) -> dict:
        return asdict(self)


def light_bucket(lum_mean: float, t: BucketThresholds) -> str:
    if lum_mean < t.dark_lum:
        return "dark"
    return "dim" if lum_mean < t.dim_lum else "bright"


def clarity_bucket(clarity: float, t: BucketThresholds) -> str:
    if clarity < t.murky_clarity:
        return "murky"
    return "clear" if clarity >= t.clear_clarity else "moderate"


def eval_slice(light: str, clarity: str) -> str:
    """The three evaluation slices from the brief. Dark takes precedence."""
    if light == "dark":
        return "dark"
    if clarity == "murky":
        return "murky"
    if clarity == "clear":
        return "clear"
    return "other"


def is_empty_water(m: dict[str, float], t: BucketThresholds) -> bool:
    return m["grad_energy"] < t.empty_grad and m["edge_density"] < t.empty_edges


def assign_buckets(df, t: BucketThresholds):
    """Add ``light``, ``clarity_bucket`` and ``slice`` columns to a metrics DataFrame (in place)."""
    df["light"] = [light_bucket(v, t) for v in df["lum_mean"].to_numpy()]
    df["clarity_bucket"] = [clarity_bucket(v, t) for v in df["clarity"].to_numpy()]
    df["slice"] = [eval_slice(a, b) for a, b in zip(df["light"], df["clarity_bucket"])]
    return df


def calibrate_thresholds(df, t: BucketThresholds | None = None, light_q=(0.15, 0.45), clarity_q=(0.3, 0.7)):
    """Percentile-based thresholds for a pool with no labels (use label-based calibration when
    you have labels; see ``fit_thresholds_from_labels``)."""
    t = BucketThresholds(**(t.to_dict() if t else {}))
    t.dark_lum, t.dim_lum = (float(np.quantile(df["lum_mean"], q)) for q in light_q)
    t.murky_clarity, t.clear_clarity = (float(np.quantile(df["clarity"], q)) for q in clarity_q)
    return t


def fit_thresholds_from_labels(metrics: dict[str, np.ndarray], light_labels, clarity_labels, t=None):
    """Grid-search thresholds maximising agreement with hand labels.

    ``light_labels`` in {dark, dim, bright}, ``clarity_labels`` in {murky, moderate, clear};
    entries may be None (not labelled). Returns (thresholds, agreement dict).
    """
    t = BucketThresholds(**(t.to_dict() if t else {}))
    lum = np.asarray(metrics["lum_mean"], dtype=np.float64)
    cla = np.asarray(metrics["clarity"], dtype=np.float64)
    ll = np.asarray(light_labels, dtype=object)
    cl = np.asarray(clarity_labels, dtype=object)

    def best(values, labels, names):
        mask = np.array([lab is not None for lab in labels])
        if mask.sum() < 5:
            return None, float("nan")
        v, lab = values[mask], labels[mask]
        grid = np.unique(np.quantile(v, np.linspace(0.01, 0.99, 99)))
        best_acc, best_pair = -1.0, None
        for i, a in enumerate(grid):
            for b in grid[i:]:
                pred = np.where(v < a, names[0], np.where(v < b, names[1], names[2]))
                acc = float((pred == lab).mean())
                if acc > best_acc:
                    best_acc, best_pair = acc, (float(a), float(b))
        return best_pair, best_acc

    pair, acc_l = best(lum, ll, ("dark", "dim", "bright"))
    if pair:
        t.dark_lum, t.dim_lum = pair
    pair, acc_c = best(cla, cl, ("murky", "moderate", "clear"))
    if pair:
        t.murky_clarity, t.clear_clarity = pair
    return t, {"light_agreement": acc_l, "clarity_agreement": acc_c}


def compute_metrics_for_path(path: str, max_side: int = 256) -> dict[str, float]:
    from talosaur.data.dedup import phash64, thumbnail
    from talosaur.data.imageio import load_rgb

    rgb = load_rgb(path, max_side=max_side)
    m = compute_metrics(rgb)
    m["phash"] = int(phash64(rgb))
    m["thumb"] = thumbnail(rgb).tobytes()
    return m
