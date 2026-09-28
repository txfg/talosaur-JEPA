"""Near-duplicate detection: 64-bit perceptual hashes + standardised thumbnails.

Why two signals: a pHash alone happily merges a dark, featureless open-water frame with the
same frame containing one small fish (low DCT frequencies barely move). The thumbnail check
compares *standardised* 24x24 thumbnails with a robust max-difference, so a small new
object keeps two frames apart while sensor noise and small camera motion do not.

Clustering uses bit-sampling LSH over the hashes (several tables of 16 sampled bits), then
verifies candidate pairs with the exact Hamming distance and the thumbnail check, then
union-find. numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

THUMB = 24


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2.0)
    return m.astype(np.float32)


_DCT32 = _dct_matrix(32)
_BIT_WEIGHTS = (np.uint64(1) << np.arange(64, dtype=np.uint64)).astype(np.uint64)


def to_gray(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    if x.max() > 1.5:
        x /= 255.0
    if x.ndim == 3:
        x = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    return x


def _area_resize(g: np.ndarray, h: int, w: int) -> np.ndarray:
    from talosaur.data.synthetic import box_blur, resize_bilinear

    k = max(1, int(round(min(g.shape[0] / h, g.shape[1] / w))))
    if k > 1:
        g = box_blur(g, k if k % 2 == 1 else k + 1)
    return resize_bilinear(g, h, w)


def phash64(img: np.ndarray) -> np.uint64:
    """DCT perceptual hash (32x32 -> top-left 8x8 vs median, DC excluded from the median)."""
    g = _area_resize(to_gray(img), 32, 32)
    d = (_DCT32 @ g @ _DCT32.T)[:8, :8].reshape(-1)
    med = np.median(d[1:])
    bits = (d > med).astype(np.uint64)
    return np.uint64(np.sum(bits * _BIT_WEIGHTS, dtype=np.uint64))


def thumbnail(img: np.ndarray, size: int = THUMB) -> np.ndarray:
    """Standardised grayscale thumbnail quantised to uint8 (z-scores clipped to +-4), followed by
    the image's mean R, G, B (3 bytes). Standardisation makes the structure comparison
    brightness-invariant; the mean colour stops a murky and a dark frame with similar layout
    from being called duplicates."""
    x = img.astype(np.float32)
    if x.max() > 1.5:
        x /= 255.0
    g = _area_resize(to_gray(x), size, size)
    z = (g - g.mean()) / (g.std() + 0.02)  # +0.02 keeps pure noise in dark frames from exploding
    zq = np.clip(np.round(z * 32.0 + 128.0), 0, 255).astype(np.uint8).reshape(-1)
    col = x.reshape(-1, x.shape[-1]).mean(axis=0) if x.ndim == 3 else np.full(3, g.mean())
    colq = np.clip(np.round(np.asarray(col)[:3] * 255.0), 0, 255).astype(np.uint8)
    return np.concatenate([zq, colq])


def hamming(a, b) -> np.ndarray:
    """Popcount of XOR, vectorised over uint64 arrays."""
    x = np.bitwise_xor(np.asarray(a, dtype=np.uint64), np.asarray(b, dtype=np.uint64))
    x = np.atleast_1d(x)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def thumb_distance(t1: np.ndarray, t2: np.ndarray, q: float = 99.5) -> np.ndarray:
    """Robust max difference of standardised thumbnails in z units, or the mean-colour difference
    (8 grey levels per unit) if that is larger. Rows = pairs."""
    n = THUMB * THUMB
    d = np.abs(t1.astype(np.int16) - t2.astype(np.int16)).astype(np.float32)
    struct = np.percentile(d[..., :n] / 32.0, q, axis=-1)
    color = d[..., n : n + 3].max(axis=-1) / 8.0 if d.shape[-1] > n else 0.0
    return np.maximum(struct, color)


@dataclass
class DedupConfig:
    max_hamming: int = 6
    max_thumb_dist: float = 1.2
    n_tables: int = 16
    bits_per_table: int = 16
    max_bucket: int = 2000
    neighbors: int = 64
    seed: int = 0


class _UnionFind:
    def __init__(self, n: int):
        self.p = np.arange(n)

    def find(self, i: int) -> int:
        p = self.p
        root = i
        while p[root] != root:
            root = p[root]
        while p[i] != root:
            p[i], i = root, p[i]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if ra < rb:
                self.p[rb] = ra
            else:
                self.p[ra] = rb

    def labels(self) -> np.ndarray:
        return np.array([self.find(i) for i in range(len(self.p))])


def _candidate_pairs(hashes: np.ndarray, cfg: DedupConfig) -> np.ndarray:
    n = len(hashes)
    rng = np.random.default_rng(cfg.seed)
    pairs: list[np.ndarray] = []
    for _ in range(cfg.n_tables):
        bits = rng.choice(64, cfg.bits_per_table, replace=False).astype(np.uint64)
        key = np.zeros(n, dtype=np.uint64)
        for j, b in enumerate(bits):
            key |= ((hashes >> b) & np.uint64(1)) << np.uint64(j)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        ends = np.r_[starts[1:], n]
        for s, e in zip(starts, ends):
            m = e - s
            if m < 2:
                continue
            members = order[s:e]
            if m <= cfg.max_bucket:
                ii, jj = np.triu_indices(m, 1)
                pairs.append(np.stack([members[ii], members[jj]], 1))
            else:  # sorted-neighbourhood fallback keeps huge buckets (dark frames) tractable
                members = members[np.argsort(hashes[members], kind="stable")]
                for off in range(1, cfg.neighbors + 1):
                    pairs.append(np.stack([members[:-off], members[off:]], 1))
    if not pairs:
        return np.zeros((0, 2), dtype=np.int64)
    p = np.concatenate(pairs).astype(np.int64)
    p.sort(axis=1)
    return np.unique(p, axis=0)


def near_duplicate_clusters(
    hashes: np.ndarray, thumbs: np.ndarray | None = None, cfg: DedupConfig | None = None, chunk: int = 200_000
) -> np.ndarray:
    """Cluster ids (the smallest member index) for near-duplicate groups."""
    cfg = cfg or DedupConfig()
    hashes = np.asarray(hashes, dtype=np.uint64)
    n = len(hashes)
    uf = _UnionFind(n)
    if n < 2:
        return np.arange(n)
    cand = _candidate_pairs(hashes, cfg)
    for s in range(0, len(cand), chunk):
        c = cand[s : s + chunk]
        ok = hamming(hashes[c[:, 0]], hashes[c[:, 1]]) <= cfg.max_hamming
        if thumbs is not None and ok.any():
            idx = np.flatnonzero(ok)
            td = thumb_distance(thumbs[c[idx, 0]], thumbs[c[idx, 1]])
            ok[idx] = td <= cfg.max_thumb_dist
        for a, b in c[ok]:
            uf.union(int(a), int(b))
    return uf.labels()


def is_near_duplicate(h1, t1, h2, t2, cfg: DedupConfig | None = None) -> bool:
    cfg = cfg or DedupConfig()
    if int(hamming(h1, h2)[0]) > cfg.max_hamming:
        return False
    return float(thumb_distance(t1[None], t2[None])[0]) <= cfg.max_thumb_dist
