"""Fixed 2-D sine-cosine position embeddings for arbitrary (rectangular) patch grids.

Half of the channels encode the row, half the column; each half uses sin/cos at geometrically
spaced frequencies (the standard Transformer formulation applied per axis). Because nothing
is learned, one encoder can run on 7x7, 10x10, 14x14 or 13x7 (16:9) grids.
"""

from __future__ import annotations

import numpy as np


def _sincos_1d(dim: int, pos: np.ndarray) -> np.ndarray:
    if dim % 2:
        raise ValueError("embedding dim per axis must be even")
    omega = 1.0 / (10000.0 ** (np.arange(dim // 2, dtype=np.float64) / (dim / 2.0)))
    out = np.outer(pos.reshape(-1).astype(np.float64), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def sincos_2d(dim: int, grid_h: int, grid_w: int) -> np.ndarray:
    """(grid_h * grid_w, dim) float32, row-major over the grid (matches patch flattening)."""
    if dim % 4:
        raise ValueError("embedding dim must be divisible by 4")
    ys, xs = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing="ij")
    emb = np.concatenate([_sincos_1d(dim // 2, ys), _sincos_1d(dim // 2, xs)], axis=1)
    return emb.astype(np.float32)


def sincos_3d(dim: int, grid_t: int, grid_h: int, grid_w: int) -> np.ndarray:
    """(t * h * w, dim): spatial sin-cos on 2/3 of the channels, temporal on 1/3 (rounded to 4s)."""
    dt = (dim // 3) // 4 * 4
    ds = dim - dt
    sp = sincos_2d(ds, grid_h, grid_w)  # (h*w, ds)
    tp = _sincos_1d(dt, np.arange(grid_t)).astype(np.float32)  # (t, dt)
    sp = np.tile(sp[None], (grid_t, 1, 1))
    tp = np.tile(tp[:, None, :], (1, grid_h * grid_w, 1))
    return np.concatenate([sp, tp], axis=-1).reshape(grid_t * grid_h * grid_w, dim).astype(np.float32)
