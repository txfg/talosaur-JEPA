"""I-JEPA multi-block masking, vectorised and grid-agnostic (square or 16:9 patch grids).

Per batch (block *sizes* are shared across the batch, as in I-JEPA):
  * ``npred`` target blocks of scale ``pred_scale`` and aspect ratio ``aspect``;
  * one context block of scale ``enc_scale``, minus the union of the targets;
  * context index lists are cut to the batch minimum so they can be stacked (>= ``min_keep``).

Defaults are the values in every shipped I-JEPA config (the collator class defaults differ).
``official_compat=True`` reproduces the reference collator's behaviour exactly:
  - one random draw sets both scale and aspect ratio of a block (correlated);
  - block top/left are drawn from ``[0, H - h)`` so blocks never touch the bottom/right edge,
    and block sizes are capped at ``H - 1`` / ``W - 1``;
  - the context block is square and truncated context masks keep the first (raster-order) indices.
With ``official_compat=False`` (default) these are fixed: independent scale/aspect draws, blocks
may touch every edge (on a 10x10 grid the official rule never predicts ~19% of patches), the
context block follows the grid's aspect ratio, and truncation keeps a random subset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class MaskConfig:
    enc_scale: tuple[float, float] = (0.85, 1.0)
    pred_scale: tuple[float, float] = (0.15, 0.2)
    aspect: tuple[float, float] = (0.75, 1.5)
    npred: int = 4
    nenc: int = 1
    min_keep: int = 10
    allow_overlap: bool = False
    official_compat: bool = False
    max_tries: int = 20


class MultiBlockMasks:
    def __init__(self, cfg: MaskConfig | None = None, **overrides):
        cfg = cfg or MaskConfig()
        for k, v in overrides.items():
            setattr(cfg, k, v)
        if cfg.nenc != 1:
            raise NotImplementedError("nenc=1 (one context block) as in all official I-JEPA configs")
        self.cfg = cfg

    # ------------------------------------------------------------------ block sizes
    def _block_size(
        self, gen: torch.Generator, H: int, W: int, scale, aspect, context: bool
    ) -> tuple[int, int]:
        c = self.cfg
        r1 = torch.rand(1, generator=gen).item()
        r2 = r1 if c.official_compat else torch.rand(1, generator=gen).item()
        s = scale[0] + r1 * (scale[1] - scale[0])
        n = H * W * s
        if context:
            ar = 1.0 if c.official_compat else H / W
        else:
            ar = aspect[0] + r2 * (aspect[1] - aspect[0])
        if c.official_compat:
            n = int(n)
            h = int(round(math.sqrt(n * ar)))
            w = int(round(math.sqrt(n / ar)))
            while h >= H:
                h -= 1
            while w >= W:
                w -= 1
            return max(h, 1), max(w, 1)
        h = min(max(int(round(math.sqrt(n * ar))), 1), H)
        w = min(max(int(round(math.sqrt(n / ar))), 1), W)
        return h, w

    def _positions(self, gen: torch.Generator, B: int, H: int, W: int, h: int, w: int):
        hi_t = H - h if self.cfg.official_compat else H - h + 1
        hi_l = W - w if self.cfg.official_compat else W - w + 1
        top = torch.randint(0, max(hi_t, 1), (B,), generator=gen)
        left = torch.randint(0, max(hi_l, 1), (B,), generator=gen)
        return top, left

    @staticmethod
    def _rect(top, left, h, w, H, W) -> torch.Tensor:
        rows = torch.arange(H)
        cols = torch.arange(W)
        r = (rows[None, :] >= top[:, None]) & (rows[None, :] < (top + h)[:, None])
        c = (cols[None, :] >= left[:, None]) & (cols[None, :] < (left + w)[:, None])
        return r[:, :, None] & c[:, None, :]  # (B, H, W) bool

    # ------------------------------------------------------------------ main
    def __call__(self, batch_size: int, grid_hw: tuple[int, int], generator: torch.Generator | None = None):
        """Returns (masks_enc: [LongTensor(B, Kc)], masks_pred: [LongTensor(B, Kt)] * npred)."""
        c = self.cfg
        gen = generator or torch.Generator().manual_seed(0)
        H, W = grid_hw
        B = batch_size
        ph, pw = self._block_size(gen, H, W, c.pred_scale, c.aspect, context=False)
        eh, ew = self._block_size(gen, H, W, c.enc_scale, (1.0, 1.0), context=True)

        preds = []
        union = torch.zeros(B, H, W, dtype=torch.bool)
        for _ in range(c.npred):
            t, lft = self._positions(gen, B, H, W, ph, pw)
            m = self._rect(t, lft, ph, pw, H, W)
            union |= m
            preds.append(m.flatten(1).nonzero()[:, 1].view(B, ph * pw))

        ctx = torch.zeros(B, H, W, dtype=torch.bool)
        todo = torch.ones(B, dtype=torch.bool)
        for _ in range(c.max_tries):
            t, lft = self._positions(gen, B, H, W, eh, ew)
            block = self._rect(t, lft, eh, ew, H, W)
            cand = block if c.allow_overlap else block & ~union
            ok = cand.flatten(1).sum(1) >= c.min_keep
            upd = todo & ok
            ctx[upd] = cand[upd]
            todo &= ~ok
            if not todo.any():
                break
        if todo.any():  # give up on the no-overlap constraint for these few samples
            t, lft = self._positions(gen, B, H, W, eh, ew)
            ctx[todo] = self._rect(t, lft, eh, ew, H, W)[todo]

        flat = ctx.flatten(1)
        k = int(flat.sum(1).min().item())
        if c.official_compat:
            score = -torch.arange(H * W, dtype=torch.float32).expand(B, -1).clone()  # first-k in raster order
        else:
            score = torch.rand(B, H * W, generator=gen)
        score = score.masked_fill(~flat, float("-inf"))
        enc = score.topk(k, dim=1).indices.sort(dim=1).values
        return [enc], preds


def coverage_stats(
    masks: MultiBlockMasks, grid_hw: tuple[int, int], n_batches: int = 50, batch_size: int = 64, seed: int = 0
):
    """How often each patch is a target / context, averaged over many batches (for tests/docs)."""
    H, W = grid_hw
    gen = torch.Generator().manual_seed(seed)
    tgt = torch.zeros(H * W)
    ctx = torch.zeros(H * W)
    for _ in range(n_batches):
        enc, preds = masks(batch_size, grid_hw, gen)
        for p in preds:
            tgt += torch.bincount(p.flatten(), minlength=H * W).float()
        ctx += torch.bincount(enc[0].flatten(), minlength=H * W).float()
    total = n_batches * batch_size
    return (tgt / total).view(H, W), (ctx / total).view(H, W)
