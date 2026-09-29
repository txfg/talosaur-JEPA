"""Curiosity: turn toward things worth a closer look while searching (docs/TWILIGHT_ZONE.md §7.4).

Two cheap cues:

* **Bioluminescent flashes.** Most twilight-zone animals make light, and touch or turbulence -
  the vehicle's own wake included - sets it off. A flash is a brief, local brightening of the
  picture's blue-green brightness, found by comparing each cell of a coarse grid with its own
  recent history. Bioluminescence is blue-green, and a far-red lamp adds little there; a steady
  lamp or a slow exposure change brightens the whole picture, which is discounted.
* **Weak detections.** Something the model scores below its detection threshold, but that stays
  in about the same place for a few frames. Single-frame specks of marine snow do not.

On a cue the vehicle turns toward its bearing and creeps closer for up to ``max_s``. If an animal
is confirmed, the state machine takes over (ACQUIRE); otherwise the search resumes and no new look
starts for ``cooldown_s``. The lamp is not changed: a light coming on is itself what makes
animals flee (lights.py).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from talosaur.guidance.heatmap import find_blobs

GLOW_GRID = (14, 26)  # rows, cols of the brightness grid (8 x 8 pixel cells at 112 x 208)


@dataclass
class CuriosityConfig:
    enabled: bool = True
    flashes: bool = True
    flash_k: float = 6.0  # a flash: a cell brighter than its recent level by this many std devs ...
    flash_min: float = 0.03  # ... and by at least this much (brightness 0-1) ...
    flash_max_share: float = 0.25  # ... in no more than this share of the cells (else global)
    flash_tau_s: float = 2.0  # memory of each cell's recent brightness
    weak: bool = True
    weak_thr: float = 0.3  # a weak detection: a blob of at least this "animal" probability ...
    weak_n: int = 2  # ... in this many of the last 3 frames ...
    weak_near: float = 0.15  # ... within this distance (image fractions)
    max_s: float = 12.0  # look this long at most ...
    lost_s: float = 2.0  # ... or until the cue has been gone this long
    cooldown_s: float = 8.0  # then no new look for this long
    creep: float = 0.15  # surge while closing in (only when roughly facing the cue)
    yaw_kp: float = 0.03  # command per degree of bearing
    pitch_kp: float = 0.03


def glow_grid(rgb: np.ndarray, grid: tuple[int, int] = GLOW_GRID) -> np.ndarray:
    """Blue-green brightness (0-1) of an RGB uint8 frame, averaged over a coarse grid of cells."""
    gh, gw = grid
    h, w = rgb.shape[:2]
    ch, cw = max(1, h // gh), max(1, w // gw)
    img = rgb[: ch * gh, : cw * gw, 1:3].astype(np.float32).mean(axis=2) / 255.0
    return img.reshape(gh, ch, gw, cw).mean(axis=(1, 3))


class FlashDetector:
    """Brief local brightenings of a brightness grid, against each cell's recent level."""

    def __init__(self, cfg: CuriosityConfig):
        self.cfg = cfg
        self.mean: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.t: float | None = None
        self.flashes = 0

    def update(self, t: float, grid: np.ndarray) -> tuple[float, float, float] | None:
        """-> (cx, cy, strength) of the strongest flash (image fractions, brightness), or None."""
        c = self.cfg
        g = np.asarray(grid, np.float64)
        if self.mean is None or self.mean.shape != g.shape:
            self.mean, self.var, self.t = g.copy(), np.full(g.shape, 1e-4), t
            return None
        dt = max(1e-3, t - (self.t if self.t is not None else t))
        self.t = t
        d = g - self.mean
        d_local = d - float(np.median(d))  # a steady lamp or an exposure change moves every cell
        z = d_local / np.sqrt(self.var + 1e-6)
        hit = (z > c.flash_k) & (d_local > c.flash_min)
        a = 1.0 - math.exp(-dt / c.flash_tau_s)
        quiet = ~hit  # a flash must not become the new normal
        self.mean[quiet] += a * d[quiet]
        self.var[quiet] += a * (np.minimum(d[quiet] ** 2, 25.0 * self.var[quiet]) - self.var[quiet])
        self.var = np.maximum(self.var, 1e-6)
        if not hit.any() or hit.mean() > c.flash_max_share:
            return None
        r, col = np.unravel_index(int(np.argmax(np.where(hit, z, -np.inf))), g.shape)
        self.flashes += 1
        gh, gw = g.shape
        return (col + 0.5) / gw, (r + 0.5) / gh, float(d_local[r, col])


class Curiosity:
    """Decides, frame by frame while searching, whether to go and look at something."""

    def __init__(self, cfg: CuriosityConfig | None = None):
        self.cfg = cfg or CuriosityConfig()
        self.flash = FlashDetector(self.cfg)
        self.looking = False
        self.cue: str | None = None  # flash | weak
        self.target: tuple[float, float] | None = None  # (cx, cy) image fractions
        self.started = 0.0
        self.last_cue = -math.inf
        self.cooldown_until = -math.inf
        self.looks = 0
        self._weak: deque[tuple[float, float] | None] = deque(maxlen=3)

    def pause(self, t: float, s: float) -> None:
        """No looks for ``s`` (e.g. while moving on from an animal just filmed)."""
        self.stop(t)
        self.cooldown_until = max(self.cooldown_until, t + s)

    def stop(self, t: float) -> None:
        if self.looking:
            self.looking = False
            self.cooldown_until = t + self.cfg.cooldown_s
        self.cue = self.target = None

    def _weak_cue(self, prob: np.ndarray) -> tuple[float, float] | None:
        c = self.cfg
        blobs, _, _ = find_blobs(prob, c.weak_thr, c.weak_thr, thr_low=0.7 * c.weak_thr)
        best = max(blobs, key=lambda b: b.peak) if blobs else None
        self._weak.append(None if best is None else (best.cx, best.cy))
        if best is None:
            return None
        near = sum(
            1
            for p in self._weak
            if p is not None and math.hypot(p[0] - best.cx, p[1] - best.cy) <= c.weak_near
        )
        return (best.cx, best.cy) if near >= c.weak_n else None

    def update(self, t: float, searching: bool, prob: np.ndarray | None, glow: np.ndarray | None):
        """Call every frame. Returns the (cx, cy) to look toward while a look is on, else None.
        Flash statistics and weak-detection history are kept up to date in every state."""
        c = self.cfg
        flash = self.flash.update(t, glow) if (c.enabled and c.flashes and glow is not None) else None
        weak = self._weak_cue(prob) if (c.enabled and c.weak and prob is not None) else None
        if not (c.enabled and searching):
            self.stop(t)
            return None
        cue, kind = (
            (flash[:2], "flash")
            if flash is not None
            else (weak, "weak")
            if weak is not None
            else (None, None)
        )
        if cue is not None and (self.looking or t >= self.cooldown_until):
            if not self.looking:
                self.looking, self.started = True, t
                self.looks += 1
            self.target, self.cue, self.last_cue = cue, kind, t
        if self.looking and (t - self.last_cue > c.lost_s or t - self.started > c.max_s):
            self.stop(t)
        return self.target if self.looking else None

    def status(self) -> dict:
        return {
            "looking": self.looking,
            "cue": self.cue,
            "looks": self.looks,
            "flashes": self.flash.flashes,
        }
