"""Unsupervised "something new in view" score from the pooled embedding.

Keeps an exponential running mean/covariance of recent embeddings (the background the vehicle
has been seeing) and scores the current frame by its shrunk Mahalanobis distance, normalised by
a running median of past scores. Useful in SEARCH for animals the classifier was never trained
on. Updates are skipped while a target is being tracked so the animal doesn't become background.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class NoveltyDetector:
    def __init__(
        self,
        dim: int,
        alpha: float = 0.02,
        shrink: float = 0.1,
        warmup: int = 30,
        history: int = 200,
        refresh_every: int = 10,
    ):
        """``refresh_every``: re-invert the covariance after this many updates (it drifts slowly;
        a 192x192 inverse per frame would cost several ms on the Pi)."""
        self.mean = np.zeros(dim)
        self.cov = np.eye(dim)
        self.alpha, self.shrink, self.warmup, self.refresh_every = alpha, shrink, warmup, refresh_every
        self.n = 0
        self.scores: deque[float] = deque(maxlen=history)
        self._inv = np.eye(dim)
        self._stale = refresh_every  # updates since the last inverse (>= refresh_every -> recompute)

    def score(self, e: np.ndarray, update: bool = True) -> float:
        e = np.asarray(e, dtype=np.float64).reshape(-1)
        if self.n < self.warmup:
            if update:
                self._update(e)
            return 0.0
        if self._stale >= self.refresh_every:
            d = self.cov.shape[0]
            c = (1 - self.shrink) * self.cov + self.shrink * np.trace(self.cov) / d * np.eye(d)
            self._inv = np.linalg.inv(c)
            self._stale = 0
        diff = e - self.mean
        raw = float(np.sqrt(max(diff @ self._inv @ diff, 0.0)))
        med = float(np.median(self.scores)) if self.scores else raw
        self.scores.append(raw)
        if update:
            self._update(e)
        return raw / (med + 1e-9)

    def _update(self, e: np.ndarray) -> None:
        a = 1.0 / (self.n + 1) if self.n < self.warmup else self.alpha
        diff = e - self.mean
        self.mean = self.mean + a * diff
        self.cov = (1 - a) * self.cov + a * np.outer(diff, diff)
        self.n += 1
        self._stale += 1
