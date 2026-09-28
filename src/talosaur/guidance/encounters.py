"""Encounters: a time budget per animal, and a memory of animals already filmed.

An *encounter* starts when the state machine locks onto a new animal (ACQUIRE -> TRACK) and ends
when the animal is lost for good or its time budget (``max_s``) is used up; then the vehicle
releases it and searches for a different one.

"Different" is decided by **appearance**, not position (underwater position is unreliable): the
model's patch tokens are pooled over the animal's heatmap blob and centred on the running mean of
the background tokens, giving a descriptor whose cosine similarity to earlier animals says whether
this is one we have already filmed (``same_sim``). A recognised animal whose budget is spent is
ignored for ``cooldown_s``; one that was lost before its budget ran out is resumed with the time it
has left, so a fish that keeps swimming in and out of view is still filmed for at most ``max_s``.

Limits: look-alikes (a school of one species) are indistinguishable by appearance; the similarity
threshold must be calibrated on your own footage (replay logs every similarity); exports without
patch tokens fall back to "turn away and move on" with no memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EncounterConfig:
    max_s: float = 60.0  # time budget per animal (TRACK + FILM + LOST); 0 = follow indefinitely
    cooldown_s: float = 300.0  # how long a filmed animal is remembered
    same_sim: float = 0.8  # cosine similarity at or above which two sightings are the same animal
    reid: bool = True  # recognise animals by appearance (needs an export with patch tokens)
    bg_prob: float = 0.2  # patches below this animal probability count as background
    bg_alpha: float = 0.05  # background mean update rate per frame
    max_memory: int = 64


@dataclass
class Remembered:
    id: int
    desc: np.ndarray | None
    engaged_s: float
    t_last: float
    exhausted: bool


@dataclass
class Encounter:
    id: int
    t_start: float
    prior_s: float = 0.0  # time already spent on this animal in earlier segments
    resumed: bool = False
    film_s: float = 0.0
    frames: int = 0
    seen: int = 0
    max_prob: float = 0.0
    size_sum: float = 0.0
    best_score: float = -1.0
    best_t: float | None = None
    desc_sum: np.ndarray | None = field(default=None, repr=False)
    desc_n: int = 0

    def engaged_s(self, t: float) -> float:
        return self.prior_s + max(0.0, t - self.t_start)

    def descriptor(self) -> np.ndarray | None:
        if self.desc_sum is None or self.desc_n == 0:
            return None
        return _unit(self.desc_sum / self.desc_n)


def _unit(v: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


class EncounterManager:
    def __init__(self, cfg: EncounterConfig | None = None):
        self.cfg = cfg or EncounterConfig()
        self.memory: list[Remembered] = []
        self.current: Encounter | None = None
        self.next_id = 1
        self.bg: np.ndarray | None = None
        self._t_prev: float | None = None

    # ------------------------------------------------------------------ appearance

    def observe_background(self, tokens: np.ndarray, prob: np.ndarray) -> None:
        """Running mean of the tokens of patches that are not animal (the water / scene)."""
        m = prob < self.cfg.bg_prob
        if not m.any():
            return
        mean = tokens[m].mean(axis=0).astype(np.float64)
        self.bg = mean if self.bg is None else (1 - self.cfg.bg_alpha) * self.bg + self.cfg.bg_alpha * mean

    def describe(
        self, tokens: np.ndarray | None, prob: np.ndarray, mask: np.ndarray | None
    ) -> np.ndarray | None:
        """Unit appearance descriptor of one blob: probability-weighted mean of its tokens minus
        the background mean (without centring, all ViT tokens share a large common direction)."""
        if tokens is None or mask is None or not self.cfg.reid or not mask.any():
            return None
        w = prob[mask].astype(np.float64) ** 2
        v = (tokens[mask].astype(np.float64) * w[:, None]).sum(axis=0) / max(w.sum(), 1e-12)
        if self.bg is not None:
            v = v - self.bg
        return _unit(v)

    def _expire(self, t: float) -> None:
        self.memory = [r for r in self.memory if t - r.t_last <= self.cfg.cooldown_s][-self.cfg.max_memory :]

    def match(self, desc: np.ndarray | None, t: float) -> tuple[Remembered | None, float]:
        """Most similar remembered animal (and its cosine similarity), or (None, nan)."""
        self._expire(t)
        if desc is None:
            return None, float("nan")
        best, best_sim = None, -1.0
        for r in self.memory:
            if r.desc is not None:
                s = float(desc @ r.desc)
                if s > best_sim:
                    best, best_sim = r, s
        return (best, best_sim) if best is not None else (None, float("nan"))

    def is_done_with(self, desc: np.ndarray | None, t: float) -> tuple[bool, float]:
        """(True, sim) when this looks like an animal whose budget is already spent."""
        r, sim = self.match(desc, t)
        return bool(r is not None and r.exhausted and sim >= self.cfg.same_sim), sim

    # ------------------------------------------------------------------ lifecycle

    @property
    def active(self) -> bool:
        return self.current is not None

    def exhausted(self, t: float) -> bool:
        c = self.current
        return c is not None and self.cfg.max_s > 0 and c.engaged_s(t) >= self.cfg.max_s

    def remaining_s(self, t: float) -> float | None:
        if self.current is None or self.cfg.max_s <= 0:
            return None
        return max(0.0, self.cfg.max_s - self.current.engaged_s(t))

    def start(self, t: float, desc: np.ndarray | None) -> Encounter:
        """New encounter, or the continuation of a remembered animal that still has budget."""
        r, sim = self.match(desc, t)
        if r is not None and not r.exhausted and sim >= self.cfg.same_sim:
            self.memory.remove(r)
            enc = Encounter(r.id, t, prior_s=r.engaged_s, resumed=True)
            if r.desc is not None:
                enc.desc_sum, enc.desc_n = r.desc.copy(), 1
        else:
            enc = Encounter(self.next_id, t)
            self.next_id += 1
        self.current = enc
        self._t_prev = t
        return enc

    def observe(self, t: float, state: str, target, frame_prob: float, desc: np.ndarray | None) -> None:
        """Accumulate one frame of the current encounter (appearance, stats, best moment)."""
        enc = self.current
        if enc is None:
            return
        dt = 0.0 if self._t_prev is None else max(0.0, t - self._t_prev)
        self._t_prev = t
        enc.frames += 1
        if state == "FILM":
            enc.film_s += dt
        enc.max_prob = max(enc.max_prob, frame_prob)
        if target is not None and target.found:
            enc.seen += 1
            enc.size_sum += target.size
            centred = 1.0 - min(1.0, 2.0 * float(np.hypot(target.cx - 0.5, target.cy - 0.5)))
            score = frame_prob * centred * min(1.0, target.size / 0.2)
            if score > enc.best_score:
                enc.best_score, enc.best_t = score, t
            if desc is not None:
                enc.desc_sum = desc.copy() if enc.desc_sum is None else enc.desc_sum + desc
                enc.desc_n += 1

    def end(self, t: float, reason: str) -> dict:
        """Close the current encounter, remember the animal, return a summary for the log."""
        enc = self.current
        if enc is None:
            return {}
        engaged = enc.engaged_s(t)
        exhausted = reason == "budget" or (self.cfg.max_s > 0 and engaged >= self.cfg.max_s)
        desc = enc.descriptor()
        self.memory.append(Remembered(enc.id, desc, engaged, t, exhausted))
        self.current = None
        return {
            "id": enc.id,
            "reason": reason,
            "t_start": round(enc.t_start, 3),
            "t_end": round(t, 3),
            "segment_s": round(t - enc.t_start, 2),
            "engaged_s": round(engaged, 2),
            "film_s": round(enc.film_s, 2),
            "resumed": enc.resumed,
            "frames": enc.frames,
            "seen_frames": enc.seen,
            "max_frame_prob": round(enc.max_prob, 4),
            "mean_size": round(enc.size_sum / enc.seen, 4) if enc.seen else None,
            "best_t": None if enc.best_t is None else round(enc.best_t, 3),
            "appearance": desc is not None,
            "done": exhausted,
        }
