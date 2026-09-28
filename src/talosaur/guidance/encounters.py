"""Encounters: how long to film each animal, and a memory of animals already filmed.

An *encounter* starts when the state machine locks onto a new animal (ACQUIRE -> TRACK) and ends
when the animal is lost for good or the vehicle decides to leave it and search for another.

**When to leave** (``rule: mvt``, the default) - the marginal value theorem of foraging theory,
applied to footage (docs/SEARCH.md):

* footage of one animal has diminishing returns: value(g) = w * (1 - exp(-g / tau_s)), where g is
  the seconds it was well framed and w > 1 for an animal unlike those already filmed;
* the *expected* marginal value rate is (fraction of recent time well framed) * w *
  exp(-g / tau_s) / tau_s;
* the long-run rate the mission earns (search + filming) is estimated online, R = total value /
  total time (with a prior);
* leave once the expected marginal rate falls below R (after ``min_s``). Where animals are rare
  this films longer; where they are plentiful, shorter; badly framed animals are left sooner.

Also: leave an animal that is fleeing (apparent size shrinking for ``flee_s`` while the vehicle
is not backing off: chasing it only disturbs it), leave if there was no good shot within
``giveup_s`` (or the rule gives up on an animal never well framed), and never exceed ``max_s``.
``rule: fixed`` keeps just the ``max_s`` budget.

Novelty is judged against every animal filmed so far this run (up to ``archive_size``), not only
the ones still remembered for the cooldown.

**"Different animal"** is decided by appearance, not position (underwater position is
unreliable): the model's patch tokens are pooled over the animal's heatmap blob and centred on the
running mean of the background tokens; the cosine similarity to earlier animals says whether this
one was already filmed (``same_sim``). An animal the vehicle left is ignored for ``cooldown_s``;
one that was only lost resumes with its footage so far (its diminishing returns continue).

Limits: look-alikes (a school of one species) are indistinguishable by appearance; ``same_sim``
and ``tau_s`` must be set from your own footage and goals; exports without patch tokens fall back
to "move on" without memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

LEAVE_REASONS = ("enough", "fled", "no_shot", "budget")  # the vehicle chose to leave


@dataclass
class EncounterConfig:
    rule: str = "mvt"  # mvt | fixed
    max_s: float = 180.0  # hard cap per animal (TRACK + FILM + LOST); 0 = none
    min_s: float = 10.0  # mvt: never leave sooner (unless it flees)
    tau_s: float = 30.0  # mvt: footage value per animal saturates over ~this many framed seconds
    novelty_bonus: float = 1.0  # mvt: value weight 1 + bonus * (1 - similarity to animals filmed)
    framed_window_s: float = 10.0  # mvt: memory of the "well framed" fraction
    prior_rate: float = 1.0 / 300.0  # mvt: prior long-run value per second (one animal per 5 min)
    prior_s: float = 1800.0  # mvt: weight of that prior, in seconds
    giveup_s: float = 30.0  # leave if no good shot at all by then
    flee_rate: float = -0.15  # d(log size)/dt below this = receding ...
    flee_s: float = 2.0  # ... for this long = fleeing: let it go
    good_center: float = 0.2  # well framed: |cx - 0.5| and |cy - 0.5| below this ...
    good_min_size: float = 0.02  # ... apparent size at least this ...
    good_min_prob: float = 0.5  # ... and "animal" probability at least this
    cooldown_s: float = 300.0  # how long a filmed animal is remembered
    same_sim: float = 0.8  # cosine similarity at or above which two sightings are the same animal
    reid: bool = True  # recognise animals by appearance (needs an export with patch tokens)
    bg_prob: float = 0.2  # patches below this animal probability count as background
    bg_alpha: float = 0.05  # background mean update rate per frame
    max_memory: int = 64
    archive_size: int = 512  # appearances of every animal filmed this run, for the novelty weight


@dataclass
class Remembered:
    id: int
    desc: np.ndarray | None
    engaged_s: float
    t_last: float
    exhausted: bool
    good_s: float = 0.0
    weight: float = 1.0


@dataclass
class Encounter:
    id: int
    t_start: float
    prior_s: float = 0.0  # time already spent on this animal in earlier segments
    resumed: bool = False
    weight: float = 1.0  # value weight (novelty)
    good_s: float = 0.0  # seconds well framed (all segments)
    good_s0: float = 0.0  # ... at the start of this segment
    framed_ema: float = 1.0  # recent fraction of time well framed (optimistic start: approaching)
    flee_s: float = 0.0
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

    def value(self, tau_s: float) -> float:
        return self.weight * (1.0 - math.exp(-self.good_s / tau_s))

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
        if self.cfg.rule not in ("mvt", "fixed"):
            raise ValueError(f"encounter.rule must be mvt or fixed, not {self.cfg.rule!r}")
        self.memory: list[Remembered] = []
        self.current: Encounter | None = None
        self.next_id = 1
        self.bg: np.ndarray | None = None
        self._t_prev: float | None = None
        self.t0: float | None = None  # mission clock start
        self.value_done = 0.0  # value of finished encounters
        self.archive: list[np.ndarray] = []  # appearance of every animal filmed (novelty)

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

    def familiarity(self, desc: np.ndarray | None) -> float:
        """Highest cosine similarity (>= 0) to any animal filmed this run; 0 when nothing is
        filmed yet or appearance is unavailable."""
        if desc is None or not self.archive:
            return 0.0
        return max(0.0, float((np.stack(self.archive) @ desc).max()))

    def is_done_with(self, desc: np.ndarray | None, t: float) -> tuple[bool, float]:
        """(True, sim) when this looks like an animal the vehicle already left."""
        r, sim = self.match(desc, t)
        return bool(r is not None and r.exhausted and sim >= self.cfg.same_sim), sim

    # ------------------------------------------------------------------ rates

    def tick(self, t: float) -> None:
        """Call every frame: starts the mission clock for the long-run rate."""
        if self.t0 is None:
            self.t0 = t

    def long_run_rate(self, t: float) -> float:
        """Value earned per second over the mission so far (search + filming), with a prior."""
        c = self.cfg
        elapsed = 0.0 if self.t0 is None else max(0.0, t - self.t0)
        return (self.value_done + c.prior_rate * c.prior_s) / (elapsed + c.prior_s)

    def marginal_rate(self) -> float:
        """Expected value per second of filming the current animal a little longer."""
        e, c = self.current, self.cfg
        if e is None:
            return 0.0
        return e.framed_ema * e.weight * math.exp(-e.good_s / c.tau_s) / c.tau_s

    # ------------------------------------------------------------------ lifecycle

    @property
    def active(self) -> bool:
        return self.current is not None

    def exhausted(self, t: float) -> bool:
        """The hard time budget is used up."""
        c = self.current
        return c is not None and self.cfg.max_s > 0 and c.engaged_s(t) >= self.cfg.max_s

    def leave_reason(self, t: float) -> str | None:
        """Why the vehicle should leave the current animal now, or None to keep filming."""
        e, c = self.current, self.cfg
        if e is None:
            return None
        if self.exhausted(t):
            return "budget"
        if c.rule == "fixed":
            return None
        if e.flee_s >= c.flee_s:
            return "fled"
        engaged = e.engaged_s(t)
        if engaged >= c.giveup_s and e.good_s < 1.0:
            return "no_shot"
        if engaged >= c.min_s and self.marginal_rate() < self.long_run_rate(t):
            return "enough" if e.good_s >= 1.0 else "no_shot"  # "enough" of nothing is no shot
        return None

    def remaining_s(self, t: float) -> float | None:
        if self.current is None or self.cfg.max_s <= 0:
            return None
        return max(0.0, self.cfg.max_s - self.current.engaged_s(t))

    def start(self, t: float, desc: np.ndarray | None) -> Encounter:
        """New encounter, or the continuation of a remembered animal that was only lost."""
        c = self.cfg
        r, sim = self.match(desc, t)
        if r is not None and not r.exhausted and sim >= c.same_sim:
            self.memory.remove(r)
            self.value_done -= r.weight * (1.0 - math.exp(-r.good_s / c.tau_s))  # counted again at the end
            enc = Encounter(
                r.id, t, prior_s=r.engaged_s, resumed=True, weight=r.weight, good_s=r.good_s, good_s0=r.good_s
            )
            if r.desc is not None:
                enc.desc_sum, enc.desc_n = r.desc.copy(), 1
        else:
            weight = 1.0 + c.novelty_bonus * (1.0 - self.familiarity(desc)) if desc is not None else 1.0
            enc = Encounter(self.next_id, t, weight=weight)
            if desc is not None:  # its appearance so far, even if it is gone at once
                enc.desc_sum, enc.desc_n = desc.copy(), 1
            self.next_id += 1
        self.current = enc
        self._t_prev = t
        return enc

    def observe(
        self,
        t: float,
        state: str,
        target,
        frame_prob: float,
        desc: np.ndarray | None,
        track=None,
        own_surge: float = 0.0,
    ) -> None:
        """Accumulate one frame of the current encounter (footage value, flight, appearance).
        ``own_surge``: the vehicle's current surge command - backing off also shrinks the animal."""
        enc, c = self.current, self.cfg
        if enc is None:
            return
        dt = 0.0 if self._t_prev is None else max(0.0, t - self._t_prev)
        self._t_prev = t
        enc.frames += 1
        if state == "FILM":
            enc.film_s += dt
        enc.max_prob = max(enc.max_prob, frame_prob)
        found = target is not None and target.found
        framed = (
            found
            and frame_prob >= c.good_min_prob
            and abs(target.cx - 0.5) < c.good_center
            and abs(target.cy - 0.5) < c.good_center
            and target.size >= c.good_min_size
        )
        if framed:
            enc.good_s += dt
        a = 1.0 - math.exp(-dt / c.framed_window_s) if dt > 0 else 0.0
        enc.framed_ema += a * ((1.0 if framed else 0.0) - enc.framed_ema)
        receding = track is not None and track.active and track.size_rate < c.flee_rate and own_surge >= 0.0
        enc.flee_s = enc.flee_s + dt if receding else 0.0
        if found:
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
        enc, c = self.current, self.cfg
        if enc is None:
            return {}
        engaged = enc.engaged_s(t)
        exhausted = reason in LEAVE_REASONS or (c.max_s > 0 and engaged >= c.max_s)
        desc = enc.descriptor()
        value = enc.value(c.tau_s)
        self.value_done += value
        self.memory.append(Remembered(enc.id, desc, engaged, t, exhausted, enc.good_s, enc.weight))
        if desc is not None and not enc.resumed:  # a resumed animal is archived already
            self.archive = (self.archive + [desc])[-self.cfg.archive_size :]
        self.current = None
        return {
            "id": enc.id,
            "reason": reason,
            "t_start": round(enc.t_start, 3),
            "t_end": round(t, 3),
            "segment_s": round(t - enc.t_start, 2),
            "engaged_s": round(engaged, 2),
            "good_s": round(enc.good_s, 2),
            "value": round(value, 3),
            "novelty_weight": round(enc.weight, 3),
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
