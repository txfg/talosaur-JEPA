"""Search planning: where to look when no animal is in view (SEARCH state).

Built from what stays reliable underwater - depth, heading and time - never from geographic x/y
(dead reckoning drifts; see docs/SEARCH.md for the evidence behind each rule):

* **Vertical: find the layer, then work it.** Mesopelagic animals live in depth layers that move
  with the time of day (diel vertical migration; in the northern Gulf of Mexico the 100-200 m band
  is mostly a corridor crossed at dusk and dawn). The planner first profiles the allowed depth
  range, then chooses depth bands by Thompson sampling on each band's detection rate (Gamma-Poisson,
  with forgetting because layers move), with an optional time-of-day prior.
* **Horizontal: composite search.** Long straight relocation legs ("extensive") at a slow speed;
  after any detection, a tight local search ("intensive": short legs, sharper turns) that gives
  up after ``giveup_s`` without a new detection - for targets that stay where they were found
  this beats Levy walks. Leg headings avoid water already searched: a coarse map in the *water*
  frame (heading + commanded speed; animals drift with the same water), blurred and decayed.
* **No navigation input:** scan (slow turn) and hop (short straight transit), no depth control.

The planner returns heading / depth setpoints (for an autopilot with heading / depth hold) plus
rate commands that steer toward them, so either kind of bridge works.
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass

import numpy as np

from talosaur.guidance.controller import Command
from talosaur.guidance.nav import NavState, wrap180


@dataclass
class SearchConfig:
    # adaptive = depth bands + composite horizontal search; depth = depth bands only (scan in
    # place); legs = horizontal legs only (no depth control); scan = the old behaviour (turn in
    # place at the current depth) - the last three are for comparisons in the simulator
    strategy: str = "adaptive"
    onset_gap_s: float = 3.0  # a detection counts as new only after this long without one
    # vertical
    min_depth_m: float = 20.0
    max_depth_m: float = 200.0
    start_depth_m: float = 50.0
    band_m: float = 10.0  # depth bins for detection statistics
    profile_first: bool = True  # sweep the whole range once before choosing bands
    stall_s: float = 60.0  # a depth not reached after this long without progress counts as reached ...
    min_altitude_m: float = 5.0  # ... e.g. the bottom: never ask to go closer to it (needs an altimeter)
    dwell_s: float = 180.0  # time in a chosen band before choosing again
    halflife_s: float = 1200.0  # forgetting of detection statistics (layers migrate)
    prior_per_min: float = 0.05  # prior detection rate per band
    prior_s: float = 60.0  # prior weight, in seconds of observation
    diel_prior: bool = True  # favour the migration corridor at dusk / dawn, shallow bands at night
    sunrise_h: float = 6.5  # at the dive site, in hours on the Pi's clock
    sunset_h: float = 18.5
    corridor_m: tuple[float, float] = (100.0, 200.0)
    night_depth_m: float = 60.0  # where the migrators are expected at night
    # horizontal
    surge: float = 0.35  # search speed (slow: fewer animals flee)
    leg_s: float = 90.0  # extensive: long relocation legs ...
    turn_deg: float = 110.0  # ... with large turns (never straight back)
    ars_leg_s: float = 15.0  # intensive (after a detection): short legs ...
    ars_turn_deg: float = 70.0  # ... sharper turns ...
    ars_surge: float = 0.2  # ... slower
    giveup_s: float = 120.0  # intensive search ends after this long without a detection
    speed_mps_per_unit: float = 0.6  # surge 1.0 in m/s (from the vehicle calibration)
    cell_m: float = 20.0  # water-frame coverage map resolution
    coverage_halflife_s: float = 900.0
    headings: int = 12  # candidate headings tried for a new leg
    # control
    heading_kp: float = 0.02  # yaw command per degree of heading error (bridges without heading hold)
    depth_kp: float = 0.15  # heave command per metre of depth error (bridges without depth hold)
    # no navigation input
    scan_s: float = 20.0
    hop_s: float = 30.0
    scan_yaw: float = 0.15
    seed: int = 0


class SearchPlanner:
    def __init__(self, cfg: SearchConfig | None = None):
        self.cfg = c = cfg or SearchConfig()
        self.rng = np.random.default_rng(c.seed)
        edges = np.arange(c.min_depth_m, c.max_depth_m + 1e-6, c.band_m)
        self.bands = (edges[:-1] + edges[1:]) / 2 if len(edges) > 1 else np.array([c.min_depth_m])
        n = len(self.bands)
        self.exposure = np.zeros(n)  # seconds spent in each band (decayed)
        self.events = np.zeros(n)  # detection onsets in each band (decayed)
        self.t_prev: float | None = None
        self.target_band: int | None = None
        self.band_until = 0.0
        self.profile = c.profile_first
        self.profile_dir = 1.0  # +1 = going down
        self.depth_lo, self.depth_hi = c.min_depth_m, c.max_depth_m  # the part the vehicle can reach
        self._goal_best = math.inf  # profile progress: closest approach to the current goal ...
        self._goal_t = 0.0  # ... and when it last improved
        self.floor_m: float | None = None  # deepest depth allowed by the altimeter, if any
        self.mode = "extensive"
        self.last_find = -1e9
        self.leg_heading: float | None = None
        self.leg_until = 0.0
        self.turn_sign = 1.0
        self.pos = np.zeros(2)  # water-frame dead reckoning (x east, y north), metres
        self.visited: dict[tuple[int, int], float] = {}
        self.last_detect_t = -1e9
        if c.strategy not in ("adaptive", "depth", "legs", "scan"):
            raise ValueError(f"unknown search strategy {c.strategy!r}")
        self.last_depth: float | None = None
        self.phase_until = 0.0  # no-nav scan / hop
        self.hopping = True  # flips on the first call: look around first, then move
        self.last_cmd = Command()
        self.hour_fn = None  # hour of the day; default: the computer's clock (the simulator overrides)

    # ------------------------------------------------------------------ bookkeeping

    def _band_of(self, depth: float) -> int:
        return int(
            np.clip(
                np.searchsorted(self.bands - self.cfg.band_m / 2, depth, side="right") - 1,
                0,
                len(self.bands) - 1,
            )
        )

    def observe(
        self,
        t: float,
        nav: NavState | None,
        detected: bool,
        surge_cmd: float,
        heading: float | None,
        searching: bool = True,
    ) -> None:
        """Every frame, in every state: water-frame dead reckoning and coverage; while
        ``searching`` (SEARCH / ACQUIRE), also the search time and new detections per depth band -
        time spent filming is not search time, and the animal being filmed is not a new find."""
        c = self.cfg
        dt = 0.0 if self.t_prev is None else max(0.0, t - self.t_prev)
        self.t_prev = t
        decay = 0.5 ** (dt / c.halflife_s) if c.halflife_s > 0 else 1.0
        self.exposure *= decay
        self.events *= decay
        depth = None if nav is None else nav.depth(t)
        onset = detected and t - self.last_detect_t > c.onset_gap_s  # flicker is not a new animal
        if detected:
            self.last_detect_t = t
        alt = None if nav is None or not nav.fresh(t) else nav.altitude_m
        self.floor_m = None if depth is None or alt is None else depth + alt - c.min_altitude_m
        if depth is not None:
            self.last_depth = depth
            if searching:
                b = self._band_of(depth)
                self.exposure[b] += dt
                if onset:
                    self.events[b] += 1.0
        if onset and searching:
            self.on_find(t)
        if heading is not None:
            v = surge_cmd * c.speed_mps_per_unit
            h = math.radians(heading)
            self.pos += v * dt * np.array([math.sin(h), math.cos(h)])
            self.visited[(int(self.pos[0] // c.cell_m), int(self.pos[1] // c.cell_m))] = t

    def on_find(self, t: float) -> None:
        """A detection (or the end of an encounter): search intensively around here."""
        self.last_find = t
        if self.mode != "intensive":
            self.mode = "intensive"
            self.leg_until = t  # start a short leg now

    # ------------------------------------------------------------------ vertical

    def _diel_prior(self, hour: float) -> np.ndarray:
        """Prior multiplier per band from the time of day (1 = neutral)."""
        c = self.cfg
        w = np.ones(len(self.bands))
        if not c.diel_prior:
            return w
        near = lambda h0: abs(((hour - h0) + 12) % 24 - 12) <= 1.5  # noqa: E731
        lo, hi = c.corridor_m
        if near(c.sunset_h) or near(c.sunrise_h):  # migrators crossing the corridor
            w[(self.bands >= lo) & (self.bands <= hi)] = 3.0
        elif not (c.sunrise_h <= hour < c.sunset_h):  # night: migrators shallow
            w *= np.exp(-0.5 * ((self.bands - c.night_depth_m) / 40.0) ** 2) * 2.0 + 0.5
        return w

    def rates_per_min(self) -> np.ndarray:
        c = self.cfg
        return 60.0 * (self.events + c.prior_per_min / 60.0 * c.prior_s) / (self.exposure + c.prior_s)

    def choose_band(self, t: float, hour: float | None = None) -> int:
        """Thompson sampling on the Gamma-Poisson posterior of each band's detection rate."""
        c = self.cfg
        if hour is None:
            hour = self.hour_fn() if self.hour_fn is not None else self.local_hour()
        prior_w = self._diel_prior(hour)
        shape = self.events + c.prior_per_min / 60.0 * c.prior_s * prior_w
        rate = self.exposure + c.prior_s
        samples = self.rng.gamma(np.maximum(shape, 1e-3), 1.0 / rate)
        half = c.band_m / 2
        reachable = (self.bands >= self.depth_lo - half) & (self.bands <= self.depth_hi + half)
        if reachable.any():
            samples = np.where(reachable, samples, -1.0)
        return int(np.argmax(samples))

    @staticmethod
    def local_hour() -> float:
        lt = _time.localtime()
        return lt.tm_hour + lt.tm_min / 60.0

    def _depth_target(self, t: float) -> float | None:
        c = self.cfg
        depth = self.last_depth
        if depth is None:
            return None
        target = None
        if self.profile:  # one sweep: down to the bottom of the range, then up to the top
            goal = c.max_depth_m if self.profile_dir > 0 else c.min_depth_m
            dist = abs(depth - goal)
            if dist < self._goal_best - 1.0:
                self._goal_best, self._goal_t = dist, t
            reached = dist < c.band_m / 2
            if not reached and t - self._goal_t > c.stall_s:  # the bottom, or the autopilot's limit
                reached = True
                if self.profile_dir > 0:
                    self.depth_hi = max(depth, c.min_depth_m)
                else:
                    self.depth_lo = min(depth, c.max_depth_m)
            if reached:
                self._goal_best, self._goal_t = math.inf, t
                if self.profile_dir > 0:
                    self.profile_dir = -1.0
                else:
                    self.profile = False
                    self.band_until = t
            else:
                target = goal
        if target is None:
            if self.target_band is None or t >= self.band_until:
                self.target_band = self.choose_band(t)
                self.band_until = t + c.dwell_s
            target = float(self.bands[self.target_band])
        if self.floor_m is not None:
            target = min(target, self.floor_m)
        return target

    # ------------------------------------------------------------------ horizontal

    def _coverage(self, heading: float, t: float, leg_s: float, surge: float) -> float:
        """Recently visited water along a candidate leg (decayed)."""
        c = self.cfg
        h = math.radians(heading)
        step = np.array([math.sin(h), math.cos(h)])
        dist = surge * c.speed_mps_per_unit * leg_s
        score = 0.0
        for s in np.linspace(c.cell_m, max(c.cell_m, dist), 6):
            p = self.pos + s * step
            last = self.visited.get((int(p[0] // c.cell_m), int(p[1] // c.cell_m)))
            if last is not None:
                score += 0.5 ** ((t - last) / c.coverage_halflife_s)
        return score

    def _new_leg(self, t: float, heading: float) -> None:
        c = self.cfg
        intensive = self.mode == "intensive"
        turn = c.ars_turn_deg if intensive else c.turn_deg
        leg = c.ars_leg_s if intensive else c.leg_s
        surge = c.ars_surge if intensive else c.surge
        if intensive:  # keep circling one way: a loose loiter around the find
            cands = [heading + self.turn_sign * turn]
        else:  # turn by roughly `turn` either way, or anywhere else, preferring unsearched water
            cands = [
                heading + d
                for d in np.linspace(-180, 180, c.headings, endpoint=False)
                if abs(wrap180(d)) >= 45
            ]
            cands += [heading + turn, heading - turn]
        costs = [self._coverage(h % 360.0, t, leg, surge) + 0.05 * self.rng.random() for h in cands]
        self.leg_heading = float(cands[int(np.argmin(costs))] % 360.0)
        self.leg_until = t + leg * (0.7 + 0.6 * self.rng.random())

    def command(self, t: float, nav: NavState | None) -> Command:
        c = self.cfg
        if self.mode == "intensive" and t - self.last_find > c.giveup_s:
            self.mode = "extensive"
            self.leg_until = t
        heading = None if nav is None else nav.heading(t)
        depth = None if nav is None else nav.depth(t)
        if c.strategy in ("scan", "depth"):  # turn in place (the old behaviour), alternating
            direction = 1.0 if int(t // c.scan_s) % 2 == 0 else -1.0
            cmd = Command(direction * c.scan_yaw, 0.0, 0.0)
        elif heading is None:  # scan and hop
            if t >= self.phase_until:
                self.hopping = not self.hopping
                self.phase_until = t + (c.hop_s if self.hopping else c.scan_s)
            cmd = Command(0.0, 0.0, c.surge) if self.hopping else Command(c.scan_yaw, 0.0, 0.0)
        else:
            if self.leg_heading is None or t >= self.leg_until:
                self._new_leg(t, heading)
            err = wrap180(self.leg_heading - heading)
            yaw = float(np.clip(c.heading_kp * err, -0.6, 0.6))
            surge = c.ars_surge if self.mode == "intensive" else c.surge
            surge = surge if abs(err) < 45 else 0.0  # turn first, then go
            cmd = Command(yaw, 0.0, surge, heading_deg=self.leg_heading)
        vertical = c.strategy in ("adaptive", "depth")
        depth_target = self._depth_target(t) if (depth is not None and vertical) else None
        if depth_target is not None:
            cmd.depth_m = depth_target
            cmd.heave = float(np.clip(-c.depth_kp * (depth_target - depth), -0.6, 0.6))  # heave > 0 = up
        self.last_cmd = cmd
        return cmd

    def status(self) -> dict:
        r = self.rates_per_min()
        best = int(np.argmax(r))
        return {
            "mode": "profile" if self.profile else self.mode,
            "band_m": None if self.target_band is None else float(self.bands[self.target_band]),
            "leg_heading": None if self.leg_heading is None else round(self.leg_heading, 1),
            "best_band_m": float(self.bands[best]),
            "best_rate_per_min": round(float(r[best]), 3),
        }
