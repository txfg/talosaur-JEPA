"""Search planning: where to look when no animal is in view (SEARCH state).

The behaviour follows the biology in docs/TWILIGHT_ZONE.md. It is built from what stays reliable
underwater - depth, heading and time - never from geographic x/y (no DVL: dead reckoning drifts):

* **Vertical first: find the layer, then work it.** Twilight-zone animals live in layers that
  move with the time of day (diel vertical migration). The planner profiles the allowed depth
  range once, then chooses depth bands by Thompson sampling on each band's rate of finds
  (Gamma-Poisson, with forgetting because layers move), with a time-of-day prior.
* **During the dusk and dawn crossings: hover, and let the layer come to you.** While migrators
  cross the corridor (``dusk_h`` / ``dawn_h``), hold depth and heading with no forward thrust for
  ``hover_s``, then one short slow leg (``relocate_s``) out of the vehicle's own disturbance, and
  hover again. A motionless vehicle frightens least.
* **Otherwise: slow transects and silent drifts.** A straight transect of ``leg_s`` (about 10 min),
  then a drift of ``drift_s`` with no forward thrust so that shy fishes settle back, then a turn
  of about ``turn_deg`` toward water not visited recently. Visited water is a coarse map in the
  *water* frame (heading + commanded speed), which is the frame the animals drift in.
* **After an animal, what next depends on how it behaved** (:meth:`after_encounter`):
  - a swarm or bloom (several animals in view): widening loops around the spot;
  - an animal that swam away: move on ``move_on_m``, beyond its avoidance halo;
  - a drifter (jelly-like: it stayed): keep to its depth for ``layer_hold_s`` and carry on, since
    drifters gather along the same layer;
  - a find that did not become an encounter: short loops (``giveup_s``).
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
    # when migrators cross the corridor, in hours relative to sunset / sunrise: the ascent starts
    # about an hour before sunset, the descent about two hours before sunrise (docs/SEARCH.md §1)
    dusk_h: tuple[float, float] = (-1.0, 1.5)
    dawn_h: tuple[float, float] = (-2.5, 0.5)
    corridor_m: tuple[float, float] = (100.0, 200.0)
    night_depth_m: float = 60.0  # where the migrators are expected at night
    # horizontal (docs/TWILIGHT_ZONE.md §7.3)
    surge: float = 0.35  # transect speed; slow, because fast approaches scatter animals
    leg_s: float = 600.0  # a straight transect (+-20%) ...
    drift_s: float = 240.0  # ... then a silent drift with no forward thrust (0: none) ...
    turn_deg: float = 120.0  # ... then a turn of about this much, toward water not visited recently
    hover_in_crossings: bool = True  # dusk and dawn crossings: hover while the layer passes ...
    hover_s: float = 420.0  # ... this long at a time ...
    relocate_s: float = 150.0  # ... then one slow leg out of the vehicle's own disturbance
    # after a find or an encounter (docs/TWILIGHT_ZONE.md §7.4)
    loop_r0_m: float = 5.0  # widening loops around the spot: first radius ...
    loop_spacing_m: float = 5.0  # ... growth per turn ...
    loop_s: float = 180.0  # ... this long after a swarm or bloom ...
    giveup_s: float = 90.0  # ... or this long after a find that did not become an encounter
    loop_surge: float = 0.2
    move_on_m: float = 40.0  # an animal that swam away: go at least this far before searching
    layer_hold_s: float = 600.0  # a drifter: keep to its depth this long
    speed_mps_per_unit: float = 0.6  # surge 1.0 in m/s (from the vehicle calibration)
    cell_m: float = 20.0  # water-frame coverage map resolution
    coverage_halflife_s: float = 900.0
    headings: int = 12  # candidate headings tried for a new leg
    # control
    heading_kp: float = 0.02  # yaw command per degree of heading error (bridges without heading hold)
    depth_kp: float = 0.15  # heave command per metre of depth error (bridges without depth hold)
    loop_kp: float = 8.0  # loops: heading correction in degrees per metre off the loop radius
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
        self._depth_cmd_t: float | None = None  # last time the planner steered depth (SEARCH only)
        self.floor_m: float | None = None  # deepest depth allowed by the altimeter, if any
        self.phase = "transect"  # transect | drift | hover | relocate
        self.phase_until = 0.0
        self.local: str | None = None  # after a find: loops | move_on (None: the normal pattern)
        self.local_until = 0.0
        self.local_t0 = 0.0
        self.local_center = np.zeros(2)
        self.local_heading: float | None = None
        self.move_from: np.ndarray | None = None
        self.loop_dir = 1.0  # +1 clockwise
        self.layer_depth: float | None = None  # a drifter's depth, held until layer_until
        self.layer_until = 0.0
        self.crossing = False
        self.last_find = -1e9
        self.leg_heading: float | None = None
        self.pos = np.zeros(2)  # water-frame dead reckoning (x east, y north), metres
        self.visited: dict[tuple[int, int], float] = {}
        self.last_detect_t = -1e9
        if c.strategy not in ("adaptive", "depth", "legs", "scan"):
            raise ValueError(f"unknown search strategy {c.strategy!r}")
        self.last_depth: float | None = None
        self.hop_until = 0.0  # no-nav scan / hop
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
        """A find (a detection the state machine accepted): loop around here. If it becomes an
        encounter, :meth:`after_encounter` decides again when it ends."""
        self.last_find = t
        self._start_loops(t, t + self.cfg.giveup_s)

    def after_encounter(self, t: float, behaviour: str | None, depth: float | None) -> None:
        """An encounter ended: what to do next depends on how the animal behaved
        (``encounters.py`` classifies it as swarm, mobile or drifter)."""
        c = self.cfg
        self.last_find = t
        if behaviour == "swarm":  # krill, salps, pyrosomes: more are around
            self._start_loops(t, t + c.loop_s)
        elif behaviour == "mobile":  # it swam away: neighbours have moved off too
            self.local, self.local_heading, self.move_from = "move_on", None, None
            self.local_until = t + 3.0 * c.move_on_m / max(1e-3, c.surge * c.speed_mps_per_unit)
        else:  # a drifter: they gather along the same layer; hold its depth and carry on
            self.local = None
            if depth is not None and behaviour == "drifter":
                self.layer_depth, self.layer_until = depth, t + c.layer_hold_s

    def _start_loops(self, t: float, until: float) -> None:
        self.local, self.local_t0, self.local_until = "loops", t, until
        self.local_center = self.pos.copy()
        self.loop_dir = 1.0 if self.rng.random() < 0.5 else -1.0

    def in_crossing(self, hour: float) -> bool:
        """Migrators are crossing the corridor (dusk or dawn windows around sunset / sunrise)."""
        c = self.cfg
        rel = lambda h0: ((hour - h0) + 12) % 24 - 12  # noqa: E731 - hours after h0, in [-12, 12)
        return c.dusk_h[0] <= rel(c.sunset_h) <= c.dusk_h[1] or c.dawn_h[0] <= rel(c.sunrise_h) <= c.dawn_h[1]

    def _hour(self) -> float:
        return self.hour_fn() if self.hour_fn is not None else self.local_hour()

    # ------------------------------------------------------------------ vertical

    def _diel_prior(self, hour: float) -> np.ndarray:
        """Prior multiplier per band from the time of day (1 = neutral)."""
        c = self.cfg
        w = np.ones(len(self.bands))
        if not c.diel_prior:
            return w
        lo, hi = c.corridor_m
        if self.in_crossing(hour):  # migrators crossing the corridor
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
            hour = self._hour()
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
        if self._depth_cmd_t is None or t - self._depth_cmd_t > 2.0:
            self._goal_best = math.inf  # back in control (after filming): restart the progress check
        self._depth_cmd_t = t
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
        if target is None and self.layer_depth is not None and t < self.layer_until:
            target = float(np.clip(self.layer_depth, self.depth_lo, self.depth_hi))  # a drifter's layer
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

    def _new_leg(self, t: float, heading: float, duration: float) -> None:
        """Choose a heading for a leg of ``duration``: a turn of about ``turn_deg`` either way, or
        anywhere else at least 45 deg off, preferring water not visited recently."""
        c = self.cfg
        cands = [
            heading + d for d in np.linspace(-180, 180, c.headings, endpoint=False) if abs(wrap180(d)) >= 45
        ]
        cands += [heading + c.turn_deg, heading - c.turn_deg]
        costs = [self._coverage(h % 360.0, t, duration, c.surge) + 0.05 * self.rng.random() for h in cands]
        self.leg_heading = float(cands[int(np.argmin(costs))] % 360.0)

    def _pattern(self, t: float, heading: float) -> float:
        """The normal pattern: hover and relocate during the crossings, transects and drifts
        otherwise. Sets ``phase`` / ``leg_heading``; returns the surge for this moment."""
        c = self.cfg
        if self.crossing and c.hover_in_crossings:
            if self.phase not in ("hover", "relocate") or t >= self.phase_until:
                if self.phase == "hover":  # out of our own disturbance, then hover again
                    self.phase, self.phase_until = "relocate", t + c.relocate_s
                    self._new_leg(t, heading, c.relocate_s)
                else:
                    self.phase, self.phase_until = "hover", t + c.hover_s
                    self.leg_heading = heading
            return c.surge if self.phase == "relocate" else 0.0
        if self.phase not in ("transect", "drift") or t >= self.phase_until:
            if self.phase == "transect" and c.drift_s > 0:  # a silent drift: shy fishes settle back
                self.phase, self.phase_until = "drift", t + c.drift_s
                self.leg_heading = heading
            else:
                leg = c.leg_s * (0.8 + 0.4 * self.rng.random())
                self.phase, self.phase_until = "transect", t + leg
                self._new_leg(t, heading, leg)
        return c.surge if self.phase == "transect" else 0.0

    def _loop_heading(self, t: float, heading: float) -> float:
        """Widening loops (an Archimedean spiral in the water frame) around ``local_center``."""
        c = self.cfg
        v = max(1e-3, c.loop_surge * c.speed_mps_per_unit)
        r_want = math.sqrt(c.loop_r0_m**2 + c.loop_spacing_m * v * max(0.0, t - self.local_t0) / math.pi)
        d = self.pos - self.local_center
        dist = float(np.hypot(d[0], d[1]))
        if dist < 0.5:  # at the centre: head out as we are
            return heading
        bearing = math.degrees(math.atan2(d[0], d[1]))  # compass bearing from the centre to us
        corr = float(np.clip(c.loop_kp * (dist - r_want), -60.0, 60.0))  # outside the radius: turn in
        return (bearing + self.loop_dir * (90.0 + corr)) % 360.0

    def command(self, t: float, nav: NavState | None) -> Command:
        c = self.cfg
        if self.local is not None and t >= self.local_until:
            self.local = None
            self.phase_until = t  # choose afresh
        heading = None if nav is None else nav.heading(t)
        depth = None if nav is None else nav.depth(t)
        self.crossing = c.diel_prior and self.in_crossing(self._hour())
        if c.strategy in ("scan", "depth"):  # turn in place (the old behaviour), alternating
            direction = 1.0 if int(t // c.scan_s) % 2 == 0 else -1.0
            cmd = Command(direction * c.scan_yaw, 0.0, 0.0)
        elif heading is None:  # scan and hop
            if t >= self.hop_until:
                self.hopping = not self.hopping
                self.hop_until = t + (c.hop_s if self.hopping else c.scan_s)
            cmd = Command(0.0, 0.0, c.surge) if self.hopping else Command(c.scan_yaw, 0.0, 0.0)
        else:
            if self.local == "loops":
                self.leg_heading, surge = self._loop_heading(t, heading), c.loop_surge
            elif self.local == "move_on":
                if self.local_heading is None:  # the first command after RELEASE: keep going this way
                    self.local_heading, self.move_from = heading, self.pos.copy()
                gone = float(np.hypot(*(self.pos - self.move_from)))
                if gone >= c.move_on_m:
                    self.local, self.phase_until = None, t
                    surge = self._pattern(t, heading)
                else:
                    self.leg_heading, surge = self.local_heading, c.surge
            else:
                surge = self._pattern(t, heading)
            err = wrap180(self.leg_heading - heading)
            yaw = float(np.clip(c.heading_kp * err, -0.6, 0.6))
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
        layer = (
            self.layer_depth
            if self.layer_depth is not None and (self.t_prev or 0.0) < self.layer_until
            else None
        )
        return {
            "mode": "profile" if self.profile else (self.local or self.phase),
            "crossing": self.crossing,
            "layer_hold_m": None if layer is None else round(layer, 1),
            "band_m": None if self.target_band is None else float(self.bands[self.target_band]),
            "leg_heading": None if self.leg_heading is None else round(self.leg_heading, 1),
            "best_band_m": float(self.bands[best]),
            "best_rate_per_min": round(float(r[best]), 3),
        }
