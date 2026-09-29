"""Model outputs -> target -> bearing -> track -> state machine -> command (+ telemetry).

Used by the Pi onboard app and by the desktop replay tool, so what you tune on recorded video
is exactly what runs on the vehicle. Each animal gets a time budget (``encounter.max_s``); then the
vehicle releases it and looks for a *different* one, recognised by appearance (encounters.py).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from talosaur.guidance.camera_model import CameraModel
from talosaur.guidance.controller import Command, Controller, ControllerConfig
from talosaur.guidance.curiosity import Curiosity, CuriosityConfig
from talosaur.guidance.encounters import EncounterConfig, EncounterManager
from talosaur.guidance.heatmap import Target, find_blobs, order_blobs, sigmoid
from talosaur.guidance.lights import LightPolicy, LightsConfig
from talosaur.guidance.nav import NavState
from talosaur.guidance.novelty import NoveltyDetector
from talosaur.guidance.search import SearchConfig, SearchPlanner
from talosaur.guidance.state_machine import ENGAGED, FSMConfig, GuidanceFSM, State
from talosaur.guidance.tracker import TargetTracker, TrackerConfig


@dataclass
class GuidanceConfig:
    heat_thr: float = 0.5  # a blob needs one patch at or above this probability ...
    heat_thr_low: float = 0.35  # ... and extends over connected patches above this one
    gate: float = 0.35
    min_mass: float = 0.6
    smooth: bool = False
    frame_index: int = 0  # which frame_logit output means "animal present"
    novelty: bool = True
    camera: CameraModel = field(default_factory=CameraModel)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    fsm: FSMConfig = field(default_factory=FSMConfig)
    encounter: EncounterConfig = field(default_factory=EncounterConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    lights: LightsConfig = field(default_factory=LightsConfig)
    curiosity: CuriosityConfig = field(default_factory=CuriosityConfig)

    @classmethod
    def from_dict(cls, d: dict | None) -> GuidanceConfig:
        d = dict(d or {})
        sub = {
            "camera": CameraModel,
            "tracker": TrackerConfig,
            "controller": ControllerConfig,
            "fsm": FSMConfig,
            "encounter": EncounterConfig,
            "search": SearchConfig,
            "lights": LightsConfig,
            "curiosity": CuriosityConfig,
        }
        kw: dict[str, Any] = {}
        for k, v in d.items():
            if k in sub:
                kw[k] = sub[k](
                    **{kk: (tuple(vv) if isinstance(vv, list) else vv) for kk, vv in (v or {}).items()}
                )
            elif k in cls.__dataclass_fields__:
                kw[k] = v
        return cls(**kw)


def _num(v: float, nd: int = 3) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v), nd)


class Guidance:
    def __init__(self, cfg: GuidanceConfig | None = None, embed_dim: int | None = None):
        self.cfg = cfg or GuidanceConfig()
        self.tracker = TargetTracker(self.cfg.tracker)
        self.controller = Controller(self.cfg.controller)
        self.fsm = GuidanceFSM(self.cfg.fsm)
        self.encounters = EncounterManager(self.cfg.encounter)
        self.novelty = NoveltyDetector(embed_dim) if (self.cfg.novelty and embed_dim) else None
        self.last_xy: tuple[float, float] | None = None
        self.last_yaw = 0.0
        self.release_away = 1.0
        self.release_heading: float | None = None
        self.nav: NavState | None = None
        self.search = SearchPlanner(self.cfg.search)
        self.lights = LightPolicy(self.cfg.lights)
        self.curiosity = Curiosity(self.cfg.curiosity)

    def step(
        self,
        t: float,
        frame_logit,
        heatmap_logit,
        embedding=None,
        tokens=None,
        nav: NavState | None = None,
        luma: float | None = None,
        glow: np.ndarray | None = None,
    ) -> tuple[Command, dict[str, Any], list[str]]:
        """One frame. ``tokens``: the model's patch tokens (h, w, D) if the export has them;
        ``nav``: depth / heading / turn rate if the vehicle provides them (search planning);
        ``luma``: the frame's mean brightness, 0-1 (lamp control: lights.py); ``glow``: a coarse
        grid of the frame's blue-green brightness (``curiosity.glow_grid``), for flashes."""
        self.nav = nav if nav is not None and nav.fresh(t) else None
        c = self.cfg
        enc = self.encounters
        enc.tick(t)
        frame_prob = float(sigmoid(np.asarray(frame_logit).reshape(-1)[c.frame_index]))
        prob = sigmoid(np.asarray(heatmap_logit, dtype=np.float32))
        tok = None if tokens is None else np.asarray(tokens)
        if tok is not None and c.encounter.reid:
            enc.observe_background(tok, prob)
        blobs, peak, n = find_blobs(prob, c.heat_thr, c.min_mass, smooth=c.smooth, thr_low=c.heat_thr_low)
        st0 = self.fsm.state
        # while looking for a new animal, skip the ones already filmed to the end of their budget
        choosing = st0 in (State.SEARCH, State.ACQUIRE)
        target, desc, skipped = Target(False, peak=peak, n_blobs=n), None, 0
        for b in order_blobs(blobs, self.last_xy, c.gate):
            d = enc.describe(tok, prob, b.mask)
            if choosing and enc.is_done_with(d, t)[0]:
                skipped += 1
                continue
            target, desc = b, d
            break
        sim = enc.match(desc, t)[1] if desc is not None else float("nan")

        meas = None
        yaw = pitch = None
        if target.found and st0 != State.RELEASE:
            yaw, pitch = c.camera.angles(target.cx, target.cy)
            meas = (yaw, pitch, max(target.size, 1e-3))
            self.last_xy = (target.cx, target.cy)
        track = self.tracker.step(t, meas)
        if not track.active:
            self.last_xy = None
        heading = None if self.nav is None else self.nav.heading(t)
        release = enc.leave_reason(t) if self.fsm.state in ENGAGED else None
        events = self.fsm.update(t, frame_prob, target, track, release=release)
        # a find for the search planner: the state machine's evidence (acquire_n of acquire_m frames),
        # so single-frame specks of marine snow do not count
        found = st0 == State.SEARCH and self.fsm.state == State.ACQUIRE
        self.search.observe(t, self.nav, found, self.controller.last.surge, heading, searching=choosing)

        summary = None
        for ev in events:
            if ev == "encounter_start":
                enc.start(t, desc)
            elif ev == "encounter_end":
                summary = enc.end(t, self.fsm.end_reason or "lost")
                depth = None if self.nav is None else self.nav.depth(t)
                self.search.after_encounter(t, summary.get("behaviour"), depth)  # where to search next
                self.curiosity.pause(t, c.fsm.release_s + c.curiosity.cooldown_s)  # not the same animal
        st = self.fsm.state
        if st == State.RELEASE and st0 != State.RELEASE:
            self.release_away = -1.0 if self.last_yaw > 0 else 1.0  # turn away from the animal's side
            self.release_heading = (
                None
                if heading is None
                else (heading + self.release_away * c.controller.release_turn_deg) % 360.0
            )
            self.tracker.reset()
            self.last_xy = None
        if st in ENGAGED:
            enc.observe(t, st.value, target, frame_prob, desc, track, own_surge=self.controller.last.surge)

        look = self.curiosity.update(t, st == State.SEARCH, prob, glow)
        if st == State.SEARCH and look is not None:  # something worth a closer look: turn and creep
            cc = c.curiosity
            lyaw, lpitch = c.camera.angles(*look)
            cmd = self.controller.shape(
                Command(
                    float(np.clip(cc.yaw_kp * lyaw, -0.6, 0.6)),
                    float(np.clip(cc.pitch_kp * lpitch, -0.4, 0.4)),
                    cc.creep if abs(lyaw) < 15.0 else 0.0,
                ),
                t,
            )
        elif st == State.SEARCH:
            cmd = self.controller.shape(self.search.command(t, self.nav), t)
        elif st == State.ACQUIRE:
            cmd = self.controller.hold(t)
        elif st == State.TRACK:  # approach - unless it is swimming away: never chase an animal
            cmd = self.controller.track(track, t, approach=not enc.fleeing())
        elif st == State.FILM:
            cmd = self.controller.track(track, t, approach=False)
        elif st == State.LOST:
            cmd = self.controller.lost(self.last_yaw, t)
        else:  # RELEASE
            cmd = self.controller.release(
                t, t - self.fsm.since, self.release_away, heading, self.release_heading
            )
        close = st in ENGAGED and track.active and track.size >= c.lights.near_size
        cmd.light = self.lights.update(t, luma, close, engaged=st in ENGAGED)
        if track.active:
            self.last_yaw = track.yaw
        nov = None
        if self.novelty is not None and embedding is not None:
            nov = self.novelty.score(embedding, update=st == State.SEARCH)
        cur = enc.current
        tele = {
            "t": round(t, 3),
            "state": st.value,
            "frame_prob": round(frame_prob, 4),
            "novelty": None if nov is None else round(nov, 3),
            "target": {**target.as_dict(), "yaw": yaw, "pitch": pitch},
            "track": track.as_dict(),
            "cmd": cmd.as_dict(),
            "recording": self.fsm.recording,
            "events": events,
            "encounter": None
            if cur is None
            else {
                "id": cur.id,
                "engaged_s": round(cur.engaged_s(t), 2),
                "remaining_s": _num(enc.remaining_s(t), 2),
                "good_s": round(cur.good_s, 2),
                "marginal_rate": _num(enc.marginal_rate(), 5),
                "long_run_rate": _num(enc.long_run_rate(t), 5),
            },
            "reid": {"sim": _num(sim), "skipped": skipped, "remembered": len(enc.memory)},
            "nav": None if self.nav is None else self.nav.as_dict(),
            "lights": self.lights.status(),
            "search": self.search.status() if st == State.SEARCH else None,
            "curiosity": self.curiosity.status(),
        }
        if summary:
            tele["encounter_summary"] = summary
        return cmd, tele, events

    def close(self, t: float, reason: str = "shutdown") -> dict | None:
        """End an encounter still open when the program stops (or the arm switch is turned off:
        ``reason="disarmed"``); returns its summary."""
        if not self.encounters.active:
            return None
        return self.encounters.end(t, reason)

    def config_dict(self) -> dict:
        return asdict(self.cfg)
