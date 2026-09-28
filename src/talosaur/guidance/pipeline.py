"""Model outputs -> target -> bearing -> track -> state machine -> command (+ telemetry).

Used by the Pi onboard app and by the desktop replay tool, so what you tune on recorded video
is exactly what runs on the vehicle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from talosaur.guidance.camera_model import CameraModel
from talosaur.guidance.controller import Command, Controller, ControllerConfig
from talosaur.guidance.heatmap import Target, find_target, sigmoid
from talosaur.guidance.novelty import NoveltyDetector
from talosaur.guidance.state_machine import FSMConfig, GuidanceFSM, State
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

    @classmethod
    def from_dict(cls, d: dict | None) -> GuidanceConfig:
        d = dict(d or {})
        sub = {
            "camera": CameraModel,
            "tracker": TrackerConfig,
            "controller": ControllerConfig,
            "fsm": FSMConfig,
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


class Guidance:
    def __init__(self, cfg: GuidanceConfig | None = None, embed_dim: int | None = None):
        self.cfg = cfg or GuidanceConfig()
        self.tracker = TargetTracker(self.cfg.tracker)
        self.controller = Controller(self.cfg.controller)
        self.fsm = GuidanceFSM(self.cfg.fsm)
        self.novelty = NoveltyDetector(embed_dim) if (self.cfg.novelty and embed_dim) else None
        self.last_xy: tuple[float, float] | None = None
        self.last_yaw = 0.0

    def step(
        self, t: float, frame_logit, heatmap_logit, embedding=None
    ) -> tuple[Command, dict[str, Any], list[str]]:
        c = self.cfg
        frame_prob = float(sigmoid(np.asarray(frame_logit).reshape(-1)[c.frame_index]))
        prob = sigmoid(np.asarray(heatmap_logit, dtype=np.float32))
        target: Target = find_target(
            prob, c.heat_thr, self.last_xy, c.gate, c.min_mass, smooth=c.smooth, thr_low=c.heat_thr_low
        )
        meas = None
        yaw = pitch = None
        if target.found:
            yaw, pitch = c.camera.angles(target.cx, target.cy)
            meas = (yaw, pitch, max(target.size, 1e-3))
            self.last_xy = (target.cx, target.cy)
        track = self.tracker.step(t, meas)
        if not track.active:
            self.last_xy = None
        events = self.fsm.update(t, frame_prob, target, track)
        st = self.fsm.state
        if st == State.SEARCH:
            cmd = self.controller.search(t)
        elif st == State.ACQUIRE:
            cmd = self.controller.hold(t)
        elif st == State.TRACK:
            cmd = self.controller.track(track, t, approach=True)
        elif st == State.FILM:
            cmd = self.controller.track(track, t, approach=False)
        else:  # LOST
            cmd = self.controller.lost(self.last_yaw, t)
        if track.active:
            self.last_yaw = track.yaw
        nov = None
        if self.novelty is not None and embedding is not None:
            nov = self.novelty.score(embedding, update=st == State.SEARCH)
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
        }
        return cmd, tele, events

    def config_dict(self) -> dict:
        return asdict(self.cfg)
