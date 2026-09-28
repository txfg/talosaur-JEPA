"""Constant-velocity Kalman filter on (yaw, pitch, log apparent size) with outlier gating.

Smooths the 3-10 Hz detections into a steady track, predicts through short dropouts, and
rejects jumps (another animal, a bright speck) with a chi-square gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CHI2_3DOF_99 = 11.345


@dataclass
class TrackerConfig:
    meas_std: tuple[float, float, float] = (3.0, 3.0, 0.2)  # deg, deg, log-size
    accel_std: tuple[float, float, float] = (20.0, 20.0, 0.5)  # per s^2
    init_vel_std: tuple[float, float, float] = (30.0, 30.0, 0.5)
    gate_chi2: float = CHI2_3DOF_99
    max_misses: int = 8
    confirm_hits: int = 3


@dataclass
class TrackState:
    active: bool = False
    confirmed: bool = False
    yaw: float = 0.0
    pitch: float = 0.0
    size: float = 0.0
    yaw_rate: float = 0.0
    pitch_rate: float = 0.0
    confidence: float = 0.0
    hits: int = 0
    misses: int = 0

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class TargetTracker:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.x = np.zeros(6)
        self.P = np.eye(6)
        self.t: float | None = None
        self.state = TrackState()
        self.R = np.diag(np.square(self.cfg.meas_std))
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))])

    def reset(self) -> None:
        self.__init__(self.cfg)

    def _predict(self, dt: float) -> None:
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        q = np.square(self.cfg.accel_std)
        G = np.vstack([np.eye(3) * 0.5 * dt * dt, np.eye(3) * dt])
        Q = G @ np.diag(q) @ G.T
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def step(self, t: float, meas: tuple[float, float, float] | None) -> TrackState:
        """``meas`` = (yaw_deg, pitch_deg, size) or None when nothing was detected."""
        c = self.cfg
        st = self.state
        if self.t is not None and st.active:
            self._predict(max(1e-3, t - self.t))
        self.t = t
        z = None if meas is None else np.array([meas[0], meas[1], math.log(max(meas[2], 1e-3))])
        if not st.active:
            if z is not None:
                self.x = np.r_[z, 0.0, 0.0, 0.0]
                self.P = np.diag(np.r_[np.square(c.meas_std), np.square(c.init_vel_std)])
                st.active, st.hits, st.misses = True, 1, 0
        elif z is not None:
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            d2 = float(y @ np.linalg.solve(S, y))
            if d2 <= c.gate_chi2:
                K = self.P @ self.H.T @ np.linalg.inv(S)
                self.x = self.x + K @ y
                self.P = (np.eye(6) - K @ self.H) @ self.P
                st.hits += 1
                st.misses = 0
            else:
                st.misses += 1
        else:
            st.misses += 1
        if st.active and st.misses > c.max_misses:
            self.reset()
            return self.state
        st.confirmed = st.active and st.hits >= c.confirm_hits
        st.yaw, st.pitch = float(self.x[0]), float(self.x[1])
        st.size = float(math.exp(self.x[2]))
        st.yaw_rate, st.pitch_rate = float(self.x[3]), float(self.x[4])
        st.confidence = float(
            min(1.0, st.hits / (c.confirm_hits + 2)) * (1.0 - st.misses / (c.max_misses + 1))
        )
        return st
