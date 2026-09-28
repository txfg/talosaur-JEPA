"""A simple vehicle that follows the guidance commands with first-order lags.

Heading 0 = north (+y), clockwise positive; depth positive down. When a command carries a heading
or depth setpoint the simulated autopilot holds it (as ArduSub-style heading / depth hold would);
otherwise the normalised rates are scaled by the vehicle's maximum rates. The autopilot also
enforces hard depth limits, as the real one should.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from talosaur.guidance.controller import Command
from talosaur.guidance.nav import NavState, wrap180


@dataclass
class VehicleConfig:
    max_yaw_dps: float = 40.0
    max_speed_mps: float = 0.6
    max_reverse_mps: float = 0.3
    max_vspeed_mps: float = 0.3
    tau_s: float = 0.6  # response time constant
    heading_kp: float = 2.0  # heading hold: deg/s per deg of error
    depth_kp: float = 0.5  # depth hold: m/s per m of error
    min_depth_m: float = 5.0
    max_depth_m: float = 200.0
    heading_noise_deg: float = 1.0  # compass noise in the navigation output
    depth_noise_m: float = 0.05


class Vehicle:
    def __init__(
        self,
        cfg: VehicleConfig | None = None,
        pos=(1000.0, 1000.0, 100.0),
        heading: float = 0.0,
        seed: int = 0,
    ):
        self.cfg = cfg or VehicleConfig()
        self.pos = np.array(pos, float)
        self.heading = heading % 360.0
        self.yaw_rate = 0.0  # deg/s
        self.speed = 0.0  # m/s forward
        self.vspeed = 0.0  # m/s, positive = down
        self.rng = np.random.default_rng(seed)
        self.distance_m = 0.0

    def forward(self) -> np.ndarray:
        h = math.radians(self.heading)
        return np.array([math.sin(h), math.cos(h), 0.0])

    def right(self) -> np.ndarray:
        h = math.radians(self.heading)
        return np.array([math.cos(h), -math.sin(h), 0.0])

    def step(self, cmd: Command, dt: float) -> None:
        c = self.cfg
        if cmd.heading_deg is not None:
            want_yaw = float(
                np.clip(c.heading_kp * wrap180(cmd.heading_deg - self.heading), -c.max_yaw_dps, c.max_yaw_dps)
            )
        else:
            want_yaw = float(np.clip(cmd.yaw_rate, -1, 1)) * c.max_yaw_dps
        if cmd.depth_m is not None:
            want_v = float(
                np.clip(c.depth_kp * (cmd.depth_m - self.pos[2]), -c.max_vspeed_mps, c.max_vspeed_mps)
            )
        else:
            want_v = -float(np.clip(cmd.heave, -1, 1)) * c.max_vspeed_mps  # heave > 0 = ascend
        s = float(np.clip(cmd.surge, -1, 1))
        want_speed = s * (c.max_speed_mps if s >= 0 else c.max_reverse_mps)
        a = 1.0 - math.exp(-dt / c.tau_s)
        self.yaw_rate += a * (want_yaw - self.yaw_rate)
        self.vspeed += a * (want_v - self.vspeed)
        self.speed += a * (want_speed - self.speed)
        self.heading = (self.heading + self.yaw_rate * dt) % 360.0
        step = self.forward() * self.speed * dt
        self.pos[:2] += step[:2]
        self.pos[2] = float(np.clip(self.pos[2] + self.vspeed * dt, c.min_depth_m, c.max_depth_m))
        self.distance_m += abs(self.speed) * dt

    def nav(self, t: float) -> NavState:
        c = self.cfg
        return NavState(
            t,
            depth_m=float(self.pos[2] + self.rng.normal(0, c.depth_noise_m)),
            heading_deg=float((self.heading + self.rng.normal(0, c.heading_noise_deg)) % 360.0),
            yaw_rate_dps=float(self.yaw_rate),
        )
