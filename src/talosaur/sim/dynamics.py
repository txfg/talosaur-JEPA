"""Vehicle dynamics for the digital twin (docs/DIGITAL_TWIN.md, milestone T1).

One description file (``configs/vehicle/*.yaml``) sets the physics. Each axis (surge, sway, heave,
yaw) follows the one-degree-of-freedom model the pool fit will use::

    (mass + added mass) * acceleration = thrust - linear * v - quadratic * v * |v| + offset

where the offset is the net buoyancy on the heave axis. Cross-coupling is left out, as in the fit.
An autopilot stands in for the bridge: heading and depth holds when a command carries setpoints,
otherwise the normalised rates are scaled by its limits. Thrusters lag with one time constant.

``DynamicVehicle`` has the same interface as :class:`talosaur.sim.vehicle.Vehicle`, so the world,
cameras and scoring work with either. Frames: heading 0 = north (+y), clockwise positive; depth
positive down. The simulation runs in the water's frame, so a current moves the vehicle and the
animals together and is left out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from talosaur.guidance.controller import Command
from talosaur.guidance.nav import NavState, wrap180

AXES = ("surge", "sway", "heave", "yaw")


@dataclass
class VehicleDescription:
    name: str = "talosaur_v0"
    mass_kg: float = 11.0
    yaw_inertia_kgm2: float = 0.26
    net_buoyancy_n: float = 2.0
    added_mass: dict[str, float] = field(
        default_factory=lambda: {"surge": 5.5, "sway": 12.7, "heave": 14.57, "yaw": 0.12}
    )
    linear_damping: dict[str, float] = field(
        default_factory=lambda: {"surge": 4.03, "sway": 6.22, "heave": 5.18, "yaw": 0.07}
    )
    quadratic_damping: dict[str, float] = field(
        default_factory=lambda: {"surge": 18.18, "sway": 21.66, "heave": 36.99, "yaw": 1.55}
    )
    thrust: dict[str, float] = field(
        default_factory=lambda: {
            "surge_n": 9.0,
            "reverse_n": 5.0,
            "heave_n": 8.0,
            "yaw_nm": 1.2,
            "response_s": 0.15,
        }
    )
    autopilot: dict[str, float] = field(
        default_factory=lambda: {
            "max_yaw_dps": 40.0,
            "max_vspeed_mps": 0.3,
            "heading_kp": 2.0,
            "yaw_rate_kp": 3.0,
            "depth_kp": 0.5,
            "vspeed_kp": 40.0,
            "vspeed_ki": 20.0,
            "min_depth_m": 5.0,
            "max_depth_m": 250.0,
        }
    )
    sensors: dict[str, float] = field(
        default_factory=lambda: {
            "heading_noise_deg": 1.0,
            "heading_bias_deg": 0.0,
            "gyro_noise_dps": 0.5,
            "depth_noise_m": 0.05,
        }
    )

    @classmethod
    def load(cls, path: str | Path) -> VehicleDescription:
        from talosaur.utils.io import load_yaml

        d = load_yaml(path) or {}
        base = cls()
        kw = {}
        for k in cls.__dataclass_fields__:
            if k not in d:
                continue
            v = d[k]
            kw[k] = {**getattr(base, k), **v} if isinstance(v, dict) else v  # partial overrides
        return cls(**kw)

    def inertia(self, axis: str) -> float:
        rigid = self.yaw_inertia_kgm2 if axis == "yaw" else self.mass_kg
        return rigid + self.added_mass[axis]

    def drag(self, axis: str, v: float) -> float:
        return self.linear_damping[axis] * v + self.quadratic_damping[axis] * v * abs(v)

    def steady_speed(self, axis: str, force: float) -> float:
        """Speed where drag balances a constant force (m/s or rad/s)."""
        a, b = self.quadratic_damping[axis], self.linear_damping[axis]
        s = math.copysign(1.0, force)
        f = abs(force)
        return s * ((-b + math.sqrt(b * b + 4 * a * f)) / (2 * a) if a > 0 else f / b)


class DynamicVehicle:
    """Drop-in for :class:`talosaur.sim.vehicle.Vehicle` with thrust, drag, inertia and buoyancy."""

    def __init__(
        self,
        desc: VehicleDescription | None = None,
        pos=(1000.0, 1000.0, 150.0),
        heading: float = 0.0,
        seed: int = 0,
        substep_s: float = 0.01,
    ):
        self.desc = desc or VehicleDescription()
        self.pos = np.array(pos, float)
        self.heading = heading % 360.0
        self.vel = np.zeros(4)  # body: surge, sway, heave (m/s, + = down), yaw rate (rad/s)
        self.force = np.zeros(4)  # thruster output after the spin-up lag
        self._vz_int = 0.0
        self.substep_s = substep_s
        self.rng = np.random.default_rng(seed)
        self.distance_m = 0.0

    # --- the kinematic Vehicle's interface
    @property
    def speed(self) -> float:
        return float(self.vel[0])

    @property
    def vspeed(self) -> float:
        return float(self.vel[2])

    @property
    def yaw_rate(self) -> float:
        return math.degrees(self.vel[3])

    def forward(self) -> np.ndarray:
        h = math.radians(self.heading)
        return np.array([math.sin(h), math.cos(h), 0.0])

    def right(self) -> np.ndarray:
        h = math.radians(self.heading)
        return np.array([math.cos(h), -math.sin(h), 0.0])

    # --- autopilot: command -> wanted thrust on each axis
    def _wanted_force(self, cmd: Command, dt: float) -> np.ndarray:
        d, ap, th = self.desc, self.desc.autopilot, self.desc.thrust
        max_r = math.radians(ap["max_yaw_dps"])
        if cmd.heading_deg is not None:
            r_des = math.radians(ap["heading_kp"] * wrap180(cmd.heading_deg - self.heading))
        else:
            r_des = float(np.clip(cmd.yaw_rate, -1, 1)) * max_r
        r_des = float(np.clip(r_des, -max_r, max_r))
        tau = d.drag("yaw", r_des) + ap["yaw_rate_kp"] * (r_des - self.vel[3])

        max_w = ap["max_vspeed_mps"]
        if cmd.depth_m is not None:
            target = float(np.clip(cmd.depth_m, ap["min_depth_m"], ap["max_depth_m"]))
            w_des = ap["depth_kp"] * (target - self.pos[2])
        else:
            w_des = -float(np.clip(cmd.heave, -1, 1)) * max_w  # heave > 0 = ascend
        if self.pos[2] <= ap["min_depth_m"]:
            w_des = max(w_des, 0.0)
        if self.pos[2] >= ap["max_depth_m"]:
            w_des = min(w_des, 0.0)
        w_des = float(np.clip(w_des, -max_w, max_w))
        err = w_des - self.vel[2]
        self._vz_int = float(np.clip(self._vz_int + err * dt, -1.0, 1.0))
        fz = d.drag("heave", w_des) + ap["vspeed_kp"] * err + ap["vspeed_ki"] * self._vz_int

        s = float(np.clip(cmd.surge, -1, 1))
        fx = s * (th["surge_n"] if s >= 0 else th["reverse_n"])
        return np.array(
            [
                fx,
                0.0,
                float(np.clip(fz, -th["heave_n"], th["heave_n"])),
                float(np.clip(tau, -th["yaw_nm"], th["yaw_nm"])),
            ]
        )

    def step(self, cmd: Command, dt: float) -> None:
        d = self.desc
        n = max(1, int(math.ceil(dt / self.substep_s)))
        h = dt / n
        lag = 1.0 - math.exp(-h / max(1e-3, d.thrust["response_s"]))
        for _ in range(n):
            self.force += lag * (self._wanted_force(cmd, h) - self.force)
            buoy = np.array([0.0, 0.0, -d.net_buoyancy_n, 0.0])  # + buoyancy pushes up (depth -)
            for k, ax in enumerate(AXES):
                acc = (self.force[k] + buoy[k] - d.drag(ax, self.vel[k])) / d.inertia(ax)
                self.vel[k] += acc * h
            self.heading = (self.heading + math.degrees(self.vel[3]) * h) % 360.0
            move = (self.forward() * self.vel[0] + self.right() * self.vel[1]) * h
            self.pos[:2] += move[:2]
            self.pos[2] += self.vel[2] * h
            if self.pos[2] < 0.0:  # the surface
                self.pos[2], self.vel[2] = 0.0, max(0.0, self.vel[2])
            self.distance_m += math.hypot(self.vel[0], self.vel[1]) * h

    def nav(self, t: float) -> NavState:
        s = self.desc.sensors
        return NavState(
            t,
            depth_m=float(self.pos[2] + self.rng.normal(0, s["depth_noise_m"])),
            heading_deg=float(
                (self.heading + s["heading_bias_deg"] + self.rng.normal(0, s["heading_noise_deg"])) % 360.0
            ),
            yaw_rate_dps=float(self.yaw_rate + self.rng.normal(0, s["gyro_noise_dps"])),
        )
