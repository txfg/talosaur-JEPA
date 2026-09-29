"""What the vehicle knows about its own state, for search planning.

Underwater, horizontal position from dead reckoning drifts quickly, but three things stay reliable:
depth (pressure sensor, ~cm), heading (compass / gyro) and turn rate (gyro). Search patterns are
therefore built from depth bands, heading legs and time - never from x/y positions.

Everything is optional: with no navigation input the planner falls back to patterns that need no
sensors (scan and hop).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class NavState:
    t: float  # app clock when the sample was received
    depth_m: float | None = None  # positive down
    heading_deg: float | None = None  # 0-360, any consistent reference (magnetic is fine)
    yaw_rate_dps: float | None = None  # positive = turning right
    altitude_m: float | None = None  # above the bottom, if an altimeter / echo sounder exists
    temp_c: float | None = None  # water temperature (most depth sensors measure it): logged, not steered by
    valid_for_s: float = 1.5

    def fresh(self, t: float) -> bool:
        return t - self.t <= self.valid_for_s

    def depth(self, t: float) -> float | None:
        return self.depth_m if self.fresh(t) else None

    def heading(self, t: float) -> float | None:
        return self.heading_deg if self.fresh(t) else None

    def as_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if k != "valid_for_s"}


def wrap180(deg: float) -> float:
    """Angle difference folded into [-180, 180)."""
    return (deg + 180.0) % 360.0 - 180.0
