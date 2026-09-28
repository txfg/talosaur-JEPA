"""Track -> normalised vehicle commands (yaw rate, heave, surge in [-1, 1]).

The autopilot interface is still open, so commands are *normalised requests*; a backend maps
them to MAVLink / serial later. Vehicle-level safety (depth limits, obstacle avoidance, battery)
stays with the autopilot; this controller only adds conservative limits of its own:
rate limits, a slew limit, and a hard stand-off (it backs off when the animal fills too much
of the frame).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from talosaur.guidance.tracker import TrackState


@dataclass
class ControllerConfig:
    yaw_kp: float = 0.035  # command per degree of bearing error
    yaw_kd: float = 0.004  # command per deg/s of bearing rate
    pitch_kp: float = 0.03
    surge_kp: float = 3.0  # command per unit of apparent-size error
    target_size: float = 0.22  # desired sqrt(area fraction) of the animal (~ fills 20-25% width)
    standoff_size: float = 0.40  # back off above this size
    max_yaw: float = 0.8
    max_heave: float = 0.6
    max_surge: float = 0.5
    max_reverse: float = 0.3
    deadband_deg: float = 2.0
    slew_per_s: float = 2.0
    search_yaw: float = 0.15  # slow scan while searching
    search_period_s: float = 20.0
    lost_yaw: float = 0.3
    # RELEASE (move on after an animal's time budget): back off, turn away, swim on. Time-based
    # until a vehicle calibration turns these into angles and distances.
    release_backoff_s: float = 1.5
    release_reverse: float = 0.2
    release_turn_s: float = 4.0  # without a heading input: turn this long ...
    release_turn_deg: float = 120.0  # ... with one: turn this far away from the animal
    release_yaw: float = 0.6
    release_surge: float = 0.3
    heading_kp: float = 0.02  # yaw command per degree of heading error (heading setpoints)


@dataclass
class Command:
    """Normalised rate requests (always set), plus optional setpoints for an autopilot with
    heading / depth hold (search legs and depth bands). A bridge that cannot use a setpoint
    ignores it: the rates already steer toward it when navigation data is available."""

    yaw_rate: float = 0.0
    heave: float = 0.0
    surge: float = 0.0
    heading_deg: float | None = None
    depth_m: float | None = None
    light: float | None = None  # lamp level 0-1; None = leave the lights to the vehicle

    def as_dict(self) -> dict:
        return {
            "yaw_rate": self.yaw_rate,
            "heave": self.heave,
            "surge": self.surge,
            "heading_deg": self.heading_deg,
            "depth_m": self.depth_m,
            "light": self.light,
        }


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class Controller:
    def __init__(self, cfg: ControllerConfig | None = None):
        self.cfg = cfg or ControllerConfig()
        self.last = Command()
        self.t: float | None = None

    def _slew(self, cmd: Command, t: float) -> Command:
        dt = 0.1 if self.t is None else max(1e-3, t - self.t)
        self.t = t
        m = self.cfg.slew_per_s * dt

        def lim(new: float, old: float) -> float:
            return old + _clip(new - old, -m, m)

        out = Command(
            lim(cmd.yaw_rate, self.last.yaw_rate),
            lim(cmd.heave, self.last.heave),
            lim(cmd.surge, self.last.surge),
            cmd.heading_deg,  # setpoints pass through (the autopilot's own loops shape them)
            cmd.depth_m,
        )
        self.last = out
        return out

    def shape(self, cmd: Command, t: float) -> Command:
        """Apply the slew limit to a command made elsewhere (e.g. the search planner)."""
        return self._slew(cmd, t)

    def track(self, st: TrackState, t: float, approach: bool = True) -> Command:
        c = self.cfg
        yaw_err = 0.0 if abs(st.yaw) < c.deadband_deg else st.yaw - math.copysign(c.deadband_deg, st.yaw)
        pitch_err = (
            0.0 if abs(st.pitch) < c.deadband_deg else st.pitch - math.copysign(c.deadband_deg, st.pitch)
        )
        yaw = _clip(c.yaw_kp * yaw_err + c.yaw_kd * st.yaw_rate, -c.max_yaw, c.max_yaw)
        heave = _clip(c.pitch_kp * pitch_err, -c.max_heave, c.max_heave)
        surge = 0.0
        if st.size > c.standoff_size:
            surge = -c.max_reverse  # too close: back off, whatever else is happening
        elif approach:
            # approach only when roughly centred, to avoid bumping into the animal sideways
            centred = abs(st.yaw) < 15.0 and abs(st.pitch) < 15.0
            surge = (
                _clip(c.surge_kp * (c.target_size - st.size), -c.max_reverse, c.max_surge) if centred else 0.0
            )
        return self._slew(Command(yaw, heave, surge), t)

    def search(self, t: float) -> Command:
        c = self.cfg
        direction = 1.0 if int(t // c.search_period_s) % 2 == 0 else -1.0
        return self._slew(Command(direction * c.search_yaw, 0.0, 0.0), t)

    def lost(self, last_yaw: float, t: float) -> Command:
        return self._slew(Command(math.copysign(self.cfg.lost_yaw, last_yaw or 1.0), 0.0, 0.0), t)

    def hold(self, t: float) -> Command:
        return self._slew(Command(), t)

    def release(
        self,
        t: float,
        elapsed: float,
        away: float,
        heading: float | None = None,
        heading_sp: float | None = None,
    ) -> Command:
        """Move on from an animal: back off, turn toward ``away`` (+1 right / -1 left, i.e. away
        from the side the animal was on), then swim on (the rest of the RELEASE state). With a
        heading input the turn goes to ``heading_sp`` (``release_turn_deg`` away); without one it
        is timed (``release_turn_s``)."""
        c = self.cfg
        if elapsed < c.release_backoff_s:
            cmd = Command(0.0, 0.0, -c.release_reverse)
        elif heading is not None and heading_sp is not None:
            err = (heading_sp - heading + 180.0) % 360.0 - 180.0
            yaw = _clip(c.heading_kp * err, -c.release_yaw, c.release_yaw)
            cmd = Command(yaw, 0.0, c.release_surge if abs(err) < 20.0 else 0.0, heading_deg=heading_sp)
        elif elapsed < c.release_backoff_s + c.release_turn_s:
            cmd = Command(math.copysign(c.release_yaw, away), 0.0, 0.0)
        else:
            cmd = Command(0.0, 0.0, c.release_surge)
        return self._slew(cmd, t)
