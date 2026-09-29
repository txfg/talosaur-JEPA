"""Arm switch and start gate: the vehicle never thrusts on deck.

A magnetic reed switch sits inside the hull against the wall, wired between a GPIO pin and ground
(the pin's internal pull-up holds it high). A magnet outside, in a holder that cannot come loose,
closes it. **Magnet on = armed.** A broken wire reads as "magnet off", and so does a magnet lost
during the dive: both stop the thrusters.

  SAFE   switch off. Commands are all zero and the lamp is off (many underwater LEDs overheat in
         air). The camera, the model and the recording keep running, so deck checks work and no
         footage is lost.
  ARMED  switch on. A countdown (``arm_delay_s``) gives time to put the vehicle in the water; the
         mission then starts once the depth sensor has read at least ``start_depth_m`` for
         ``start_hold_s`` (in the water, not on deck). Still all zero.
  RUN    the mission: guidance steers. Switching off at any time returns to SAFE at once.

Each RUN starts a fresh mission: nothing that happened on deck or in an earlier run leaks into
the search planner (a planner that "descended" on deck would conclude the bottom is at 0 m).

``kind: none`` (the default when the config has no ``arming:`` section) has no switch and no
gate: RUN from the first frame. Use it only for replay, the simulator and bench tests.
``kind: file`` arms while a file exists (``touch /run/talosaur/arm``), for pool tests over SSH.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from talosaur.guidance.controller import Command
from talosaur.guidance.heatmap import sigmoid
from talosaur.guidance.nav import NavState
from talosaur.utils.log import get_logger

log = get_logger("arming")

SAFE, ARMED, RUN = "SAFE", "ARMED", "RUN"


@dataclass
class ArmingConfig:
    kind: str = "none"  # none | gpio | file
    pin: int = 17  # gpio: BCM number of the reed switch input (switch to ground, internal pull-up)
    invert: bool = False  # gpio: True for a normally-closed switch (the magnet opens it)
    path: str = "/run/talosaur/arm"  # file: armed while this file exists
    debounce_s: float = 0.5  # the switch must read the same this long before a change counts
    arm_delay_s: float = 20.0  # after arming: time to put the vehicle in the water
    start_depth_m: float | None = 0.2  # in the water: depth at least this ... (None = countdown only)
    start_hold_s: float = 3.0  # ... for this long


def _gpio_reader(pin: int, invert: bool) -> tuple[Callable[[], bool], Callable[[], None]]:
    # gpiozero ships with Raspberry Pi OS and drives the Pi 5's GPIO through lgpio
    from gpiozero import DigitalInputDevice

    dev = DigitalInputDevice(pin, pull_up=True)  # active (value 1) while the switch pulls the pin low
    return (lambda: bool(dev.value) != invert), dev.close


class Arming:
    """``update(t, depth)`` once per frame; ``state`` is SAFE, ARMED or RUN."""

    def __init__(
        self,
        cfg: ArmingConfig | None = None,
        read_switch: Callable[[], bool] | None = None,
        close: Callable[[], None] | None = None,
        fault: str | None = None,
    ):
        self.cfg = cfg or ArmingConfig()
        self.kind = self.cfg.kind
        self._read = read_switch
        self._close = close
        self.fault = fault  # why the switch cannot be read (the vehicle then stays SAFE)
        self.faults = 0
        self.state = RUN if self.kind == "none" else SAFE
        self.switch = False  # debounced
        self._raw: bool | None = None
        self._raw_t = 0.0
        self.armed_t: float | None = None
        self._wet_t: float | None = None  # since when the depth reads "in the water"
        self.wait: str | None = None
        self.missions = 0

    def _read_switch(self) -> bool:
        if self._read is None:
            return False
        try:
            return bool(self._read())
        except Exception as e:  # a read error counts as "switch off": stop, never start
            self.faults += 1
            if self.faults == 1:
                log.error(f"arm switch read failed ({e}); treating it as off")
            return False

    def update(self, t: float, depth_m: float | None) -> list[str]:
        """Advance the state machine; returns the events (``armed``, ``disarmed``, ``mission_start``)."""
        if self.kind == "none":
            return []
        c = self.cfg
        events: list[str] = []
        raw = self._read_switch()
        if raw != self._raw:
            self._raw, self._raw_t = raw, t
        if raw != self.switch and t - self._raw_t >= c.debounce_s:
            self.switch = raw
        if not self.switch:
            if self.state != SAFE:
                events.append("disarmed")
                log.info("disarmed: thrusters stop" if self.state == RUN else "disarmed")
            self.state, self.armed_t, self._wet_t, self.wait = SAFE, None, None, None
            return events
        if self.state == SAFE:
            self.state, self.armed_t = ARMED, t
            events.append("armed")
            log.info(f"armed: the mission starts in {c.arm_delay_s:.0f} s once the vehicle is in the water")
        if self.state == ARMED:
            if c.start_depth_m is not None:
                if depth_m is not None and depth_m >= c.start_depth_m:
                    self._wet_t = t if self._wet_t is None else self._wet_t
                else:
                    self._wet_t = None
            if t - self.armed_t < c.arm_delay_s:
                self.wait = "countdown"
            elif c.start_depth_m is None or (self._wet_t is not None and t - self._wet_t >= c.start_hold_s):
                self.state, self.wait = RUN, None
                self.missions += 1
                events.append("mission_start")
                log.info(f"mission {self.missions} started")
            else:
                self.wait = "no depth" if depth_m is None else "in the water"
        return events

    def status(self, t: float) -> dict[str, Any]:
        s: dict[str, Any] = {"state": self.state, "switch": self.switch, "missions": self.missions}
        if self.state == ARMED:
            s["wait"] = self.wait
            s["countdown_s"] = round(max(0.0, self.cfg.arm_delay_s - (t - self.armed_t)), 1)
        if self.fault:
            s["fault"] = self.fault
        if self.faults:
            s["read_errors"] = self.faults
        return s

    def close(self) -> None:
        if self._close is not None:
            self._close()


def make_arming(spec: dict | None) -> Arming:
    """``arming: {kind: gpio, pin: 17}`` in the onboard config; absent or ``kind: none`` = no gate.
    If the switch cannot be set up, the vehicle stays SAFE (it records, it never thrusts) and the
    reason is in the telemetry, rather than the whole program failing to start."""
    cfg = ArmingConfig(**(spec or {}))
    if cfg.kind == "none":
        return Arming(cfg)
    if cfg.kind == "file":
        return Arming(cfg, read_switch=lambda: os.path.exists(cfg.path))
    if cfg.kind == "gpio":
        try:
            read, close = _gpio_reader(cfg.pin, cfg.invert)
        except Exception as e:
            fault = f"GPIO {cfg.pin} unavailable: {e}"
            log.error(f"{fault}. The vehicle stays SAFE: it records, it never thrusts.")
            return Arming(cfg, fault=fault)
        return Arming(cfg, read_switch=read, close=close)
    raise ValueError(f"unknown arming kind {cfg.kind!r} (none | gpio | file)")


def safe_output(
    t: float, frame_logit, heatmap_logit, frame_index: int, state: str, nav: NavState | None
) -> tuple[Command, dict[str, Any]]:
    """The command and telemetry while not running a mission: all zero, lamp off, plus what the
    model sees (a deck check of camera and model)."""
    cmd = Command(light=0.0)
    frame_prob = float(sigmoid(np.asarray(frame_logit).reshape(-1)[frame_index]))
    peak = float(np.max(sigmoid(np.asarray(heatmap_logit, dtype=np.float32))))
    tele = {
        "t": round(t, 3),
        "state": state,
        "frame_prob": round(frame_prob, 4),
        "peak": round(peak, 4),
        "cmd": cmd.as_dict(),
        "events": [],
        "nav": None if nav is None else nav.as_dict(),
    }
    return cmd, tele
