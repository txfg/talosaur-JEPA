"""Navigation input for the onboard loop: depth, heading, turn rate, altitude and water temperature.

The autopilot is still undecided, so the default is the mirror image of the command output: the
autopilot bridge sends small JSON datagrams to a UDP port, e.g.

  {"depth_m": 152.3, "heading_deg": 41.0, "yaw_rate_dps": -2.5, "altitude_m": null, "temp_c": 17.4}

at 5-50 Hz. Several senders may share the port (say the autopilot bridge for depth and heading, a
separate echosounder reader for altitude): each field keeps its latest value and goes stale on its
own after ``valid_for_s``. A field left out of a datagram keeps its previous value; ``null``
clears it. A MAVLink source (ATTITUDE + depth) can be added here once the vehicle stack is chosen.
"""

from __future__ import annotations

import json
import socket

from talosaur.guidance.nav import NavState

FIELDS = ("depth_m", "heading_deg", "yaw_rate_dps", "altitude_m", "temp_c")


class NullNav:
    """No navigation data: search falls back to sensor-free patterns."""

    def poll(self, t: float) -> NavState | None:
        return None

    def close(self) -> None:
        pass


class UdpNavSource:
    def __init__(self, host: str = "0.0.0.0", port: int = 14601, valid_for_s: float = 1.5):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.sock.setblocking(False)
        self.port = self.sock.getsockname()[1]
        self.valid_for_s = valid_for_s
        self.fields: dict[str, tuple[float | None, float]] = {}  # field -> (value, time received)
        self.bad = 0

    def poll(self, t: float) -> NavState | None:
        """Drain pending datagrams (never blocks); new values are stamped with ``t``. Returns the
        fields still fresh, or None if none is."""
        while True:
            try:
                data = self.sock.recv(65536)
            except (BlockingIOError, OSError):
                break
            try:
                msg = json.loads(data)
                vals = {k: (None if msg[k] is None else float(msg[k])) for k in FIELDS if k in msg}
            except (ValueError, TypeError, AttributeError):
                self.bad += 1
                continue
            for k, v in vals.items():
                self.fields[k] = (v, t)
        fresh = {k: v for k, (v, tr) in self.fields.items() if t - tr <= self.valid_for_s}
        return NavState(t, valid_for_s=self.valid_for_s, **fresh) if fresh else None

    def close(self) -> None:
        self.sock.close()


def make_nav(spec: dict | None):
    """``nav: {kind: udp, port: 14601}`` in the onboard config; absent or ``kind: none`` = no input."""
    spec = spec or {}
    kind = spec.get("kind", "none")
    if kind in (None, "none"):
        return NullNav()
    if kind == "udp":
        return UdpNavSource(
            spec.get("host", "0.0.0.0"), int(spec.get("port", 14601)), float(spec.get("valid_for_s", 1.5))
        )
    if kind == "mavlink":
        raise NotImplementedError("MAVLink navigation input: pending the autopilot choice (docs/PI5.md)")
    raise ValueError(f"unknown nav input {kind!r}")
