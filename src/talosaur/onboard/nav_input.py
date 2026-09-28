"""Navigation input for the onboard loop: depth, heading, turn rate (and altitude if available).

The autopilot is still undecided, so the default is the mirror image of the command output: the
autopilot bridge sends small JSON datagrams to a UDP port, e.g.

  {"depth_m": 152.3, "heading_deg": 41.0, "yaw_rate_dps": -2.5, "altitude_m": null}

at 5-50 Hz. Missing fields are fine. A MAVLink source (ATTITUDE + depth) can be added here once
the vehicle stack is chosen.
"""

from __future__ import annotations

import json
import socket

from talosaur.guidance.nav import NavState

FIELDS = ("depth_m", "heading_deg", "yaw_rate_dps", "altitude_m")


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
        self.latest: NavState | None = None
        self.bad = 0

    def poll(self, t: float) -> NavState | None:
        """Drain pending datagrams (never blocks); the newest sample is stamped with ``t``."""
        while True:
            try:
                data = self.sock.recv(65536)
            except (BlockingIOError, OSError):
                break
            try:
                msg = json.loads(data)
                vals = {k: (None if msg.get(k) is None else float(msg[k])) for k in FIELDS}
            except (ValueError, TypeError, AttributeError):
                self.bad += 1
                continue
            self.latest = NavState(t, valid_for_s=self.valid_for_s, **vals)
        return self.latest if self.latest is not None and self.latest.fresh(t) else None

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
