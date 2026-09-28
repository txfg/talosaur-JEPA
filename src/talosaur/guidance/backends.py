"""Where guidance output goes. The autopilot link is still undecided, so the defaults are a JSONL
log and JSON-over-UDP (easy to consume from BlueOS/ROS/a microcontroller bridge). A MAVLink
(ArduSub) backend can be added here once the vehicle stack is chosen."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any


class NullBackend:
    def publish(self, msg: dict[str, Any]) -> None:
        pass

    def close(self) -> None:
        pass


class JsonlBackend:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a", buffering=1)

    def publish(self, msg: dict[str, Any]) -> None:
        self.f.write(json.dumps(msg, separators=(",", ":")) + "\n")

    def close(self) -> None:
        self.f.close()


class UdpJsonBackend:
    def __init__(self, host: str = "127.0.0.1", port: int = 14600):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)

    def publish(self, msg: dict[str, Any]) -> None:
        try:
            self.sock.sendto(json.dumps(msg, separators=(",", ":")).encode(), self.addr)
        except (BlockingIOError, OSError):
            pass  # never block the vision loop on the network

    def close(self) -> None:
        self.sock.close()


class MultiBackend:
    def __init__(self, backends):
        self.backends = list(backends)

    def publish(self, msg):
        for b in self.backends:
            b.publish(msg)

    def close(self):
        for b in self.backends:
            b.close()


def make_backend(specs: list[dict] | None):
    """specs: e.g. [{"kind": "jsonl", "path": "logs/guidance.jsonl"}, {"kind": "udp", "port": 14600}]"""
    out = []
    for s in specs or []:
        kind = s.get("kind")
        if kind == "jsonl":
            out.append(JsonlBackend(s["path"]))
        elif kind == "udp":
            out.append(UdpJsonBackend(s.get("host", "127.0.0.1"), int(s.get("port", 14600))))
        elif kind == "mavlink":
            raise NotImplementedError("MAVLink backend: pending the autopilot choice (see docs/PI5.md)")
        elif kind in (None, "null"):
            out.append(NullBackend())
        else:
            raise ValueError(f"unknown backend {kind!r}")
    return MultiBackend(out) if out else NullBackend()
