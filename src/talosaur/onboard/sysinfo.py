"""Raspberry Pi system probes (memory, temperature, clock, throttling); all degrade gracefully
on non-Pi Linux machines."""

from __future__ import annotations

import os
import re
import resource
import shutil
import subprocess


def rss_mb() -> float:
    """Current resident set size of this process (MB)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def peak_rss_mb() -> float:
    """Peak resident set size of this process (MB) - ru_maxrss is in KB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def mem_available_mb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def mem_total_mb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return float("nan")


def cpu_temp_c() -> float:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return float("nan")


def cpu_freq_mhz(cpu: int = 0) -> float:
    try:
        with open(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        return float("nan")


THROTTLE_BITS = {
    0: "under-voltage now",
    1: "arm frequency capped now",
    2: "throttled now",
    3: "soft temperature limit now",
    16: "under-voltage occurred",
    17: "arm frequency capping occurred",
    18: "throttling occurred",
    19: "soft temperature limit occurred",
}


def throttled() -> dict:
    """Decode ``vcgencmd get_throttled`` (Raspberry Pi only)."""
    if not shutil.which("vcgencmd"):
        return {"available": False}
    try:
        out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True, timeout=2)
        m = re.search(r"0x([0-9a-fA-F]+)", out)
        v = int(m.group(1), 16) if m else 0
        return {
            "available": True,
            "raw": hex(v),
            "flags": [n for b, n in THROTTLE_BITS.items() if v & (1 << b)],
        }
    except Exception:
        return {"available": False}


def snapshot() -> dict:
    return {
        "rss_mb": round(rss_mb(), 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "mem_available_mb": round(mem_available_mb(), 1),
        "cpu_temp_c": cpu_temp_c(),
        "cpu_freq_mhz": cpu_freq_mhz(),
        "loadavg": os.getloadavg()[0] if hasattr(os, "getloadavg") else float("nan"),
    }


def is_raspberry_pi() -> bool:
    try:
        with open("/proc/device-tree/model") as f:
            return "Raspberry Pi" in f.read()
    except OSError:
        return False
