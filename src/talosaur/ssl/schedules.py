"""Per-step schedules (learning rate, weight decay, EMA momentum), as in I-JEPA."""

from __future__ import annotations

import math


def warmup_cosine(step: int, total: int, warmup: int, peak: float, start: float, final: float) -> float:
    """Linear warmup start -> peak over ``warmup`` steps, then cosine peak -> final."""
    if warmup > 0 and step < warmup:
        return start + (peak - start) * step / warmup
    t = min(1.0, (step - warmup) / max(1, total - warmup))
    return final + 0.5 * (peak - final) * (1.0 + math.cos(math.pi * t))


def cosine(step: int, total: int, start: float, end: float) -> float:
    t = min(1.0, step / max(1, total))
    return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * t))


def linear(step: int, total: int, start: float, end: float) -> float:
    t = min(1.0, step / max(1, total))
    return start + (end - start) * t


def weight_decay(step: int, total: int, start: float, end: float) -> float:
    """I-JEPA increases weight decay along a cosine from ``start`` to ``end``."""
    t = min(1.0, step / max(1, total))
    return start + 0.5 * (end - start) * (1.0 - math.cos(math.pi * t))
