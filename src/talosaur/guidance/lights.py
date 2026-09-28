"""Lamp policy: as little light as possible, but enough to see.

Midwater animals avoid artificial light - red included - often tens of metres away; dim far-red
light disturbs least (docs/SEARCH.md §1). So the lamp stays off while searching as long as the
camera can see by ambient light (a lake by day, the upper twilight zone by day), and goes to a
dim level only when it is too dark. Close to an animal it uses the ``track`` level.

"Too dark" comes from the frame's mean brightness with the lamp off: with auto-exposure, a scene
the camera can expose properly sits near the exposure target, so a mean well below it means the
exposure and gain are at their limits. While the lamp is on for darkness, it is switched off
every ``check_s`` for ``check_len_s`` to see whether ambient light is back (dawn, shallower
water). Switching a lamp *off* does not disturb animals.

Without a brightness input (``luma=None``) the lamp simply uses ``search`` while searching.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class LightsConfig:
    control: bool = True  # False: leave the lamp to the vehicle (Command.light = None)
    search: float = 0.0  # lamp level while searching when the camera can see by ambient light ...
    search_dark: float = 0.3  # ... and when it cannot (needs the frame brightness)
    track: float = 0.3  # once an animal is close (apparent size >= near_size)
    near_size: float = 0.08
    dark_luma: float = 0.08  # mean frame brightness (0-1) with the lamp off below which it is too dark
    settle_s: float = 1.0  # after the lamp goes off, wait this long (auto-exposure) before measuring
    check_s: float = 120.0  # lamp on for darkness: switch it off this often ...
    check_len_s: float = 3.0  # ... for this long, to see whether ambient light is back


class LightPolicy:
    def __init__(self, cfg: LightsConfig | None = None):
        self.cfg = cfg or LightsConfig()
        self.dark = False
        self.ambient: float | None = None  # smoothed brightness measured with the lamp off
        self.level = 0.0  # the level commanded last frame
        self.off_since: float | None = 0.0
        self.check_until: float | None = None
        self.next_check = math.inf
        self._t: float | None = None

    def update(self, t: float, luma: float | None, close: bool) -> float | None:
        """Lamp level (0-1) for this frame; ``close``: filming an animal near the camera."""
        c = self.cfg
        if not c.control:
            return None
        dt = 0.0 if self._t is None else max(0.0, t - self._t)
        self._t = t
        lamp_off = self.level <= 0.0 and self.off_since is not None and t - self.off_since >= c.settle_s
        if luma is not None and lamp_off:  # ambient light only: our own lamp is off (and settled)
            if self.ambient is None:
                self.ambient = float(luma)
            else:
                self.ambient += (1.0 - math.exp(-dt)) * (float(luma) - self.ambient)
        if self.dark and self.check_until is not None and t >= self.check_until:  # a lamp-off check ends
            self.check_until = None
            if self.ambient is not None and self.ambient >= 1.5 * c.dark_luma:
                self.dark = False  # enough ambient light again
            else:
                self.next_check = t + c.check_s
        elif not self.dark and self.ambient is not None and self.ambient < c.dark_luma:
            self.dark, self.next_check = True, t + c.check_s
        if close:
            level = c.track
        elif self.dark:
            if self.check_until is None and t >= self.next_check:
                self.check_until, self.ambient = t + c.check_len_s, None  # measure afresh
            level = c.search if self.check_until is not None else c.search_dark
        else:
            level = c.search
        if level <= 0.0 and self.level > 0.0:
            self.off_since = t
        elif level > 0.0:
            self.off_since = None
        self.level = level
        return level

    def status(self) -> dict:
        return {
            "dark": self.dark,
            "ambient": None if self.ambient is None else round(self.ambient, 3),
            "checking": self.check_until is not None,
        }
