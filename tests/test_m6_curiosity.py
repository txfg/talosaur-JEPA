"""Curiosity tests: bioluminescent flashes and weak detections worth a closer look. numpy only."""

from __future__ import annotations

import numpy as np
import pytest

from talosaur.guidance.curiosity import Curiosity, CuriosityConfig, FlashDetector, glow_grid
from talosaur.guidance.pipeline import Guidance, GuidanceConfig

GRID = (14, 26)


def _dark(rng, level=0.02):
    return level + rng.normal(0, 0.002, GRID)  # a dark picture with sensor noise


def test_glow_grid_is_the_blue_green_brightness_of_coarse_cells():
    rgb = np.zeros((112, 208, 3), np.uint8)
    rgb[:8, :8] = (255, 0, 0)  # red lamp light: not blue-green
    rgb[:8, -8:] = (0, 255, 255)  # a blue-green flash in the top-right cell
    g = glow_grid(rgb)
    assert g.shape == GRID and g[0, 0] == 0.0 and g[0, -1] == pytest.approx(1.0) and g[5, 5] == 0.0


def test_flash_detector_finds_a_brief_local_brightening_but_not_a_global_one():
    rng = np.random.default_rng(0)
    det = FlashDetector(CuriosityConfig())
    t = 0.0
    for _ in range(50):  # 10 s of dark water: learn the background
        t += 0.2
        assert det.update(t, _dark(rng)) is None
    frame = _dark(rng)
    frame[3, 20] += 0.2  # a flash, upper right
    t += 0.2
    cx, cy, strength = det.update(t, frame)
    assert cx == pytest.approx(20.5 / 26) and cy == pytest.approx(3.5 / 14) and strength > 0.15
    for _ in range(10):  # quiet again
        t += 0.2
        det.update(t, _dark(rng))
    t += 0.2
    assert det.update(t, _dark(rng) + 0.2) is None  # the whole picture brightens: lamp or exposure


def test_a_weak_detection_must_persist_before_it_is_worth_a_look():
    cur = Curiosity(CuriosityConfig(flashes=False))
    empty = np.full((7, 13), 0.02)
    weak = empty.copy()
    weak[3, 10] = 0.45  # below the detection threshold, right of centre
    assert cur.update(0.2, True, weak, None) is None  # one frame: could be marine snow
    assert cur.update(0.4, True, empty, None) is None
    assert cur.update(0.6, True, weak, None) is not None  # seen again in the same place: look
    assert cur.looking and cur.cue == "weak"
    t = 0.6
    for _ in range(20):  # gone for 4 s: stop looking, then a cooldown
        t += 0.2
        cur.update(t, True, empty, None)
    assert not cur.looking
    assert cur.update(t + 0.2, True, weak, None) is None and cur.update(t + 0.4, True, weak, None) is None


def test_looks_last_at_most_max_s_and_only_while_searching():
    cur = Curiosity(CuriosityConfig(flashes=False, max_s=3.0))
    weak = np.full((7, 13), 0.02)
    weak[3, 6] = 0.45
    t, looked = 0.0, []
    for _ in range(30):  # a weak thing that never becomes a detection
        t += 0.2
        looked.append(cur.update(t, True, weak, None) is not None)
    assert any(looked) and not looked[-1]  # gave up after max_s, then cooldown
    assert cur.update(t + 20.0, False, weak, None) is None  # not searching: no looks


def test_pipeline_turns_toward_a_flash_while_searching():
    g = Guidance(GuidanceConfig())
    rng = np.random.default_rng(1)
    empty = np.full((7, 13), -4.0, np.float32)
    t = 0.0
    for _ in range(40):  # 8 s of dark water, searching
        t += 0.2
        cmd, tele, _ = g.step(t, np.array([-4.0]), empty, glow=_dark(rng))
    assert tele["state"] == "SEARCH" and not tele["curiosity"]["looking"]
    flash = _dark(rng)
    flash[7, 24] += 0.3  # a flash far to the right
    t += 0.2
    g.step(t, np.array([-4.0]), empty, glow=flash)
    for _ in range(5):
        t += 0.2
        cmd, tele, _ = g.step(t, np.array([-4.0]), empty, glow=_dark(rng))
    assert tele["curiosity"]["looking"] and tele["curiosity"]["cue"] == "flash"
    assert cmd.yaw_rate > 0.0  # turning right, toward it
