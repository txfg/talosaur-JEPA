"""Arm switch and start gate (talosaur.onboard.arming): the vehicle never thrusts on deck. Torch-free."""

from __future__ import annotations

import sys
import types

import pytest

from talosaur.onboard import arming as arming_mod
from talosaur.onboard.arming import ARMED, RUN, SAFE, Arming, ArmingConfig, make_arming


class _Switch:
    def __init__(self, on: bool = False):
        self.on = on

    def __call__(self) -> bool:
        return self.on


def _arming(switch, **kw) -> Arming:
    cfg = ArmingConfig(kind="file", **{"debounce_s": 0.5, "arm_delay_s": 10.0, "start_hold_s": 3.0, **kw})
    return Arming(cfg, read_switch=switch)


def test_countdown_then_in_the_water_then_run_and_off_stops_at_once():
    sw = _Switch()
    a = _arming(sw)
    assert a.update(0.0, 0.0) == [] and a.state == SAFE
    sw.on = True
    assert a.update(1.0, 0.0) == [] and a.state == SAFE  # not yet: debounce
    assert a.update(1.6, 0.0) == ["armed"] and a.state == ARMED
    st = a.status(1.6)
    assert st["wait"] == "countdown" and st["countdown_s"] == pytest.approx(10.0)
    # on deck after the countdown: it waits for the water
    assert a.update(12.0, 0.02) == [] and a.state == ARMED and a.wait == "in the water"
    assert a.update(13.0, None) == [] and a.wait == "no depth"
    # in the water: 3 s at >= 0.2 m, and one wave that lifts the sensor restarts the hold
    for t, d in ((14.0, 0.3), (15.0, 0.1), (16.0, 0.3), (18.0, 0.35)):
        assert a.update(t, d) == [] and a.state == ARMED
    assert a.update(19.0, 0.3) == ["mission_start"] and a.state == RUN and a.missions == 1
    sw.on = False
    assert a.update(19.2, 5.0) == [] and a.state == RUN  # a knock shorter than the debounce ...
    sw.on = True
    assert a.update(19.4, 5.0) == [] and a.state == RUN  # ... changes nothing
    sw.on = False
    a.update(20.0, 5.0)
    assert a.update(20.6, 5.0) == ["disarmed"] and a.state == SAFE  # magnet off: stop
    assert a.status(20.6) == {"state": SAFE, "switch": False, "missions": 1}


def test_hold_in_the_water_can_run_during_the_countdown_and_countdown_only_mode():
    sw = _Switch(True)
    a = _arming(sw, debounce_s=0.0)
    assert a.update(0.0, 1.0) == ["armed"]
    for t in (2.0, 5.0, 9.9):
        a.update(t, 1.0)
    assert a.update(10.0, 1.0) == ["mission_start"]  # in the water for 10 s already
    b = _arming(_Switch(True), debounce_s=0.0, start_depth_m=None)
    b.update(0.0, None)
    assert b.update(9.0, None) == [] and b.update(10.0, None) == ["mission_start"]  # no depth needed


def test_switch_read_errors_count_as_off():
    def broken():
        raise OSError("gpio gone")

    a = _arming(broken, debounce_s=0.0)
    for t in (0.0, 1.0, 20.0):
        assert a.update(t, 5.0) == [] and a.state == SAFE
    assert a.status(20.0)["read_errors"] == 3


def test_make_arming_kinds(tmp_path, monkeypatch):
    assert make_arming(None).state == RUN and make_arming({"kind": "none"}).update(0.0, None) == []
    flag = tmp_path / "arm"
    a = make_arming({"kind": "file", "path": str(flag), "debounce_s": 0.0, "arm_delay_s": 0.0})
    assert a.update(0.0, 1.0) == [] and a.state == SAFE
    flag.touch()
    assert a.update(1.0, 1.0) == ["armed"]
    with pytest.raises(ValueError):
        make_arming({"kind": "magic"})
    with pytest.raises(TypeError):
        make_arming({"kind": "gpio", "pn": 17})  # a typo is an error, not a silent default

    def no_gpio(pin, invert):
        raise RuntimeError("no GPIO here")

    monkeypatch.setattr(arming_mod, "_gpio_reader", no_gpio)
    g = make_arming({"kind": "gpio", "debounce_s": 0.0})
    for t in (0.0, 30.0):
        assert g.update(t, 5.0) == [] and g.state == SAFE  # it records, it never thrusts
    assert "GPIO 17 unavailable" in g.status(30.0)["fault"]


def test_gpio_polarity_with_a_stand_in_for_gpiozero(monkeypatch):
    class FakeInput:
        level = 1  # 1 = the pin is high: switch open

        def __init__(self, pin, pull_up=False):
            assert pin == 17 and pull_up is True
            self.closed = False

        @property
        def value(self):  # gpiozero: with pull_up=True, value 1 means the pin is pulled low
            return 1 - FakeInput.level

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "gpiozero", types.SimpleNamespace(DigitalInputDevice=FakeInput))
    a = make_arming({"kind": "gpio", "debounce_s": 0.0})
    assert a.update(0.0, None) == []  # open switch: no magnet
    FakeInput.level = 0  # the magnet closes the switch to ground
    assert a.update(1.0, None) == ["armed"]
    nc = make_arming({"kind": "gpio", "invert": True, "debounce_s": 0.0})  # normally-closed switch
    assert nc.update(0.0, None) == []  # closed without the magnet
    FakeInput.level = 1
    assert nc.update(1.0, None) == ["armed"]
    a.close()
