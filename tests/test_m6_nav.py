"""Navigation input (depth / heading / turn rate) for search planning. Torch-free."""

from __future__ import annotations

import json
import socket

import pytest

from talosaur.guidance.nav import NavState, wrap180
from talosaur.onboard.nav_input import NullNav, UdpNavSource, make_nav


def test_nav_state_staleness_and_angles():
    n = NavState(10.0, depth_m=150.0, heading_deg=359.0, valid_for_s=1.0)
    assert n.depth(10.5) == 150.0 and n.heading(10.9) == 359.0
    assert n.depth(11.5) is None and not n.fresh(11.5)
    assert wrap180(350.0 - 10.0) == pytest.approx(-20.0)
    assert wrap180(10.0 - 350.0) == pytest.approx(20.0)
    assert wrap180(180.0) == pytest.approx(-180.0)
    assert n.as_dict()["depth_m"] == 150.0 and "valid_for_s" not in n.as_dict()


def test_udp_nav_source_keeps_the_newest_sample_and_skips_garbage():
    nav = UdpNavSource("127.0.0.1", 0, valid_for_s=1.0)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        assert nav.poll(0.0) is None
        for depth in (100.0, 101.5):
            tx.sendto(json.dumps({"depth_m": depth, "heading_deg": 42.0}).encode(), ("127.0.0.1", nav.port))
        tx.sendto(b"not json", ("127.0.0.1", nav.port))
        tx.sendto(json.dumps({"depth_m": "deep"}).encode(), ("127.0.0.1", nav.port))
        s = None
        for _ in range(100):  # datagrams on loopback arrive almost at once
            s = nav.poll(1.0)
            if s is not None and s.depth_m == 101.5 and nav.bad == 2:
                break
        assert s.depth_m == 101.5 and s.heading_deg == 42.0 and s.yaw_rate_dps is None
        assert nav.bad == 2
        assert nav.poll(2.5) is None  # stale
    finally:
        tx.close()
        nav.close()


def test_make_nav_variants():
    assert isinstance(make_nav(None), NullNav) and make_nav({"kind": "none"}).poll(1.0) is None
    udp = make_nav({"kind": "udp", "host": "127.0.0.1", "port": 0})
    assert isinstance(udp, UdpNavSource)
    udp.close()
    with pytest.raises(NotImplementedError):
        make_nav({"kind": "mavlink"})
    with pytest.raises(ValueError):
        make_nav({"kind": "sextant"})
