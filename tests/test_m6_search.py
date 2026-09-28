"""Search planner and closed-loop simulator tests (numpy only)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from talosaur.guidance.lights import LightPolicy, LightsConfig
from talosaur.guidance.nav import NavState
from talosaur.guidance.pipeline import Guidance, GuidanceConfig
from talosaur.guidance.search import SearchConfig, SearchPlanner
from talosaur.onboard.dive_report import load as load_logs
from talosaur.onboard.dive_report import summarise as dive_report
from talosaur.sim.camera import SimCamera
from talosaur.sim.run import build, run_episode
from talosaur.sim.vehicle import Vehicle
from talosaur.sim.world import World, WorldConfig


def _nav(t, depth=50.0, heading=0.0):
    return NavState(t, depth_m=depth, heading_deg=heading)


def test_without_navigation_it_scans_first_then_hops():
    p = SearchPlanner(SearchConfig(scan_s=5.0, hop_s=5.0))
    cmds = [p.command(k * 0.5, None) for k in range(40)]
    assert cmds[0].yaw_rate > 0 and cmds[0].surge == 0.0  # look around first
    assert any(c.surge > 0 and c.yaw_rate == 0 for c in cmds)  # then move somewhere else
    assert all(c.depth_m is None and c.heave == 0.0 for c in cmds)  # no depth control without depth


def test_profiles_the_whole_range_then_works_bands():
    c = SearchConfig(min_depth_m=20, max_depth_m=100, band_m=10)
    p = SearchPlanner(c)
    depth, t = 50.0, 0.0
    targets = []
    for _ in range(3000):
        t += 0.5
        p.observe(t, _nav(t, depth), False, 0.3, 0.0)
        cmd = p.command(t, _nav(t, depth))
        targets.append(cmd.depth_m)
        depth += float(np.clip(cmd.depth_m - depth, -0.15, 0.15))  # a vehicle holding depth at 0.3 m/s
    assert targets[0] == 100.0  # down first ...
    assert 20.0 in targets  # ... then up to the top of the range
    later = targets[-500:]
    assert not p.profile and all(t0 in set(p.bands) for t0 in later)  # then band centres
    assert p.exposure.sum() > 0


def test_profile_finishes_when_the_bottom_is_shallower_than_the_range():
    p = SearchPlanner(SearchConfig(min_depth_m=20, max_depth_m=200, band_m=10, stall_s=30))
    depth, t = 30.0, 0.0
    targets = []
    for _ in range(3000):
        t += 0.5
        p.observe(t, _nav(t, depth), False, 0.3, 0.0)
        cmd = p.command(t, _nav(t, depth))
        targets.append(cmd.depth_m)
        depth = min(45.0, depth + float(np.clip(cmd.depth_m - depth, -0.15, 0.15)))  # a lake 45 m deep
    assert not p.profile and p.depth_hi == pytest.approx(45.0)
    assert set(targets[-1000:]) <= {25.0, 35.0, 45.0}  # only bands it can reach


def test_time_spent_filming_does_not_count_as_a_stalled_profile():
    p = SearchPlanner(SearchConfig(min_depth_m=20, max_depth_m=200, stall_s=30))
    depth, t = 50.0, 0.0
    for k in range(2000):
        t += 0.5
        filming = 100 <= k < 400  # 150 s on an animal at 80 m: the planner is not in control
        p.observe(t, _nav(t, depth), False, 0.3, 0.0, searching=not filming)
        if filming:
            continue
        cmd = p.command(t, _nav(t, depth))
        depth += float(np.clip(cmd.depth_m - depth, -0.15, 0.15))
        if depth >= 190:
            break
    assert depth >= 190 and p.depth_hi == 200.0  # it went on down to the bottom of the range


def test_keeps_off_the_bottom_with_an_altimeter():
    p = SearchPlanner(SearchConfig(min_altitude_m=5.0))
    nav = NavState(1.0, depth_m=30.0, heading_deg=0.0, altitude_m=3.0)
    p.observe(1.0, nav, False, 0.0, 0.0)
    cmd = p.command(1.0, nav)  # the profile wants 200 m; the bottom is 33 m
    assert cmd.depth_m == pytest.approx(28.0) and cmd.heave > 0  # rise to 5 m above it


def test_depth_control_without_a_heading():
    p = SearchPlanner(SearchConfig())
    nav = NavState(1.0, depth_m=60.0)  # a depth sensor but no compass
    p.observe(1.0, nav, False, 0.0, None)
    cmd = p.command(1.0, nav)
    assert cmd.depth_m == 200.0 and cmd.heading_deg is None and cmd.yaw_rate > 0  # scan while profiling


def test_filming_time_is_not_search_time():
    p = SearchPlanner(SearchConfig())
    t = 0.0
    for _ in range(600):  # 5 min filming one animal at 55 m
        t += 0.5
        p.observe(t, _nav(t, 55.0), True, 0.1, 0.0, searching=False)
    assert p.exposure.sum() == 0.0 and p.events.sum() == 0.0
    p.observe(t + 0.5, _nav(t + 0.5, 55.0), False, 0.1, 0.0)
    assert p.exposure.sum() > 0.0


def test_thompson_sampling_favours_the_band_where_animals_are():
    p = SearchPlanner(SearchConfig(min_depth_m=20, max_depth_m=200, diel_prior=False, seed=1))
    t = 0.0
    for depth in np.arange(20.0, 200.0, 10.0):  # 60 s in every band, animals only at 120-130 m
        for _ in range(120):
            t += 0.5
            hit = 120 <= depth < 130 and int(t * 2) % 20 == 0  # a new animal every 10 s
            p.observe(t, _nav(t, depth), hit, 0.0, None)
    picks = [p.bands[p.choose_band(t, hour=12.0)] for _ in range(200)]
    assert np.mean(np.array(picks) == 125.0) > 0.8


def test_flicker_counts_as_one_detection():
    p = SearchPlanner(SearchConfig())
    t = 0.0
    for k in range(20):  # detected every other frame for 4 s: one animal, not ten
        t += 0.2
        p.observe(t, _nav(t, 55.0), k % 2 == 0, 0.0, None)
    assert p.events.sum() == pytest.approx(1.0, abs=0.01)


def test_diel_prior_prefers_the_corridor_at_dusk_and_shallow_water_at_night():
    p = SearchPlanner(SearchConfig(min_depth_m=20, max_depth_m=200))
    dusk, night, noon = p._diel_prior(18.5), p._diel_prior(23.0), p._diel_prior(12.0)
    b = p.bands
    assert dusk[(b >= 100) & (b <= 200)].mean() > 2 * dusk[b < 100].mean()
    corridor = (b >= 100) & (b <= 200)
    assert p._diel_prior(4.5)[corridor].mean() > 2.0  # the descent starts ~2 h before sunrise ...
    assert p._diel_prior(7.5)[corridor].mean() < 2.0  # ... and is over an hour after it
    assert night[np.argmin(abs(b - 60))] > 2 * night[np.argmin(abs(b - 190))]
    assert np.allclose(noon, 1.0)


def test_intensive_search_after_a_find_then_back_to_long_legs():
    c = SearchConfig(giveup_s=30.0)
    p = SearchPlanner(c)
    t, heading = 0.0, 0.0
    p.command(t, _nav(t, heading=heading))
    assert p.mode == "extensive"
    p.observe(1.0, _nav(1.0), True, 0.3, heading)  # an animal
    assert p.mode == "intensive"
    cmd = p.command(1.5, _nav(1.5, heading=heading))
    assert cmd.surge <= c.ars_surge
    p.command(40.0, _nav(40.0, heading=heading))  # nothing for 39 s: give up
    assert p.mode == "extensive"


def test_new_legs_avoid_water_already_searched():
    p = SearchPlanner(SearchConfig(seed=3))
    t = 0.0
    for _ in range(600):  # 5 min heading north through the water
        t += 0.5
        p.observe(t, _nav(t, heading=0.0), False, 0.35, 0.0)
    p._new_leg(t, heading=180.0)  # coming back south: do not retrace the northward track
    assert abs(((p.leg_heading - 180.0) + 180) % 360 - 180) >= 45


def test_marine_snow_does_not_count_as_a_find_but_an_animal_does():
    g = Guidance(GuidanceConfig())
    empty = np.full((7, 13), -4.0, np.float32)
    speck = empty.copy()
    speck[2, 3] = 2.0  # one bright patch in one frame
    t = 0.0
    for k in range(100):  # 20 s with a speck every 5th frame
        t += 0.2
        g.step(t, np.array([-4.0]), speck if k % 5 == 0 else empty, nav=_nav(t, depth=55.0))
    assert g.search.events.sum() == 0.0 and g.search.mode == "extensive"
    fish = empty.copy()
    fish[3, 6] = fish[3, 7] = 3.0
    for _ in range(10):  # an animal in view for 2 s
        t += 0.2
        g.step(t, np.array([2.0]), fish, nav=_nav(t, depth=55.0))
    assert g.search.events.sum() == pytest.approx(1.0, abs=0.01) and g.search.mode == "intensive"


def test_search_telemetry_and_setpoints_in_the_pipeline():
    g = Guidance(GuidanceConfig())
    empty = np.full((7, 13), -4.0, np.float32)
    _, tele, _ = g.step(0.1, np.array([-4.0]), empty, nav=_nav(0.1, depth=40.0, heading=90.0))
    assert tele["state"] == "SEARCH" and tele["search"]["mode"] == "profile"
    assert tele["cmd"]["depth_m"] == 200.0 and tele["cmd"]["heading_deg"] is not None
    assert tele["cmd"]["light"] == 0.0 and tele["nav"]["depth_m"] == 40.0
    json.dumps(tele)


# ----------------------------------------------------------------------------- lamp


def _lamp(policy, luma, seconds, t0=0.0, dt=0.2, close=False):
    """Run the lamp policy; ``luma(lamp)`` is the picture brightness under last frame's lamp level;
    ``close`` also means an animal is being filmed."""
    levels, t = [], t0
    for _ in range(int(round(seconds / dt))):
        t += dt
        levels.append(policy.update(t, luma(policy.level), close, engaged=close))
    return levels, t


def test_lamp_never_switches_on_suddenly_and_skips_checks_while_filming():
    p = LightPolicy(LightsConfig(search_dark=0.3, ramp_s=5.0, check_s=60, check_len_s=3))
    night = lambda lamp: 0.02 + 0.3 * lamp  # noqa: E731
    levels, t = _lamp(p, night, 10)
    steps = np.diff([0.0] + levels)
    assert p.dark and levels[-1] == pytest.approx(0.3) and steps.max() <= 0.2 / 5.0 + 1e-9  # ramped
    levels, t = _lamp(p, night, 120, t0=t, close=True)  # filming an animal for 2 min ...
    assert min(levels) == pytest.approx(0.3)  # ... no lamp-off check, no change when close


def test_lamp_stays_off_while_the_camera_can_see():
    p = LightPolicy(LightsConfig())
    levels, _ = _lamp(p, lambda lamp: 0.4, 60)
    assert set(levels) == {0.0} and not p.dark


def test_lamp_goes_dim_in_the_dark_and_checks_for_daylight():
    p = LightPolicy(LightsConfig(search_dark=0.3, check_s=60, check_len_s=3))
    night = lambda lamp: 0.02 + 0.3 * lamp  # noqa: E731 - our own lamp brightens the picture
    levels, t = _lamp(p, night, 30)
    assert levels[0] == 0.0 and levels[-1] == 0.3 and p.dark
    levels, t = _lamp(p, night, 70, t0=t)  # a lamp-off check after 60 s: still night
    assert 0.0 in levels and levels[-1] == 0.3 and p.dark
    dawn = lambda lamp: 0.3 + 0.3 * lamp  # noqa: E731
    levels, t = _lamp(p, dawn, 70, t0=t)  # the next check sees daylight: off for good
    assert levels[-1] == 0.0 and not p.dark and set(levels[-50:]) == {0.0}


def test_lamp_track_level_when_close_and_fixed_level_without_brightness():
    p = LightPolicy(LightsConfig(search=0.1, track=0.5, ramp_s=0.0))
    assert p.update(1.0, None, close=False) == 0.1 and p.update(1.2, None, close=True) == 0.5
    assert LightPolicy(LightsConfig(control=False)).update(1.0, 0.5, False) is None
    default = LightPolicy(LightsConfig(search=0.2, ramp_s=0.0))  # track: None
    assert default.update(1.0, None, close=False) == default.update(1.2, None, close=True) == 0.2
    g = Guidance(GuidanceConfig())
    empty = np.full((7, 13), -4.0, np.float32)
    for k in range(20):  # 4 s of dark water
        _, tele, _ = g.step(0.2 * (k + 1), np.array([-4.0]), empty, luma=0.01)
    assert tele["cmd"]["light"] == g.cfg.lights.search_dark and tele["lights"]["dark"]


# ----------------------------------------------------------------------------- simulator


def test_world_migrates_with_the_time_of_day():
    w = World(WorldConfig(size_m=200.0, layer_density=0.001, start_hour=12.0))
    assert w.layer_depth() == pytest.approx(w.cfg.day_layer_m)
    assert w.layer_depth(t=10 * 3600) == pytest.approx(w.cfg.night_layer_m)  # 22:00
    dusk = w.layer_depth(t=6.5 * 3600)  # 18:30, half way up
    assert w.cfg.night_layer_m < dusk < w.cfg.day_layer_m
    assert w.n > 1000 and 0.3 < np.mean(w.patch >= 0) < 0.7


def test_spatial_index_finds_every_animal_in_range():
    w = World(WorldConfig(size_m=200.0, layer_density=0.01, start_hour=22.0))
    rng = np.random.default_rng(0)
    p = np.array([100.0, 100.0, w.layer_depth()])
    heading = 0.0
    for k in range(240):  # 4 min: the vehicle cruises and turns with its lights on (animals flee)
        heading += rng.normal(0, 0.2)
        p[:2] = (p[:2] + 0.6 * np.array([np.sin(heading), np.cos(heading)])) % 200.0
        if k == 120:
            p[:2] = (p[:2] + 60.0) % 200.0  # a jump well outside the indexed box
        w.step(1.0, p, light=1.0)
        if k % 5 == 0:
            everyone = np.arange(w.n)
            dist = np.linalg.norm(w.offsets(everyone, p), axis=1)
            for r in (2.5, 23.5):
                assert set(everyone[dist < r].tolist()) <= set(w.near(p, r).tolist())
    assert (w.flee_until > 0).any()


def test_patches_move_together():
    w = World(WorldConfig(size_m=300.0, layer_density=0.002))
    members = np.nonzero(w.patch == w.patch[0])[0]
    cruise = max(s.speed_mps for s in w.cfg.species)
    assert len(members) > 5 and np.linalg.norm(w.v[members].std(axis=0)) < 0.15 * cruise


def test_camera_sees_an_animal_straight_ahead():
    w = World(WorldConfig(size_m=200.0, layer_density=0.0001, start_hour=22.0))
    v = Vehicle(pos=(100.0, 100.0, w.layer_depth()))
    i = 0
    w.xy0[i] = (100.0, 101.5)  # 1.5 m north (ahead), same depth
    w.v[i] = 0.0
    w.migrant[i], w.dz[i], w.size[i] = True, 0.0, 0.3
    w.invalidate_index()  # animals were moved by hand
    cam = SimCamera(seed=0)
    hits = [cam.render(v, w, light=1.0)[4] for _ in range(20)]
    seen = [h for hs in hits for h in hs if h[0] == i]
    assert len(seen) > 15
    _, u, vv, dist, conf = seen[0]
    assert (
        u == pytest.approx(0.5, abs=0.02)
        and vv == pytest.approx(0.5, abs=0.02)
        and dist == pytest.approx(1.5)
    )
    frame, heat, emb, tok, _ = cam.render(v, w, light=1.0)
    assert heat[3, 6] > 1.0 and heat.shape == (7, 13) and tok.shape == (7, 13, w.cfg.token_dim)


def test_closed_loop_episode_runs_and_scores(tmp_path):
    g, w, v, c = build({}, start_hour=22.0, seed=0, world_overrides={"size_m": 300.0, "layer_density": 0.01})
    tele = tmp_path / "tele.jsonl"
    r = run_episode(g, w, v, c, 300.0, fps=5.0, trace_every_s=30.0, telemetry_path=tele)
    for key in ("value_per_h", "animals_per_h", "detections_per_h", "state_share", "depth_share", "trace"):
        assert key in r
    assert r["distance_km"] > 0 and len(r["trace"]) == 10
    assert sum(r["state_share"].values()) == pytest.approx(1.0, abs=0.01)
    json.dumps(r)
    frames, encounters = load_logs([tele])  # the dive report reads simulator telemetry like a dive log
    assert len(frames) == 1500 and len(encounters) == r["encounters"]
    text = dive_report(frames, encounters)
    assert "| depth band |" in text and "## Encounters" in text and "Lamp: mean level" in text
