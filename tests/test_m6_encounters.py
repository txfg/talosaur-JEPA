"""Encounter tests: a time budget per animal, moving on, recognising animals already filmed by
appearance (patch tokens), resuming an animal that was only briefly lost. numpy only."""

from __future__ import annotations

import json

import numpy as np
import pytest

from talosaur.guidance.controller import Controller, ControllerConfig
from talosaur.guidance.encounters import EncounterConfig, EncounterManager
from talosaur.guidance.heatmap import Target, find_blobs, order_blobs
from talosaur.guidance.pipeline import Guidance, GuidanceConfig
from talosaur.guidance.state_machine import FSMConfig, GuidanceFSM, State
from talosaur.guidance.tracker import TrackState

GRID = (7, 13)
D = 16
_rng = np.random.default_rng(0)
WATER = _rng.normal(size=D)


def _animal(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=D)
    return WATER + 3.0 * v / np.linalg.norm(v)


FISH_A, FISH_B = _animal(1), _animal(2)


def _scene(fish, seed: int = 0):
    """fish: list of (appearance, row, col) -> (heatmap logits, tokens (h, w, D)); each fish covers
    two neighbouring patches."""
    rng = np.random.default_rng(seed)
    p = np.full(GRID, 0.02, np.float32)
    tok = WATER[None, None, :] + 0.05 * rng.normal(size=(*GRID, D))
    for vec, y, x in fish:
        p[y, x : x + 2] = 0.95
        tok[y, x : x + 2] = vec + 0.05 * rng.normal(size=(2, D))
    return np.log(p / (1 - p)).astype(np.float32), tok.astype(np.float32)


def _cfg(**enc) -> GuidanceConfig:
    return GuidanceConfig(
        fsm=FSMConfig(release_s=2.0, postroll_s=1.0, lost_timeout_s=1.0),
        encounter=EncounterConfig(**{"max_s": 3.0, **enc}),
    )


def _run(g: Guidance, frames, t0: float = 0.0, fps: float = 10.0, tokens: bool = True):
    """frames: list of fish lists; returns per-frame telemetry."""
    out = []
    for k, fish in enumerate(frames):
        heat, tok = _scene(fish, seed=k)
        frame_logit = np.array([3.0 if fish else -4.0])
        _, tele, _ = g.step(t0 + k / fps, frame_logit, heat, None, tok if tokens else None)
        json.dumps(tele)  # telemetry must stay JSON-serialisable
        out.append(tele)
    return out


def _events(teles):
    return [(t["t"], e) for t in teles for e in t["events"]]


# ----------------------------------------------------------------------------- building blocks


def test_find_blobs_returns_all_blobs_with_masks_in_preference_order():
    heat, _ = _scene([(FISH_A, 1, 1), (FISH_B, 5, 9)])
    prob = 1 / (1 + np.exp(-heat))
    blobs, peak, n = find_blobs(prob)
    assert len(blobs) == 2 and n == 2 and peak > 0.9
    assert all(b.mask.sum() == 2 for b in blobs)
    b0 = order_blobs(blobs)[0]
    near_b = order_blobs(blobs, prev_xy=(10 / 13, 5.5 / 7))[0]
    assert b0.mask[1, 1] or b0.mask[5, 9]  # equal mass: either, but deterministic
    assert near_b.mask[5, 9] and near_b.mask[5, 10]


def test_appearance_descriptor_separates_animals():
    m = EncounterManager(EncounterConfig())
    descs = {}
    for name, vec in (("a1", FISH_A), ("a2", FISH_A), ("b", FISH_B)):
        heat, tok = _scene([(vec, 3, 5)], seed=len(descs))
        prob = 1 / (1 + np.exp(-heat))
        m.observe_background(tok, prob)
        blob = find_blobs(prob)[0][0]
        descs[name] = m.describe(tok, prob, blob.mask)
    assert descs["a1"] @ descs["a2"] > 0.95  # same animal, different frames
    assert descs["a1"] @ descs["b"] < 0.6  # a different animal
    assert m.describe(None, prob, blob.mask) is None


def test_controller_release_profile():
    ctrl = Controller(ControllerConfig(release_backoff_s=1.0, release_turn_s=2.0, slew_per_s=100.0))
    assert ctrl.release(0.5, 0.5, away=-1.0).surge == pytest.approx(-0.2)
    turn = ctrl.release(1.5, 1.5, away=-1.0)
    assert turn.yaw_rate == pytest.approx(-0.6) and turn.surge == 0.0
    assert ctrl.release(3.5, 3.5, away=-1.0).surge == pytest.approx(0.3)


def test_fsm_release_ignores_detections_then_searches():
    fsm = GuidanceFSM(FSMConfig(release_s=1.0, postroll_s=0.5))
    tgt = Target(True, 0.5, 0.5, 0.2, 2.0, 0.9)
    confirmed = TrackState(active=True, confirmed=True, size=0.2, hits=5)
    events = []
    for k in range(4):
        events += fsm.update(k * 0.1, 0.9, tgt, TrackState(active=True, hits=k + 1) if k < 3 else confirmed)
    assert fsm.state == State.TRACK and "encounter_start" in events
    ev = fsm.update(0.4, 0.9, tgt, confirmed, release=True)
    assert fsm.state == State.RELEASE and ev[0] == "encounter_end" and fsm.end_reason == "budget"
    for k in range(5, 14):  # the animal is still in plain view: ignored
        events += fsm.update(k * 0.1, 0.9, tgt, confirmed)
        assert fsm.state == State.RELEASE
    assert "stop_recording" in events and not fsm.recording  # post-roll ended the video
    fsm.update(1.5, 0.9, tgt, confirmed)
    assert fsm.state == State.SEARCH


# ----------------------------------------------------------------------------- scenarios


def test_budget_release_then_ignores_filmed_animal_and_picks_a_different_one():
    g = Guidance(_cfg())
    a = (FISH_A, 3, 8)  # right of centre
    teles = _run(g, [[a]] * 80)  # 8 s with only fish A in view
    ev = _events(teles)
    starts = [t for t, e in ev if e == "encounter_start"]
    ends = [t for t, e in ev if e == "encounter_end"]
    assert starts == [pytest.approx(0.3)] and len(ends) == 1
    assert ends[0] == pytest.approx(0.3 + 3.0, abs=0.15)  # the time budget
    summary = next(t["encounter_summary"] for t in teles if "encounter_summary" in t)
    assert summary["id"] == 1 and summary["reason"] == "budget" and summary["done"] and summary["appearance"]
    stop = [t for t, e in ev if e == "stop_recording"]
    assert stop and stop[0] == pytest.approx(ends[0] + 1.0, abs=0.15)  # post-roll, then stop documenting
    rel = [t for t in teles if t["state"] == "RELEASE"]
    assert rel[0]["cmd"]["surge"] < 0  # backs off first ...
    assert rel[-1]["cmd"]["yaw_rate"] < 0  # ... then turns away (left: the fish was on the right)
    after = [t for t in teles if t["t"] > ends[0] + 2.2]
    assert after and all(t["state"] == "SEARCH" for t in after)  # A is still in view but ignored
    assert all(t["reid"]["skipped"] == 1 for t in after)

    b = (FISH_B, 5, 2)  # a different animal shows up while A is still there
    teles2 = _run(g, [[a, b]] * 20, t0=8.0)
    starts2 = [(t["t"], t["encounter"]["id"]) for t in teles2 if "encounter_start" in t["events"]]
    assert len(starts2) == 1 and starts2[0][1] == 2
    tracked = [t for t in teles2 if t["state"] == "TRACK"]
    assert tracked and all(t["target"]["cx"] < 0.4 for t in tracked)  # following B, not A


def test_briefly_lost_animal_resumes_with_the_time_it_has_left():
    g = Guidance(_cfg(max_s=5.0))
    a = (FISH_A, 3, 6)
    teles = _run(g, [[a]] * 20 + [[]] * 20 + [[a]] * 50)  # visible 2 s, gone 2 s, back
    summaries = [t["encounter_summary"] for t in teles if "encounter_summary" in t]
    assert [s["reason"] for s in summaries] == ["lost", "budget"]
    assert summaries[0]["id"] == summaries[1]["id"] == 1 and summaries[1]["resumed"]
    assert summaries[1]["engaged_s"] == pytest.approx(5.0, abs=0.15)  # total across both segments
    assert summaries[0]["engaged_s"] + summaries[1]["segment_s"] == pytest.approx(5.0, abs=0.15)


def test_forgets_animals_after_the_cooldown():
    g = Guidance(_cfg(max_s=2.0, cooldown_s=3.0))
    teles = _run(g, [[(FISH_A, 3, 6)]] * 80)
    starts = [(t["t"], t["encounter"]["id"]) for t in teles if "encounter_start" in t["events"]]
    assert [i for _, i in starts][:2] == [1, 2]  # A again, as a new encounter ...
    end1 = next(t["t"] for t in teles if "encounter_end" in t["events"])
    assert starts[1][0] > end1 + 3.0  # ... but only after it was forgotten


def test_without_patch_tokens_it_still_moves_on_but_cannot_recognise():
    g = Guidance(_cfg())
    teles = _run(g, [[(FISH_A, 3, 6)]] * 80, tokens=False)
    ids = [t["encounter"]["id"] for t in teles if "encounter_start" in t["events"]]
    assert ids[:2] == [1, 2]  # released after the budget, then (same fish, unrecognised) re-acquired
    assert any(t["state"] == "RELEASE" for t in teles)
    s = next(t["encounter_summary"] for t in teles if "encounter_summary" in t)
    assert not s["appearance"]


def test_zero_budget_follows_indefinitely():
    g = Guidance(_cfg(max_s=0.0, rule="fixed"))
    teles = _run(g, [[(FISH_A, 3, 6)]] * 300)
    assert not any(t["state"] == "RELEASE" for t in teles)
    assert teles[-1]["encounter"]["remaining_s"] is None and teles[-1]["encounter"]["engaged_s"] > 29


# ----------------------------------------------------------------------------- when to leave (MVT)


def _film_until_leave(
    mgr: EncounterManager,
    t0: float,
    dt: float = 0.2,
    framed: bool = True,
    size_rate: float = 0.0,
    own_surge: float = 0.0,
):
    """Film one animal (well framed or not) until the manager says leave; returns (reason, t)."""
    tgt = Target(True, 0.5 if framed else 0.95, 0.5, 0.15, 2.0, 0.95)
    trk = TrackState(active=True, confirmed=True, size=0.15, size_rate=size_rate, hits=10)
    mgr.start(t0, None)
    t = t0
    while t < t0 + 1000:
        t += dt
        mgr.observe(t, "FILM", tgt, 0.95, None, trk, own_surge=own_surge)
        reason = mgr.leave_reason(t)
        if reason:
            return reason, t
    return None, t


def test_mvt_films_longer_when_animals_are_rare():
    import math

    cfg = EncounterConfig(max_s=0.0, tau_s=30.0, prior_s=1.0, min_s=5.0)
    times = {}
    for name, rate in (("plentiful", 1 / 60), ("rare", 1 / 1200)):
        m = EncounterManager(cfg)
        m.tick(0.0)
        m.value_done = rate * 3600.0  # an hour of history at that rate
        reason, t = _film_until_leave(m, 3600.0)
        assert reason == "enough"
        times[name] = t - 3600.0
        # analytic optimum for continuous framing: g* = tau * ln(w / (tau * R)), w = 1 (no appearance)
        assert times[name] == pytest.approx(30.0 * math.log(1.0 / (30.0 * rate)), abs=1.5)
    assert times["plentiful"] < 25 < 80 < times["rare"]


def test_never_chases_a_fleeing_animal_and_logs_it_as_fled_when_it_gets_away():
    m = EncounterManager(EncounterConfig())
    tgt = Target(True, 0.5, 0.5, 0.15, 2.0, 0.95)
    away = TrackState(active=True, confirmed=True, size=0.15, size_rate=-0.3, hits=10)
    m.start(0.0, np.eye(4)[0])
    t = 0.0
    while not m.fleeing():  # swimming away (size shrinking 30%/s) for flee_s = 2 s
        t += 0.2
        m.observe(t, "TRACK", tgt, 0.95, None, away)
    assert t == pytest.approx(2.0, abs=0.3) and m.leave_reason(t) is None  # stop chasing, keep filming
    s = m.end(t + 5.0, "lost")  # then it got away
    assert s["reason"] == "fled" and m.is_done_with(np.eye(4)[0], t + 6.0)[0]  # leave it alone
    # shrinking because the vehicle is backing off (stand-off) is not fleeing
    m = EncounterManager(EncounterConfig(max_s=15.0))
    m.start(0.0, None)
    for k in range(20):
        m.observe(0.2 * (k + 1), "FILM", tgt, 0.95, None, away, own_surge=-0.2)
    assert not m.fleeing()


def test_pipeline_stops_approaching_an_animal_that_swims_away():
    g = Guidance(GuidanceConfig())
    p = np.full(GRID, 0.02, np.float32)
    p[3, 6] = 0.95  # a small animal (one patch) straight ahead: too small to film, so approach
    heat = np.log(p / (1 - p))
    t, cmds = 0.0, []
    for _ in range(30):
        t += 0.1
        cmd, tele, _ = g.step(t, np.array([3.0]), heat)
        cmds.append(cmd.surge)
    assert tele["state"] == "TRACK" and cmds[-1] > 0.0  # approaching
    g.encounters.fleeing = lambda: True  # as if it were swimming away (see the test above)
    for _ in range(10):
        t += 0.1
        cmd, tele, _ = g.step(t, np.array([3.0]), heat)
    assert tele["state"] == "TRACK" and cmd.surge <= 0.0  # still filming it, no longer chasing


def test_mvt_gives_up_without_a_good_shot():
    import math

    # never well framed: its expected value decays with the framed fraction (10 s memory) until it
    # drops below the long-run rate - at 10 * ln(1 / (tau * R)) = 23 s with the default prior
    reason, t = _film_until_leave(EncounterManager(EncounterConfig()), 0.0, framed=False)
    assert reason == "no_shot" and t == pytest.approx(10 * math.log(1 / (30 / 300)), abs=0.3)
    # where animals are rare the rule would wait longer, so the give-up time ends it
    rare = EncounterManager(EncounterConfig(prior_rate=1 / 3000))
    assert _film_until_leave(rare, 0.0, framed=False) == ("no_shot", pytest.approx(30.0, abs=0.3))
    m = EncounterManager(EncounterConfig(rule="fixed", max_s=20.0))
    assert _film_until_leave(m, 0.0, size_rate=-0.3) == ("budget", pytest.approx(20.0, abs=0.3))


def test_novel_looking_animals_are_worth_more():
    m = EncounterManager(EncounterConfig())
    a = np.eye(4)[0]
    first = m.start(0.0, a)
    m.end(10.0, "enough")
    lookalike = m.start(20.0, _unit_like(a, 0.95))
    m.end(30.0, "enough")
    novel = m.start(40.0, np.eye(4)[2])
    assert first.weight == pytest.approx(2.0)  # nothing filmed yet: fully novel
    assert lookalike.weight < 1.2 < novel.weight == pytest.approx(2.0)
    m.end(50.0, "enough")
    again = m.start(3650.0, _unit_like(a, 0.9))  # an hour later
    assert m.memory == []  # the cooldown memory has forgotten them ...
    assert again.weight < 1.2  # ... but novelty still knows the look


def _unit_like(v, cos):
    w = np.zeros_like(v)
    w[1] = 1.0
    out = cos * v + np.sqrt(1 - cos**2) * w
    return out / np.linalg.norm(out)


def test_pipeline_leaves_a_well_filmed_animal_when_enough():
    g = Guidance(
        GuidanceConfig(
            fsm=FSMConfig(release_s=2.0),
            encounter=EncounterConfig(max_s=0.0, tau_s=10.0, min_s=5.0, prior_rate=1 / 60, prior_s=600.0),
        )
    )
    teles = _run(g, [[(FISH_A, 3, 6)]] * 400)  # a centred fish, for 40 s
    s = next(t["encounter_summary"] for t in teles if "encounter_summary" in t)
    assert s["reason"] == "enough" and 10 < s["good_s"] < 30 and 0 < s["value"] <= s["novelty_weight"]
    live = [t["encounter"] for t in teles if t["encounter"]]
    assert live[0]["marginal_rate"] > live[0]["long_run_rate"] > live[-1]["marginal_rate"]


def test_close_reports_an_open_encounter():
    g = Guidance(_cfg(max_s=100.0))
    _run(g, [[(FISH_A, 3, 6)]] * 20)
    s = g.close(1.9)
    assert s["reason"] == "shutdown" and s["id"] == 1 and s["seen_frames"] > 10
    assert g.close(2.0) is None
