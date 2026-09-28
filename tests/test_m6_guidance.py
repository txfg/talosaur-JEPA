"""M6 guidance tests: heatmap -> target, camera model, tracker, controller, state machine,
novelty, backends and the full pipeline. numpy only (these also run in the torch-free CI job)."""

from __future__ import annotations

import json
import math
import socket

import numpy as np
import pytest

from talosaur.guidance.backends import JsonlBackend, NullBackend, UdpJsonBackend, make_backend
from talosaur.guidance.camera_model import CameraModel
from talosaur.guidance.controller import Controller, ControllerConfig
from talosaur.guidance.heatmap import Target, find_target, label_components, sigmoid
from talosaur.guidance.novelty import NoveltyDetector
from talosaur.guidance.pipeline import Guidance, GuidanceConfig
from talosaur.guidance.state_machine import FSMConfig, GuidanceFSM, State
from talosaur.guidance.tracker import TargetTracker, TrackerConfig, TrackState

GRID = (7, 13)  # 112x208 input, 16 px patches


def _prob(cells: dict[tuple[int, int], float], bg: float = 0.02) -> np.ndarray:
    p = np.full(GRID, bg, dtype=np.float32)
    for (y, x), v in cells.items():
        p[y, x] = v
    return p


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)


# ----------------------------------------------------------------------------- heatmap


def test_label_components_8_connected():
    m = np.zeros((5, 5), bool)
    m[0, 0] = m[1, 1] = True  # diagonal neighbours: one component
    m[3, 3] = m[3, 4] = True
    m[0, 4] = True
    labels, n = label_components(m)
    assert n == 3
    assert labels[0, 0] == labels[1, 1] != labels[3, 3]


def test_find_target_centroid_size_and_bbox():
    t = find_target(_prob({(2, 4): 0.9, (2, 5): 0.9, (3, 4): 0.9, (3, 5): 0.9}))
    assert t.found and t.n_blobs == 1
    assert t.cx == pytest.approx(5.0 / 13) and t.cy == pytest.approx(3.0 / 7)
    assert t.size == pytest.approx(math.sqrt(4 / 91))
    assert t.mass == pytest.approx(3.6, rel=1e-5)
    assert t.bbox == pytest.approx((4 / 13, 2 / 7, 6 / 13, 4 / 7))


def test_find_target_sub_patch_centroid_leans_to_stronger_cell():
    t = find_target(_prob({(3, 4): 0.95, (3, 5): 0.6}))
    assert 4.5 / 13 < t.cx < 5.0 / 13


def test_single_weak_speck_rejected_but_small_strong_target_kept():
    assert not find_target(_prob({(1, 1): 0.55})).found  # marine snow: mass < min_mass
    t = find_target(_prob({(1, 1): 0.9}))  # a small, distant animal on one patch
    assert t.found and t.cx == pytest.approx(1.5 / 13)


def test_hysteresis_extends_blob_over_weaker_neighbours():
    p = _prob({(3, 6): 0.8, (3, 7): 0.4, (3, 8): 0.4})
    assert find_target(p, thr=0.5).size == pytest.approx(math.sqrt(1 / 91))
    assert find_target(p, thr=0.5, thr_low=0.35).size == pytest.approx(math.sqrt(3 / 91))
    # a blob of only weak cells has no seed, whatever its mass
    assert not find_target(_prob({(3, 6): 0.45, (3, 7): 0.45, (4, 6): 0.45}), thr=0.5, thr_low=0.35).found


def test_smoothing_suppresses_one_patch_target():
    p = _prob({(3, 6): 0.95})
    assert find_target(p).found
    assert not find_target(p, smooth=True).found


def test_find_target_prefers_blob_near_previous_position():
    p = _prob({(1, 1): 0.9, (1, 2): 0.9, (5, 10): 0.8})  # heavy blob top-left, lighter bottom-right
    assert find_target(p).cx < 0.3
    near = find_target(p, prev_xy=(10.5 / 13, 5.5 / 7))
    assert near.cx == pytest.approx(10.5 / 13)
    far = find_target(p, prev_xy=(0.5, 0.5), gate=0.05)  # nothing within the gate: heaviest wins
    assert far.cx < 0.3


# ----------------------------------------------------------------------------- camera model


def test_camera_air_matches_nominal_fov_and_signs():
    cam = CameraModel(port="air")
    assert cam.effective_fov() == pytest.approx((102.0, 67.0))
    assert cam.angles(0.5, 0.5) == pytest.approx((0.0, 0.0))
    yaw, pitch = cam.angles(0.8, 0.2)  # right of and above the centre
    assert yaw > 0 and pitch > 0
    assert cam.angles(0.2, 0.8) == pytest.approx((-yaw, -pitch))


def test_camera_flat_port_narrows_fov_to_about_71x49():
    hf, vf = CameraModel(port="flat").effective_fov()
    assert hf == pytest.approx(2 * math.degrees(math.asin(math.sin(math.radians(51)) / 1.333)), abs=1e-6)
    assert 70.5 < hf < 72.0 and 48.0 < vf < 50.0
    assert CameraModel(port="dome").effective_fov() == pytest.approx((102.0, 67.0))


@pytest.mark.parametrize(
    "cam",
    [
        CameraModel(port="air"),
        CameraModel(port="flat"),
        CameraModel(
            port="calibrated", fx=0.55, fy=0.97, cx=0.51, cy=0.49, dist=(-0.12, 0.03, 0.001, -0.0005, 0.0)
        ),
    ],
)
def test_camera_project_inverts_angles(cam):
    for u, v in [(0.5, 0.5), (0.1, 0.2), (0.9, 0.75), (0.02, 0.97), (0.7, 0.4)]:
        assert cam.project(*cam.angles(u, v)) == pytest.approx((u, v), abs=1e-6)


def test_calibrated_distortion_changes_edge_bearings_only():
    plain = CameraModel(port="calibrated", fx=0.6, fy=1.0)
    barrel = CameraModel(port="calibrated", fx=0.6, fy=1.0, dist=(-0.2, 0.0, 0.0, 0.0))
    assert barrel.angles(0.5, 0.5) == pytest.approx((0.0, 0.0))
    assert barrel.angles(0.52, 0.5)[0] == pytest.approx(plain.angles(0.52, 0.5)[0], rel=1e-2)
    assert barrel.angles(0.98, 0.5)[0] > plain.angles(0.98, 0.5)[0] + 1.0  # undistortion widens the edges


def test_camera_rejects_bad_config():
    with pytest.raises(ValueError):
        CameraModel(port="fisheye")
    with pytest.raises(ValueError):
        CameraModel(port="calibrated")


# ----------------------------------------------------------------------------- tracker


def test_tracker_confirms_and_smooths_noise():
    rng = np.random.default_rng(0)
    trk = TargetTracker()
    errs_meas, errs_trk = [], []
    for k in range(60):
        z = (10 + rng.normal(0, 2.0), -5 + rng.normal(0, 2.0), 0.2 * math.exp(rng.normal(0, 0.1)))
        st = trk.step(k * 0.1, z)
        if k == 1:
            assert st.active and not st.confirmed
        if k == 2:
            assert st.confirmed
        if k >= 20:
            errs_meas.append(z[0] - 10)
            errs_trk.append(st.yaw - 10)
    assert np.std(errs_trk) < 0.6 * np.std(errs_meas)
    assert abs(np.mean(errs_trk)) < 1.0
    assert st.size == pytest.approx(0.2, rel=0.15)


def test_tracker_rejects_outlier_and_coasts_through_dropouts():
    trk = TargetTracker()
    for k in range(20):  # target sweeping right at 20 deg/s
        st = trk.step(k * 0.1, (-10 + 2.0 * k, 0.0, 0.2))
    yaw_before = st.yaw
    st = trk.step(2.0, (-60.0, 30.0, 0.2))  # another animal / a speck far away: gated out
    assert st.misses == 1 and st.yaw > yaw_before  # kept predicting along the sweep
    for k in range(21, 25):
        st = trk.step(k * 0.1, None)
    assert st.active and st.yaw > 35  # coasting at ~20 deg/s
    assert st.yaw_rate == pytest.approx(20.0, abs=4.0)


def test_tracker_drops_track_after_max_misses():
    trk = TargetTracker(TrackerConfig(max_misses=3))
    for k in range(5):
        trk.step(k * 0.1, (0.0, 0.0, 0.2))
    for k in range(5, 8):
        assert trk.step(k * 0.1, None).active
    assert not trk.step(0.8, None).active


# ----------------------------------------------------------------------------- controller


def _track(yaw=0.0, pitch=0.0, size=0.2, yaw_rate=0.0) -> TrackState:
    return TrackState(active=True, confirmed=True, yaw=yaw, pitch=pitch, size=size, yaw_rate=yaw_rate, hits=5)


def _settle(ctrl: Controller, st: TrackState, approach: bool = True, n: int = 20):
    for k in range(n):
        cmd = ctrl.track(st, k * 0.1, approach=approach)
    return cmd


def test_controller_turns_and_heaves_toward_target():
    assert _settle(Controller(), _track(yaw=20)).yaw_rate > 0
    assert _settle(Controller(), _track(yaw=-20)).yaw_rate < 0
    assert _settle(Controller(), _track(pitch=10)).heave > 0  # target above -> ascend
    assert _settle(Controller(), _track(pitch=-10)).heave < 0
    assert _settle(Controller(), _track(yaw=1.5, pitch=-1.5)).yaw_rate == 0.0  # inside the deadband


def test_controller_limits_and_standoff():
    c = ControllerConfig()
    cmd = _settle(Controller(c), _track(yaw=200, pitch=-200))
    assert cmd.yaw_rate == pytest.approx(c.max_yaw) and cmd.heave == pytest.approx(-c.max_heave)
    assert _settle(Controller(c), _track(size=0.5)).surge == pytest.approx(-c.max_reverse)  # too close
    assert _settle(Controller(c), _track(size=0.5), approach=False).surge == pytest.approx(-c.max_reverse)
    assert _settle(Controller(c), _track(size=0.08)).surge > 0  # small, centred: approach
    assert _settle(Controller(c), _track(size=0.08, yaw=30)).surge == 0.0  # not centred: turn first
    assert _settle(Controller(c), _track(size=0.08), approach=False).surge == 0.0  # FILM holds range


def test_controller_slew_limit():
    ctrl = Controller(ControllerConfig(slew_per_s=2.0))
    ctrl.track(_track(), 0.0)
    cmd = ctrl.track(_track(yaw=200), 0.1)
    assert abs(cmd.yaw_rate) <= 0.2 + 1e-9


def test_controller_search_sweeps_both_ways():
    c = ControllerConfig(search_period_s=2.0)
    ctrl = Controller(c)
    first = [ctrl.search(k * 0.1).yaw_rate for k in range(20)][-1]
    second = [ctrl.search(2.0 + k * 0.1).yaw_rate for k in range(20)][-1]
    assert first == pytest.approx(c.search_yaw) and second == pytest.approx(-c.search_yaw)


# ----------------------------------------------------------------------------- state machine


def test_fsm_full_cycle_with_recording_events():
    fsm = GuidanceFSM(FSMConfig(film_hold_s=1.0, lost_timeout_s=2.0, postroll_s=1.0))
    seen: list[tuple[float, str]] = []
    tgt = Target(True, 0.5, 0.5, 0.2, 2.0, 0.9)
    none = Target(False, peak=0.05)

    def step(t, frame_prob, target, track):
        for ev in fsm.update(t, frame_prob, target, track):
            seen.append((round(t, 1), ev))
        return fsm.state

    t = 0.0
    assert step(t, 0.9, tgt, TrackState(active=True, hits=1)) == State.SEARCH  # 1 of 5
    t += 0.1
    step(t, 0.9, tgt, TrackState(active=True, hits=2))
    t += 0.1
    assert step(t, 0.9, tgt, TrackState(active=True, hits=3)) == State.ACQUIRE  # 3 of 5
    t += 0.1
    assert step(t, 0.9, tgt, _track(yaw=20, size=0.1)) == State.TRACK
    assert (0.3, "start_recording") in seen
    for _ in range(5):  # off-centre: stays in TRACK
        t += 0.1
        assert step(t, 0.9, tgt, _track(yaw=20, size=0.1)) == State.TRACK
    for _ in range(12):  # centred and at a good size for > film_hold_s -> FILM
        t += 0.1
        st = step(t, 0.9, tgt, _track(yaw=2, size=0.2))
    assert st == State.FILM
    t += 0.1
    assert step(t, 0.9, tgt, _track(yaw=20, size=0.2)) == State.TRACK  # drifted off-centre
    t += 0.1
    assert step(t, 0.1, none, TrackState(active=False)) == State.LOST
    assert fsm.recording  # LOST keeps recording (it may come back)
    while fsm.state == State.LOST:
        t += 0.1
        step(t, 0.1, none, TrackState(active=False))
    assert fsm.state == State.SEARCH and t == pytest.approx(2.2 + 2.0, abs=0.15)  # LOST at 2.2 + timeout
    while fsm.recording:
        t += 0.1
        step(t, 0.1, none, TrackState(active=False))
    stops = [tt for tt, ev in seen if ev == "stop_recording"]
    assert len(stops) == 1 and stops[0] == pytest.approx(4.2 + 1.0, abs=0.15)  # post-roll after SEARCH
    states = [ev for _, ev in seen if ev.startswith("state:")]
    assert states == [
        "state:ACQUIRE",
        "state:TRACK",
        "state:FILM",
        "state:TRACK",
        "state:LOST",
        "state:SEARCH",
    ]


def test_fsm_ignores_isolated_detections_and_rolls_over_long_recordings():
    fsm = GuidanceFSM(FSMConfig())
    tgt = Target(True, 0.5, 0.5, 0.2, 2.0, 0.9)
    none = Target(False, peak=0.05)
    for k in range(30):  # a speck every third frame never reaches 3-of-5
        fsm.update(k * 0.1, 0.9 if k % 3 == 0 else 0.1, tgt if k % 3 == 0 else none, TrackState())
    assert fsm.state == State.SEARCH

    fsm = GuidanceFSM(FSMConfig(max_record_s=1.0))
    events = []
    for k in range(40):
        track = _track() if k >= 3 else TrackState(active=True, hits=k + 1)
        events += fsm.update(k * 0.1, 0.9, tgt, track)
    assert events.count("start_recording") >= 3 and events.count("stop_recording") >= 2


def test_acquire_times_out_back_to_search_without_flapping():
    c = FSMConfig(acquire_timeout_s=0.5)
    fsm = GuidanceFSM(c)
    tgt = Target(True, 0.5, 0.5, 0.2, 2.0, 0.9)
    states = []
    for k in range(30):  # detections every frame, but the tracker never confirms
        fsm.update(k * 0.1, 0.9, tgt, TrackState(active=True, hits=1))
        states.append(fsm.state)
    assert State.TRACK not in states
    i = states.index(State.SEARCH, states.index(State.ACQUIRE))  # timed out
    assert states[i : i + c.acquire_n - 1] == [State.SEARCH] * (c.acquire_n - 1)  # fresh 3-of-5 needed


# ----------------------------------------------------------------------------- novelty


def test_novelty_scores_outliers_high():
    rng = np.random.default_rng(0)
    d = 16
    basis = rng.normal(size=(d, d))
    nd = NoveltyDetector(d, warmup=30)
    for _ in range(300):
        nd.score(basis @ rng.normal(size=d))
    typical = np.median([nd.score(basis @ rng.normal(size=d), update=False) for _ in range(50)])
    outlier = nd.score(basis @ rng.normal(size=d) + 25 * rng.normal(size=d), update=False)
    assert 0.6 < typical < 1.5
    assert outlier > 3 * typical


def test_novelty_is_zero_during_warmup_and_frozen_without_update():
    nd = NoveltyDetector(4, warmup=5)
    assert all(nd.score(np.ones(4) * k) == 0.0 for k in range(5))
    n = nd.n
    nd.score(np.zeros(4), update=False)
    assert nd.n == n


# ----------------------------------------------------------------------------- backends


def test_jsonl_and_udp_backends(tmp_path):
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    port = rx.getsockname()[1]
    be = make_backend(
        [{"kind": "jsonl", "path": str(tmp_path / "logs" / "g.jsonl")}, {"kind": "udp", "port": port}]
    )
    msg = {"t": 1.0, "state": "TRACK", "cmd": {"yaw_rate": 0.25, "heave": 0.0, "surge": 0.1}}
    be.publish(msg)
    be.close()
    assert json.loads((tmp_path / "logs" / "g.jsonl").read_text().strip()) == msg
    assert json.loads(rx.recv(65536)) == msg
    rx.close()


def test_udp_backend_never_raises_without_listener():
    be = UdpJsonBackend("127.0.0.1", 9)  # discard port; nothing listening
    for _ in range(3):
        be.publish({"x": 1})
    be.close()


def test_make_backend_variants(tmp_path):
    assert isinstance(make_backend(None), NullBackend)
    assert isinstance(make_backend([{"kind": "null"}]).backends[0], NullBackend)
    with pytest.raises(NotImplementedError):
        make_backend([{"kind": "mavlink"}])
    with pytest.raises(ValueError):
        make_backend([{"kind": "carrier-pigeon"}])
    be = JsonlBackend(tmp_path / "a.jsonl")
    be.close()


# ----------------------------------------------------------------------------- pipeline


def test_guidance_config_from_nested_dict():
    cfg = GuidanceConfig.from_dict(
        {
            "heat_thr": 0.6,
            "camera": {"port": "air"},
            "fsm": {"film_size": [0.1, 0.3], "acquire_n": 2},
            "controller": {"max_surge": 0.2},
            "unknown_key": 1,
        }
    )
    assert cfg.heat_thr == 0.6 and cfg.camera.port == "air"
    assert cfg.fsm.film_size == (0.1, 0.3) and cfg.fsm.acquire_n == 2
    assert cfg.controller.max_surge == 0.2
    assert GuidanceConfig.from_dict(None).camera.port == "flat"


def _blob_logits(cx_cell: int, cy_cell: int) -> np.ndarray:
    p = np.full(GRID, 0.02, np.float32)
    p[cy_cell, cx_cell : cx_cell + 2] = 0.95
    return _logit(p)


@pytest.mark.parametrize("side,cell", [("right", 10), ("left", 1)])
def test_pipeline_steers_toward_blob(side, cell):
    g = Guidance(GuidanceConfig())
    states = []
    for k in range(30):
        cmd, tele, _ = g.step(k * 0.1, np.array([3.0]), _blob_logits(cell, 3), np.zeros(8))
        states.append(tele["state"])
    assert "TRACK" in states and tele["recording"]
    assert (cmd.yaw_rate > 0.1) if side == "right" else (cmd.yaw_rate < -0.1)
    assert tele["target"]["found"] and tele["track"]["confirmed"]
    json.dumps(tele)  # telemetry must be JSON-serialisable


def test_pipeline_quiet_scene_stays_in_search_and_scans():
    g = Guidance(GuidanceConfig())
    empty = _logit(np.full(GRID, 0.02, np.float32))
    for k in range(50):
        cmd, tele, events = g.step(k * 0.1, np.array([-4.0]), empty)
        assert tele["state"] == "SEARCH" and not events
    assert abs(cmd.yaw_rate) > 0 and cmd.surge == 0.0


def test_pipeline_lost_then_search_turns_toward_last_bearing():
    g = Guidance(GuidanceConfig())
    for k in range(20):
        g.step(k * 0.1, np.array([3.0]), _blob_logits(1, 3))  # target on the left
    empty = _logit(np.full(GRID, 0.02, np.float32))
    seen = set()
    for k in range(20, 40):
        cmd, tele, _ = g.step(k * 0.1, np.array([-4.0]), empty)
        seen.add(tele["state"])
        if tele["state"] == "LOST":
            lost_cmd = cmd
    assert "LOST" in seen
    assert lost_cmd.yaw_rate < 0  # keeps turning left, where the animal went


def test_sigmoid_is_stable():
    assert sigmoid(np.array([-1e6, 0.0, 1e6])) == pytest.approx([0.0, 0.5, 1.0], abs=1e-12)
