"""Digital twin: vehicle dynamics, rendered camera, closed loop with the model and ground truth."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import requires_av, requires_pil, requires_torch

from talosaur.guidance.controller import Command
from talosaur.sim.dynamics import DynamicVehicle, VehicleDescription


def test_dynamics_steady_speed_turn_and_holds():
    d = VehicleDescription()
    v = DynamicVehicle(d, pos=(0, 0, 150))
    for _ in range(300):
        v.step(Command(surge=1.0, depth_m=150), 0.1)
    assert v.speed == pytest.approx(d.steady_speed("surge", d.thrust["surge_n"]), rel=1e-3)
    assert abs(v.pos[2] - 150) < 0.1 and v.pos[1] > 10  # heading 0 = north (+y)
    v = DynamicVehicle(d, pos=(0, 0, 150))
    for _ in range(200):
        v.step(Command(yaw_rate=1.0), 0.05)
    assert v.yaw_rate == pytest.approx(d.autopilot["max_yaw_dps"], rel=0.02)
    v = DynamicVehicle(d, pos=(0, 0, 150), heading=350)
    for _ in range(600):  # 5 m at the 0.3 m/s limit, then settling
        v.step(Command(heading_deg=80, depth_m=155), 0.05)
    assert abs((v.heading - 80 + 180) % 360 - 180) < 1.0 and abs(v.pos[2] - 155) < 0.3
    v = DynamicVehicle(d, pos=(0, 0, 150))
    for _ in range(600):
        v.step(Command(), 0.05)  # no setpoint: the vertical-speed hold carries the buoyancy
    assert abs(v.pos[2] - 150) < 0.3


def test_vehicle_description_partial_override(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text("mass_kg: 20\nthrust: {surge_n: 30}\n")
    d = VehicleDescription.load(p)
    assert d.mass_kg == 20 and d.thrust["surge_n"] == 30 and d.thrust["heave_n"] == 8.0


class FixedWorld:
    """Animals at fixed offsets from the vehicle."""

    def __init__(self, animals, hour=23.0):
        from talosaur.sim.world import WorldConfig

        self.cfg = WorldConfig()
        names = [s.name for s in self.cfg.species]
        self._rel = np.array([a[1] for a in animals], float).reshape(-1, 3)
        self.sp = np.array([names.index(a[0]) for a in animals], int)
        self.size = np.array([a[2] for a in animals], float)
        self.appearance = np.eye(len(animals), 16)
        self._hour = hour

    def hour(self, t=None):
        return self._hour

    def near(self, p, r):
        return np.arange(len(self._rel))

    def offsets(self, idx, p):
        return self._rel[idx].copy()


def _bank():
    from talosaur.sim.render import SpriteBank

    yy, xx = np.mgrid[0:40, 0:24]
    a = ((((xx - 12) / 11) ** 2 + ((yy - 20) / 19) ** 2) < 1).astype(np.uint8) * 255
    sprite = np.dstack([np.full((40, 24, 3), 200, np.uint8), a])
    return SpriteBank({g: [sprite] for g in ("fish", "shrimp", "medusa", "siphonophore", "squid")},
                      {g: ["synthetic"] for g in ("fish", "shrimp", "medusa", "siphonophore", "squid")})  # fmt: skip


@requires_pil
def test_cut_out_keeps_the_animal_and_drops_the_background():
    from talosaur.sim.render import cut_out

    rng = np.random.default_rng(0)
    crop = (rng.normal(40, 3, size=(60, 60, 3))).clip(0, 255).astype(np.uint8)
    crop[20:40, 25:35] = 220
    s = cut_out(crop, (20, 15, 40, 45))
    assert s is not None and s.shape == (30, 20, 4)
    assert s[10, 7, 3] > 200 and s[1, 1, 3] < 30  # animal opaque, background clear
    assert cut_out(np.full((40, 40, 3), 50, np.uint8), (10, 10, 30, 30)) is None  # no contrast


@requires_pil
def test_render_places_a_lamp_lit_animal_ahead_and_night_is_dark():
    from talosaur.sim.render import RenderConfig, Renderer

    veh = DynamicVehicle(pos=(0, 0, 200), heading=0)
    world = FixedWorld([("squid", (0.0, 1.5, 0.0), 0.3)])
    r = Renderer(_bank(), cfg=RenderConfig(snow_per_m3=0.0, lamp="white"))
    for k in range(8):
        rgb, small, vis, luma = r.render(veh, world, 0.5, t=0.2 * k)
    assert rgb.shape == (224, 416, 3) and small.shape == (112, 208, 3)
    assert len(vis) == 1 and vis[0][6] == "squid"
    _, u, v, dist, contrast, box, _ = vis[0]
    assert abs(u - 0.5) < 0.03 and abs(v - 0.5) < 0.03 and dist == pytest.approx(1.5)
    assert box[0] < 0.5 < box[2]
    dark = Renderer(_bank(), cfg=RenderConfig(snow_per_m3=0.0))
    for k in range(8):
        _, _, vis_off, luma_off = dark.render(veh, world, 0.0, t=0.2 * k)  # night, lamp off
    assert not vis_off and luma_off < 0.1 < luma


@requires_torch
@requires_pil
def test_twin_loop_with_truth_and_model_cameras(tmp_path):
    from talosaur.export.onnx_export import build_net
    from talosaur.guidance.pipeline import Guidance, GuidanceConfig
    from talosaur.sim.render import RenderConfig, Renderer
    from talosaur.sim.twin import ModelCamera, TruthCamera, run_twin
    from talosaur.sim.world import World, WorldConfig

    def episode(camera, seconds):
        g = Guidance(GuidanceConfig())
        world = World(WorldConfig(size_m=60.0, start_hour=19.5, seed=1))
        veh = DynamicVehicle(pos=(30.0, 30.0, world.layer_depth()), seed=1)
        r = Renderer(_bank(), g.cfg.camera, RenderConfig(seed=1))
        return run_twin(g, world, veh, r, camera, seconds, fps=5.0, telemetry_path=tmp_path / "t.jsonl")

    res = episode(TruthCamera((7, 13)), 30.0)
    assert set(res["state_s"]) and res["distance_m"] >= 0 and (tmp_path / "t.jsonl").exists()
    if res["frames_with_animal"]:
        assert res["peak_hit"] == 1.0  # the truth heatmap peaks on an animal by construction
    res = episode(ModelCamera(build_net("random:vit_tiny", None, (112, 208)), threads=2), 2.0)
    assert sum(res["state_s"].values()) == pytest.approx(2.0, abs=0.21)


@requires_av
@requires_pil
def test_video_writer(tmp_path):
    import av

    from talosaur.sim.twin import VideoWriter

    w = VideoWriter(tmp_path / "v.mp4", 5.0, (832, 512))
    for _ in range(3):
        w.write(np.zeros((224, 416, 3), np.uint8), np.zeros((7, 13)), [], {}, ["status"])
    w.close()
    with av.open(str(tmp_path / "v.mp4")) as c:
        assert sum(1 for _ in c.decode(video=0)) == 3
