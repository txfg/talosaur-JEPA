"""The digital twin: the unchanged guidance code and the Pi's exact network, in closed loop with a
physics vehicle (``sim.dynamics``) and a rendered camera (``sim.render``) in the simulated
twilight zone (``sim.world``). docs/DIGITAL_TWIN.md, milestones T1 and T2 (in one process).

  python -m talosaur.sim.twin --encoder runs/<run>/encoder_target.pt \\
      --heads reports/eval/<name>/heads_<backbone>_112x208.pt \\
      --minutes 20 --start-hour 18.9 --video runs/twin/dusk.mp4 --out runs/twin/dusk.json
  python -m talosaur.sim.twin --camera truth ...     # ground-truth heatmaps: a perfect detector

Per frame: the world moves (animals, migration, avoidance of the vehicle and its lamp) -> the
renderer draws what the camera sees -> the network (``--camera model``, the same TalosaurNet the
ONNX export wraps) or the ground truth (``--camera truth``) gives the frame logit, heatmap, embedding
and patch tokens -> ``Guidance.step`` (with the frame's brightness and glow grid, as on the Pi)
-> commands -> the vehicle's autopilot and thrusters.

Scores: the simulator's footage ``value`` and ``animals`` (docs: ``sim.run``), plus the model in
the loop: ``peak_hit`` (frames with a visible animal where the heatmap peak is on one),
``engaged_empty_s`` (ACQUIRE/TRACK/FILM with no animal visible), and ``target_err_deg`` (median
angle between the tracked target and the nearest visible animal).

Frames and videos contain FathomNet-derived animals: keep them local (no re-hosting).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from talosaur.guidance.curiosity import glow_grid
from talosaur.guidance.pipeline import Guidance, GuidanceConfig
from talosaur.sim.dynamics import DynamicVehicle, VehicleDescription
from talosaur.sim.render import RenderConfig, Renderer, SpriteBank
from talosaur.sim.world import World, WorldConfig

ENGAGED = ("ACQUIRE", "TRACK", "FILM")


class ModelCamera:
    """The Pi's network (encoder + frame head + heatmap head) on the rendered frames."""

    def __init__(self, net, device: str = "cpu", threads: int = 4):
        import torch

        torch.set_num_threads(threads)
        self.torch, self.device = torch, torch.device(device)
        self.net = net.to(self.device).eval()
        self.grid = net.grid_hw

    def __call__(self, small: np.ndarray, visible, world):
        torch = self.torch
        x = torch.from_numpy(small).permute(2, 0, 1)[None].float().div(255.0).to(self.device)
        with torch.no_grad():
            f, h, e, tok = self.net(x)
        gh, gw = self.grid
        return (
            f[0].cpu().numpy(),
            h[0].cpu().numpy().reshape(gh, gw),
            e[0].cpu().numpy(),
            tok[0].cpu().numpy().reshape(gh, gw, -1),
        )


class TruthCamera:
    """A perfect detector: each visible animal's box painted into the heatmap, its appearance
    vector into the tokens (the plan's "mask mode")."""

    def __init__(self, grid=(7, 13)):
        self.grid = tuple(grid)

    def __call__(self, small, visible, world):
        gh, gw = self.grid
        D = world.appearance.shape[1]
        water = np.zeros(D, np.float32)
        water[0] = 1.0
        cov = np.zeros((gh, gw), np.float32)
        tok = np.tile(water, (gh, gw, 1))
        for i, _u, _v, _d, _c, box, _sp in visible:
            x0, y0, x1, y1 = box
            xs = (
                np.clip(
                    np.minimum((np.arange(gw) + 1) / gw, x1) - np.maximum(np.arange(gw) / gw, x0), 0, None
                )
                * gw
            )
            ys = (
                np.clip(
                    np.minimum((np.arange(gh) + 1) / gh, y1) - np.maximum(np.arange(gh) / gh, y0), 0, None
                )
                * gh
            )
            c = np.outer(ys, xs)
            if c.max() > 0 and c.max() < 0.3:  # smaller than a cell: mark the cell it is in
                c = (c == c.max()).astype(np.float32) * 0.3
            cov = np.maximum(cov, c)
            tok[c >= 0.3] = world.appearance[i] * 2.0 + water
        heat = -4.0 + 8.0 * np.clip(cov / 0.3, 0.0, 1.0)
        frame = np.array([4.0 if visible else -4.0], np.float32)
        return frame, heat.astype(np.float32), tok.mean(axis=(0, 1)), tok


class VideoWriter:
    """Annotated MP4: the rendered frame (2x), heatmap cells > 0.5, ground-truth boxes, the tracked
    target, and a status line."""

    def __init__(self, path: str | Path, fps: float, size_wh: tuple[int, int]):
        import av

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.c = av.open(str(path), "w")
        for codec in ("libx264", "mpeg4"):
            try:
                self.s = self.c.add_stream(codec, rate=int(round(fps)))
                break
            except Exception:
                continue
        self.s.width, self.s.height = size_wh
        self.s.pix_fmt = "yuv420p"
        self.av = av

    def write(self, rgb: np.ndarray, heat_prob, visible, tele, lines: list[str]) -> None:
        from PIL import Image, ImageDraw

        H, W = rgb.shape[:2]
        im = Image.fromarray(rgb).resize((2 * W, 2 * H), Image.Resampling.NEAREST)
        canvas = Image.new("RGB", (self.s.width, self.s.height), (16, 16, 20))
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        gh, gw = heat_prob.shape
        cw, ch = 2 * W / gw, 2 * H / gh
        for r, c in zip(*np.nonzero(heat_prob > 0.5)):
            d.rectangle([c * cw, r * ch, (c + 1) * cw, (r + 1) * ch], outline=(0, 220, 255))
        for _i, _u, _v, _dd, _con, box, sp in visible:
            d.rectangle(
                [box[0] * 2 * W, box[1] * 2 * H, box[2] * 2 * W, box[3] * 2 * H], outline=(80, 255, 120)
            )
            d.text((box[0] * 2 * W, box[3] * 2 * H + 1), sp, fill=(80, 255, 120))
        tg = tele.get("target") or {}
        if tg.get("found"):
            x, y = tg["cx"] * 2 * W, tg["cy"] * 2 * H
            d.line([x - 12, y, x + 12, y], fill=(255, 60, 220), width=3)
            d.line([x, y - 12, x, y + 12], fill=(255, 60, 220), width=3)
        for k, line in enumerate(lines):
            d.text((8, 2 * H + 6 + 16 * k), line, fill=(235, 235, 235))
        frame = self.av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
        for p in self.s.encode(frame):
            self.c.mux(p)

    def close(self) -> None:
        for p in self.s.encode():
            self.c.mux(p)
        self.c.close()


def _peak_on_animal(heat_logit: np.ndarray, visible) -> bool:
    gh, gw = heat_logit.shape
    r, c = np.unravel_index(int(np.argmax(heat_logit)), heat_logit.shape)
    for *_, box, _sp in visible:
        # the peak cell overlaps the animal's box (widened by half a cell for sub-cell animals)
        if (c + 1) / gw > box[0] - 0.5 / gw and c / gw < box[2] + 0.5 / gw:
            if (r + 1) / gh > box[1] - 0.5 / gh and r / gh < box[3] + 0.5 / gh:
                return True
    return False


def run_twin(
    guidance: Guidance,
    world: World,
    vehicle,
    renderer: Renderer,
    camera,
    duration_s: float,
    fps: float = 5.0,
    tau_s: float = 20.0,
    good_range_m: float = 3.0,
    min_good_s: float = 3.0,
    video: VideoWriter | None = None,
    video_every: int = 1,
    telemetry_path: str | Path | None = None,
    trace_every_s: float = 10.0,
) -> dict:
    dt = 1.0 / fps
    tele_f = open(telemetry_path, "w") if telemetry_path else None
    hfov, vfov = renderer.hfov, renderer.vfov
    good: dict[int, float] = defaultdict(float)
    seen: set[int] = set()
    states: Counter = Counter()
    encounters, trace, target_err = [], [], []
    n_vis = n_hit = 0
    engaged_empty_s = light_s = 0.0
    t, next_trace, light = 0.0, 0.0, 0.0
    if hasattr(guidance.search, "hour_fn"):
        guidance.search.hour_fn = world.hour
    for k in range(int(duration_s * fps)):
        t += dt
        world.step(dt, vehicle.pos, light)
        rgb, small, visible, luma = renderer.render(vehicle, world, light, t)
        frame, heat, emb, tok = camera(small, visible, world)
        cmd, tele, _ = guidance.step(
            t, frame, heat, emb, tok, nav=vehicle.nav(t), luma=luma, glow=glow_grid(small)
        )
        if tele_f:
            tele_f.write(json.dumps({"kind": "guidance", **tele}) + "\n")
        vehicle.step(cmd, dt)
        light = 0.0 if cmd.light is None else float(cmd.light)
        light_s += light * dt
        st = tele["state"]
        states[st] += dt
        if visible:
            n_vis += 1
            n_hit += _peak_on_animal(heat, visible)
        elif st in ENGAGED:
            engaged_empty_s += dt
        tg = tele.get("target") or {}
        if tg.get("found") and visible:
            target_err.append(
                min(math.hypot((tg["cx"] - u) * hfov, (tg["cy"] - v) * vfov) for _i, u, v, *_ in visible)
            )
        for i, u, v, dd, *_ in visible:
            seen.add(i)
            if dd <= good_range_m and abs(u - 0.5) < 0.2 and abs(v - 0.5) < 0.2:
                good[i] += dt
        if "encounter_summary" in tele:
            encounters.append(tele["encounter_summary"])
        if video is not None and k % video_every == 0:
            hour = world.hour()
            hp = 1.0 / (1.0 + np.exp(-heat))
            video.write(
                rgb,
                hp,
                visible,
                tele,
                [
                    f"t {int(t // 60):02d}:{int(t % 60):02d}  local {int(hour):02d}:{int(hour % 1 * 60):02d}  "
                    f"depth {vehicle.pos[2]:6.1f} m  heading {vehicle.heading:5.1f}  speed {vehicle.speed:4.2f} m/s",
                    f"state {st:8s}  lamp {light:.2f}  p(animal) {1 / (1 + math.exp(-float(frame[0]))):.2f}  "
                    f"visible {len(visible)}  encounters {len(encounters)}  layer {world.layer_depth():5.0f} m",
                    "green: animals in view (truth)   cyan: heatmap > 0.5   magenta: tracked target",
                ],
            )
        if trace_every_s and t >= next_trace:
            next_trace += trace_every_s
            trace.append(
                {
                    "t": round(t, 1),
                    "state": st,
                    "depth": round(float(vehicle.pos[2]), 1),
                    "heading": round(vehicle.heading, 1),
                    "visible": len(visible),
                    "lamp": round(light, 2),
                    "layer": round(world.layer_depth(), 1),
                }
            )
    if tele_f:
        tele_f.close()
    hours = duration_s / 3600.0
    value = sum(1.0 - math.exp(-g / tau_s) for g in good.values())
    return {
        "hours": hours,
        "value_per_h": value / hours,
        "animals_per_h": sum(g >= min_good_s for g in good.values()) / hours,
        "seen_per_h": len(seen) / hours,
        "encounters": len(encounters),
        "encounter_ends": dict(Counter(e.get("reason") for e in encounters)),
        "state_s": {k: round(v, 1) for k, v in states.items()},
        "peak_hit": n_hit / n_vis if n_vis else float("nan"),
        "frames_with_animal": n_vis,
        "engaged_empty_s": round(engaged_empty_s, 1),
        "target_err_deg_median": float(np.median(target_err)) if target_err else float("nan"),
        "lamp_mean": light_s / duration_s,
        "distance_m": round(float(vehicle.distance_m), 1),
        "trace": trace,
    }


def main(argv=None) -> int:
    from talosaur.utils.io import load_yaml

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", choices=["model", "truth"], default="model")
    ap.add_argument("--encoder", default=None, help="encoder_target.pt (``--camera model``)")
    ap.add_argument("--heads", default=None, help="an eval heads_*.pt fitted at --input")
    ap.add_argument("--input", default="112x208", help="model input HxW (the Pi's lores stream)")
    ap.add_argument("--config", default="configs/onboard/pi5.yaml", help="guidance settings")
    ap.add_argument("--vehicle", default="configs/vehicle/talosaur_v0.yaml")
    ap.add_argument("--sprites", default="data/cache/twin_sprites.npz")
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--start-hour", type=float, default=18.9, help="local time (the dusk crossing ~18.5-20)")
    ap.add_argument("--depth", type=float, default=150.0)
    ap.add_argument("--fps", type=float, default=5.0, help="guidance / model rate")
    ap.add_argument("--lamp", choices=["red", "white"], default="red")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", default=None)
    ap.add_argument("--video-every", type=int, default=1)
    ap.add_argument("--out", default=None, help="results JSON")
    ap.add_argument("--telemetry", default=None, help="guidance telemetry JSONL (for onboard.dive_report)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)

    h, w = (int(v) for v in a.input.lower().split("x"))
    cfg = load_yaml(a.config) or {}
    guidance = Guidance(GuidanceConfig.from_dict(cfg.get("guidance")))
    world = World(WorldConfig(start_hour=a.start_hour, seed=a.seed))
    vehicle = DynamicVehicle(VehicleDescription.load(a.vehicle), pos=(1000.0, 1000.0, a.depth), seed=a.seed)
    renderer = Renderer(
        SpriteBank.load(a.sprites),
        guidance.cfg.camera,
        RenderConfig(hw=(2 * h, 2 * w), model_hw=(h, w), lamp=a.lamp, seed=a.seed),
    )
    if a.camera == "model":
        if not a.encoder:
            ap.error("--camera model needs --encoder (and --heads)")
        from talosaur.export.onnx_export import build_net

        camera = ModelCamera(build_net(a.encoder, a.heads, (h, w)), a.device, a.threads)
    else:
        camera = TruthCamera((h // 16, w // 16))
    video = VideoWriter(a.video, a.fps, (4 * w, 4 * h + 64)) if a.video else None
    try:
        res = run_twin(guidance, world, vehicle, renderer, camera, a.minutes * 60, a.fps, video=video,
                       video_every=a.video_every, telemetry_path=a.telemetry)  # fmt: skip
    finally:
        if video is not None:
            video.close()
    res["args"] = vars(a)
    summary = {k: v for k, v in res.items() if k not in ("trace", "args")}
    print(json.dumps(summary, indent=1, default=str))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
