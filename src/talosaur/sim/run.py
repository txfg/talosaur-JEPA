"""Closed-loop episodes: world + vehicle + simulated camera + the real guidance code.

  python -m talosaur.sim.run --hours 1 --start-hour 21 --config configs/onboard/pi5.yaml

Scores (per hour):
* ``value``: footage value with diminishing returns per animal, sum_a (1 - exp(-good_a / tau)),
  where good_a is the seconds animal a was well framed (in view, detected, near the centre, within
  ``good_range_m``). The first seconds of a new animal are worth more than more of the same one;
  this is the quantity a "how long to film" rule should maximise.
* ``animals``: distinct animals with at least ``min_good_s`` of good footage (per species too).
* ``detections``: animals seen at all; ``encounters`` / state time from the guidance telemetry.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from talosaur.guidance.pipeline import Guidance, GuidanceConfig
from talosaur.sim.camera import SimCamera, SimCameraConfig
from talosaur.sim.vehicle import Vehicle, VehicleConfig
from talosaur.sim.world import World, WorldConfig


def run_episode(
    guidance: Guidance,
    world: World,
    vehicle: Vehicle,
    camera: SimCamera,
    duration_s: float,
    fps: float = 5.0,
    use_nav: bool = True,
    tau_s: float = 20.0,
    good_range_m: float = 3.0,
    min_good_s: float = 3.0,
    trace_every_s: float = 0.0,
    telemetry_path: str | Path | None = None,
) -> dict:
    """``telemetry_path``: also write the guidance telemetry as JSONL, like the onboard app's log
    (for ``python -m talosaur.onboard.dive_report``)."""
    dt = 1.0 / fps
    tele_f = open(telemetry_path, "w") if telemetry_path else None
    good: dict[int, float] = defaultdict(float)
    seen: set[int] = set()
    states: Counter = Counter()
    encounters = []
    trace = []
    depth_time: Counter = Counter()
    t = 0.0
    next_trace = 0.0
    light = 0.0
    light_s = 0.0
    if hasattr(guidance.search, "hour_fn"):
        guidance.search.hour_fn = world.hour  # the planner's clock = the simulated local time
    for _ in range(int(duration_s * fps)):
        t += dt
        world.step(dt, vehicle.pos, light)
        frame, heat, emb, tok, visible = camera.render(vehicle, world, light)
        nav = vehicle.nav(t) if use_nav else None
        cmd, tele, _ = guidance.step(t, frame, heat, emb, tok, nav=nav)
        if tele_f:
            tele_f.write(json.dumps({"kind": "guidance", **tele}) + "\n")
        vehicle.step(cmd, dt)
        light = 0.0 if cmd.light is None else float(cmd.light)
        light_s += light * dt
        states[tele["state"]] += dt
        depth_time[int(vehicle.pos[2] // 10) * 10] += dt
        for i, u, v, dist, conf in visible:
            seen.add(i)
            if conf >= 0.5 and dist <= good_range_m and abs(u - 0.5) < 0.2 and abs(v - 0.5) < 0.2:
                good[i] += dt
        if "encounter_summary" in tele:
            encounters.append(tele["encounter_summary"])
        if trace_every_s and t >= next_trace:
            next_trace += trace_every_s
            trace.append(
                {
                    "t": round(t, 1),
                    "state": tele["state"],
                    "x": round(float(vehicle.pos[0]), 1),
                    "y": round(float(vehicle.pos[1]), 1),
                    "depth": round(float(vehicle.pos[2]), 1),
                    "heading": round(vehicle.heading, 1),
                    "layer": round(world.layer_depth(), 1),
                }
            )
    last = guidance.close(t)
    if last:
        encounters.append(last)
    if tele_f:
        if last:
            tele_f.write(json.dumps({"kind": "encounter", **last}) + "\n")
        tele_f.close()
    hours = duration_s / 3600.0
    documented = [i for i, s in good.items() if s >= min_good_s]
    species = Counter(world.cfg.species[world.sp[i]].name for i in documented)
    value = sum(1.0 - math.exp(-s / tau_s) for s in good.values())
    return {
        "hours": round(hours, 3),
        "value_per_h": round(value / hours, 2),
        "animals_per_h": round(len(documented) / hours, 2),
        "detections_per_h": round(len(seen) / hours, 1),
        "good_footage_s_per_h": round(sum(good.values()) / hours, 1),
        "mean_good_s_per_animal": round(float(np.mean([good[i] for i in documented])), 1)
        if documented
        else 0.0,
        "encounters": len(encounters),
        "species": dict(species),
        "state_share": {k: round(v / duration_s, 3) for k, v in states.items()},
        "depth_share": {k: round(v / duration_s, 3) for k, v in sorted(depth_time.items())},
        "distance_km": round(vehicle.distance_m / 1000.0, 2),
        "light_duty": round(light_s / duration_s, 3),
        "trace": trace,
    }


def build(cfg: dict | None, start_hour: float, seed: int = 0, world_overrides: dict | None = None):
    cfg = cfg or {}
    guidance = Guidance(GuidanceConfig.from_dict(cfg.get("guidance")))
    world = World(WorldConfig(start_hour=start_hour, seed=seed, **(world_overrides or {})))
    lim = (cfg.get("guidance", {}) or {}).get("search", {}) or {}
    vcfg = VehicleConfig(max_depth_m=float(lim.get("max_depth_m", 200.0)))
    vehicle = Vehicle(
        vcfg,
        pos=(world.cfg.size_m / 2, world.cfg.size_m / 2, float(lim.get("start_depth_m", 50.0))),
        seed=seed,
    )
    camera = SimCamera(SimCameraConfig(), guidance.cfg.camera, seed=seed)
    return guidance, world, vehicle, camera


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/onboard/pi5.yaml", help="guidance settings")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument(
        "--start-hour", type=float, default=21.0, help="local time at the start (night: animals shallow)"
    )
    ap.add_argument("--fps", type=float, default=5.0, help="guidance rate (the Pi's loop fps)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--no-nav", action="store_true", help="no depth / heading input")
    ap.add_argument(
        "--set", nargs="*", default=[], help="override config values, e.g. guidance.encounter.max_s=30"
    )
    ap.add_argument("--out", default=None, help="write results (with traces) as JSON")
    ap.add_argument(
        "--telemetry", default=None, help="write each run's guidance telemetry to DIR/seed<N>.jsonl"
    )
    a = ap.parse_args(argv)
    import yaml

    cfg = yaml.safe_load(Path(a.config).read_text()) if Path(a.config).exists() else {}
    for kv in a.set:
        key, val = kv.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    results = []
    for seed in a.seeds:
        g, w, v, c = build(cfg, a.start_hour, seed)
        tele = None
        if a.telemetry:
            Path(a.telemetry).mkdir(parents=True, exist_ok=True)
            tele = Path(a.telemetry) / f"seed{seed}.jsonl"
        r = run_episode(
            g, w, v, c, a.hours * 3600, a.fps, use_nav=not a.no_nav, trace_every_s=10.0, telemetry_path=tele
        )
        results.append(r)
        print(json.dumps({"seed": seed, **{k: val for k, val in r.items() if k != "trace"}}))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
