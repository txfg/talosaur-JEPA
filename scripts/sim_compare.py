#!/usr/bin/env python
"""Compare search and filming settings in the closed-loop simulator (docs/SEARCH.md §7).

  python scripts/sim_compare.py --out reports/sim/compare.jsonl            # ~40 min on 4 cores
  python scripts/sim_compare.py --report reports/sim/compare.jsonl         # tables only

Three experiments, each repeated with ``--seeds`` different random worlds:

* **A. Search strategy x search lights x time of day** (one animal per 100 m^3 in the layer).
  Strategies: ``scan`` (turn in place, the old behaviour), ``legs`` (horizontal legs only),
  ``depth`` (depth bands only), ``adaptive`` (both; the default). Lights while searching: off,
  half, and (adaptive only) full. Start at 21:00 (night) or 18:18 (dusk, the layer is rising).
* **B. How long to film each animal**: fixed 60 s, fixed 180 s, and the marginal value rule
  (``mvt``), where animals are scarce (0.003 per m^3, twice the seeds) and plentiful
  (0.03 per m^3).
* **C. Navigation input**: the adaptive planner without depth and heading (scan and hop).

Each line of the output is one run (partial results survive an interruption). The report prints
the mean and standard deviation over seeds of the footage value per hour (the simulator's
score, see ``talosaur.sim.run``), distinct animals well filmed per hour, and animals detected per
hour. The simulator is a caricature: use it to rank settings, not to predict a dive.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from talosaur.sim.run import build, run_episode
from talosaur.utils.io import load_yaml

KEYS = ("batch", "strategy", "light", "start", "rule", "density", "nav")
METRICS = ("value_per_h", "animals_per_h", "detections_per_h", "mean_good_s_per_animal")


def plan(seeds: list[int]) -> list[dict]:
    base = dict(strategy="adaptive", light=0.5, start=21.0, rule="mvt", density=0.01, nav=True)
    jobs = []
    for s, light, start, seed in itertools.product(
        ["scan", "legs", "depth", "adaptive"], [0.0, 0.5], [21.0, 18.3], seeds
    ):
        jobs.append({**base, "batch": "A", "strategy": s, "light": light, "start": start, "seed": seed})
    for start, seed in itertools.product([21.0, 18.3], seeds):
        jobs.append({**base, "batch": "A", "light": 1.0, "start": start, "seed": seed})
    for rule in ["fixed60", "fixed180", "mvt"]:
        # scarce animals: few encounters per run, so twice the seeds (these runs are cheap)
        for dens, ss in ((0.003, seeds + [s + 1000 for s in seeds]), (0.03, seeds)):
            for seed in ss:
                jobs.append({**base, "batch": "B", "rule": rule, "density": dens, "seed": seed})
    for seed in seeds:
        jobs.append({**base, "batch": "C", "nav": False, "seed": seed})
    return sorted(jobs, key=lambda j: -j["density"] * (1 + j["light"]))  # slowest first


def run_one(job: dict, cfg: dict, hours: float, fps: float) -> dict:
    cfg = copy.deepcopy(cfg)
    g = cfg.setdefault("guidance", {})
    g.setdefault("search", {})["strategy"] = job["strategy"]
    g.setdefault("lights", {})["search"] = job["light"]
    rules = {"fixed60": dict(rule="fixed", max_s=60.0), "fixed180": dict(rule="fixed", max_s=180.0)}
    g.setdefault("encounter", {}).update(rules.get(job["rule"], dict(rule="mvt", max_s=180.0)))
    t0 = time.time()
    guidance, world, vehicle, camera = build(
        cfg, job["start"], seed=job["seed"], world_overrides={"layer_density": job["density"]}
    )
    r = run_episode(guidance, world, vehicle, camera, hours * 3600.0, fps=fps, use_nav=job["nav"])
    r.pop("trace", None)
    return {**job, **r, "wall_s": round(time.time() - t0, 1)}


def report(rows: list[dict]) -> str:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if "error" not in r:
            groups[tuple(r[k] for k in KEYS)].append(r)
    lines = [
        "| batch | strategy | search light | start | rule | density /m³ | nav | runs | "
        + " | ".join(METRICS)
        + " |",
        "|" + "---|" * (8 + len(METRICS)),
    ]
    for key in sorted(groups, key=lambda k: tuple(str(v) for v in k)):
        rs = groups[key]
        cells = []
        for m in METRICS:
            v = np.array([r[m] for r in rs], float)
            cells.append(f"{v.mean():.1f} ± {v.std(ddof=1) if len(v) > 1 else 0.0:.1f}")
        b, s, light, start, rule, dens, nav = key
        lines.append(
            f"| {b} | {s} | {light} | {start} | {rule} | {dens} | {nav} | {len(rs)} | "
            + " | ".join(cells)
            + " |"
        )
    failed = sum("error" in r for r in rows)
    if failed:
        lines.append(f"\n{failed} run(s) failed; see the output file.")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/onboard/pi5.yaml", help="guidance settings")
    ap.add_argument("--out", default="reports/sim/compare.jsonl")
    ap.add_argument("--report", default=None, help="only print the tables for this output file")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--hours", type=float, default=1.5, help="length of each run")
    ap.add_argument("--fps", type=float, default=5.0, help="guidance rate (the Pi's loop fps)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None, help="batches to run, e.g. --only B C")
    a = ap.parse_args(argv)
    if a.report:
        rows = [json.loads(line) for line in Path(a.report).read_text().splitlines() if line.strip()]
        print(report(rows))
        return 0
    cfg = load_yaml(a.config) if Path(a.config).exists() else {}  # read once for every run
    jobs = [j for j in plan(a.seeds) if a.only is None or j["batch"] in a.only]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(a.workers) as ex, out.open("w") as f:
        futures = [ex.submit(run_one, j, cfg, a.hours, a.fps) for j in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                row = fut.result()
            except Exception as e:  # keep the other runs
                row = {"error": repr(e)}
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(f"{k}/{len(jobs)} runs done ({time.time() - t0:.0f} s)", flush=True)
    print(report(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
