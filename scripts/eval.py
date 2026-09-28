#!/usr/bin/env python
"""Evaluate encoders and baselines with frozen-feature probes, sliced by dark / murky / clear.

  python scripts/eval.py --config configs/eval/default.yaml
  python scripts/eval.py --config configs/eval/default.yaml --only jepa_tiny_ctx_target dinov2_vits14 --sizes 160x160
  python scripts/eval.py --config configs/eval/default.yaml --pi-bench reports/pi5/bench.json   # adds Pi fps

Writes reports/eval/<name>/report.md, results.json and heads_<backbone>_<HxW>.pt (deployable heads).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from talosaur.data.index import read_table
from talosaur.eval.backbones import build_backbone, count_params_and_gmacs
from talosaur.eval.harness import EvalConfig, evaluate_backbone
from talosaur.eval.report import write_report
from talosaur.utils.io import load_yaml
from talosaur.utils.log import get_logger

log = get_logger("eval")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/eval/default.yaml")
    ap.add_argument("--only", nargs="*", default=None, help="backbone names to run")
    ap.add_argument("--sizes", nargs="*", default=None, help="e.g. 224x224 160x160 112x208")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-robustness", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true", help="baselines with random weights (tests)")
    ap.add_argument("--pi-bench", default=None, help="JSON from the Pi benchmark to add fps to the table")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    cfg = load_yaml(a.config)
    out = Path(a.out or cfg.get("out", "reports/eval/v1"))
    sizes = (
        [tuple(int(v) for v in s.split("x")) for s in a.sizes]
        if a.sizes
        else [tuple(s) for s in cfg["sizes"]]
    )
    ecfg = EvalConfig(
        **{k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.get("probe", {}).items()}
    )
    df = read_table(cfg["index"])
    device = torch.device(a.device)
    results, costs = [], {}
    for spec in cfg["backbones"]:
        if a.only and spec["name"] not in a.only:
            continue
        if spec["kind"] == "jepa" and not Path(spec["path"]).exists():
            log.warning(f"skipping {spec['name']}: {spec['path']} not found")
            continue
        bb = build_backbone(spec, pretrained=not a.no_pretrained)
        for hw in sizes:
            res = evaluate_backbone(
                bb,
                df,
                cfg["root"],
                hw,
                ecfg,
                device,
                out,
                cfg.get("hfov", 102.0),
                cfg.get("vfov", 67.0),
                robustness=cfg.get("robustness", True) and not a.no_robustness,
            )
            results.append(res)
            p, g = count_params_and_gmacs(bb.cpu(), tuple(res["input_hw"]))
            costs[f"{spec['name']}_{hw[0]}x{hw[1]}"] = {"params_m": p, "gmacs": g}
    pi = None
    if a.pi_bench and Path(a.pi_bench).exists():
        rows = json.loads(Path(a.pi_bench).read_text()).get("results", [])
        pi = {f"{r['model']}_{r['size']}": r["fps"] for r in rows if r.get("runtime", "").endswith("int8")}
    path = write_report(results, out, costs, pi)
    log.info(f"report -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
