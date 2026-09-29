"""Markdown + JSON comparison report across backbones and input sizes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NAN = float("nan")


def _g(d: dict, *keys, default=NAN):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _fmt(v, ci=None) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return "–"
    s = f"{v:.3f}" if isinstance(v, float) else str(v)
    if ci and all(c == c for c in ci):
        s += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    return s


def _auroc(block: dict) -> str:
    a = block.get("auroc") if isinstance(block, dict) else None
    if not isinstance(a, dict):
        return "–"
    return _fmt(a.get("value"), (a.get("ci_lo"), a.get("ci_hi"))) + f" (n={a.get('n')})"


def write_report(
    results: list[dict[str, Any]],
    out_dir: Path,
    costs: dict[str, dict] | None = None,
    pi_fps: dict[str, float] | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    costs = costs or {}
    pi_fps = pi_fps or {}
    md = [
        "# Evaluation report",
        "",
        "Frozen features. Linear probes are fitted on train, weight decay is selected on val, and results are on test. Brackets show bootstrap 95% CIs. Slices are defined by the dataset's condition buckets.",
        "",
    ]
    by_size: dict[str, list[dict]] = {}
    for r in results:
        by_size.setdefault("x".join(map(str, r.get("requested_hw", r["input_hw"]))), []).append(r)
    for size, rs in by_size.items():
        md += [f"## Input {size}", "", "### Frame probe: animal present (AUROC)", ""]
        md += ["| backbone | all | dark | murky | clear | TPR@5%FPR (all) |", "|---|---|---|---|---|---|"]
        for r in rs:
            f = r.get("frame", {})
            md.append(
                f"| {r['backbone']} | {_auroc(f.get('all', {}))} | {_auroc(f.get('dark', {}))} | {_auroc(f.get('murky', {}))} | "
                f"{_auroc(f.get('clear', {}))} | {_fmt(_g(f, 'all', 'tpr_at_5fpr'))} |"
            )
        md += ["", "### Patch probe and steering", ""]
        md += [
            "| backbone | patch AUROC all / dark / murky / clear | within-image AUROC | centroid err ° (median) all / dark / murky / clear | p90 ° | size log-err | peak hit | found |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rs:
            p = r.get("patch", {})
            au = " / ".join(_fmt(_g(p, s, "patch_auroc")) for s in ("all", "dark", "murky", "clear"))
            ce = " / ".join(
                _fmt(_g(p, s, "centroid_err_deg_median")) for s in ("all", "dark", "murky", "clear")
            )
            md.append(
                f"| {r['backbone']} | {au} | {_fmt(_g(p, 'all', 'within_image_auroc'))} | {ce} | {_fmt(_g(p, 'all', 'centroid_err_deg_p90'))} | "
                f"{_fmt(_g(p, 'all', 'size_log_err_median'))} | {_fmt(_g(p, 'all', 'peak_hit_rate'))} | {_fmt(_g(p, 'all', 'found_frac'))} |"
            )
        if any(r.get("frame", {}).get("by_source") for r in rs):
            md += ["", "### Frame AUROC by source", ""]
            srcs = sorted({s for r in rs for s in r.get("frame", {}).get("by_source", {})})
            md += ["| backbone | " + " | ".join(srcs) + " |", "|---" * (len(srcs) + 1) + "|"]
            for r in rs:
                bs = r.get("frame", {}).get("by_source", {})
                md.append(
                    f"| {r['backbone']} | "
                    + " | ".join(_fmt(_g(bs.get(s, {}), "auroc", "value")) for s in srcs)
                    + " |"
                )
        md += [
            "",
            "### Cost",
            "",
            "| backbone | actual input | params (M) | GMACs | Pi 5 fps (int8) |",
            "|---|---|---|---|---|",
        ]
        for r in rs:
            key = f"{r['backbone']}_{size}"
            c = costs.get(key, {})
            actual = "x".join(map(str, r["input_hw"]))
            md.append(
                f"| {r['backbone']} | {actual} | {_fmt(c.get('params_m'))} | {_fmt(c.get('gmacs'))} | {_fmt(pi_fps.get(key))} |"
            )
        rob = [r for r in rs if r.get("robustness")]
        if rob:
            md += ["", "### Underwater-C (held-out synthetic degradations; secondary to the real slices)", ""]
            keys = list(rob[0]["robustness"].keys())
            ops = sorted({k.split("/")[0] for k in keys})
            lvls = sorted({int(k.split("/")[1]) for k in keys})
            for metric in ("frame_auroc", "centroid_err_deg_median"):
                md += [
                    f"{metric} per severity level (1..{max(lvls)})",
                    "",
                    "| backbone | op | " + " | ".join(map(str, lvls)) + " |",
                    "|---|---|" + "---|" * len(lvls),
                ]
                for r in rob:
                    for op in ops:
                        md.append(
                            f"| {r['backbone']} | {op} | "
                            + " | ".join(_fmt(_g(r["robustness"], f"{op}/{lv}", metric)) for lv in lvls)
                            + " |"
                        )
                md.append("")
        md.append("")
    path = out_dir / "report.md"
    path.write_text("\n".join(md))
    (out_dir / "results.json").write_text(
        json.dumps({"results": results, "costs": costs}, indent=1, default=float)
    )
    return path
