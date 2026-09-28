"""Dataset report: Markdown + JSON + figures (histograms, contact sheets) for a curated index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from talosaur.utils.io import save_json
from talosaur.utils.log import get_logger

log = get_logger("data.stats")

DEPLOY_SIZES = {"112x112": (112, 112), "160x160": (160, 160), "224x224": (224, 224), "208x112 (16:9)": (112, 208)}
PATCH = 16


def _md_table(rows: list[list[Any]], header: list[str]) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.3g}"
        return str(v)

    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def box_size_stats(df) -> dict[str, dict[str, float]]:
    """Fraction of animal boxes smaller than one 16-px patch at each deploy input size."""
    ws, hs = [], []
    for boxes, animal in zip(df["boxes"], df["box_is_animal"]):
        for b, a in zip(boxes, animal):
            if a:
                ws.append(b[2] - b[0])
                hs.append(b[3] - b[1])
    ws, hs = np.asarray(ws), np.asarray(hs)
    out = {}
    if len(ws) == 0:
        return out
    for name, (H, W) in DEPLOY_SIZES.items():
        pw, ph = ws * W / PATCH, hs * H / PATCH
        out[name] = {
            "n_boxes": int(len(ws)),
            "frac_max_side_lt_1_patch": float(np.mean(np.maximum(pw, ph) < 1.0)),
            "frac_area_lt_1_patch": float(np.mean(pw * ph < 1.0)),
            "frac_area_lt_4_patches": float(np.mean(pw * ph < 4.0)),
            "median_patches_w": float(np.median(pw)),
            "median_patches_h": float(np.median(ph)),
        }
    return out


def label_stats(df) -> list[list[Any]]:
    rows = []
    for (src, split), g in df.groupby(["source", "split"]):
        pos = int((g["frame_label"] == 1).sum())
        neg = int((g["frame_label"] == 0).sum())
        nbox = int(sum(len(b) for b in g["boxes"]))
        nmask = int(g["mask_path"].map(lambda v: isinstance(v, str) and len(v) > 0).sum())
        rows.append([src, split, len(g), pos, neg, int((g["frame_label"] == -1).sum()), nbox, nmask])
    return rows


def contact_sheet(root: Path, paths: list[str], out: Path, cols: int = 8, tile: int = 128, title: str = "") -> Path | None:
    if not paths:
        return None
    from PIL import Image, ImageDraw

    from talosaur.data.imageio import load_rgb

    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile, rows * tile + (18 if title else 0)), (20, 20, 20))
    if title:
        ImageDraw.Draw(sheet).text((4, 2), title, fill=(230, 230, 230))
    y0 = 18 if title else 0
    for i, p in enumerate(paths):
        try:
            im = Image.fromarray(load_rgb(root / p, max_side=tile))
        except Exception:
            continue
        im.thumbnail((tile, tile))
        sheet.paste(im, ((i % cols) * tile, y0 + (i // cols) * tile))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def _hist_figure(df, col: str, out: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=110)
    for src, g in df.groupby("source"):
        v = g[col].dropna().to_numpy()
        if len(v):
            ax.hist(v, bins=40, histtype="step", density=True, label=f"{src} (n={len(v)})")
    ax.set_xlabel(col)
    ax.set_ylabel("density")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def write_report(df, removed, curation_log: dict, root: Path, out_dir: Path, figures: bool = True, seed: int = 0) -> Path:
    """Write ``report.md`` + ``report.json`` (+ ``figures/``) into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    js: dict[str, Any] = {"n_images": int(len(df)), "curation": curation_log.get("steps", [])}
    md: list[str] = [f"# Dataset report: {curation_log.get('config', {}).get('name', '')}", ""]

    # 1. overview per source
    rows = []
    rem_counts = removed.groupby(["source", "removed_reason"]).size() if len(removed) else None
    for src, g in df.groupby("source"):
        r = [src, len(g)] + [int((g["split"] == s).sum()) for s in ("train", "val", "test")]
        for reason in ("near_dup", "dup_of_eval", "empty_water"):
            r.append(int(rem_counts.get((src, reason), 0)) if rem_counts is not None else 0)
        rows.append(r)
    md += ["## Images per source", "", _md_table(rows, ["source", "kept", "train", "val", "test", "removed: near-dup", "removed: dup of eval", "removed: empty water"]), ""]
    js["per_source"] = rows

    # 2. licenses
    lic = df.groupby("license").agg(n=("image_id", "size"), commercial=("train_commercial_ok", "mean"), sources=("source", lambda s: ",".join(sorted(set(s)))))
    md += ["## Licenses", "", _md_table([[i, int(r.n), float(r.commercial), r.sources] for i, r in lic.iterrows()], ["license", "images", "train_commercial_ok (fraction)", "sources"]), ""]
    js["licenses"] = lic.reset_index().to_dict("records")

    # 3. conditions
    tr = df[df["split"] == "train"]
    grid = tr.groupby(["light", "clarity_bucket"]).size().unstack(fill_value=0)
    grid_rows = [[k, *[int(x) for x in v.tolist()]] for k, v in grid.iterrows()]
    md += ["## Condition buckets (train pool)", "", _md_table(grid_rows, ["light \\ clarity", *grid.columns.tolist()]) if len(grid) else "_empty_", ""]
    ev = df[df["split"] != "train"]
    sl = ev.groupby(["source", "split", "slice"]).size().unstack(fill_value=0)
    md += ["## Evaluation slices (val/test)", "", _md_table([[*k, *v.tolist()] for k, v in sl.iterrows()], ["source", "split", *sl.columns.tolist()]) if len(sl) else "_no evaluation images_", ""]
    js["train_condition_grid"] = {f"{a}/{b}": int(v) for (a, b), v in tr.groupby(["light", "clarity_bucket"]).size().items()}
    md += ["Thresholds used:", "", "```", *[f"{k}: {v:.4g}" for k, v in curation_log.get("thresholds", {}).items()], "```", ""]

    # 4. labels
    md += ["## Labels", "", _md_table(label_stats(df), ["source", "split", "images", "frame=1", "frame=0", "frame=unknown", "boxes", "masks"]), ""]
    bs = box_size_stats(ev if len(ev) else df)
    js["box_sizes"] = bs
    if bs:
        md += [
            "## Animal box size vs. 16-px patch grid (val/test)",
            "",
            "These numbers show how much detail each deploy input size keeps. If many animals are smaller than one patch at 112 px, prefer 160 px or 208x112.",
            "",
            _md_table([[k, v["n_boxes"], v["frac_max_side_lt_1_patch"], v["frac_area_lt_1_patch"], v["frac_area_lt_4_patches"], v["median_patches_w"], v["median_patches_h"]] for k, v in bs.items()],
                      ["input", "boxes", "max side < 1 patch", "area < 1 patch", "area < 4 patches", "median width (patches)", "median height (patches)"]),
            "",
        ]

    # 5. metric distributions
    cols = ["lum_mean", "clarity", "norm_contrast", "noise", "cast_mag", "red_ratio"]
    rows = []
    for src, g in df.groupby("source"):
        rows.append([src] + [f"{g[c].quantile(0.1):.3f} / {g[c].median():.3f} / {g[c].quantile(0.9):.3f}" for c in cols])
    md += ["## Metric distributions (p10 / median / p90)", "", _md_table(rows, ["source", *cols]), ""]

    # 6. figures
    if figures:
        fig_dir = out_dir / "figures"
        links = []
        for c in ("lum_mean", "clarity"):
            p = _hist_figure(df, c, fig_dir / f"hist_{c}.png")
            if p:
                links.append(f"![{c}](figures/{p.name})")
        for s in ("dark", "murky", "clear"):
            sub = df[df["slice"] == s]
            pick = sub.sample(min(32, len(sub)), random_state=int(rng.integers(1 << 31)))["path"].tolist() if len(sub) else []
            p = contact_sheet(root, pick, fig_dir / f"slice_{s}.png", title=f"slice={s} (random {len(pick)})")
            if p:
                links.append(f"![{s}](figures/{p.name})")
        if len(removed) and "removed_reason" in removed:
            emp = removed[removed["removed_reason"] == "empty_water"]
            pick = emp.sample(min(32, len(emp)), random_state=0)["path"].tolist() if len(emp) else []
            p = contact_sheet(root, pick, fig_dir / "removed_empty_water.png", title="removed as empty open water (audit me)")
            if p:
                links.append(f"![empty](figures/{p.name})")
        md += ["## Figures", "", *links, ""]

    md += ["## Curation steps", "", "```", *[str(s) for s in curation_log.get("steps", [])], "```", ""]
    report = out_dir / "report.md"
    report.write_text("\n".join(md))
    save_json(js, out_dir / "report.json")
    log.info(f"report -> {report}")
    return report
