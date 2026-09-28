"""fp32 PyTorch vs ONNX fp32 vs int8 on labelled test images: does quantisation hurt the metrics
the vehicle relies on (frame AUROC, patch AUROC, steering centroid error)?"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from talosaur.eval.metrics import angular_error_deg, roc_auc, soft_centroid


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def labelled_batch(df, root, hw, patch: int = 16, max_images: int = 400, seed: int = 0):
    from talosaur.data.torch_datasets import LabeledFrames

    sub = (
        df.sample(min(len(df), max_images), random_state=seed).reset_index(drop=True)
        if len(df) > max_images
        else df.reset_index(drop=True)
    )
    ds = LabeledFrames(sub, root, hw, patch)
    items = [ds[i] for i in range(len(ds))]
    x = np.stack([it["image"].numpy().astype(np.float32) / 255.0 for it in items])
    lab = {
        k: np.stack([it[k].numpy() for it in items])
        for k in ("frame_label", "coverage", "patch_valid", "centroid")
    }
    return x, lab


def _metrics(frame_logit, heat_logit, lab, pos_thr=0.3, hfov=102.0, vfov=67.0) -> dict[str, float]:
    out: dict[str, float] = {}
    fl = lab["frame_label"]
    m = fl >= 0
    if m.any() and len(np.unique(fl[m])) == 2:
        out["frame_auroc"] = roc_auc(fl[m], frame_logit[m, 0])
    prob = _sigmoid(heat_logit)
    cov, valid = lab["coverage"], lab["patch_valid"]
    pl = np.where(cov >= pos_thr, 1, np.where(cov <= 0, 0, -1))
    pl[~valid] = np.where(pl[~valid] == 1, 1, -1)
    mm = pl >= 0
    if mm.any() and (pl[mm] == 1).any() and (pl[mm] == 0).any():
        out["patch_auroc"] = roc_auc(pl[mm], prob[mm])
    errs = []
    for i in range(len(prob)):
        c = lab["centroid"][i]
        if c[0] < 0:
            continue
        pc = soft_centroid(prob[i])
        if pc is not None:
            errs.append(angular_error_deg(pc, c, hfov, vfov))
    if errs:
        out["centroid_err_deg_median"] = float(np.median(errs))
    return out


def run_parity(
    net,
    runners: dict[str, Any],
    x: np.ndarray,
    lab: dict,
    pos_thr: float = 0.3,
    sizes_mb: dict[str, float] | None = None,
) -> dict[str, Any]:
    """``runners``: name -> callable mapping one (3, H, W) image to (frame_logit, heatmap_logit,
    embedding) - the same runner classes the Pi uses (talosaur.onboard.runtime)."""
    with torch.no_grad():
        ref = [t.numpy() for t in net(torch.from_numpy(x))]
    res: dict[str, Any] = {"n_images": int(len(x)), "torch_fp32": _metrics(ref[0], ref[1], lab, pos_thr)}
    for name, run in runners.items():
        outs = [[], [], []]
        t0 = time.perf_counter()
        for i in range(len(x)):
            o = run(x[i])
            for k in range(3):
                outs[k].append(np.asarray(o[k]).reshape(ref[k].shape[1:]))
        dt = (time.perf_counter() - t0) / max(1, len(x))
        fo, ho, eo = (np.stack(v) for v in outs)
        m = _metrics(fo, ho, lab, pos_thr)
        hp, rp = _sigmoid(ho).reshape(len(x), -1), _sigmoid(ref[1]).reshape(len(x), -1)
        corr = np.mean(
            [np.corrcoef(a, b)[0, 1] if a.std() > 1e-6 and b.std() > 1e-6 else 1.0 for a, b in zip(hp, rp)]
        )
        m.update(
            {
                "heatmap_corr_vs_torch": float(corr),
                "frame_logit_max_abs_diff": float(np.abs(fo - ref[0]).max()),
                "heatmap_prob_mean_abs_diff": float(np.abs(hp - rp).mean()),
                "embedding_cosine_vs_torch": float(
                    np.mean(
                        np.sum(eo * ref[2], 1)
                        / (np.linalg.norm(eo, axis=1) * np.linalg.norm(ref[2], axis=1) + 1e-9)
                    )
                ),
                "latency_ms_this_machine": 1000 * dt,
                "file_mb": (sizes_mb or {}).get(name, float("nan")),
            }
        )
        for k in ("frame_auroc", "patch_auroc", "centroid_err_deg_median"):
            if k in m and k in res["torch_fp32"]:
                m[f"delta_{k}"] = m[k] - res["torch_fp32"][k]
        res[name] = m
    return res


def parity_markdown(res: dict[str, Any], tolerances: dict[str, float]) -> str:
    rows = [
        "| model | frame AUROC | patch AUROC | centroid err ° | heatmap corr | Δ frame AUROC | Δ centroid ° | size MB | ms (this machine) | ok |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    def f(v):
        return "–" if v is None or v != v else f"{v:.3f}"

    for name, m in res.items():
        if not isinstance(m, dict):
            continue
        ok = ""
        if name != "torch_fp32":
            bad = (
                abs(m.get("delta_frame_auroc", 0.0)) > tolerances["auroc"]
                or m.get("delta_centroid_err_deg_median", 0.0) > tolerances["centroid_deg"]
            )
            ok = "NO" if bad else "yes"
        rows.append(
            f"| {name} | {f(m.get('frame_auroc'))} | {f(m.get('patch_auroc'))} | {f(m.get('centroid_err_deg_median'))} | {f(m.get('heatmap_corr_vs_torch'))} | "
            f"{f(m.get('delta_frame_auroc'))} | {f(m.get('delta_centroid_err_deg_median'))} | {f(m.get('file_mb'))} | {f(m.get('latency_ms_this_machine'))} | {ok} |"
        )
    return "\n".join(rows)
