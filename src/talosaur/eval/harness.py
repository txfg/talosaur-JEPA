"""Frozen-feature evaluation harness (M3/M4).

For each backbone and input size:
  1. extract patch tokens for the labelled train / val / test images (grid of that backbone);
  2. **frame probe**: logistic regression on mean+max pooled tokens (weight decay chosen on val)
     -> test AUROC, AP, TPR@5% FPR, overall / per slice (dark, murky, clear) / per source,
     with bootstrap 95% CIs;
  3. **patch probe**: logistic regression on patch tokens (positives: coverage >= pos_thr;
     negatives: empty patches of masked / exhaustively boxed / empty frames, plus patches
     >= neg_margin cells from every box in other boxed images, see ``patch_negatives``)
     -> patch AUROC/AP, within-image AUROC, best IoU, and the steering metrics: centroid error
     (deg), size error, hit rate, heatmap-max frame AUROC, overall and per slice;
  4. the two fitted probes are saved as deployable heads (``heads_<backbone>_<H>x<W>.pt``);
  5. optional **Underwater-C** sweep: held-out synthetic degradations at severities 1-5 on test
     images, re-using the trained heads (secondary to the real slices).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from talosaur.data.labels import patch_negatives, sample_presence_crops, within_image_auroc
from talosaur.data.torch_datasets import LabeledFrames
from talosaur.eval.backbones import Backbone
from talosaur.eval.linear import fit_logreg
from talosaur.eval.metrics import (
    angular_error_deg,
    average_precision,
    best_iou,
    bootstrap_ci,
    roc_auc,
    soft_centroid,
    tpr_at_fpr,
)
from talosaur.models.heads import pool_meanmax
from talosaur.utils.log import get_logger
from talosaur.utils.seed import derive_seed

log = get_logger("eval")
SLICES = LabeledFrames.SLICES


@dataclass
class EvalConfig:
    pos_thr: float = 0.3
    crop_sources: tuple[str, ...] = ("fathomnet",)  # boxes but no empty frames -> presence crops
    crops_per_image: int = 2
    max_frame_train: int = 30000
    max_patch_images_train: int = 8000
    max_patch_samples: int = 400_000
    neg_margin: int | None = 1  # None: background only from masked / exhaustive / empty frames
    wd_grid: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1)
    batch_size: int = 128
    workers: int = 8
    bootstrap: int = 500
    seed: int = 0
    robustness_ops: tuple[str, ...] = (
        "haze",
        "low_light",
        "noise",
        "particles",
        "blur",
        "compression",
        "light",
    )
    robustness_levels: tuple[int, ...] = (1, 2, 3, 4, 5)
    robustness_max_images: int = 800
    extra: dict[str, Any] = field(default_factory=dict)


ROBUST_OPS = {  # Underwater-C op groups -> degradation ops
    "haze": {"attenuation": True, "backscatter": True},
    "low_light": {"low_light": True, "noise": True},
    "noise": {"noise": True},
    "particles": {"particles": True},
    "blur": {"blur": True},
    "compression": {"compression": True},
    "light": {"light": True, "attenuation": True, "backscatter": True},
}


# --------------------------------------------------------------------------- data selection


def frame_rows(df, cfg: EvalConfig, split: str):
    """Rows with frame labels, plus presence crops for box-only sources (e.g. FathomNet)."""
    import pandas as pd

    sub = df[df["split"] == split]
    labelled = sub[sub["frame_label"].isin([0, 1]) & ~sub["source"].isin(cfg.crop_sources)].copy()
    labelled["crop"] = None
    rows = [labelled]
    crop_src = sub[sub["source"].isin(cfg.crop_sources) & sub["boxes"].map(len).gt(0)]
    if len(crop_src):
        rng = np.random.default_rng(derive_seed(cfg.seed, "presence_crops", split))
        recs = []
        for _, r in crop_src.iterrows():
            for crop, lab in sample_presence_crops(
                r["boxes"], r["box_is_animal"], rng, n=cfg.crops_per_image
            ):
                rr = r.copy()
                rr["crop"] = crop
                rr["frame_label"] = lab
                rr["source"] = f"{r['source']}_crops"
                recs.append(rr)
        if recs:
            rows.append(pd.DataFrame(recs))
    out = pd.concat(rows, ignore_index=True) if rows else sub.iloc[:0]
    if split == "train" and len(out) > cfg.max_frame_train:
        out = out.sample(cfg.max_frame_train, random_state=cfg.seed).reset_index(drop=True)
    return out


def patch_rows(df, cfg: EvalConfig, split: str):
    sub = df[df["split"] == split]
    has_mask = sub["mask_path"].map(lambda v: isinstance(v, str) and len(v) > 0)
    has_box = sub["boxes"].map(len).gt(0)
    m = has_mask | has_box | (sub["frame_label"] == 0)
    out = sub[m].copy()
    out["crop"] = None
    if split == "train" and len(out) > cfg.max_patch_images_train:
        out = out.sample(cfg.max_patch_images_train, random_state=cfg.seed)
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- extraction


@dataclass
class Features:
    tokens: torch.Tensor  # (N, h, w, D) float16
    frame_label: np.ndarray
    coverage: np.ndarray  # (N, h, w)
    patch_valid: np.ndarray
    centroid: np.ndarray  # (N, 3)
    slice: np.ndarray
    source: np.ndarray

    @property
    def pooled(self) -> torch.Tensor:
        n, h, w, d = self.tokens.shape
        return pool_meanmax(self.tokens.float().reshape(n, h * w, d))


@torch.no_grad()
def extract(
    backbone: Backbone, df, root, hw, cfg: EvalConfig, device, degrade=None, degrade_kw=None
) -> Features:
    ds = LabeledFrames(df, root, hw, backbone.info.patch)
    dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.workers, shuffle=False)
    mean = torch.tensor(backbone.info.mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(backbone.info.std, device=device).view(1, 3, 1, 1)
    toks, fl, cov, val, cen, sl = [], [], [], [], [], []
    backbone.eval().to(device)
    amp = device.type == "cuda"
    for i, b in enumerate(dl):
        x = b["image"].to(device).float() / 255.0
        if degrade is not None:
            g = torch.Generator(device=device).manual_seed(cfg.seed * 7919 + i)
            x = degrade(x, g, **(degrade_kw or {}))
        x = (x - mean) / std
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            t = backbone(x)
        toks.append(t.half().cpu())
        fl.append(b["frame_label"].numpy())
        cov.append(b["coverage"].numpy())
        val.append(b["patch_valid"].numpy())
        cen.append(b["centroid"].numpy())
        sl.append(b["slice"].numpy())
    if not toks:
        raise ValueError("no images to extract")
    return Features(
        torch.cat(toks), np.concatenate(fl), np.concatenate(cov), np.concatenate(val),
        np.concatenate(cen), np.concatenate(sl), df["source"].to_numpy(),
    )  # fmt: skip


# --------------------------------------------------------------------------- probes


def _pick_wd(Xtr, ytr, Xva, yva, grid, device) -> tuple[torch.nn.Linear, float]:
    import torch.nn.functional as F

    best, best_wd, best_loss = None, None, float("inf")
    for wd in grid:
        lin = fit_logreg(Xtr.to(device), ytr.to(device), weight_decay=wd)
        if len(yva) and len(torch.unique(yva)) == 2:
            with torch.no_grad():
                loss = F.binary_cross_entropy_with_logits(
                    lin(Xva.to(device)).squeeze(-1), yva.float().to(device)
                ).item()
        else:
            loss = wd  # no usable val set: prefer the smallest wd
        if loss < best_loss:
            best, best_wd, best_loss = lin, wd, loss
    return best.cpu(), best_wd


def _with_ci(metric_fn, n: int, reps: int, seed: int) -> dict[str, float]:
    v = metric_fn(np.arange(n))
    lo, hi = bootstrap_ci(metric_fn, n, reps, seed) if reps else (float("nan"), float("nan"))
    return {"value": v, "ci_lo": lo, "ci_hi": hi, "n": n}


def frame_probe(
    tr: Features, va: Features, te: Features, cfg: EvalConfig, device
) -> tuple[torch.nn.Linear, dict]:
    Xtr, ytr = tr.pooled, torch.from_numpy(tr.frame_label).long()
    Xva, yva = va.pooled, torch.from_numpy(va.frame_label).long()
    lin, wd = _pick_wd(Xtr, ytr, Xva, yva, cfg.wd_grid, device)
    with torch.no_grad():
        s = lin(te.pooled).squeeze(-1).numpy()
    y = te.frame_label
    res: dict[str, Any] = {"weight_decay": wd, "n_train": int(len(ytr)), "n_test": int(len(y))}

    def block(mask) -> dict:
        yy, ss = y[mask], s[mask]
        n = int(mask.sum())
        if n < 5 or len(np.unique(yy)) < 2:
            return {"n": n}
        return {
            "auroc": _with_ci(lambda i: roc_auc(yy[i], ss[i]), n, cfg.bootstrap, cfg.seed),
            "ap": average_precision(yy, ss),
            "tpr_at_5fpr": tpr_at_fpr(yy, ss, 0.05),
            "n": n,
            "pos_frac": float(yy.mean()),
        }

    res["all"] = block(np.ones(len(y), dtype=bool))
    for k, name in enumerate(SLICES[:3]):
        res[name] = block(te.slice == k)
    res["by_source"] = {src: block(te.source == src) for src in np.unique(te.source)}
    return lin, res


def _patch_training_set(f: Features, cfg: EvalConfig, rng) -> tuple[torch.Tensor, torch.Tensor]:
    n, h, w, d = f.tokens.shape
    cov = f.coverage.reshape(n, h * w)
    pos = cov >= cfg.pos_thr
    neg = patch_negatives(f.coverage, f.patch_valid, cfg.neg_margin).reshape(n, h * w)
    pi = np.flatnonzero(pos.reshape(-1))
    ni = np.flatnonzero(neg.reshape(-1))
    half = cfg.max_patch_samples // 2
    pi = rng.permutation(pi)[:half]
    ni = rng.permutation(ni)[: max(half, len(pi))]
    flat = f.tokens.reshape(-1, d)
    X = torch.cat([flat[torch.from_numpy(pi)], flat[torch.from_numpy(ni)]]).float()
    y = torch.cat([torch.ones(len(pi)), torch.zeros(len(ni))]).long()
    return X, y


def patch_probe(tr: Features, va: Features, te: Features, cfg: EvalConfig, device, hfov: float, vfov: float):
    rng = np.random.default_rng(cfg.seed)
    Xtr, ytr = _patch_training_set(tr, cfg, rng)
    if len(torch.unique(ytr)) < 2:
        return None, {"error": "patch probe needs positive and negative patches in the train split"}
    Xva, yva = _patch_training_set(va, cfg, rng) if len(va.frame_label) else (Xtr[:0], ytr[:0])
    lin, wd = _pick_wd(Xtr, ytr, Xva, yva, cfg.wd_grid, device)
    return lin, patch_metrics(lin, te, cfg, hfov, vfov) | {
        "weight_decay": wd,
        "n_train_patches": int(len(ytr)),
    }


def heatmaps(lin: torch.nn.Linear, f: Features) -> np.ndarray:
    n, h, w, d = f.tokens.shape
    with torch.no_grad():
        logit = lin(f.tokens.float().reshape(-1, d)).reshape(n, h, w)
    return torch.sigmoid(logit).numpy()


def patch_metrics(lin, te: Features, cfg: EvalConfig, hfov: float, vfov: float) -> dict:
    prob = heatmaps(lin, te)
    lab = np.where(te.coverage >= cfg.pos_thr, 1, np.where(te.coverage <= 0.0, 0, -1))
    lab[~te.patch_valid] = np.where(lab[~te.patch_valid] == 1, 1, -1)
    res: dict[str, Any] = {}

    def block(mask) -> dict:
        out: dict[str, Any] = {"n_images": int(mask.sum())}
        m = lab[mask] >= 0
        pl, pp = lab[mask][m], prob[mask][m]
        if len(pl) and (pl == 1).any() and (pl == 0).any():
            out["patch_auroc"] = roc_auc(pl, pp)
            out["patch_ap"] = average_precision(pl, pp)
            valid_imgs = mask & te.patch_valid
            if valid_imgs.any():
                out["best_iou"], out["iou_thr"] = best_iou(
                    prob[valid_imgs], te.coverage[valid_imgs] >= cfg.pos_thr
                )
        out["within_image_auroc"] = within_image_auroc(prob[mask], te.coverage[mask], cfg.pos_thr)
        errs, size_err, hits = [], [], []
        for i in np.flatnonzero(mask):
            c = te.centroid[i]
            if c[0] < 0:
                continue
            pc = soft_centroid(prob[i])
            if pc is None:
                errs.append(np.nan)
                continue
            errs.append(angular_error_deg(pc, c, hfov, vfov))
            if pc[2] > 0 and c[2] > 0:
                size_err.append(abs(np.log(pc[2] / c[2])))
            am = np.unravel_index(np.argmax(prob[i]), prob[i].shape)
            hits.append(te.coverage[i][am] > 0)
        if errs:
            e = np.asarray(errs, dtype=np.float64)
            found = np.isfinite(e)
            out["centroid_err_deg_median"] = float(np.median(e[found])) if found.any() else float("nan")
            out["centroid_err_deg_p90"] = float(np.quantile(e[found], 0.9)) if found.any() else float("nan")
            out["found_frac"] = float(found.mean())
            out["size_log_err_median"] = float(np.median(size_err)) if size_err else float("nan")
            out["peak_hit_rate"] = float(np.mean(hits)) if hits else float("nan")
            out["n_with_animal"] = len(errs)
        fl = te.frame_label[mask]
        if (fl >= 0).any() and len(np.unique(fl[fl >= 0])) == 2:
            out["heatmap_max_frame_auroc"] = roc_auc(
                fl[fl >= 0], prob[mask].reshape(mask.sum(), -1).max(1)[fl >= 0]
            )
        return out

    res["all"] = block(np.ones(len(te.frame_label), dtype=bool))
    for k, name in enumerate(SLICES[:3]):
        res[name] = block(te.slice == k)
    return res


# --------------------------------------------------------------------------- top level


def evaluate_backbone(
    backbone: Backbone,
    df,
    root,
    hw,
    cfg: EvalConfig,
    device,
    out_dir: Path,
    hfov=102.0,
    vfov=67.0,
    robustness: bool = False,
) -> dict:
    requested = tuple(hw)
    hw = backbone.input_size(hw)
    tag = f"{backbone.info.name}_{requested[0]}x{requested[1]}"
    log.info(f"[{tag}] extracting features")
    fr = {s: frame_rows(df, cfg, s) for s in ("train", "val", "test")}
    pr = {s: patch_rows(df, cfg, s) for s in ("train", "val", "test")}
    res: dict[str, Any] = {
        "backbone": backbone.info.name,
        "requested_hw": list(requested),
        "input_hw": list(hw),  # snapped to the backbone's stride (e.g. 224 -> 224 for /16, 70 for /14 at 64)
        "grid": list(backbone.grid(hw)),
    }
    frame_lin = patch_lin = None
    if all(len(fr[s]) for s in ("train", "test")):
        F = {s: extract(backbone, fr[s], root, hw, cfg, device) if len(fr[s]) else None for s in fr}
        if F["val"] is None:
            F["val"] = F["test"]
        frame_lin, res["frame"] = frame_probe(F["train"], F["val"], F["test"], cfg, device)
    else:
        res["frame"] = {"error": "no frame-labelled train/test images"}
    if all(len(pr[s]) for s in ("train", "test")):
        P = {s: extract(backbone, pr[s], root, hw, cfg, device) if len(pr[s]) else None for s in pr}
        if P["val"] is None:
            P["val"] = P["test"]
        patch_lin, res["patch"] = patch_probe(P["train"], P["val"], P["test"], cfg, device, hfov, vfov)
    else:
        res["patch"] = {"error": "no patch-labelled train/test images"}
    if frame_lin is not None and patch_lin is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backbone": backbone.info.name,
                "input_hw": list(hw),
                "dim": backbone.info.dim,
                "frame_head": frame_lin.state_dict(),
                "heatmap_head": patch_lin.state_dict(),
                "pos_thr": cfg.pos_thr,
                "mean": list(backbone.info.mean),
                "std": list(backbone.info.std),
            },
            out_dir / f"heads_{tag}.pt",
        )
    if robustness and frame_lin is not None and patch_lin is not None:
        res["robustness"] = underwater_c(
            backbone, fr["test"], pr["test"], root, hw, cfg, device, frame_lin, patch_lin, hfov, vfov
        )
    (out_dir / f"result_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def underwater_c(
    backbone, frame_df, patch_df, root, hw, cfg: EvalConfig, device, frame_lin, patch_lin, hfov, vfov
) -> dict:
    """Held-out synthetic degradations (eval variant) at severity levels 1..5."""
    from talosaur.augment.degrade import DegradeConfig, UnderwaterDegradation

    deg = UnderwaterDegradation(DegradeConfig(p=1.0, variant="eval")).to(device)

    def pick(df):
        clear = df[df["slice"] == "clear"]
        base = clear if len(clear) >= 50 else df
        return base.sample(min(len(base), cfg.robustness_max_images), random_state=cfg.seed).reset_index(
            drop=True
        )

    fdf, pdf = pick(frame_df), pick(patch_df)
    out: dict[str, Any] = {}
    for op in cfg.robustness_ops:
        ops = ROBUST_OPS[op]
        for lvl in cfg.robustness_levels:
            sev = lvl / 5.0

            def kw(n, sev=sev, ops=ops):
                return {"ops": ops, "severity": torch.full((n,), sev, device=device)}

            f = extract(backbone, fdf, root, hw, cfg, device, degrade=_Sized(deg, kw))
            p = extract(backbone, pdf, root, hw, cfg, device, degrade=_Sized(deg, kw))
            with torch.no_grad():
                s = frame_lin(f.pooled).squeeze(-1).numpy()
            pm = patch_metrics(patch_lin, p, cfg, hfov, vfov)["all"]
            out[f"{op}/{lvl}"] = {
                "frame_auroc": roc_auc(f.frame_label, s)
                if len(np.unique(f.frame_label)) == 2
                else float("nan"),
                "patch_auroc": pm.get("patch_auroc", float("nan")),
                "centroid_err_deg_median": pm.get("centroid_err_deg_median", float("nan")),
            }
    return out


class _Sized:
    """Adapter so ``extract`` can call ``degrade(x, generator)`` with per-batch severity tensors."""

    def __init__(self, deg, kw_fn):
        self.deg, self.kw_fn = deg, kw_fn

    def __call__(self, x, g):
        return self.deg(x, g, **self.kw_fn(x.shape[0]))
