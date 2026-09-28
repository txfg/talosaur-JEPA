"""Curation: records from all sources -> one balanced, deduplicated, grouped-split index.

Steps (each logged with counts for the dataset report):
 1. load ``interim/<source>/records.parquet`` for the configured sources
 2. optional license filter (``commercial``: keep rows with ``train_commercial_ok``)
 3. condition metrics per image (cached) -> light / clarity buckets -> eval slices
 4. grouped train/val/test splits (hash of group id; official splits optional per source)
 5. near-duplicate clusters across *all* sources: drop redundant copies, and drop any training
    image that duplicates a val/test image (leakage)
 6. empty open-water downsampling (unlabelled training frames only; keep fractions per bucket)
 7. optional per-group cap on training frames (one long dive can't dominate)
 8. sampling weights: inverse light x clarity bucket frequency (capped) + per-source share caps
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np

from talosaur.data.dedup import DedupConfig, near_duplicate_clusters
from talosaur.data.index import concat_record_tables, read_table, write_table
from talosaur.data.quality import BucketThresholds, assign_buckets, calibrate_thresholds, is_empty_water
from talosaur.data.schema import METRIC_COLUMNS
from talosaur.utils.io import load_yaml, save_json
from talosaur.utils.log import get_logger
from talosaur.utils.seed import derive_seed

log = get_logger("data.curate")


@dataclass
class CurateConfig:
    name: str = "underwater_v1"
    root: str = "data"
    sources: list[str] = field(default_factory=lambda: ["fathomnet", "noaa_oer", "deepfish"])
    license_filter: str = "none"  # none | commercial
    metrics_max_side: int = 256
    workers: int = max(1, (os.cpu_count() or 2) - 1)
    thresholds: dict[str, float] = field(default_factory=dict)
    thresholds_file: str | None = None  # YAML written by scripts/data/label_frames.py calibrate
    calibrate: str = "none"  # none | percentile
    dedup: dict[str, Any] = field(default_factory=lambda: {"enabled": True})
    empty_water: dict[str, Any] = field(
        default_factory=lambda: {"enabled": True, "keep_frac": {"default": 0.15, "murky": 0.5, "dark": 0.5}}
    )
    val_frac: float = 0.05
    test_frac: float = 0.10
    split_seed: int = 0
    official_splits: list[str] = field(default_factory=list)  # sources whose split_hint is honoured
    max_train_per_group: int | None = None
    weight_cap: float = 8.0
    source_max_frac: dict[str, float] = field(default_factory=dict)
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> CurateConfig:
        d = load_yaml(path) or {}
        d.update({k: v for k, v in overrides.items() if v is not None})
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


# --------------------------------------------------------------------------- metrics (cached)


def _metrics_worker(args: tuple[str, int]) -> dict:
    from talosaur.data.quality import compute_metrics_for_path

    path, max_side = args
    try:
        return compute_metrics_for_path(path, max_side)
    except Exception as e:  # unreadable/corrupt image
        return {"error": str(e)}


def compute_metrics_table(df, root: Path, workers: int, max_side: int, cache_path: Path):
    """Add METRIC_COLUMNS + phash + thumb columns; cache keyed by (path, size, mtime)."""
    import pandas as pd

    root = Path(root)
    keys = []
    for p in df["path"]:
        st = (root / p).stat()
        keys.append(f"{p}|{st.st_size}|{int(st.st_mtime)}")
    cache = read_table(cache_path) if cache_path.exists() else pd.DataFrame(columns=["key"])
    cache = cache.set_index("key") if len(cache) else cache
    have = set(cache.index) if len(cache) else set()
    todo = [(i, str(root / p)) for i, (p, k) in enumerate(zip(df["path"], keys)) if k not in have]
    log.info(f"metrics: {len(df) - len(todo)} cached, {len(todo)} to compute with {workers} workers")
    new_rows = []
    if todo:
        args = [(p, max_side) for _, p in todo]
        if workers > 1:
            with Pool(workers) as pool:
                results = pool.map(_metrics_worker, args, chunksize=64)
        else:
            results = [_metrics_worker(a) for a in args]
        for (i, _), r in zip(todo, results):
            r = dict(r)
            r["key"] = keys[i]
            new_rows.append(r)
    if new_rows:
        new = pd.DataFrame(new_rows).set_index("key")
        cache = pd.concat([cache, new]) if len(cache) else new
        cache = cache[~cache.index.duplicated(keep="last")]
        out = cache.reset_index()
        write_table(out, cache_path)
    m = cache.loc[keys]
    for c in [*METRIC_COLUMNS, "phash", "thumb"]:
        df[c] = m[c].to_numpy() if c in m.columns else np.nan
    if "error" in m.columns:
        df["metric_error"] = m["error"].to_numpy()
    return df


# --------------------------------------------------------------------------- splits


def _hash01(*parts) -> float:
    return (derive_seed(*parts) % 1_000_003) / 1_000_003.0


def assign_splits(df, cfg: CurateConfig):
    splits = []
    for src, gid, hint in zip(df["source"], df["group_id"], df["split_hint"]):
        if src in cfg.official_splits and isinstance(hint, str) and hint in ("train", "val", "test"):
            splits.append(hint)
            continue
        u = _hash01(cfg.split_seed, "split", gid)
        splits.append("test" if u < cfg.test_frac else "val" if u < cfg.test_frac + cfg.val_frac else "train")
    df["split"] = splits
    mixed = df.groupby("group_id")["split"].nunique()
    if (mixed > 1).any():
        log.warning(f"{int((mixed > 1).sum())} groups span several splits (official splits); see report")
    return df


# --------------------------------------------------------------------------- dedup


def _priority(row) -> tuple:
    """Higher is better: richer supervision, then larger image."""
    richness = (
        (4 if isinstance(row["mask_path"], str) and row["mask_path"] else 0)
        + (2 if len(row["boxes"]) else 0)
        + (1 if row["frame_label"] != -1 else 0)
    )
    return (richness, row["width"] * row["height"])


def apply_dedup(df, cfg: CurateConfig):
    """Return (keep mask, reason array). Leakage rule: training members of a cluster that
    contains any val/test image are dropped; within a split only the best member is kept."""
    dcfg_d = {k: v for k, v in cfg.dedup.items() if k != "enabled"}
    dcfg = DedupConfig(**dcfg_d)
    thumbs = np.stack([np.frombuffer(t, dtype=np.uint8) for t in df["thumb"]])
    clusters = near_duplicate_clusters(df["phash"].to_numpy(dtype=np.uint64), thumbs, dcfg)
    df["dup_cluster"] = clusters
    keep = np.ones(len(df), dtype=bool)
    reason = np.array([""] * len(df), dtype=object)
    order = np.argsort(clusters, kind="stable")
    cs = clusters[order]
    starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
    ends = np.r_[starts[1:], len(cs)]
    for s, e in zip(starts, ends):
        if e - s < 2:
            continue
        members = order[s:e]
        sub = df.iloc[members]
        has_eval = (sub["split"] != "train").any()
        for split in ("train", "val", "test"):
            idx = members[(sub["split"] == split).to_numpy()]
            if len(idx) == 0:
                continue
            if split == "train" and has_eval:
                keep[idx] = False
                reason[idx] = "dup_of_eval"
                continue
            best = max(idx, key=lambda i: _priority(df.iloc[i]))
            for i in idx:
                if i != best:
                    keep[i] = False
                    reason[i] = "near_dup"
        # a val image duplicating a test image would double count: keep the test copy
        vals = members[(sub["split"] == "val").to_numpy()]
        if len(vals) and (sub["split"] == "test").any():
            keep[vals] = False
            reason[vals] = "dup_of_test"
    return keep, reason


# --------------------------------------------------------------------------- weights


def compute_weights(df, cfg: CurateConfig) -> np.ndarray:
    w = np.zeros(len(df), dtype=np.float64)
    train = (df["split"] == "train").to_numpy()
    if not train.any():
        return w
    bucket = (df["light"] + "/" + df["clarity_bucket"]).to_numpy()
    tb = bucket[train]
    uniq, counts = np.unique(tb, return_counts=True)
    freq = dict(zip(uniq, counts / counts.sum()))
    raw = np.array([1.0 / freq[b] for b in tb])
    raw = raw / np.median(raw)
    raw = np.clip(raw, 1.0 / cfg.weight_cap, cfg.weight_cap)
    w[train] = raw
    src = df["source"].to_numpy()
    for _ in range(5):  # iterate: capping one source changes the others' shares
        total = w[train].sum()
        changed = False
        for s, maxf in cfg.source_max_frac.items():
            m = train & (src == s)
            share = w[m].sum() / total if total else 0
            if share > maxf + 1e-9:
                others = total - w[m].sum()
                target = maxf * others / (1 - maxf) if maxf < 1 else w[m].sum()
                w[m] *= target / w[m].sum()
                changed = True
        if not changed:
            break
    w[train] /= w[train].mean()
    return w


# --------------------------------------------------------------------------- main entry


def run_curation(cfg: CurateConfig):
    """Returns (index DataFrame, removed DataFrame, curation log dict) and writes them to disk."""
    import pandas as pd

    root = Path(cfg.root)
    paths = []
    for s in cfg.sources:
        p = root / "interim" / s / "records.parquet"
        if p.exists():
            paths.append(p)
        else:
            log.warning(f"source {s!r} has no records at {p}; skipping (run scripts/data/fetch.py {s} first)")
    df = concat_record_tables(paths)
    steps: list[dict[str, Any]] = [{"step": "loaded", "n": len(df), "by_source": df["source"].value_counts().to_dict()}]
    removed = []

    if cfg.license_filter == "commercial":
        m = df["train_commercial_ok"].astype(bool)
        removed.append(df[~m].assign(removed_reason="license_filter"))
        df = df[m].reset_index(drop=True)
        steps.append({"step": "license_filter", "n": len(df)})

    df = compute_metrics_table(df, root, cfg.workers, cfg.metrics_max_side, root / "cache" / "metrics.parquet")
    bad = df.get("metric_error")
    if bad is not None:
        m = bad.isna() | (bad == "")
        if (~m).any():
            removed.append(df[~m].assign(removed_reason="unreadable"))
            df = df[m].reset_index(drop=True)
    steps.append({"step": "metrics", "n": len(df)})

    t = BucketThresholds(**cfg.thresholds)
    if cfg.thresholds_file:
        t = BucketThresholds(**{**t.to_dict(), **(load_yaml(cfg.thresholds_file) or {})})
    if cfg.calibrate == "percentile":
        t = calibrate_thresholds(df, t)
    assign_buckets(df, t)
    df = assign_splits(df, cfg)
    steps.append({"step": "splits", "counts": df["split"].value_counts().to_dict()})

    if cfg.dedup.get("enabled", True):
        keep, reason = apply_dedup(df, cfg)
        removed.append(df[~keep].assign(removed_reason=reason[~keep]))
        df = df[keep].reset_index(drop=True)
        steps.append({"step": "dedup", "n": len(df), "removed": pd.Series(reason[~keep]).value_counts().to_dict()})
    else:
        df["dup_cluster"] = np.arange(len(df))

    df["is_empty"] = [is_empty_water(r, t) for r in df[["grad_energy", "edge_density"]].to_dict("records")]
    ew = cfg.empty_water
    if ew.get("enabled", True):
        rng = np.random.default_rng(derive_seed(cfg.seed, "empty_water"))
        kf = ew.get("keep_frac", {"default": 0.15})
        cand = (
            (df["split"] == "train")
            & df["is_empty"]
            & ~df["labeled"].astype(bool)
            & (df["frame_label"] != 1)
        ).to_numpy()
        keep_p = np.array(
            [kf.get(s, kf.get(lb, kf.get("default", 0.15))) for s, lb in zip(df["slice"], df["light"])]
        )
        drop = cand & (rng.random(len(df)) >= keep_p)
        removed.append(df[drop].assign(removed_reason="empty_water"))
        steps.append({"step": "empty_water", "candidates": int(cand.sum()), "dropped": int(drop.sum())})
        df = df[~drop].reset_index(drop=True)

    if cfg.max_train_per_group:
        rng = np.random.default_rng(derive_seed(cfg.seed, "group_cap"))
        drop = np.zeros(len(df), dtype=bool)
        tr = df["split"] == "train"
        for _, idx in df[tr].groupby("group_id").groups.items():
            idx = np.asarray(idx)
            if len(idx) > cfg.max_train_per_group:
                drop[rng.choice(idx, len(idx) - cfg.max_train_per_group, replace=False)] = True
        removed.append(df[drop].assign(removed_reason="group_cap"))
        df = df[~drop].reset_index(drop=True)
        steps.append({"step": "group_cap", "dropped": int(drop.sum())})

    df["weight"] = compute_weights(df, cfg)
    steps.append({"step": "final", "n": len(df), "by_split": df["split"].value_counts().to_dict()})

    rem = pd.concat(removed, ignore_index=True) if removed else df.iloc[:0].assign(removed_reason="")
    out_dir = root / "index"
    index_path = out_dir / f"{cfg.name}.parquet"
    write_table(df.drop(columns=["thumb"]), index_path)
    write_table(rem.drop(columns=["thumb"], errors="ignore"), out_dir / f"{cfg.name}.removed.parquet")
    curation_log = {"config": asdict(cfg), "thresholds": t.to_dict(), "steps": steps}
    save_json(curation_log, out_dir / f"{cfg.name}.curation.json")
    log.info(f"index -> {index_path} ({len(df)} images)")
    return df, rem, curation_log
