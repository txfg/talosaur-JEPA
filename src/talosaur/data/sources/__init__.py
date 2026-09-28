"""Dataset sources. Use ``scripts/data/fetch.py <source> --root DATA --accept-license ID`` or
:func:`run_fetch` from Python. Each fetch writes ``interim/<source>/records.parquet`` and a
manifest; ``scripts/data/build_dataset.py`` then curates all sources into one index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from talosaur.data.schema import write_records
from talosaur.data.sources.base import (
    LicenseNotAccepted,
    SourceInfo,
    records_path,
    require_license_acceptance,
    safe_extract,
    write_manifest,
)
from talosaur.utils.log import get_logger

log = get_logger("data.fetch")

SOURCES = ["fathomnet", "noaa_oer", "deepfish", "kakadu", "river_herring", "brackish", "ozfish", "onc", "own", "synthetic"]

__all__ = ["SOURCES", "run_fetch", "LicenseNotAccepted", "SourceInfo"]


def _find_coco_jsons(d: Path) -> list[Path]:
    out = []
    for p in sorted(Path(d).rglob("*.json")):
        try:
            head = json.loads(p.read_text())
        except Exception:
            continue
        if isinstance(head, dict) and "images" in head and "annotations" in head:
            out.append(p)
    return out


def _split_from_name(p: Path) -> str | None:
    s = p.as_posix().lower()
    for key, split in (("train", "train"), ("val", "val"), ("valid", "val"), ("test", "test")):
        if key in Path(s).stem or f"/{key}/" in s:
            return split
    return None


def run_fetch(source: str, root: str | Path, accept: str | None, **opts: Any) -> Path:
    """Fetch/ingest one source; returns the records.parquet path."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; choose from {SOURCES}")
    root = Path(root)
    info = SourceInfo.load(source)
    require_license_acceptance(info, accept)
    extra: dict[str, Any] = {"options": {k: (str(v) if isinstance(v, Path) else v) for k, v in opts.items()}}

    if source == "fathomnet":
        from talosaur.data.sources.fathomnet import fetch_fathomnet

        records = fetch_fathomnet(info, root, **opts)
    elif source == "noaa_oer":
        from talosaur.data.sources.noaa_oer import fetch_noaa_oer

        records = fetch_noaa_oer(info, root, **opts)
    elif source == "deepfish":
        from talosaur.data.sources.deepfish import fetch_deepfish, ingest_deepfish

        d = opts.pop("from_dir", None) or fetch_deepfish(info, root, opts.pop("archive", None), opts.pop("url", None))
        records = ingest_deepfish(info, root, Path(d), short_side=opts.get("short_side", 384))
    elif source in ("kakadu", "brackish"):
        from talosaur.data.sources.generic import ingest_coco, ingest_yolo

        d = opts.get("from_dir")
        if d is None and source == "kakadu":
            from talosaur.data.sources.zenodo import download_record

            raw = root / "raw" / source
            for p in download_record(info.access.get("record", 7250921), raw):
                if p.suffix in (".zip", ".tar", ".gz", ".tgz"):
                    safe_extract(p, raw / "extracted")
            d = raw
        if d is None:
            raise ValueError(f"{source}: pass from_dir=<extracted download>")
        records = []
        cocos = _find_coco_jsons(Path(d))
        for cj in cocos:
            records += ingest_coco(
                info,
                root,
                cj,
                image_root=opts.get("image_root") or cj.parent,
                short_side=opts.get("short_side", 384),
                boxes_exhaustive=opts.get("boxes_exhaustive", False),
                split_hint=_split_from_name(cj),
                id_prefix=f"{cj.stem}/",
            )
        if not cocos and source == "brackish":
            names = opts.get("class_names") or ["fish", "crab", "shrimp", "starfish", "small_fish", "jellyfish"]
            records = ingest_yolo(
                info,
                root,
                Path(opts.get("images_dir") or Path(d) / "images"),
                Path(opts.get("labels_dir") or Path(d) / "labels"),
                names,
                short_side=opts.get("short_side", 384),
            )
        if not records:
            raise RuntimeError(f"{source}: found no COCO JSON (or YOLO labels) under {d}")
    elif source == "river_herring":
        from talosaur.data.sources.generic import ingest_camera_traps

        records = ingest_camera_traps(
            info,
            root,
            Path(opts["metadata"]),
            images_dir=Path(opts["images_dir"]) if opts.get("images_dir") else None,
            image_base_url=opts.get("image_base_url"),
            short_side=opts.get("short_side", 384),
            max_empty=opts.get("max_empty"),
            max_nonempty=opts.get("max_nonempty"),
        )
    elif source in ("ozfish", "onc", "own"):
        from talosaur.data.sources.generic import ingest_folder

        if not opts.get("from_dir"):
            raise ValueError(f"{source}: pass from_dir=<folder with images and/or videos>")
        records, reports = ingest_folder(
            info,
            root,
            Path(opts["from_dir"]),
            short_side=opts.get("short_side", 288),
            fps=opts.get("fps", 1.0),
            group_by=opts.get("group_by", "parent"),
        )
        extra["videos"] = reports
    elif source == "synthetic":
        records = _fetch_synthetic(info, root, **opts)
    else:  # pragma: no cover
        raise AssertionError(source)

    out = records_path(root, source)
    write_records(records, out)
    write_manifest(root, info, {**extra, "n_records": len(records)})
    log.info(f"{source}: {len(records)} records -> {out}")
    return out


def _fetch_synthetic(info: SourceInfo, root: Path, n_images: int = 120, n_videos: int = 2, seed: int = 0, **_):
    import numpy as np

    from talosaur.data import synthetic as syn
    from talosaur.data.sources.generic import ingest_coco, ingest_folder

    raw = root / "raw" / "synthetic"
    ann = syn.write_image_dataset(raw / "stills", n=n_images, seed=seed, size=(240, 320))
    rng = np.random.default_rng(seed + 1)
    for i in range(n_videos):
        frames, _ = syn.make_video_frames(
            rng,
            n_frames=80,
            h=144,
            w=256,
            condition=syn.CONDITIONS[i % 3],
            empty_span=(0.35, 0.75),
            overlay_text=(i % 2 == 0),
            static_span=(0.1, 0.25),
        )
        syn.write_video(raw / "videos" / f"dive{i:02d}" / f"clip{i:02d}.mp4", frames, fps=10)
    records = ingest_coco(info, root, ann, short_side=240, boxes_exhaustive=True)
    vrec, _ = ingest_folder(info, root, raw / "videos", short_side=144, fps=4.0, group_by="parent")
    return records + vrec
