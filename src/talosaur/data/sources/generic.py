"""Generic ingesters: COCO detection JSON, YOLO txt labels, COCO Camera Traps, and plain
folders of images/videos. Dataset-specific modules are thin wrappers around these."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from talosaur.data.schema import Record
from talosaur.data.sources.base import SourceInfo, download
from talosaur.data.sources.common import (
    IMAGE_EXTS,
    VIDEO_EXTS,
    find_files,
    ingest_image,
    ingest_mask,
    make_record,
    video_id_from_frame_name,
    xywh_to_norm_xyxy,
)
from talosaur.data.video import ExtractConfig, extract_frames
from talosaur.utils.log import get_logger

log = get_logger("data.ingest")


def _safe_rel(p: str) -> str:
    return p.replace("\\", "/").lstrip("/").replace("..", "_")


def ingest_coco(
    info: SourceInfo,
    root: Path,
    coco_json: Path,
    image_root: Path | None = None,
    short_side: int = 384,
    animal_categories: set[str] | None = None,
    boxes_exhaustive: bool = True,
    group_fn: Callable[[dict], str] | None = None,
    split_hint: str | None = None,
    id_prefix: str = "",
) -> list[Record]:
    """COCO detection JSON -> records. ``animal_categories=None`` means every category is an animal."""
    coco_json = Path(coco_json)
    image_root = Path(image_root) if image_root else coco_json.parent
    data = json.loads(coco_json.read_text())
    cats = {c["id"]: c.get("name", str(c["id"])) for c in data.get("categories", [])}
    anns = defaultdict(list)
    for a in data.get("annotations", []):
        anns[a["image_id"]].append(a)
    records, missing = [], 0
    for img in data.get("images", []):
        src = image_root / img["file_name"]
        if not src.exists():
            missing += 1
            continue
        rel = _safe_rel(f"{id_prefix}{img['file_name']}")
        path, ow, oh, w, h = ingest_image(src, root, info.name, rel, short_side)
        W, H = float(img.get("width") or ow), float(img.get("height") or oh)
        boxes, labels, animal = [], [], []
        for a in anns.get(img["id"], []):
            if "bbox" not in a or a.get("iscrowd", 0):
                continue
            name = cats.get(a.get("category_id"), "object")
            boxes.append(xywh_to_norm_xyxy(a["bbox"], W, H))
            labels.append(name)
            animal.append(animal_categories is None or name in animal_categories)
        mask_path = None
        if img.get("mask_file") and (image_root / img["mask_file"]).exists():
            mask_path = ingest_mask(image_root / img["mask_file"], root, info.name, rel, (w, h))
        if any(animal):
            frame_label = 1
        elif boxes_exhaustive:
            frame_label = 0
        else:
            frame_label = -1
        group = group_fn(img) if group_fn else str(img.get("group") or video_id_from_frame_name(img["file_name"]))
        extra = {k: img[k] for k in ("condition", "has_animal") if k in img}
        records.append(
            make_record(
                info,
                image_id=f"{info.name}:{id_prefix}{img['file_name']}",
                path=path,
                group_id=f"{info.name}:{group}",
                width=w,
                height=h,
                split_hint=img.get("split") or split_hint,
                frame_label=frame_label,
                boxes=boxes,
                box_labels=labels,
                box_is_animal=animal,
                boxes_exhaustive=boxes_exhaustive,
                mask_path=mask_path,
                extra=extra,
            )
        )
    if missing:
        log.warning(f"{info.name}: {missing} images listed in {coco_json.name} were not found under {image_root}")
    return records


def ingest_yolo(
    info: SourceInfo,
    root: Path,
    images_dir: Path,
    labels_dir: Path,
    class_names: list[str],
    short_side: int = 384,
    animal_classes: set[str] | None = None,
    boxes_exhaustive: bool = False,
    split_hint: str | None = None,
) -> list[Record]:
    """YOLO layout: one ``<stem>.txt`` per image with ``cls cx cy w h`` (normalised) lines."""
    records = []
    for src in find_files(images_dir, IMAGE_EXTS):
        rel = _safe_rel(str(src.relative_to(images_dir)))
        lab = Path(labels_dir) / src.relative_to(images_dir).with_suffix(".txt")
        boxes, labels, animal = [], [], []
        if lab.exists():
            for line in lab.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c = int(float(parts[0]))
                cx, cy, bw, bh = (float(v) for v in parts[1:5])
                name = class_names[c] if c < len(class_names) else str(c)
                boxes.append([max(0.0, cx - bw / 2), max(0.0, cy - bh / 2), min(1.0, cx + bw / 2), min(1.0, cy + bh / 2)])
                labels.append(name)
                animal.append(animal_classes is None or name in animal_classes)
        path, _, _, w, h = ingest_image(src, root, info.name, rel, short_side)
        frame_label = 1 if any(animal) else (0 if boxes_exhaustive and lab.exists() else -1)
        records.append(
            make_record(
                info,
                image_id=f"{info.name}:{rel}",
                path=path,
                group_id=f"{info.name}:{video_id_from_frame_name(src.name)}",
                width=w,
                height=h,
                split_hint=split_hint,
                frame_label=frame_label,
                boxes=boxes,
                box_labels=labels,
                box_is_animal=animal,
                boxes_exhaustive=boxes_exhaustive,
            )
        )
    return records


def ingest_camera_traps(
    info: SourceInfo,
    root: Path,
    metadata: Path,
    images_dir: Path | None = None,
    image_base_url: str | None = None,
    short_side: int = 384,
    empty_names: tuple[str, ...] = ("empty", "blank", "none"),
    max_empty: int | None = None,
    max_nonempty: int | None = None,
    seed: int = 0,
) -> list[Record]:
    """COCO Camera Traps JSON (LILA). Frames whose annotations are all 'empty' become negatives.

    Either ``images_dir`` (already downloaded) or ``image_base_url`` (fetched per file) is needed.
    ``max_empty`` subsamples the (often very numerous) empty frames before downloading.
    """
    import numpy as np

    data = json.loads(Path(metadata).read_text())
    cats = {c["id"]: c["name"].lower() for c in data.get("categories", [])}
    anns = defaultdict(list)
    for a in data.get("annotations", []):
        anns[a["image_id"]].append(a)
    imgs = data.get("images", [])
    rng = np.random.default_rng(seed)

    def is_empty(img) -> bool:
        a = anns.get(img["id"], [])
        return bool(a) and all(cats.get(x.get("category_id"), "") in empty_names for x in a)

    empty = [im for im in imgs if is_empty(im)]
    nonempty = [im for im in imgs if anns.get(im["id"]) and not is_empty(im)]
    if max_empty is not None and len(empty) > max_empty:
        empty = [empty[i] for i in sorted(rng.choice(len(empty), max_empty, replace=False))]
    if max_nonempty is not None and len(nonempty) > max_nonempty:
        nonempty = [nonempty[i] for i in sorted(rng.choice(len(nonempty), max_nonempty, replace=False))]

    records = []
    for img, label in [(im, 0) for im in empty] + [(im, 1) for im in nonempty]:
        fn = img["file_name"]
        if images_dir is not None:
            src = Path(images_dir) / fn
        elif image_base_url:
            src = download(image_base_url.rstrip("/") + "/" + fn, root / "raw" / info.name / _safe_rel(fn))
        else:
            raise ValueError("need images_dir or image_base_url")
        if not Path(src).exists():
            continue
        rel = _safe_rel(fn)
        path, ow, oh, w, h = ingest_image(src, root, info.name, rel, short_side)
        boxes, labels = [], []
        for a in anns.get(img["id"], []):
            if "bbox" in a and cats.get(a.get("category_id"), "") not in empty_names:
                boxes.append(xywh_to_norm_xyxy(a["bbox"], img.get("width") or ow, img.get("height") or oh))
                labels.append(cats.get(a.get("category_id"), "animal"))
        loc = str(img.get("location", "unknown"))
        day = str(img.get("datetime", ""))[:10]
        records.append(
            make_record(
                info,
                image_id=f"{info.name}:{fn}",
                path=path,
                group_id=f"{info.name}:{loc}:{day}",
                width=w,
                height=h,
                frame_label=label,
                boxes=boxes,
                box_labels=labels,
                box_is_animal=[True] * len(boxes),
                boxes_exhaustive=False,
                extra={
                    "location": loc,
                    "datetime": img.get("datetime"),
                    "seq_id": img.get("seq_id"),
                    # condition labels some LILA sets carry (River Herring, Puget Sound)
                    **{k: img[k] for k in ("time_of_day", "visibility", "habitat_type", "filter") if k in img},
                },
            )
        )
    return records


def ingest_folder(
    info: SourceInfo,
    root: Path,
    folder: Path,
    short_side: int = 288,
    fps: float = 1.0,
    group_by: str = "parent",
    frame_label: int = -1,
    extract_cfg: ExtractConfig | None = None,
) -> tuple[list[Record], list[dict]]:
    """A folder of stills and/or videos (own footage, OzFish frames, ONC downloads).

    ``group_by``: ``parent`` (sub-folder = dive/deployment) or ``file`` (one group per video /
    frame-name prefix). Returns (records, per-video extraction reports).
    """
    folder = Path(folder)
    records, reports = [], []
    for src in find_files(folder, IMAGE_EXTS):
        rel = _safe_rel(str(src.relative_to(folder)))
        path, _, _, w, h = ingest_image(src, root, info.name, rel, short_side)
        group = src.parent.name if group_by == "parent" else video_id_from_frame_name(src.name)
        records.append(
            make_record(
                info,
                image_id=f"{info.name}:{rel}",
                path=path,
                group_id=f"{info.name}:{group}",
                width=w,
                height=h,
                frame_label=frame_label,
            )
        )
    cfg = extract_cfg or ExtractConfig(fps=fps, short_side=short_side)
    for vid in find_files(folder, VIDEO_EXTS):
        rel = _safe_rel(str(vid.relative_to(folder)))
        video_id = Path(rel).with_suffix("").as_posix().replace("/", "__")
        out_dir = root / "images" / info.name / video_id
        res = extract_frames(vid, out_dir, video_id, cfg)
        reports.append({"video": rel, **res["stats"], "overlay": res["overlay"]})
        group = vid.parent.name if group_by == "parent" and vid.parent != folder else video_id
        for fr in res["frames"]:
            records.append(
                make_record(
                    info,
                    image_id=f"{info.name}:{video_id}:{fr['t_sec']:.2f}",
                    path=f"images/{info.name}/{video_id}/{fr['path']}",
                    group_id=f"{info.name}:{group}",
                    width=fr["width"],
                    height=fr["height"],
                    frame_label=frame_label,
                    video_id=f"{info.name}:{video_id}",
                    t_sec=fr["t_sec"],
                )
            )
    return records, reports
