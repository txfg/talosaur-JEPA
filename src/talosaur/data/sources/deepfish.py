"""DeepFish (JCU): fish/no-fish frame labels, point labels and segmentation masks.

Layout (from the official loader code, github.com/alzayats/DeepFish):
  Classification/{train,val,test}.csv  columns ID, labels      image: Classification/<ID>.jpg
  Localization/{split}.csv            columns ID, labels, counts image: Localization/images/<ID>.jpg,
                                                                  points: Localization/masks/<ID>.png
  Segmentation/{split}.csv            columns ID, labels         image: Segmentation/images/<ID>.jpg,
                                                                  mask: Segmentation/masks/<ID>.png
Habitat = ``ID.split('/')[0]`` for Classification, ``ID.split('/')[1].split('_')[0]`` otherwise;
it is used as the group id so whole habitats can be held out.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from talosaur.data.schema import Record
from talosaur.data.sources.base import SourceInfo, download, safe_extract
from talosaur.data.sources.common import ingest_image, ingest_mask, make_record
from talosaur.utils.log import get_logger

log = get_logger("data.deepfish")
SPLIT_FILES = {"train": "train", "val": "val", "test": "test"}


def _find_base(d: Path) -> Path:
    for cand in [d, *sorted(p for p in d.rglob("Classification") if p.is_dir())]:
        base = cand if (cand / "Classification").is_dir() else cand.parent
        if (base / "Classification").is_dir():
            return base
    raise FileNotFoundError(f"no DeepFish 'Classification' directory under {d}")


def _rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def _habitat(id_: str, subset: str) -> str:
    parts = id_.split("/")
    if subset == "Classification":
        return parts[0]
    return (parts[1] if len(parts) > 1 else parts[0]).split("_")[0]


def _mask_boxes(mask_png: Path, min_px: int = 16) -> tuple[list[list[float]], bool]:
    from PIL import Image
    from scipy import ndimage as ndi

    with Image.open(mask_png) as m:
        arr = np.asarray(m.convert("L")) > 0
    h, w = arr.shape
    lab, n = ndi.label(arr)
    boxes = []
    for sl in ndi.find_objects(lab):
        if sl is None:
            continue
        ys, xs = sl
        if (ys.stop - ys.start) * (xs.stop - xs.start) < min_px:
            continue
        boxes.append([xs.start / w, ys.start / h, xs.stop / w, ys.stop / h])
    return boxes, bool(arr.any())


def _points(mask_png: Path) -> list[list[float]]:
    from PIL import Image

    with Image.open(mask_png) as m:
        arr = np.asarray(m)
    if arr.ndim == 3:
        arr = arr[..., 0]
    ys, xs = np.nonzero(arr > 0)
    h, w = arr.shape
    return [[round(float(x + 0.5) / w, 5), round(float(y + 0.5) / h, 5)] for y, x in zip(ys, xs)]


def ingest_deepfish(info: SourceInfo, root: Path, deepfish_dir: Path, short_side: int = 384) -> list[Record]:
    base = _find_base(Path(deepfish_dir))
    records: list[Record] = []
    n_missing = 0
    for split in SPLIT_FILES:
        for row in _rows(base / "Classification" / f"{split}.csv"):
            id_ = row["ID"]
            src = base / "Classification" / f"{id_}.jpg"
            if not src.exists():
                n_missing += 1
                continue
            path, _, _, w, h = ingest_image(src, root, info.name, f"cls/{id_}.jpg", short_side)
            records.append(
                make_record(
                    info,
                    image_id=f"deepfish:cls:{id_}",
                    path=path,
                    group_id=f"deepfish:{_habitat(id_, 'Classification')}",
                    width=w,
                    height=h,
                    split_hint=split,
                    frame_label=int(float(row["labels"]) > 0),
                    extra={"subset": "classification"},
                )
            )
        for row in _rows(base / "Segmentation" / f"{split}.csv"):
            id_ = row["ID"]
            src = base / "Segmentation" / "images" / f"{id_}.jpg"
            msk = base / "Segmentation" / "masks" / f"{id_}.png"
            if not src.exists() or not msk.exists():
                n_missing += 1
                continue
            path, _, _, w, h = ingest_image(src, root, info.name, f"seg/{id_}.jpg", short_side)
            mask_path = ingest_mask(msk, root, info.name, f"seg/{id_}", (w, h))
            boxes, any_fish = _mask_boxes(msk)
            records.append(
                make_record(
                    info,
                    image_id=f"deepfish:seg:{id_}",
                    path=path,
                    group_id=f"deepfish:{_habitat(id_, 'Segmentation')}",
                    width=w,
                    height=h,
                    split_hint=split,
                    frame_label=int(any_fish or float(row.get("labels", 0) or 0) > 0),
                    boxes=boxes,
                    box_labels=["fish"] * len(boxes),
                    box_is_animal=[True] * len(boxes),
                    boxes_exhaustive=True,
                    mask_path=mask_path,
                    extra={"subset": "segmentation"},
                )
            )
        for row in _rows(base / "Localization" / f"{split}.csv"):
            id_ = row["ID"]
            src = base / "Localization" / "images" / f"{id_}.jpg"
            pts = base / "Localization" / "masks" / f"{id_}.png"
            if not src.exists():
                n_missing += 1
                continue
            path, _, _, w, h = ingest_image(src, root, info.name, f"loc/{id_}.jpg", short_side)
            points = _points(pts) if pts.exists() else []
            count = int(float(row.get("counts", len(points)) or 0))
            records.append(
                make_record(
                    info,
                    image_id=f"deepfish:loc:{id_}",
                    path=path,
                    group_id=f"deepfish:{_habitat(id_, 'Localization')}",
                    width=w,
                    height=h,
                    split_hint=split,
                    frame_label=int(count > 0),
                    extra={"subset": "localization", "points": points, "count": count},
                )
            )
    if n_missing:
        log.warning(f"DeepFish: {n_missing} CSV rows had no matching image/mask file")
    return records


def fetch_deepfish(info: SourceInfo, root: Path, archive: Path | None = None, url: str | None = None) -> Path:
    """Download (or take a local copy of) DeepFish.tar, verify SHA-256, extract. Returns the dir."""
    raw = Path(root) / "raw" / info.name
    out = raw / "extracted"
    if out.exists() and any(out.rglob("Classification")):
        return out
    sha = info.access.get("sha256")
    tar = raw / "DeepFish.tar"
    if archive:
        tar = download(str(archive), tar, sha256=sha)
    else:
        urls = [url] if url else list(info.access.get("urls", []))
        last = None
        for u in urls:
            try:
                log.info(f"downloading {u}")
                tar = download(u, tar, sha256=sha)
                break
            except Exception as e:  # try the next mirror
                last = e
                log.warning(f"failed: {e}")
        else:
            raise RuntimeError(f"all DeepFish mirrors failed: {last}")
    safe_extract(tar, out)
    return out
