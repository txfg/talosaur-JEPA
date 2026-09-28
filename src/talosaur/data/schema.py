"""The record/index schema shared by every source and by the curation pipeline.

One row per image (a still or a frame extracted from video). Boxes are normalised
``[x0, y0, x1, y1]`` in [0, 1] relative to the stored image. Paths are relative to the data root.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SPLITS = ("train", "val", "test")
FRAME_LABEL_UNKNOWN = -1


@dataclass
class Record:
    image_id: str
    path: str
    source: str
    group_id: str
    license: str
    train_commercial_ok: bool
    width: int
    height: int
    attribution: str = ""
    split_hint: str | None = None  # official split if the source defines one
    frame_label: int = FRAME_LABEL_UNKNOWN  # 1 animal present, 0 absent, -1 unknown
    boxes: list[list[float]] = field(default_factory=list)
    box_labels: list[str] = field(default_factory=list)
    box_is_animal: list[bool] = field(default_factory=list)
    boxes_exhaustive: bool = False  # True only if every animal in the image is boxed
    mask_path: str | None = None  # binary animal mask (same size as image)
    video_id: str | None = None
    t_sec: float | None = None
    url: str | None = None
    labeled: bool = False  # carries any supervision (frame label, boxes or mask)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d["extra"] = _json_dumps(d["extra"])
        return d


def _json_dumps(o: Any) -> str:
    import json

    return json.dumps(o, sort_keys=True, default=str)


# Columns added by the curation pipeline (quality.py / dedup.py / curate.py).
METRIC_COLUMNS = [
    "lum_mean",
    "lum_p95",
    "contrast",
    "norm_contrast",
    "haze",
    "cast_mag",
    "cast_hue",
    "red_ratio",
    "sharpness",
    "noise",
    "uciqe",
    "grad_energy",
    "edge_density",
    "colorfulness",
    "clarity",
]
CURATION_COLUMNS = [
    "phash",
    "dup_cluster",
    "is_empty",
    "light",
    "clarity_bucket",
    "slice",
    "split",
    "weight",
]


def records_to_frame(records: list[Record]):
    import pandas as pd

    cols = list(Record.__dataclass_fields__)
    if not records:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in cols})
    return pd.DataFrame([r.to_row() for r in records], columns=cols)


def write_records(records: list[Record], path) -> None:
    from pathlib import Path

    from talosaur.data.index import write_table

    write_table(records_to_frame(records), Path(path))
