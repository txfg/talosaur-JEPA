"""NOAA Ocean Exploration (Okeanos Explorer) dive video -> training frames.

There is no bulk API: export segment URLs from the NCEI Ocean Exploration Video Portal
(https://www.ncei.noaa.gov/access/ocean-exploration/video/) into a CSV with a ``url`` column
(optional ``dive`` / ``notes`` columns). Local paths also work (for ordered full-resolution
files you already downloaded). Each video is downloaded, frames are extracted (de-interlaced
when needed, overlays cropped, near-duplicates dropped) and the video is deleted unless
``keep_video``.

Only files whose name starts with an Okeanos cruise id (``EX`` + 4 digits) are accepted, because
the portal also hosts non-NOAA footage (e.g. E/V Nautilus) under different terms.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

from talosaur.data.schema import Record
from talosaur.data.sources.base import SourceInfo, download
from talosaur.data.sources.common import make_record
from talosaur.data.video import ExtractConfig, extract_frames
from talosaur.utils.io import save_json
from talosaur.utils.log import get_logger

log = get_logger("data.noaa")

_EX = re.compile(r"^(EX\d{4})", re.IGNORECASE)
_DIVE = re.compile(r"DIVE[_-]?(\d+)", re.IGNORECASE)
_DATE = re.compile(r"(20\d{2})(\d{2})(\d{2})T?")


def parse_name(filename: str) -> dict:
    stem = Path(filename).stem
    m = _EX.match(stem)
    d = _DIVE.search(stem)
    t = _DATE.search(stem)
    return {
        "cruise": m.group(1).upper() if m else None,
        "dive": f"DIVE{int(d.group(1)):02d}" if d else None,
        "date": "".join(t.groups()) if t else None,
        "stem": stem,
    }


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
    rows = [r for r in csv.DictReader(lines) if (r.get("url") or "").strip()]
    if not rows:
        raise ValueError(f"{path}: no rows with a 'url' column")
    return rows


def fetch_noaa_oer(
    info: SourceInfo,
    root: Path,
    manifest: Path,
    fps: float = 1.0,
    short_side: int = 288,
    keep_video: bool = False,
    allow_non_ex: bool = False,
    max_videos: int | None = None,
) -> list[Record]:
    root = Path(root)
    rows = read_manifest(Path(manifest))
    raw = root / "raw" / info.name
    reports: list[dict] = []
    records: list[Record] = []
    cfg = ExtractConfig(fps=fps, short_side=short_side, deinterlace="auto", overlay="auto")
    for i, row in enumerate(rows):
        if max_videos is not None and i >= max_videos:
            break
        url = row["url"].strip()
        fname = unquote(Path(urlparse(url).path).name) if "://" in url else Path(url).name
        meta = parse_name(fname)
        if meta["cruise"] is None and not allow_non_ex:
            log.warning(f"skipping {fname}: not an Okeanos Explorer (EX....) file")
            reports.append({"video": fname, "skipped": "not an EX cruise file"})
            continue
        video_id = meta["stem"]
        out_dir = root / "images" / info.name / video_id
        done_flag = out_dir / "_done.json"
        if done_flag.exists():
            from talosaur.utils.io import load_json

            res = load_json(done_flag)
        else:
            local = download(url, raw / fname)
            res = extract_frames(local, out_dir, video_id, cfg)
            save_json(res, done_flag)
            if not keep_video and local.exists() and "://" in url:
                local.unlink()
        reports.append({"video": fname, **res["stats"], "overlay": res["overlay"]})
        group = f"{info.name}:{meta['cruise'] or 'NA'}:{meta['dive'] or meta['date'] or video_id}"
        for fr in res["frames"]:
            records.append(
                make_record(
                    info,
                    image_id=f"{info.name}:{video_id}:{fr['t_sec']:.2f}",
                    path=f"images/{info.name}/{video_id}/{fr['path']}",
                    group_id=group,
                    width=fr["width"],
                    height=fr["height"],
                    video_id=f"{info.name}:{video_id}",
                    t_sec=fr["t_sec"],
                    url=url if "://" in url else None,
                    extra={"cruise": meta["cruise"], "dive": meta["dive"], "notes": row.get("notes")},
                )
            )
    save_json({"videos": reports}, root / "interim" / info.name / "extraction_report.json")
    return records
