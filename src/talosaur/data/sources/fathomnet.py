"""FathomNet fetcher (fathomnet-py client; https://database.fathomnet.org/api).

* enumerates images by concept list, by filtered query (depth / owner institution), or all;
* downloads each image URL (<= ``workers`` concurrent, exponential backoff), resizes on the fly
  and never keeps the ~3.8 MB PNG originals;
* resolves the per-image license from its image-set upload's Darwin Core record (cached);
* maps concepts to animal / not-animal through FathomNet's WoRMS service (cached);
* is resumable: finished images are appended to ``records.jsonl`` and skipped on restart.

FathomNet annotations are not exhaustive, so images without an animal box get frame_label -1.
Images must never be redistributed; this repo only stores derived training copies locally.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from talosaur.data.imageio import decode_resize_save
from talosaur.data.licenses import normalize_license
from talosaur.data.schema import Record
from talosaur.data.sources.base import USER_AGENT, SourceInfo
from talosaur.data.sources.common import make_record, video_id_from_frame_name
from talosaur.utils.log import get_logger

log = get_logger("data.fathomnet")


def _retry(fn, *args, retries: int = 5, base: float = 2.0, **kwargs):
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # the client raises ValueError on HTTP errors
            msg = str(e)
            client_error = "Client error (4" in msg and "Client error (429" not in msg
            if attempt == retries - 1 or client_error:
                raise
            wait = base ** (attempt + 1)
            log.warning(f"{getattr(fn, '__name__', 'call')} failed ({msg[:120]}); retry in {wait:.0f}s")
            time.sleep(wait)
    return None


class _JsonCache:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.data: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}

    def get(self, k: str, fn):
        with self.lock:
            if k in self.data:
                return self.data[k]
        v = fn()
        with self.lock:
            self.data[k] = v
        return v

    def save(self) -> None:
        with self.lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data))
            tmp.replace(self.path)


def enumerate_images(
    concepts: list[str] | None = None,
    owner_codes: list[str] | None = None,
    min_depth: float | None = None,
    max_depth: float | None = None,
    max_images: int | None = None,
    page_size: int = 500,
    sample_seed: int | None = None,
) -> list[Any]:
    """Return fathomnet ``AImageDTO`` objects (deduplicated by uuid).

    By default the first ``max_images`` in API order, which cluster in time (the first 16k MBARI
    images below 50 m all date from 1989-2006). With ``sample_seed``, every matching image is
    listed and ``max_images`` of them are drawn at random, reproducibly for a given listing.
    """
    import numpy as np
    from fathomnet import dto
    from fathomnet.api import images

    seen: dict[str, Any] = {}
    cap = None if sample_seed is not None else max_images

    def add(batch) -> bool:
        for im in batch or []:
            if im.uuid and im.uuid not in seen and im.url:
                seen[im.uuid] = im
                if cap and len(seen) >= cap:
                    return False
        return True

    if concepts:
        for c in concepts:
            if not add(_retry(images.find_by_concept, c)):
                break
    elif owner_codes or min_depth is not None or max_depth is not None:
        offset = 0
        while True:
            cons = dto.GeoImageConstraints(
                ownerInstitutionCodes=owner_codes,
                minDepth=min_depth,
                maxDepth=max_depth,
                limit=page_size,
                offset=offset,
            )
            batch = _retry(images.find, cons)
            if not batch or not add(batch) or len(batch) < page_size:
                break
            offset += page_size
    else:
        page = 0
        while True:
            batch = _retry(images.find_all, dto.Pageable(size=page_size, number=page))
            if not batch or not add(batch):
                break
            page += 1
    out = list(seen.values())
    if sample_seed is not None and max_images and len(out) > max_images:
        pick = np.random.default_rng(sample_seed).choice(len(out), max_images, replace=False)
        out = [out[i] for i in sorted(pick)]
    return out


def _group_id(im) -> str:
    name = Path(urlparse(im.url).path).name
    vid = video_id_from_frame_name(name)
    if vid != Path(name).stem or name.upper().startswith("EX"):
        return f"fathomnet:{vid}"
    day = (im.timestamp or "")[:10]
    lat = f"{im.latitude:.1f}" if im.latitude is not None else "na"
    lon = f"{im.longitude:.1f}" if im.longitude is not None else "na"
    return f"fathomnet:{im.contributorsEmail or 'na'}:{day}:{lat},{lon}"


def fetch_fathomnet(
    info: SourceInfo,
    root: Path,
    concepts: list[str] | None = None,
    owner_codes: list[str] | None = None,
    min_depth: float | None = None,
    max_depth: float | None = None,
    max_images: int | None = None,
    short_side: int = 384,
    workers: int = 4,
    resolve_license: bool = True,
    sample_seed: int | None = None,
) -> list[Record]:
    import requests
    from fathomnet.api import imagesetuploads, worms

    root = Path(root)
    work = root / "interim" / info.name
    work.mkdir(parents=True, exist_ok=True)
    done_path = work / "records.jsonl"
    done: dict[str, dict] = {}
    if done_path.exists():
        for line in done_path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                done[d["image_id"]] = d
    worms_cache = _JsonCache(work / "worms_animalia.json")
    lic_cache = _JsonCache(work / "image_licenses.json")

    log.info("enumerating FathomNet images ...")
    ims = enumerate_images(concepts, owner_codes, min_depth, max_depth, max_images, sample_seed=sample_seed)
    todo = [im for im in ims if f"fathomnet:{im.uuid}" not in done]
    log.info(f"{len(ims)} images listed, {len(done)} already fetched, {len(todo)} to go")

    def is_animal(concept: str) -> bool | None:
        def lookup():
            try:
                anc = _retry(worms.get_ancestors_names, concept)
                return bool(concept == "Animalia" or (anc and "Animalia" in anc))
            except Exception:
                return None

        return worms_cache.get(concept, lookup)

    def license_of(uuid: str) -> dict:
        def lookup():
            try:
                ups = _retry(imagesetuploads.find_by_image_uuid, uuid) or []
                dc = ups[0].darwinCore if ups else None
                return {
                    "raw": getattr(dc, "license", None),
                    "owner": getattr(dc, "ownerInstitutionCode", None),
                    "rights": getattr(dc, "rightsHolder", None),
                }
            except Exception:
                return {"raw": None, "owner": None, "rights": None}

        return lic_cache.get(uuid, lookup)

    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    write_lock = threading.Lock()

    def work_one(im) -> dict | None:
        out_rel = f"images/{info.name}/{im.uuid}.jpg"
        r = _retry(session.get, im.url, timeout=60)
        r.raise_for_status()
        ow, oh, w, h = decode_resize_save(r.content, root / out_rel, short_side)
        boxes, labels, animal = [], [], []
        for bb in im.boundingBoxes or []:
            if bb.x is None or bb.width is None or getattr(bb, "rejected", False):
                continue
            x0, y0 = max(0.0, bb.x / ow), max(0.0, bb.y / oh)
            x1, y1 = min(1.0, (bb.x + bb.width) / ow), min(1.0, (bb.y + bb.height) / oh)
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append([round(x0, 6), round(y0, 6), round(x1, 6), round(y1, 6)])
            labels.append(bb.concept or "unknown")
            animal.append(bool(is_animal(bb.concept)) if bb.concept else False)
        lic = license_of(im.uuid) if resolve_license else {"raw": None, "owner": None, "rights": None}
        rec = make_record(
            info,
            image_id=f"fathomnet:{im.uuid}",
            path=out_rel,
            group_id=_group_id(im),
            width=w,
            height=h,
            license_id=normalize_license(lic["raw"]),
            frame_label=1 if any(animal) else -1,
            boxes=boxes,
            box_labels=labels,
            box_is_animal=animal,
            boxes_exhaustive=False,
            url=im.url,
            extra={
                "uuid": im.uuid,
                "depth_m": im.depthMeters,
                "lat": im.latitude,
                "lon": im.longitude,
                "imaging_type": im.imagingType,
                "timestamp": im.timestamp,
                "license_raw": lic["raw"],
                "owner": lic["owner"],
                "rights_holder": lic["rights"],
                "orig_size": [ow, oh],
            },
        )
        row = rec.to_row()
        with write_lock, open(done_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 5))) as ex:
        futs = [ex.submit(work_one, im) for im in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                row = fut.result()
                done[row["image_id"]] = row
            except Exception as e:
                failures += 1
                log.warning(f"image failed: {e}")
            if i % 500 == 0:
                log.info(f"{i}/{len(todo)} fetched ({failures} failed)")
                worms_cache.save()
                lic_cache.save()
    worms_cache.save()
    lic_cache.save()
    if failures:
        log.warning(f"{failures} images failed; re-run the same command to retry them")
    return [_row_to_record(r) for r in done.values()]


def _row_to_record(row: dict) -> Record:
    row = dict(row)
    extra = row.get("extra")
    if isinstance(extra, str):
        row["extra"] = json.loads(extra)
    return Record(**row)
