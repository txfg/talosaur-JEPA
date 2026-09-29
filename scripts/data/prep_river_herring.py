#!/usr/bin/env python
"""Turn LILA's MIT Sea Grant River Herring metadata into the COCO Camera Traps JSON the ingester needs.

LILA's ``mit_sea_grant_river_herring.json`` only has boxes: a frame without a box carries no
annotation at all, so ``fetch.py river_herring`` would drop every empty frame (~200k of ~262k).
The per-location CSVs next to it say which clips were fully reviewed (``status == completed``;
their ``frames_with_fish`` equals the number of boxed frames). This script writes a copy where

* every unboxed frame of a completed clip gets an explicit ``empty`` annotation (a reviewed
  negative); unboxed frames of clips still in annotation are left out;
* every image gets ``datetime`` (from the clip's file name), ``seq_id`` (the clip) and
  ``time_of_day`` (day / night / dawn), so splits group by location and day.

  python scripts/data/prep_river_herring.py --zip data/raw/river_herring/mit_river_herring.zip \\
      --out data/raw/river_herring/river_herring_cct.json
  python scripts/data/fetch.py river_herring --root data --accept-license CDLA-Permissive-1.0 \\
      --metadata data/raw/river_herring/river_herring_cct.json \\
      --images-dir data/raw/river_herring/extracted/mit_river_herring --max-empty 20000
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

META = "mit_river_herring/mit_sea_grant_river_herring.json"
_DT = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})")


def clip_of(file_name: str) -> str:
    """``Ipswich/541384/images/default/frame_000047.PNG`` -> ``Ipswich/541384``."""
    return "/".join(file_name.split("/")[:2])


def convert(meta: dict, clips: dict[str, dict]) -> tuple[dict, Counter]:
    boxed = {a["image_id"] for a in meta["annotations"]}
    empty_id = max(c["id"] for c in meta["categories"]) + 1
    images, annotations, n = [], list(meta["annotations"]), Counter()
    for im in meta["images"]:
        clip = clips.get(clip_of(im["file_name"]))
        if clip is None:
            n["no_clip_row"] += 1
            continue
        if im["id"] not in boxed:
            if clip["status"] != "completed":
                n["unreviewed_dropped"] += 1
                continue
            annotations.append({"id": f"empty:{im['id']}", "image_id": im["id"], "category_id": empty_id})
            n["empty"] += 1
        else:
            n["boxed"] += 1
        m = _DT.search(clip["name"])
        images.append(
            {
                **im,
                "datetime": f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}" if m else None,
                "seq_id": clip_of(im["file_name"]),
                "time_of_day": clip.get("time_of_day"),
            }
        )
    cats = [*meta["categories"], {"id": empty_id, "name": "empty"}]
    return {"info": meta.get("info"), "categories": cats, "images": images, "annotations": annotations}, n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", required=True, help="mit_river_herring.zip from LILA")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with zipfile.ZipFile(a.zip) as z:
        meta = json.loads(z.read(META))
        clips = {}
        for name in (n for n in z.namelist() if n.endswith(".csv")):
            loc = Path(name).name.split("_")[0]  # Ipswich_12152024.csv -> Ipswich
            for row in csv.DictReader(io.StringIO(z.read(name).decode())):
                clips[f"{loc}/{row['id']}"] = row
    out, n = convert(meta, clips)
    Path(a.out).write_text(json.dumps(out))
    print(
        f"{len(out['images'])} images ({n['boxed']} with boxes, {n['empty']} reviewed empty); "
        f"dropped {n['unreviewed_dropped']} unboxed frames of unfinished clips, {n['no_clip_row']} without a clip row"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
