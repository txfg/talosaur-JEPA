#!/usr/bin/env python
"""Fetch / ingest one dataset source into DATA_ROOT (license-gated).

Examples
--------
  python scripts/data/fetch.py synthetic   --root data --accept-license synthetic
  python scripts/data/fetch.py fathomnet   --root data --accept-license fathomnet-tou --max-images 2000
  python scripts/data/fetch.py fathomnet   --root data --accept-license fathomnet-tou --owner-codes MBARI --min-depth 100
  python scripts/data/fetch.py noaa_oer    --root data --accept-license noaa-public-domain --manifest dives.csv
  python scripts/data/fetch.py deepfish    --root data --accept-license CC-BY-4.0
  python scripts/data/fetch.py kakadu      --root data --accept-license CC-BY-4.0
  python scripts/data/fetch.py river_herring --root data --accept-license CDLA-Permissive-1.0 \
        --metadata herring.json --images-dir /path/to/images --max-empty 20000
  python scripts/data/fetch.py brackish    --root data --accept-license CC-BY-SA-4.0 --from-dir ~/Downloads/brackish
  python scripts/data/fetch.py own         --root data --accept-license own --from-dir ~/talosaur_dives
"""

from __future__ import annotations

import argparse
import sys

from talosaur.data.sources import SOURCES, LicenseNotAccepted, run_fetch


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=SOURCES)
    ap.add_argument("--root", default="data", help="data root (default: data)")
    ap.add_argument("--accept-license", default=None, help="license acceptance id printed by a dry run")
    ap.add_argument("--short-side", type=int, default=None, help="resize stored images to this short side")
    # FathomNet
    ap.add_argument("--concepts", nargs="*", default=None)
    ap.add_argument("--owner-codes", nargs="*", default=None)
    ap.add_argument("--min-depth", type=float, default=None)
    ap.add_argument("--max-depth", type=float, default=None)
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-license-lookup", action="store_true", help="FathomNet: skip per-image license calls")
    # NOAA
    ap.add_argument("--manifest", default=None, help="NOAA: CSV with a `url` column")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--keep-video", action="store_true")
    ap.add_argument("--max-videos", type=int, default=None)
    # archives / folders
    ap.add_argument("--from-dir", default=None)
    ap.add_argument("--archive", default=None, help="DeepFish: local DeepFish.tar")
    ap.add_argument("--url", default=None)
    ap.add_argument("--group-by", default="parent", choices=["parent", "file"])
    # LILA camera traps
    ap.add_argument("--metadata", default=None)
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--image-base-url", default=None)
    ap.add_argument("--max-empty", type=int, default=None)
    ap.add_argument("--max-nonempty", type=int, default=None)
    # synthetic
    ap.add_argument("--n-images", type=int, default=120)
    ap.add_argument("--n-videos", type=int, default=2)
    a = ap.parse_args(argv)

    opts: dict = {}
    s = a.source
    if a.short_side:
        opts["short_side"] = a.short_side
    if s == "fathomnet":
        opts.update(concepts=a.concepts, owner_codes=a.owner_codes, min_depth=a.min_depth, max_depth=a.max_depth,
                    max_images=a.max_images, workers=a.workers, resolve_license=not a.no_license_lookup)
    elif s == "noaa_oer":
        if not a.manifest:
            ap.error("noaa_oer needs --manifest")
        opts.update(manifest=a.manifest, fps=a.fps, keep_video=a.keep_video, max_videos=a.max_videos)
    elif s == "deepfish":
        opts.update({k: v for k, v in dict(from_dir=a.from_dir, archive=a.archive, url=a.url).items() if v})
    elif s in ("kakadu", "brackish"):
        opts.update({k: v for k, v in dict(from_dir=a.from_dir, images_dir=a.images_dir).items() if v})
    elif s == "river_herring":
        if not a.metadata:
            ap.error("river_herring needs --metadata (COCO Camera Traps JSON from the LILA page)")
        opts.update(metadata=a.metadata, images_dir=a.images_dir, image_base_url=a.image_base_url,
                    max_empty=a.max_empty, max_nonempty=a.max_nonempty)
    elif s in ("ozfish", "onc", "own"):
        opts.update(from_dir=a.from_dir, fps=a.fps, group_by=a.group_by)
    elif s == "synthetic":
        opts.update(n_images=a.n_images, n_videos=a.n_videos)
    try:
        run_fetch(s, a.root, a.accept_license, **opts)
    except LicenseNotAccepted as e:
        print(f"\n{e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
