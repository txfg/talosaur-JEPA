#!/usr/bin/env python
"""Pack each source's stored images and masks into HDF5 (data/h5/<source>/part-NNN.h5).

Records keep their paths; every reader falls back to the pack when a loose file is gone, so
fetching, curation, training, eval and export work the same before and after packing. Re-run
after fetching more: only new or changed files go into a new part.

Examples
--------
  python scripts/data/pack_h5.py --root data --sources deepfish kakadu            # pack, keep loose files
  python scripts/data/pack_h5.py --root data --sources deepfish kakadu --delete   # pack, then delete them
  python scripts/data/pack_h5.py --root data --verify                             # re-check every part's CRCs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from talosaur.data.h5store import PACK_DIR, PACKED_KINDS, pack_source, verify_part
from talosaur.utils.log import get_logger

log = get_logger("data.pack_h5")


def check_records(root: Path, source: str) -> int:
    """Log whether every image and mask in the source's records resolves; returns the number missing."""
    from talosaur.data import h5store
    from talosaur.data.index import read_table

    rec = root / "interim" / source / "records.parquet"
    if not rec.exists():
        log.warning(f"{source}: no {rec}; skipping the record check")
        return 0
    df = read_table(rec)
    paths = list(df["path"]) + [p for p in df["mask_path"] if isinstance(p, str) and p]
    missing = [p for p in paths if not h5store.exists(root / p)]
    loose = len(h5store.loose_files(root, source))
    (log.warning if missing else log.info)(
        f"{source}: {len(df)} records, {len(paths)} files, {len(missing)} missing, {loose} still loose"
    )
    return len(missing)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data", help="data root (default: data)")
    ap.add_argument("--sources", nargs="*", default=None, help="default: every source with stored files")
    ap.add_argument(
        "--delete", action="store_true", help="delete loose files once a verified part holds them"
    )
    ap.add_argument("--workers", type=int, default=16, help="file-reading threads")
    ap.add_argument("--verify", action="store_true", help="only re-check the CRCs of existing parts")
    a = ap.parse_args(argv)
    root = Path(a.root)

    if a.verify:
        bad = 0
        for s in a.sources or sorted(p.name for p in (root / PACK_DIR).iterdir() if p.is_dir()):
            for part in sorted((root / PACK_DIR / s).glob("part-*.h5")):
                log.info(f"{part}: {verify_part(part)} entries OK")
            bad += check_records(root, s)
        return 1 if bad else 0

    sources = a.sources or sorted({p.name for k in PACKED_KINDS for p in (root / k).glob("*") if p.is_dir()})
    for s in sources:
        st = pack_source(root, s, delete=a.delete, workers=a.workers, log=log)
        log.info(
            f"{s}: {st['loose']} loose files, {st['already_packed']} already packed, "
            f"{st['packed']} newly packed ({st['bytes'] / 1e9:.2f} GB), {st['deleted']} deleted"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
