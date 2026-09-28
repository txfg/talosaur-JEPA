#!/usr/bin/env python
"""Re-render the dataset report for an existing index (no recomputation).

  python scripts/data/dataset_report.py --root data --name underwater_v1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from talosaur.data.index import read_table
from talosaur.data.stats import write_report
from talosaur.utils.io import load_json


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="data")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default=None, help="default: reports/dataset/<name>")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    idx = Path(a.root) / "index"
    df = read_table(idx / f"{a.name}.parquet")
    removed = read_table(idx / f"{a.name}.removed.parquet")
    log = load_json(idx / f"{a.name}.curation.json")
    write_report(df, removed, log, Path(a.root), Path(a.out or f"reports/dataset/{a.name}"), figures=not a.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
