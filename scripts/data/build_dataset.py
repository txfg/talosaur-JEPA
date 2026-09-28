#!/usr/bin/env python
"""Curate all fetched sources into one index + dataset report.

  python scripts/data/build_dataset.py --config configs/curate/v1.yaml --root data
  python scripts/data/build_dataset.py --config configs/curate/debug.yaml --root /tmp/tal
"""

from __future__ import annotations

import argparse
from pathlib import Path

from talosaur.data.curate import CurateConfig, run_curation
from talosaur.data.stats import write_report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--root", default=None, help="override `root` from the config")
    ap.add_argument("--name", default=None, help="override the index name")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--report-dir", default=None, help="default: reports/dataset/<name>")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    cfg = CurateConfig.from_yaml(a.config, root=a.root, name=a.name, workers=a.workers)
    df, removed, log = run_curation(cfg)
    out = Path(a.report_dir or f"reports/dataset/{cfg.name}")
    write_report(df, removed, log, Path(cfg.root), out, figures=not a.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
