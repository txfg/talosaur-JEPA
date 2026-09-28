#!/usr/bin/env python
"""Preview the underwater degradation on your own images (sanity-check realism).

python scripts/viz_degrade.py --index data/index/underwater_v1.parquet --root data --out degrade.png
python scripts/viz_degrade.py --synthetic --out degrade.png --variant eval
python scripts/viz_degrade.py --index ... --ops particles,light --severity 0.8
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from talosaur.augment.degrade import OPS, DegradeConfig, UnderwaterDegradation


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=None)
    ap.add_argument("--root", default="data")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--n", type=int, default=6, help="source images (rows)")
    ap.add_argument("--samples", type=int, default=5, help="degraded samples per image (columns)")
    ap.add_argument("--size", type=int, nargs=2, default=[180, 320])
    ap.add_argument("--variant", default="train", choices=["train", "eval"])
    ap.add_argument("--ops", default=None, help=f"comma list to force, from {','.join(OPS)}")
    ap.add_argument("--severity", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="degrade_preview.png")
    a = ap.parse_args(argv)
    from PIL import Image

    H, W = a.size
    if a.synthetic or not a.index:
        from talosaur.data.synthetic import make_scene

        rng = np.random.default_rng(a.seed)
        imgs = [make_scene(rng, H, W, condition="clear", n_animals=2).image for _ in range(a.n)]
    else:
        from talosaur.data.imageio import load_rgb
        from talosaur.data.index import read_table

        df = read_table(a.index)
        df = df[df["slice"] == "clear"] if (df["slice"] == "clear").sum() >= a.n else df
        paths = df.sample(a.n, random_state=a.seed)["path"].tolist()
        imgs = [np.asarray(Image.fromarray(load_rgb(f"{a.root}/{p}")).resize((W, H))) for p in paths]
    x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255
    deg = UnderwaterDegradation(DegradeConfig(p=1.0, variant=a.variant))
    ops = {o: True for o in a.ops.split(",")} if a.ops else None
    cols = [x]
    for k in range(a.samples):
        sev = torch.full((len(x),), a.severity) if a.severity is not None else None
        cols.append(deg(x, torch.Generator().manual_seed(a.seed * 1000 + k), ops=ops, severity=sev))
    grid = torch.cat([torch.cat(list(c), dim=1) for c in cols], dim=2)  # rows = images, cols = samples
    Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).save(a.out)
    print(f"wrote {a.out} (first column = original)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
