#!/usr/bin/env python
"""Measure I-JEPA training throughput and peak memory on this GPU (run this first on a new box).

  python scripts/bench_train_throughput.py --model vit_tiny --size 224 --batch-sizes 128 192 256
  python scripts/bench_train_throughput.py --model vit_small --size 224 --batch-sizes 64 96 128 --degrade context_only
  # include JPEG decoding + cropping from disk (needs a curated index):
  python scripts/bench_train_throughput.py --index data/index/underwater_v1.parquet --root data --workers 24

Prints img/s per GPU, peak memory, and the time for 100 epochs over --dataset-size images on
--gpus GPUs. The largest batch that fits is what to put in train.batch_size.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from talosaur.augment.degrade import UnderwaterDegradation, make_views
from talosaur.models.predictor import Predictor
from talosaur.models.vit import build_vit
from talosaur.ssl.ijepa import IJEPA
from talosaur.ssl.masks import MultiBlockMasks

PRED = {"vit_tiny": (128, 6, 4), "vit_small": (192, 6, 6), "vit_base": (384, 6, 6)}


def bench(model_name, size, bs, steps, warmup, degrade_mode, loader_iter, device, grad_ckpt):
    enc = build_vit(model_name, grad_checkpointing=grad_ckpt)
    pd, pdepth, pheads = PRED[model_name]
    model = IJEPA(enc, Predictor(enc.embed_dim, pd, pdepth, pheads)).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    masks = MultiBlockMasks()
    deg = UnderwaterDegradation().to(device) if degrade_mode != "none" else None
    H, W = size
    grid = (H // 16, W // 16)
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    t0 = None
    for i in range(warmup + steps):
        if i == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
        if loader_iter is not None:
            x = next(loader_iter).to(device, non_blocking=True).float() / 255
        else:
            x = torch.rand(bs, 3, H, W, device=device)
        me, mp = masks(bs, grid, torch.Generator().manual_seed(i))
        me = [m.to(device) for m in me]
        mp = [m.to(device) for m in mp]
        xc, xt = make_views(x, degrade_mode, deg)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss, _ = model(xc, xt, me, mp)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        model.momentum_update(0.996)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.perf_counter() - t0
    mem = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else float("nan")
    return bs * steps / dt, mem


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="vit_tiny", choices=list(PRED))
    ap.add_argument("--size", type=int, nargs="+", default=[224], help="H [W]")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 192, 256])
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--degrade", default="none", choices=["none", "shared", "context_only", "independent"])
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--index", default=None, help="benchmark the real data loader too")
    ap.add_argument("--root", default="data")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dataset-size", type=int, default=400_000)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument(
        "--gpus", type=int, default=4, help="for the wall-clock estimate (assumes ~88%% DDP scaling)"
    )
    a = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = (a.size[0], a.size[-1])
    rows = []
    for bs in a.batch_sizes:
        it = None
        if a.index:
            from torch.utils.data import DataLoader

            from talosaur.data.index import read_table
            from talosaur.data.torch_datasets import PretrainImages

            df = read_table(a.index)
            df = df[df["split"] == "train"]
            dl = DataLoader(
                PretrainImages(df, a.root, size),
                batch_size=bs,
                shuffle=True,
                num_workers=a.workers,
                drop_last=True,
                pin_memory=True,
                persistent_workers=True,
            )
            it = iter(dl)
        try:
            ips, mem = bench(a.model, size, bs, a.steps, a.warmup, a.degrade, it, device, a.grad_ckpt)
            hours = a.dataset_size * a.epochs / (ips * a.gpus * (0.88 if a.gpus > 1 else 1.0)) / 3600
            rows.append(
                {
                    "batch": bs,
                    "img_per_s": round(ips, 1),
                    "peak_mem_gb": round(mem, 2),
                    f"hours_{a.epochs}ep_{a.gpus}gpu": round(hours, 1),
                }
            )
        except torch.OutOfMemoryError:
            rows.append({"batch": bs, "img_per_s": None, "peak_mem_gb": "OOM"})
            torch.cuda.empty_cache()
        print(json.dumps(rows[-1]))
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    print(
        json.dumps(
            {
                "gpu": gpu,
                "model": a.model,
                "size": size,
                "degrade": a.degrade,
                "loader": bool(a.index),
                "results": rows,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
