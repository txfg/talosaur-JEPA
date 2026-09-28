"""I-JEPA training engine: fp16 AMP (Turing-safe), DDP, per-step schedules, EMA target,
resumable atomic checkpoints, JSONL/TensorBoard/W&B logging, collapse monitors, periodic probes.

Launch through ``scripts/train.py`` (Hydra). Single GPU: ``python scripts/train.py ...``;
4 GPUs: ``torchrun --standalone --nproc_per_node=4 scripts/train.py ...``.
"""

from __future__ import annotations

import math
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from talosaur.augment.degrade import DegradeConfig, UnderwaterDegradation, make_views
from talosaur.data.torch_datasets import (
    DistributedWeightedSampler,
    MultiResBatchSampler,
    PretrainImages,
    SyntheticImages,
)
from talosaur.models.predictor import Predictor
from talosaur.models.vit import build_vit, load_timm_weights
from talosaur.monitor.collapse import collapse_report, is_collapsed
from talosaur.ssl import schedules
from talosaur.ssl.ijepa import IJEPA
from talosaur.ssl.masks import MaskConfig, MultiBlockMasks
from talosaur.utils import dist
from talosaur.utils.io import atomic_torch_save, save_json, save_yaml
from talosaur.utils.log import get_logger
from talosaur.utils.metrics_writer import MetricWriter
from talosaur.utils.seed import derive_seed, seed_everything, worker_init_fn

log = get_logger("train")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_container(cfg) -> dict:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    return dict(cfg)


class _Cfg(dict):
    """dict with attribute access (keeps the engine independent of OmegaConf at runtime)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return _Cfg(v) if isinstance(v, dict) else v


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


class Trainer:
    def __init__(self, cfg):
        self.raw_cfg = _to_container(cfg)
        c = self.cfg = _Cfg(self.raw_cfg)
        self.rank, self.world, self.local_rank = dist.init_distributed()
        self.is_main = self.rank == 0
        cuda = torch.cuda.is_available() and c.train.get("device", "auto") != "cpu"
        self.device = torch.device(f"cuda:{self.local_rank}" if cuda else "cpu")
        self.out = Path(c.out_dir)
        if self.is_main:
            self.out.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        seed_everything(c.seed + self.rank, deterministic=c.train.get("deterministic", False))

        amp = c.train.get("amp", "fp16") if cuda else "none"
        self.amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "none": None}[amp]
        if amp == "bf16" and cuda and torch.cuda.get_device_capability(self.device)[0] < 8:
            log.warning(
                "bf16 requested on a pre-Ampere GPU: it is emulated and slower than fp32. Use amp=fp16."
            )
        self.mean = torch.tensor(c.data.get("mean", IMAGENET_MEAN), device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(c.data.get("std", IMAGENET_STD), device=self.device).view(1, 3, 1, 1)

        self._build_data()
        self._build_model()
        self._build_optim()
        self.writer = MetricWriter(
            self.out,
            tensorboard=c.logging.get("tensorboard", True),
            wandb=c.logging.get("wandb"),
            enabled=self.is_main,
        )
        self.global_step = 0
        self._resume()
        self.probe = self._build_probe() if self.is_main and c.probe.get("enabled", False) else None
        self._monitor_batch = None
        if self.is_main:
            save_yaml(self.raw_cfg, self.out / "config.yaml")
            save_json(
                {
                    "git_sha": _git_sha(),
                    "python": sys.version,
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(self.device) if cuda else "cpu",
                    "world_size": self.world,
                    "platform": platform.platform(),
                    "argv": sys.argv,
                },
                self.out / "env.json",
            )

    # ------------------------------------------------------------------ builders
    def _sizes(self) -> list[tuple[int, int]]:
        t = self.cfg.train
        sizes = t.get("sizes") or [t.get("size", [224, 224])]
        return [tuple(int(v) for v in (s if isinstance(s, (list, tuple)) else (s, s))) for s in sizes]

    def _build_data(self) -> None:
        c = self.cfg
        d = c.data
        sizes = self._sizes()
        if d.kind == "synthetic":
            ds = SyntheticImages(n=int(d.get("n_images", 512)), size=sizes[0], seed=c.seed)
            weights = np.ones(len(ds))
        elif d.kind == "index":
            from talosaur.data.index import read_table

            df = read_table(d.index)
            df = df[df["split"] == "train"]
            if d.get("license_filter") == "commercial":
                df = df[df["train_commercial_ok"].astype(bool)]
            if d.get("sources"):
                df = df[df["source"].isin(list(d.sources))]
            df = df.reset_index(drop=True)
            if len(df) == 0:
                raise RuntimeError(f"no training images in {d.index}")
            ds = PretrainImages(
                df, d.root, sizes[0], tuple(d.get("crop_scale", (0.3, 1.0))), hflip=float(d.get("hflip", 0.0))
            )
            weights = df["weight"].to_numpy() if d.get("weighted", True) else np.ones(len(df))
            weights = np.where(weights > 0, weights, 1e-6)
        else:
            raise ValueError(f"unknown data.kind {d.kind!r}")
        self.dataset = ds
        n = int(d.get("samples_per_epoch") or len(ds))
        self.sampler = DistributedWeightedSampler(weights, n, self.rank, self.world, seed=c.seed)
        bs = int(c.train.batch_size)
        self.batch_sampler = MultiResBatchSampler(self.sampler, bs, sizes, seed=c.seed)
        self.ipe = len(self.batch_sampler)
        if self.ipe == 0:
            raise RuntimeError("fewer samples than one batch per GPU; lower train.batch_size")
        nw = int(c.train.get("num_workers", 8))
        self.loader = DataLoader(
            ds,
            batch_sampler=self.batch_sampler,
            num_workers=nw,
            pin_memory=self.device.type == "cuda",
            persistent_workers=nw > 0,
            prefetch_factor=4 if nw > 0 else None,
            worker_init_fn=worker_init_fn,
        )

    def _build_model(self) -> None:
        c = self.cfg
        m = c.model
        enc = build_vit(
            m.name,
            drop_path_rate=float(m.get("drop_path_rate", 0.0)),
            attn_impl=m.get("attn_impl", "sdpa"),
            grad_checkpointing=bool(m.get("grad_checkpointing", False)),
        )
        init = m.get("init", "scratch")
        if init == "imagenet":
            rep = load_timm_weights(enc, m.get("timm_name", "vit_tiny_patch16_224.augreg_in21k_ft_in1k"))
            log.info(f"ImageNet init: {rep['loaded']} tensors loaded; dropped {len(rep['ignored'])}")
        elif init not in ("scratch", None) and os.path.exists(str(init)):
            sd = torch.load(init, map_location="cpu", weights_only=False)
            sd = sd.get("encoder", sd.get("target_encoder", sd))
            rep = enc.load_state_dict(sd, strict=False)
            log.info(f"init from {init}: missing={rep.missing_keys}")
        p = m.predictor
        pred = Predictor(
            enc.embed_dim,
            int(p.dim),
            int(p.depth),
            int(p.heads),
            float(p.get("mlp_ratio", 4.0)),
            float(p.get("drop_path_rate", 0.0)),
            m.get("attn_impl", "sdpa"),
        )
        model = IJEPA(enc, pred, loss=c.ssl.get("loss", "smooth_l1")).to(self.device)
        self.model = model
        self.ddp = (
            DDP(
                model,
                device_ids=[self.local_rank] if self.device.type == "cuda" else None,
                broadcast_buffers=False,
            )
            if self.world > 1
            else None
        )
        self.masks = MultiBlockMasks(
            MaskConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in c.mask.items()})
        )
        dg = c.degrade
        self.degrade_mode = dg.get("mode", "none")
        self.degrade = None
        if self.degrade_mode != "none":
            dcfg = DegradeConfig(p=float(dg.get("p", 0.8)), severity=tuple(dg.get("severity", (0.2, 1.0))))
            if dg.get("op_p"):
                dcfg.op_p.update(dict(dg.op_p))
            self.degrade = UnderwaterDegradation(dcfg).to(self.device)
        if self.is_main:
            n_enc = sum(p.numel() for p in enc.parameters())
            n_pred = sum(p.numel() for p in pred.parameters())
            log.info(
                f"encoder {m.name}: {n_enc / 1e6:.2f} M params, predictor {n_pred / 1e6:.2f} M; degrade={self.degrade_mode}"
            )

    def _build_optim(self) -> None:
        c = self.cfg
        o = c.optim
        decay, no_decay = [], []
        for mod in (self.model.encoder, self.model.predictor):
            for n, p in mod.named_parameters():
                if not p.requires_grad:
                    continue
                (no_decay if p.ndim < 2 or n.endswith(".bias") or "mask_token" in n else decay).append(p)
        self.opt = torch.optim.AdamW(
            [{"params": decay, "wd_scale": 1.0}, {"params": no_decay, "wd_scale": 0.0, "weight_decay": 0.0}],
            lr=0.0,
            betas=tuple(o.get("betas", (0.9, 0.999))),
            eps=float(o.get("eps", 1e-8)),
        )
        gbs = int(c.train.batch_size) * self.world * int(c.train.get("accum", 1))
        lr = float(o.lr)
        if o.get("scale_lr", True):
            lr = lr * gbs / float(o.get("batch_ref", 256))
        self.peak_lr = lr
        self.start_lr = lr * float(o.get("start_lr_frac", 0.2))
        self.final_lr = float(o.get("final_lr", 1e-6))
        self.total_steps = int(c.train.epochs) * self.ipe
        self.warmup_steps = int(round(float(o.get("warmup_epochs", 10)) * self.ipe))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype == torch.float16)
        if self.is_main:
            log.info(f"global batch {gbs}, peak lr {lr:.2e}, {self.ipe} it/epoch, {self.total_steps} steps")

    def _build_probe(self):
        from talosaur.monitor.probe_hook import ProbeHook

        pc = self.cfg.probe
        size = tuple(pc.get("size", self._sizes()[0]))
        mean, std = self.cfg.data.get("mean", IMAGENET_MEAN), self.cfg.data.get("std", IMAGENET_STD)
        if pc.get("kind", "synthetic") == "synthetic":
            return ProbeHook.synthetic(
                int(pc.get("n_train", 256)), int(pc.get("n_val", 128)), size, mean, std
            )
        return ProbeHook.from_index(
            pc.index,
            pc.get("root", self.cfg.data.get("root")),
            size,
            mean,
            std,
            int(pc.get("n_train", 4000)),
            int(pc.get("n_val", 2000)),
            workers=int(pc.get("workers", 8)),
        )

    # ------------------------------------------------------------------ checkpointing
    def _state(self) -> dict[str, Any]:
        return {
            "encoder": self.model.encoder.state_dict(),
            "target_encoder": self.model.target_encoder.state_dict(),
            "predictor": self.model.predictor.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "config": self.raw_cfg,
            "model_config": {"name": self.cfg.model.name, **self.model.encoder.cfg.to_dict()},
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }

    def save(self, name: str = "latest.pt") -> None:
        if not self.is_main:
            return
        atomic_torch_save(self._state(), self.out / name)

    def export_encoder(self, name: str = "encoder_target.pt") -> Path:
        """Weights-only file for evaluation/export (target encoder, the one I-JEPA evaluates)."""
        path = self.out / name
        if self.is_main:
            atomic_torch_save(
                {
                    "model_config": {"name": self.cfg.model.name, **self.model.encoder.cfg.to_dict()},
                    "target_encoder": self.model.target_encoder.state_dict(),
                    "encoder": self.model.encoder.state_dict(),
                    "global_step": self.global_step,
                    "mean": list(self.cfg.data.get("mean", IMAGENET_MEAN)),
                    "std": list(self.cfg.data.get("std", IMAGENET_STD)),
                    "degrade_mode": self.degrade_mode,
                },
                path,
            )
        return path

    def _resume(self) -> None:
        path = self.out / "latest.pt"
        if not path.exists() or not self.cfg.train.get("resume", True):
            return
        sd = torch.load(path, map_location="cpu", weights_only=False)
        self.model.encoder.load_state_dict(sd["encoder"])
        self.model.target_encoder.load_state_dict(sd["target_encoder"])
        self.model.predictor.load_state_dict(sd["predictor"])
        self.opt.load_state_dict(sd["optimizer"])
        self.scaler.load_state_dict(sd["scaler"])
        self.global_step = int(sd["global_step"])
        r = sd.get("rng", {})
        if self.world == 1 and r:
            random.setstate(r["python"])
            np.random.set_state(r["numpy"])
            torch.set_rng_state(r["torch"])
            if r.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(r["cuda"])
        if self.is_main:
            log.info(f"resumed from {path} at step {self.global_step}")

    # ------------------------------------------------------------------ schedules
    def _apply_schedules(self, step: int) -> tuple[float, float, float]:
        o = self.cfg.optim
        lr = schedules.warmup_cosine(
            step, self.total_steps, self.warmup_steps, self.peak_lr, self.start_lr, self.final_lr
        )
        wd = schedules.weight_decay(
            step, self.total_steps, float(o.get("wd_start", 0.04)), float(o.get("wd_end", 0.4))
        )
        ema = schedules.linear(
            step, self.total_steps, *[float(v) for v in self.cfg.ssl.get("ema", (0.996, 1.0))]
        )
        for g in self.opt.param_groups:
            g["lr"] = lr
            if g.get("wd_scale", 1.0) > 0:
                g["weight_decay"] = wd * g["wd_scale"]
        return lr, wd, ema

    # ------------------------------------------------------------------ monitoring
    @torch.no_grad()
    def _monitor(self, images: torch.Tensor) -> dict[str, float]:
        if self._monitor_batch is None:
            self._monitor_batch = images[: min(len(images), 256)].detach().clone()
        x = self._monitor_batch
        out = {}
        with torch.autocast(
            device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None
        ):
            enc = self.model.encoder
            was = enc.training
            enc.eval()
            out.update(collapse_report(enc(x), prefix="context/"))
            enc.train(was)
            out.update(collapse_report(self.model.target_encoder(x), prefix="target/"))
        return out

    # ------------------------------------------------------------------ training
    def _prep(self, batch: torch.Tensor) -> torch.Tensor:
        x = batch.to(self.device, non_blocking=True).float().div_(255.0)
        return x

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def train(self) -> dict[str, float]:
        c = self.cfg
        t = c.train
        log_every = int(t.get("log_every", 50))
        mon_every = int(t.get("monitor_every", 500))
        ckpt_every = int(t.get("ckpt_every", 2000))
        max_steps = t.get("max_steps")
        accum = int(t.get("accum", 1))
        clip = t.get("grad_clip")
        net = self.ddp or self.model
        start_epoch = self.global_step // self.ipe
        skip = self.global_step % self.ipe
        last: dict[str, float] = {}
        t_last = time.perf_counter()
        seen_since = 0
        data_time = 0.0
        loss_acc, n_acc = torch.zeros((), device=self.device), 0
        nonfinite = torch.zeros((), dtype=torch.int32, device=self.device)
        bad_steps = 0
        stop = False
        for epoch in range(start_epoch, int(t.epochs)):
            self.batch_sampler.set_epoch(epoch)
            if skip:
                self.batch_sampler.set_start(skip)
                skip = 0
            t_data = time.perf_counter()
            for it, batch in enumerate(self.loader):
                data_time += time.perf_counter() - t_data
                step = self.global_step
                lr, wd, ema = self._apply_schedules(step)
                x = self._prep(batch)
                B, _, H, W = x.shape
                grid = (H // self.model.encoder.patch_size, W // self.model.encoder.patch_size)
                gen_m = torch.Generator().manual_seed(derive_seed(c.seed, "masks", step, self.rank))
                masks_enc, masks_pred = self.masks(B, grid, gen_m)
                masks_enc = [m.to(self.device, non_blocking=True) for m in masks_enc]
                masks_pred = [m.to(self.device, non_blocking=True) for m in masks_pred]
                gen_d = None
                if self.degrade is not None:
                    gen_d = torch.Generator(device=self.device).manual_seed(
                        derive_seed(c.seed, "degrade", step, self.rank)
                    )
                x_ctx, x_tgt = make_views(x, self.degrade_mode, self.degrade, gen_d)
                same = x_tgt is x_ctx
                x_ctx = self._normalize(x_ctx)
                x_tgt = x_ctx if same else self._normalize(x_tgt)
                sync = self.ddp is None or (it + 1) % accum == 0
                with nullcontext() if sync else self.ddp.no_sync():
                    with torch.autocast(
                        device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None
                    ):
                        loss, stats = net(x_ctx, x_tgt, masks_enc, masks_pred)
                    # Always backward: under DDP every rank must join the all-reduce. Non-finite
                    # gradients are then skipped identically on all ranks (GradScaler, or the
                    # all-reduced grad-norm check below).
                    self.scaler.scale(loss / accum).backward()
                gnorm = None
                if sync:
                    params = [p for g in self.opt.param_groups for p in g["params"]]
                    if self.scaler.is_enabled():
                        if clip or step % log_every == 0:
                            self.scaler.unscale_(self.opt)
                            gnorm = torch.nn.utils.clip_grad_norm_(
                                params, float(clip) if clip else float("inf")
                            )
                        self.scaler.step(self.opt)  # skips the update if any grad is inf/nan
                        self.scaler.update()
                        self.model.momentum_update(ema)
                    else:
                        gnorm = torch.nn.utils.clip_grad_norm_(params, float(clip) if clip else float("inf"))
                        if torch.isfinite(gnorm):
                            self.opt.step()
                            self.model.momentum_update(ema)
                        else:
                            bad_steps += 1
                            log.warning(f"non-finite gradients at step {step}; update skipped")
                    self.opt.zero_grad(set_to_none=True)
                ld = loss.detach()
                nonfinite = nonfinite + (~torch.isfinite(ld)).int()
                loss_acc = loss_acc + torch.nan_to_num(ld, nan=0.0, posinf=0.0, neginf=0.0)
                n_acc += 1
                seen_since += B * self.world
                self.global_step += 1

                if self.is_main and self.global_step % log_every == 0:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize(self.device)
                    dt = time.perf_counter() - t_last
                    nf = int(nonfinite)
                    bad_steps += nf
                    if bad_steps > 50:
                        raise RuntimeError("too many non-finite losses; lower the lr or check the data")
                    last = {
                        "train/loss": float(loss_acc) / max(1, n_acc - nf),
                        "train/nonfinite_losses": nf,
                        "train/lr": lr,
                        "train/wd": wd,
                        "train/ema": ema,
                        "train/img_per_s": seen_since / dt,
                        "train/data_time_frac": data_time / dt,
                        "train/epoch": epoch + it / self.ipe,
                        "train/amp_scale": float(self.scaler.get_scale())
                        if self.scaler.is_enabled()
                        else 1.0,
                        "train/pred_std": float(stats["pred_std"]),
                        "train/target_std": float(stats["target_std"]),
                        "train/ctx_tokens": float(stats["ctx_tokens"]),
                        "train/input_hw": H * 1000 + W,
                    }
                    if gnorm is not None:
                        last["train/grad_norm"] = float(gnorm)
                    if self.device.type == "cuda":
                        last["train/max_mem_gb"] = torch.cuda.max_memory_allocated(self.device) / 1e9
                    self.writer.log(last, self.global_step)
                    log.info(
                        f"ep {epoch} it {it}/{self.ipe} loss {last['train/loss']:.4f} lr {lr:.2e} "
                        f"{last['train/img_per_s']:.0f} img/s data {100 * last['train/data_time_frac']:.0f}%"
                    )
                    loss_acc, n_acc, seen_since, data_time = torch.zeros((), device=self.device), 0, 0, 0.0
                    nonfinite = torch.zeros((), dtype=torch.int32, device=self.device)
                    t_last = time.perf_counter()

                if self.is_main and self.global_step % mon_every == 0:
                    ms = self._monitor(x_tgt)
                    self.writer.log(ms, self.global_step)
                    if is_collapsed(ms, prefix="target/"):
                        log.warning(f"possible representation collapse at step {self.global_step}: {ms}")
                        if t.get("stop_on_collapse", False):
                            stop = True
                if self.global_step % ckpt_every == 0:
                    self.save()
                if max_steps and self.global_step >= int(max_steps):
                    stop = True
                if stop:
                    break
                t_data = time.perf_counter()

            # end of epoch
            self.save()
            if self.is_main and (epoch + 1) % int(t.get("save_every_epochs", 10)) == 0:
                self.save(f"ckpt_ep{epoch + 1:04d}.pt")
            if self.probe is not None and (
                (epoch + 1) % int(c.probe.get("every_epochs", 5)) == 0 or stop or epoch + 1 == int(t.epochs)
            ):
                pm = self.probe.run(
                    self.model.target_encoder, self.device, self.amp_dtype, prefix="probe/target/"
                )
                if self.degrade_mode == "context_only":
                    pm.update(
                        self.probe.run(
                            self.model.encoder, self.device, self.amp_dtype, prefix="probe/context/"
                        )
                    )
                self.writer.log(pm, self.global_step)
                log.info(
                    "probe "
                    + " ".join(
                        f"{k.removeprefix('probe/')}={v:.3f}"
                        for k, v in pm.items()
                        if isinstance(v, float) and not math.isnan(v)
                    )
                )
                last.update(pm)
            dist.barrier()
            if stop:
                break
        self.export_encoder()
        self.writer.close()
        return last
