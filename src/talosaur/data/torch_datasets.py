"""PyTorch datasets and samplers for pretraining and probing (needs torch + Pillow).

* :class:`PretrainImages` - random-resized crops (I-JEPA uses crop scale 0.3-1.0, no other
  augmentation) returned as uint8 CHW; degradation + normalisation happen later on the GPU.
* :class:`LabeledFrames` - full frames resized to a fixed (H, W) with frame labels and per-patch
  animal coverage (boxes or masks) for probes and evaluation.
* :class:`SyntheticImages` / :func:`synthetic_labeled_frame` - procedural stand-ins for tests/debug.
* :class:`DistributedWeightedSampler` - curation weights, deterministic per epoch, sharded over
  ranks, resumable mid-epoch; :class:`MultiResBatchSampler` - one input size per batch.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from talosaur.data.h5store import open_image
from talosaur.data.labels import boxes_to_coverage, box_centroid_and_size, mask_to_coverage


def _load_pil(path: Path):
    from PIL import Image

    with Image.open(open_image(path)) as im:
        return im.convert("RGB")


def random_resized_crop_params(w: int, h: int, scale, ratio, rng: np.random.Generator) -> tuple[int, int, int, int]:
    """(top, left, crop_h, crop_w) like torchvision's RandomResizedCrop, with an explicit RNG."""
    area = w * h
    log_r = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target = area * rng.uniform(scale[0], scale[1])
        ar = math.exp(rng.uniform(*log_r))
        cw = int(round(math.sqrt(target * ar)))
        ch = int(round(math.sqrt(target / ar)))
        if 0 < cw <= w and 0 < ch <= h:
            return int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1)), ch, cw
    in_ratio = w / h
    if in_ratio < ratio[0]:
        cw, ch = w, int(round(w / ratio[0]))
    elif in_ratio > ratio[1]:
        ch, cw = h, int(round(h * ratio[1]))
    else:
        cw, ch = w, h
    return (h - ch) // 2, (w - cw) // 2, ch, cw


class PretrainImages(Dataset):
    """Images from a curated index. Items may be ``idx`` or ``(idx, (H, W))`` (multi-res batches)."""

    def __init__(self, df, root: str | Path, size=(224, 224), scale=(0.3, 1.0), ratio=(3 / 4, 4 / 3), hflip: float = 0.0, seed: int = 0):
        self.paths = [str(Path(root) / p) for p in df["path"].tolist()]
        self.size = tuple(size)
        self.scale, self.ratio, self.hflip = tuple(scale), tuple(ratio), hflip
        self.seed = seed
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.paths)

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            base = torch.initial_seed() if info is None else info.seed
            self._rng = np.random.default_rng(base % (2**63))
        return self._rng

    def __getitem__(self, item):
        from PIL import Image

        idx, size = item if isinstance(item, tuple) else (item, self.size)
        rng = self._get_rng()
        img = _load_pil(Path(self.paths[idx]))
        w, h = img.size
        top, left, ch, cw = random_resized_crop_params(w, h, self.scale, self.ratio, rng)
        H, W = size
        img = img.resize((W, H), Image.Resampling.BILINEAR, box=(left, top, left + cw, top + ch), reducing_gap=2.0)
        arr = np.array(img, dtype=np.uint8)  # writable copy
        if self.hflip and rng.random() < self.hflip:
            arr = np.ascontiguousarray(arr[:, ::-1])
        return torch.from_numpy(arr).permute(2, 0, 1)


class SyntheticImages(Dataset):
    """Procedural scenes generated on the fly (deterministic per index)."""

    def __init__(self, n: int = 512, size=(224, 224), seed: int = 0):
        self.n, self.size, self.seed = n, tuple(size), seed

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, item):
        from talosaur.data.synthetic import make_scene

        idx, size = item if isinstance(item, tuple) else (item, self.size)
        sc = make_scene(np.random.default_rng((self.seed, idx)), size[0], size[1])
        return torch.from_numpy(sc.image).permute(2, 0, 1).contiguous()


# --------------------------------------------------------------------------- labelled frames


def _extra(v) -> dict:
    if isinstance(v, dict):
        return v
    try:
        return json.loads(v) if isinstance(v, str) and v else {}
    except json.JSONDecodeError:
        return {}


class LabeledFrames(Dataset):
    """Full frames resized to ``size`` plus supervision on the ``size // patch`` grid.

    Returns dict: image uint8 (3, H, W); frame_label (-1/0/1); coverage (h, w) float in [0, 1];
    patch_valid (bool: negatives in ``coverage`` can be trusted - mask, exhaustive boxes, or an
    empty frame); centroid (cx, cy, size) or (-1, -1, -1); slice id; row index.
    """

    SLICES = ("clear", "murky", "dark", "other")

    def __init__(self, df, root: str | Path, size=(224, 224), patch: int = 16):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.size = tuple(size)
        self.grid = (self.size[0] // patch, self.size[1] // patch)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        from PIL import Image

        r = self.df.iloc[i]
        H, W = self.size
        img = _load_pil(self.root / r["path"])
        boxes = [b for b, a in zip(r["boxes"], r["box_is_animal"]) if a]
        crop = r.get("crop") if "crop" in self.df.columns else None
        mask_path = r.get("mask_path")
        if crop is not None and not (isinstance(crop, float) and np.isnan(crop)) and len(crop) == 4:
            # crop-level sample (e.g. FathomNet presence crops): re-express boxes in crop coords
            cx0, cy0, cx1, cy1 = (float(v) for v in crop)
            iw, ih = img.size
            img = img.crop((int(cx0 * iw), int(cy0 * ih), int(cx1 * iw), int(cy1 * ih)))
            cw, ch = cx1 - cx0, cy1 - cy0
            nb = []
            for x0, y0, x1, y1 in boxes:
                b = [(x0 - cx0) / cw, (y0 - cy0) / ch, (x1 - cx0) / cw, (y1 - cy0) / ch]
                b = [min(max(v, 0.0), 1.0) for v in b]
                if b[2] > b[0] and b[3] > b[1]:
                    nb.append(b)
            boxes = nb
            mask_path = None
        img = img.resize((W, H), Image.Resampling.BILINEAR)
        if isinstance(mask_path, str) and mask_path:
            with Image.open(open_image(self.root / mask_path)) as m:
                cov = mask_to_coverage(np.asarray(m.convert("L")) > 0, self.grid)
            valid = True
        else:
            cov = boxes_to_coverage(boxes, self.grid)
            valid = bool(r.get("boxes_exhaustive", False)) or int(r["frame_label"]) == 0
        c = box_centroid_and_size(boxes) if boxes else None
        sl = r.get("slice", "other")
        return {
            "image": torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1),
            "frame_label": torch.tensor(int(r["frame_label"])),
            "coverage": torch.from_numpy(cov),
            "patch_valid": torch.tensor(valid),
            "centroid": torch.tensor(c if c else (-1.0, -1.0, -1.0), dtype=torch.float32),
            "slice": torch.tensor(self.SLICES.index(sl) if sl in self.SLICES else 3),
            "row": torch.tensor(i),
        }


def synthetic_labeled_frames(n: int, size=(224, 224), patch: int = 16, seed: int = 0) -> list[dict]:
    """In-memory labelled frames with exact masks (debug/probe tests)."""
    from talosaur.data.synthetic import CONDITIONS, make_scene

    rng = np.random.default_rng(seed)
    grid = (size[0] // patch, size[1] // patch)
    out = []
    for i in range(n):
        k = int(rng.integers(1, 3)) if rng.random() < 0.6 else 0
        sc = make_scene(rng, size[0], size[1], condition=CONDITIONS[i % 3], n_animals=k)
        cov = mask_to_coverage(sc.mask, grid)
        c = box_centroid_and_size(sc.boxes.tolist()) if len(sc.boxes) else None
        out.append(
            {
                "image": torch.from_numpy(sc.image).permute(2, 0, 1).contiguous(),
                "frame_label": torch.tensor(int(sc.has_animal)),
                "coverage": torch.from_numpy(cov),
                "patch_valid": torch.tensor(True),
                "centroid": torch.tensor(c if c else (-1.0, -1.0, -1.0), dtype=torch.float32),
                "slice": torch.tensor(LabeledFrames.SLICES.index(sc.condition)),
                "row": torch.tensor(i),
            }
        )
    return out


# --------------------------------------------------------------------------- samplers


class DistributedWeightedSampler(Sampler[int]):
    """Weighted sampling with replacement, identical across ranks for a given epoch, then sharded.

    ``set_start(k)`` skips the first k of this rank's samples (exact mid-epoch resume).
    """

    def __init__(self, weights, num_samples: int | None = None, rank: int = 0, world: int = 1, seed: int = 0):
        w = torch.as_tensor(np.asarray(weights, dtype=np.float64))
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be non-negative with positive sum")
        self.weights = w
        self.total = int(num_samples or len(w))
        self.total -= self.total % world
        self.rank, self.world, self.seed = rank, world, seed
        self.epoch = 0
        self.start = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_start(self, start: int) -> None:
        self.start = start

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
        idx = torch.multinomial(self.weights, self.total, replacement=True, generator=g)
        mine = idx[self.rank :: self.world].tolist()
        return iter(mine[self.start :])

    def __len__(self) -> int:
        return self.total // self.world - self.start


class MultiResBatchSampler(Sampler[list]):
    """Wraps a sampler; every batch gets one (H, W) drawn from ``sizes`` (seeded per epoch).

    ``set_start(k)`` skips the first k batches of the epoch (exact mid-epoch resume, including
    the sequence of input sizes)."""

    def __init__(self, sampler, batch_size: int, sizes: list[tuple[int, int]], seed: int = 0, drop_last: bool = True):
        self.sampler, self.batch_size, self.sizes = sampler, batch_size, [tuple(s) for s in sizes]
        self.seed, self.drop_last = seed, drop_last
        self.epoch = 0
        self.skip = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.skip = 0
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
        if hasattr(self.sampler, "set_start"):
            self.sampler.set_start(0)

    def set_start(self, batches: int) -> None:
        self.skip = batches
        if hasattr(self.sampler, "set_start"):
            self.sampler.set_start(batches * self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng((self.seed, self.epoch))
        for _ in range(self.skip):  # keep the size sequence aligned after a resume
            rng.integers(len(self.sizes))
        batch = []
        for i in self.sampler:
            batch.append(i)
            if len(batch) == self.batch_size:
                size = self.sizes[int(rng.integers(len(self.sizes)))]
                yield [(j, size) for j in batch]
                batch = []
        if batch and not self.drop_last:
            size = self.sizes[int(rng.integers(len(self.sizes)))]
            yield [(j, size) for j in batch]

    def __len__(self) -> int:
        n = len(self.sampler)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)
