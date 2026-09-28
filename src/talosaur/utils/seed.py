"""Seeding helpers. Every stochastic component derives its seed from a base seed plus a
stable description of *what* it is seeding (see :func:`derive_seed`), so results do not depend
on the order in which components are constructed."""

from __future__ import annotations

import hashlib
import os
import random

import numpy as np


def derive_seed(*parts: object) -> int:
    """Stable 63-bit seed from arbitrary parts, e.g. ``derive_seed(base, "masks", step)``."""
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "little") & ((1 << 63) - 1)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and (if installed) torch.

    ``deterministic=True`` trades speed for bitwise reproducibility on GPU (disables cuDNN
    autotuning and enables deterministic algorithms where available).
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import torch
    except ImportError:  # onboard / data-only installs
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker init: give numpy/random a per-worker seed derived from torch's."""
    import torch

    seed = torch.initial_seed() % (2**32)
    np.random.seed((seed + worker_id) % (2**32))
    random.seed(seed + worker_id)
