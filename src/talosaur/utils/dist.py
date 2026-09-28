"""Minimal torch.distributed helpers for single- or multi-GPU (torchrun) training."""

from __future__ import annotations

import os
from datetime import timedelta


def is_dist() -> bool:
    import torch.distributed as dist

    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    if not is_dist():
        return 0
    import torch.distributed as dist

    return dist.get_rank()


def get_world_size() -> int:
    if not is_dist():
        return 1
    import torch.distributed as dist

    return dist.get_world_size()


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(backend: str | None = None) -> tuple[int, int, int]:
    """Initialise the process group when launched by torchrun. Returns (rank, world, local_rank)."""
    import torch
    import torch.distributed as dist

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) <= 1:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    return rank, world, local_rank


def barrier() -> None:
    if is_dist():
        import torch.distributed as dist

        dist.barrier()


def all_reduce_mean(t):
    """In-place mean over ranks (no-op when not distributed). Returns the tensor."""
    if is_dist():
        import torch.distributed as dist

        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= get_world_size()
    return t


def cleanup() -> None:
    if is_dist():
        import torch.distributed as dist

        dist.destroy_process_group()
