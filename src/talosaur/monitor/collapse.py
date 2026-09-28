"""Representation-collapse monitors.

* per-dimension std of L2-normalised embeddings, times sqrt(D): ~1 for isotropic features,
  -> 0 under collapse;
* RankMe effective rank (Garrido et al., 2023): exp(entropy of normalised singular values).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def embedding_std(z: torch.Tensor) -> float:
    zn = F.normalize(z.float(), dim=-1)
    return float(zn.std(dim=0).mean().item() * (z.shape[-1] ** 0.5))


@torch.no_grad()
def rankme(z: torch.Tensor, eps: float = 1e-7) -> float:
    s = torch.linalg.svdvals(z.float())
    p = s / (s.sum() + eps) + eps
    return float(torch.exp(-(p * p.log()).sum()).item())


@torch.no_grad()
def collapse_report(tokens: torch.Tensor, max_patches: int = 4096, prefix: str = "") -> dict[str, float]:
    """tokens (B, N, D) -> stats on pooled and on (subsampled) patch embeddings."""
    B, N, D = tokens.shape
    pooled = tokens.mean(dim=1)
    patches = tokens.reshape(B * N, D)
    if patches.shape[0] > max_patches:
        idx = torch.randperm(patches.shape[0], device=patches.device)[:max_patches]
        patches = patches[idx]
    out = {
        "pooled_std_norm": embedding_std(pooled),
        "patch_std_norm": embedding_std(patches),
        "pooled_rankme": rankme(pooled),
        "patch_rankme": rankme(patches),
    }
    out["pooled_rank_frac"] = out["pooled_rankme"] / min(B, D)
    out["patch_rank_frac"] = out["patch_rankme"] / min(patches.shape[0], D)
    return {f"{prefix}{k}": v for k, v in out.items()}


def is_collapsed(
    stats: dict[str, float], min_rank_frac: float = 0.1, min_std: float = 0.1, prefix: str = ""
) -> bool:
    return (
        stats.get(f"{prefix}patch_rank_frac", 1.0) < min_rank_frac
        or stats.get(f"{prefix}patch_std_norm", 1.0) < min_std
    )
