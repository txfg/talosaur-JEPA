"""Deployable heads and the fused onboard network.

``TalosaurNet``: float image in [0, 1] (B, 3, H, W) ->
  * ``frame_logit``   (B, K)    - "animal present" (+ optional extra outputs such as "interesting")
  * ``heatmap_logit`` (B, h, w) - per-patch animal-likeness for steering
  * ``embedding``     (B, D)    - mean-pooled features (for onboard novelty detection)
Normalisation happens inside the graph, so the Pi only scales uint8 to [0, 1].
"""

from __future__ import annotations

import torch
import torch.nn as nn


def pool_meanmax(tokens: torch.Tensor) -> torch.Tensor:
    """(B, N, D) -> (B, 2D): mean keeps context, max keeps small/sparse objects visible."""
    return torch.cat([tokens.mean(dim=1), tokens.amax(dim=1)], dim=-1)


class FrameHead(nn.Module):
    def __init__(self, dim: int, n_out: int = 1):
        super().__init__()
        self.linear = nn.Linear(2 * dim, n_out)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.linear(pool_meanmax(tokens))


class HeatmapHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor, grid_hw: tuple[int, int]) -> torch.Tensor:
        B = tokens.shape[0]
        return self.linear(tokens).view(B, grid_hw[0], grid_hw[1])


class TalosaurNet(nn.Module):
    def __init__(self, encoder, frame_head: FrameHead, heatmap_head: HeatmapHead, mean, std, input_hw):
        super().__init__()
        self.encoder = encoder
        self.frame_head = frame_head
        self.heatmap_head = heatmap_head
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.input_hw = tuple(input_hw)
        p = encoder.patch_size
        self.grid_hw = (self.input_hw[0] // p, self.input_hw[1] // p)
        if hasattr(encoder, "freeze_pos_embed"):
            encoder.freeze_pos_embed(*self.grid_hw)

    def forward(self, x: torch.Tensor):
        x = (x - self.mean) / self.std
        tokens = self.encoder(x)
        return self.frame_head(tokens), self.heatmap_head(tokens, self.grid_hw), tokens.mean(dim=1)


def heads_from_probes(
    dim: int, frame_linear: nn.Linear, patch_linear: nn.Linear
) -> tuple[FrameHead, HeatmapHead]:
    """Wrap fitted probe ``nn.Linear`` layers (standardisation already folded in) as heads."""
    fh, hh = FrameHead(dim, frame_linear.out_features), HeatmapHead(dim)
    with torch.no_grad():
        fh.linear.weight.copy_(frame_linear.weight)
        fh.linear.bias.copy_(frame_linear.bias)
        hh.linear.weight.copy_(patch_linear.weight)
        hh.linear.bias.copy_(patch_linear.bias)
    return fh, hh
