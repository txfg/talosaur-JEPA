"""I-JEPA predictor: a narrow ViT that maps context tokens + positional mask tokens to
predictions of the target encoder's features at the target positions."""

from __future__ import annotations

import torch
import torch.nn as nn

from talosaur.models.pos_embed import sincos_2d
from talosaur.models.vit import Block, init_transformer_weights


def _gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))


class Predictor(nn.Module):
    def __init__(
        self,
        embed_dim: int = 192,
        predictor_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        attn_impl: str = "sdpa",
    ):
        super().__init__()
        self.predictor_dim = predictor_dim
        self.embed = nn.Linear(embed_dim, predictor_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            Block(predictor_dim, num_heads, mlp_ratio, True, dpr[i], attn_impl) for i in range(depth)
        )
        self.norm = nn.LayerNorm(predictor_dim, eps=1e-6)
        self.proj = nn.Linear(predictor_dim, embed_dim)
        self._pos_cache: dict[tuple, torch.Tensor] = {}
        init_transformer_weights(self, self.blocks)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _pos(self, h: int, w: int, device, dtype) -> torch.Tensor:
        key = (h, w, device, dtype)
        if key not in self._pos_cache:
            self._pos_cache[key] = (
                torch.from_numpy(sincos_2d(self.predictor_dim, h, w))
                .to(device=device, dtype=dtype)
                .unsqueeze(0)
            )
        return self._pos_cache[key]

    def forward(
        self,
        ctx: torch.Tensor,
        masks_ctx: list[torch.Tensor],
        masks_tgt: list[torch.Tensor],
        grid_hw: tuple[int, int],
    ) -> torch.Tensor:
        """ctx: (B, Kc, D) context-encoder tokens for the single context mask ``masks_ctx[0]``.

        Returns (n_tgt * B, Kt, D) predictions ordered target-mask-major, i.e. the same order as
        ``apply_masks(target_tokens, masks_tgt)``. (All shipped I-JEPA configs use one context mask.)
        """
        if len(masks_ctx) != 1:
            raise NotImplementedError("one context mask per image (as in the official I-JEPA configs)")
        B = masks_ctx[0].shape[0]
        n_tgt = len(masks_tgt)
        x = self.embed(ctx)
        pos = self._pos(*grid_hw, x.device, x.dtype).expand(B, -1, -1)
        x = x + _gather(pos, masks_ctx[0])
        kc = x.shape[1]
        x = x.repeat(n_tgt, 1, 1)  # [ctx; ctx; ...] one copy per target block
        tgt_pos = torch.cat([_gather(pos, m) for m in masks_tgt], dim=0)
        x = torch.cat([x, tgt_pos + self.mask_token], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, kc:])
        return self.proj(x)
