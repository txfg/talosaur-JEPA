"""I-JEPA objective: predict the EMA target encoder's (layer-normed) features of masked target
blocks from a context block, in representation space (Assran et al., CVPR 2023).

Re-implemented from the paper and the official configs; no code copied from the CC BY-NC
reference repository.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from talosaur.models.predictor import Predictor
from talosaur.models.vit import VisionTransformer, apply_masks


class IJEPA(nn.Module):
    def __init__(self, encoder: VisionTransformer, predictor: Predictor, loss: str = "smooth_l1"):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.loss_name = loss

    def forward(
        self,
        x_ctx: torch.Tensor,
        x_tgt: torch.Tensor,
        masks_enc: list[torch.Tensor],
        masks_pred: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        grid = self.encoder.grid(x_ctx.shape[-2], x_ctx.shape[-1])
        with torch.no_grad():
            self.target_encoder.eval()
            h = self.target_encoder(x_tgt)
            h = F.layer_norm(h, (h.size(-1),))
            h = apply_masks(h, masks_pred)
        z = self.encoder(x_ctx, masks_enc)
        p = self.predictor(z, masks_enc, masks_pred, grid)
        p32, h32 = p.float(), h.float()
        if self.loss_name == "smooth_l1":
            loss = F.smooth_l1_loss(p32, h32)
        elif self.loss_name == "l1":
            loss = F.l1_loss(p32, h32)
        else:
            loss = F.mse_loss(p32, h32)
        with torch.no_grad():
            stats = {
                "pred_std": p32.std(dim=(0, 1)).mean(),
                "target_std": h32.std(dim=(0, 1)).mean(),
                "ctx_tokens": torch.tensor(float(z.shape[1])),
            }
        return loss, stats

    @torch.no_grad()
    def momentum_update(self, m: float) -> None:
        src = [p.detach() for p in self.encoder.parameters()]
        dst = list(self.target_encoder.parameters())
        torch._foreach_mul_(dst, m)
        torch._foreach_add_(dst, src, alpha=1.0 - m)
