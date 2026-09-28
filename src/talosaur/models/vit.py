"""Vision Transformer encoders (ViT-Tiny/Small/Base with 16-px patches) for JEPA training and
Pi deployment.

* fixed 2-D sin-cos positions, no CLS token (as in I-JEPA) -> any grid size, incl. 16:9;
* ``attn_impl="sdpa"`` for training (PyTorch's memory-efficient kernel works on Turing in fp16),
  ``"math"`` for export (explicit matmul/softmax; ONNX- and PNNX-friendly);
* ``forward(x, masks)`` keeps only the listed patch indices (context encoder in I-JEPA);
* parameter names follow timm's ViT so ImageNet weights can be loaded (``load_timm_weights``).
Written from the ViT/I-JEPA papers; no code copied from CC BY-NC repositories.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from talosaur.models.pos_embed import sincos_2d


@dataclass
class ViTConfig:
    embed_dim: int = 192
    depth: int = 12
    num_heads: int = 3
    patch_size: int = 16
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_path_rate: float = 0.0
    attn_impl: str = "sdpa"  # sdpa | math
    grad_checkpointing: bool = False
    in_chans: int = 3

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    "vit_tiny": ViTConfig(embed_dim=192, depth=12, num_heads=3),
    "vit_small": ViTConfig(embed_dim=384, depth=12, num_heads=6),
    "vit_base": ViTConfig(embed_dim=768, depth=12, num_heads=12),
}


def apply_masks(x: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    """Gather tokens: x (B, N, D), masks list of (B, K) index tensors -> (len(masks) * B, K, D)."""
    out = []
    for m in masks:
        idx = m.unsqueeze(-1).expand(-1, -1, x.size(-1))
        out.append(torch.gather(x, 1, idx))
    return torch.cat(out, dim=0)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        x = self.proj(x)
        h, w = x.shape[-2:]
        return x.flatten(2).transpose(1, 2), (h, w)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, attn_impl: str = "sdpa"):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.attn_impl = attn_impl
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.attn_impl == "sdpa":
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop_path=0.0, attn_impl="sdpa"):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_impl)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def init_transformer_weights(module: nn.Module, blocks: nn.ModuleList) -> None:
    """Truncated-normal(0.02) linears, unit LayerNorms, and per-depth rescaling of the residual
    projections (1/sqrt(2 * layer)) as used by MAE/BEiT-style ViTs."""

    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    module.apply(_init)
    with torch.no_grad():
        for i, blk in enumerate(blocks, start=1):
            blk.attn.proj.weight.div_(math.sqrt(2.0 * i))
            blk.mlp.fc2.weight.div_(math.sqrt(2.0 * i))


class VisionTransformer(nn.Module):
    def __init__(self, cfg: ViTConfig | None = None, **overrides):
        super().__init__()
        cfg = cfg or ViTConfig()
        if overrides:
            cfg = ViTConfig(**{**cfg.to_dict(), **overrides})
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.patch_size = cfg.patch_size
        self.patch_embed = PatchEmbed(cfg.patch_size, cfg.in_chans, cfg.embed_dim)
        dpr = torch.linspace(0, cfg.drop_path_rate, cfg.depth).tolist()
        self.blocks = nn.ModuleList(
            Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, cfg.qkv_bias, dpr[i], cfg.attn_impl)
            for i in range(cfg.depth)
        )
        self.norm = nn.LayerNorm(cfg.embed_dim, eps=1e-6)
        self._pos_cache: dict[tuple, torch.Tensor] = {}
        init_transformer_weights(self, self.blocks)

    def set_attn_impl(self, impl: str) -> None:
        self.cfg.attn_impl = impl
        for b in self.blocks:
            b.attn.attn_impl = impl

    def pos_embed(self, h: int, w: int, device, dtype) -> torch.Tensor:
        key = (h, w, device, dtype)
        pe = self._pos_cache.get(key)
        if pe is None:
            pe = torch.from_numpy(sincos_2d(self.embed_dim, h, w)).to(device=device, dtype=dtype).unsqueeze(0)
            self._pos_cache[key] = pe
        return pe

    def grid(self, height: int, width: int) -> tuple[int, int]:
        return height // self.patch_size, width // self.patch_size

    def forward(self, x: torch.Tensor, masks: list[torch.Tensor] | None = None) -> torch.Tensor:
        """x (B, 3, H, W) -> tokens (B, N, D) [or (len(masks) * B, K, D) with masks]."""
        x, (h, w) = self.patch_embed(x)
        x = x + self.pos_embed(h, w, x.device, x.dtype)
        if masks is not None:
            x = apply_masks(x, masks)
        for blk in self.blocks:
            if self.cfg.grad_checkpointing and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.norm(x)

    def forward_grid(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Tokens reshaped to the patch grid: (B, h, w, D)."""
        h, w = self.grid(x.shape[-2], x.shape[-1])
        t = self.forward(x)
        return t.reshape(x.shape[0], h, w, -1), (h, w)


def build_vit(name: str = "vit_tiny", **overrides) -> VisionTransformer:
    if name not in PRESETS:
        raise ValueError(f"unknown ViT preset {name!r}; choose from {list(PRESETS)}")
    return VisionTransformer(PRESETS[name], **overrides)


def load_timm_weights(model: VisionTransformer, timm_name: str, pretrained: bool = True) -> dict:
    """Initialise from a timm ViT (e.g. ``vit_tiny_patch16_224.augreg_in21k_ft_in1k``).

    CLS token, learned position embeddings and the classifier head are dropped (this encoder
    uses fixed sin-cos positions), everything else maps 1:1. Returns load_state_dict's report.
    """
    import timm

    src = timm.create_model(timm_name, pretrained=pretrained, num_classes=0).state_dict()
    own = model.state_dict()
    mapped = {k: v for k, v in src.items() if k in own and own[k].shape == v.shape}
    report = model.load_state_dict(mapped, strict=False)
    return {
        "loaded": len(mapped),
        "missing": list(report.missing_keys),
        "ignored": sorted(set(src) - set(mapped)),
    }
