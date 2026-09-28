"""A common "patch-grid features" interface for our JEPA encoders and the baselines.

Every backbone maps a normalised image batch (B, 3, H, W) to tokens on its own grid
(B, h, w, D). Evaluation labels are computed on each backbone's grid, and the steering metrics
(centroid error in degrees, size error) are grid-independent, so /14, /16 and /32 models are
directly comparable.

Baselines (weights are downloaded by timm/torchvision on first use, never redistributed):
  * ``dinov2_vits14``   - DINOv2 ViT-S/14 (Apache-2.0), timm ``vit_small_patch14_dinov2.lvd142m``
  * ``timm``            - e.g. ImageNet ViT-Ti ``vit_tiny_patch16_224.augreg_in21k_ft_in1k`` (AugReg, Apache-2.0; mean/std 0.5)
  * ``mobilenetv3_large`` - torchvision ImageNet ``IMAGENET1K_V2`` (stride-16 map + upsampled stride-32 map)
  * ``random_vit_tiny`` - untrained ViT-Ti (lower bound)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
HALF = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


@dataclass
class BackboneInfo:
    name: str
    patch: int
    dim: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


class Backbone(nn.Module):
    info: BackboneInfo

    def input_size(self, hw: tuple[int, int]) -> tuple[int, int]:
        """Snap a requested (H, W) to a multiple of the stride."""
        p = self.info.patch
        return max(p, round(hw[0] / p) * p), max(p, round(hw[1] / p) * p)

    def grid(self, hw: tuple[int, int]) -> tuple[int, int]:
        return hw[0] // self.info.patch, hw[1] // self.info.patch


class JepaBackbone(Backbone):
    def __init__(
        self,
        path: str | None = None,
        which: str = "target",
        name: str = "jepa",
        encoder=None,
        mean=None,
        std=None,
    ):
        super().__init__()
        from talosaur.models.vit import PRESETS, VisionTransformer, ViTConfig

        if encoder is None:
            sd = torch.load(path, map_location="cpu", weights_only=False)
            mc = dict(sd["model_config"])
            preset = mc.pop("name", "vit_tiny")
            cfg = ViTConfig(
                **{
                    **PRESETS[preset].to_dict(),
                    **{k: v for k, v in mc.items() if k in ViTConfig.__dataclass_fields__},
                }
            )
            cfg.grad_checkpointing = False
            encoder = VisionTransformer(cfg)
            key = "target_encoder" if which == "target" else "encoder"
            encoder.load_state_dict(sd[key])
            mean = tuple(sd.get("mean", IMAGENET[0]))
            std = tuple(sd.get("std", IMAGENET[1]))
        self.encoder = encoder.eval()
        self.info = BackboneInfo(
            name, encoder.patch_size, encoder.embed_dim, tuple(mean or IMAGENET[0]), tuple(std or IMAGENET[1])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t, _ = self.encoder.forward_grid(x)
        return t


class TimmViTBackbone(Backbone):
    def __init__(self, timm_name: str, name: str, mean, std, pretrained: bool = True):
        super().__init__()
        import timm

        self.model = timm.create_model(
            timm_name, pretrained=pretrained, num_classes=0, dynamic_img_size=True
        ).eval()
        patch = self.model.patch_embed.patch_size
        patch = patch[0] if isinstance(patch, (tuple, list)) else patch
        self.info = BackboneInfo(name, int(patch), int(self.model.num_features), tuple(mean), tuple(std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        h, w = H // self.info.patch, W // self.info.patch
        t = self.model.forward_features(x)[:, self.model.num_prefix_tokens :]
        return t.reshape(B, h, w, -1)


class MobileNetV3Backbone(Backbone):
    """Stride-16 feature map concatenated with the upsampled stride-32 (960-ch) map."""

    def __init__(self, name: str = "mobilenetv3_large", pretrained: bool = True):
        super().__init__()
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

        m = mobilenet_v3_large(
            weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        ).eval()
        self.features = m.features
        # find the last block with output stride 16
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            self.idx16 = None
            for i, blk in enumerate(self.features):
                x = blk(x)
                if x.shape[-1] == 224 // 16:
                    self.idx16, c16 = i, x.shape[1]
            c32 = x.shape[1]
        self.info = BackboneInfo(name, 16, c16 + c32, *IMAGENET)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f16 = None
        for i, blk in enumerate(self.features):
            x = blk(x)
            if i == self.idx16:
                f16 = x
        up = F.interpolate(x, size=f16.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f16, up], dim=1).permute(0, 2, 3, 1)


def build_backbone(spec: dict, pretrained: bool = True) -> Backbone:
    kind = spec["kind"]
    name = spec.get("name", kind)
    if kind == "jepa":
        return JepaBackbone(spec["path"], spec.get("which", "target"), name)
    if kind == "dinov2_vits14":
        return TimmViTBackbone(
            spec.get("timm_name", "vit_small_patch14_dinov2.lvd142m"), name, *IMAGENET, pretrained=pretrained
        )
    if kind == "timm":
        mean_std = HALF if "augreg" in spec["timm_name"] else IMAGENET
        return TimmViTBackbone(spec["timm_name"], name, *mean_std, pretrained=pretrained)
    if kind == "mobilenetv3_large":
        return MobileNetV3Backbone(name, pretrained=pretrained)
    if kind == "random_vit_tiny":
        from talosaur.models.vit import build_vit

        torch.manual_seed(int(spec.get("seed", 0)))
        return JepaBackbone(encoder=build_vit("vit_tiny"), name=name, mean=IMAGENET[0], std=IMAGENET[1])
    raise ValueError(f"unknown backbone kind {kind!r}")


def count_params_and_gmacs(backbone: Backbone, hw: tuple[int, int]) -> tuple[float, float]:
    """(M params, GMACs) at input size ``hw`` via torch's FLOP counter (MACs = FLOPs / 2).

    Counted with the *math* attention backend: fused CPU attention kernels have no FLOP formula,
    and the attention matmuls are real work on the Pi."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.utils.flop_counter import FlopCounterMode

    params = sum(p.numel() for p in backbone.parameters()) / 1e6
    x = torch.zeros(1, 3, *hw)
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH), FlopCounterMode(display=False) as fc:
        backbone(x)
    return params, fc.get_total_flops() / 2e9
