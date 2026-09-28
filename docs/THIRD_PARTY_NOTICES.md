# Third-party notices

This file lists third-party code, models, and data that this repository **includes or adapts**. Libraries used only as dependencies are not listed.

| Component | Source | License | How used |
|---|---|---|---|
| I-JEPA method and hyperparameters | Assran et al., CVPR 2023; `facebookresearch/ijepa` | CC BY-NC 4.0 (code) | **Re-implemented from the paper and the configs.** No code copied. |
| V-JEPA / V-JEPA 2 method | Bardes et al. 2024; Assran et al. 2025; `facebookresearch/vjepa2` | MIT (V-JEPA 2) | Method reference. Any adapted snippet is marked in a comment and listed here. |

Pretrained weights (DINOv2, timm/AugReg ViT-Tiny, torchvision MobileNetV3, V-JEPA 2.1) are downloaded at run time from their original hosts and are **not** redistributed. Their licenses are in `docs/PLAN.md` §4.4–§5.
