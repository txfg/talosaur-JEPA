# Talosaur Vision (JEPA)

Self-supervised vision for **Talosaur**, a low-cost AUV that finds, follows, and films marine animals in dim, murky water.

**Status: planning.** The proposed architecture, data sources, milestones, and open decisions are in
[`docs/PLAN.md`](docs/PLAN.md). Dataset and license verification is in [`docs/DATASETS.md`](docs/DATASETS.md).
Code lands milestone by milestone after the plan is approved. Each milestone adds its own "how to run" section here.

## Target
- Pretraining: I-JEPA ViT-Tiny/16 (ViT-Small/16 comparison) on curated underwater imagery, trained on RTX 2080-class GPUs.
- Heads: frame-level "animal present / interesting" classifier plus a per-patch animal-likeness heatmap for steering.
- Onboard: Raspberry Pi 5 (1 GB RAM, CPU only) with Camera Module 3 Wide. Int8 ONNX / ncnn, target ≥ 3 fps at 112–160 px, well under 1 GB RAM.
