# Talosaur Vision (JEPA)

Self-supervised vision for **Talosaur**, a low-cost AUV that finds, follows, and films marine animals in dim, murky water.

- **Design and plan:** [`docs/PLAN.md`](docs/PLAN.md)
- **Dataset sources and licenses:** [`docs/DATASETS.md`](docs/DATASETS.md)

| Milestone | Status |
|---|---|
| M0 scaffold (package, CI, synthetic data) | done |
| M1 data pipeline | next |
| M2 I-JEPA pretraining | planned |
| M3 probes, heads, and condition slices | planned |
| M4 baselines | planned |
| M5 ONNX export and int8 | planned |
| M6 Pi 5 benchmark and onboard runtime | planned |

## Setup

### Training / data machine (Linux + RTX 2080 Ti)

```bash
python -m venv .venv && source .venv/bin/activate
# PyTorch 2.14.0. Use the default CUDA 13.0 build if `nvidia-smi` shows driver >= 580.
# Otherwise use the CUDA 12.6 build:
#   pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
pip install torch==2.14.0 torchvision==0.29.0
pip install -e ".[data,fathomnet,train,deploy,dev]"
pytest -q
```

Turing GPUs (RTX 20xx) have no bf16 hardware, so all training uses **fp16 autocast + GradScaler** (see `docs/PLAN.md` §2).

### Raspberry Pi 5 (1 GB)

The onboard runtime needs only `numpy`, `pyyaml`, and `onnxruntime` (or `ncnn`). **Never install torch on the Pi.** Setup instructions arrive with M6.

## Repository layout

```
src/talosaur/
  data/      curation pipeline (sources, frames, dedup, quality, splits, report) + synthetic data
  augment/   underwater degradation (GPU)                        [M2]
  models/    ViT-Ti/S (sin-cos, rectangular grids), predictor, heads, baselines
  ssl/       I-JEPA masks/objective, training engine            [M2]
  monitor/   collapse metrics, in-training probes               [M2]
  eval/      probes, slices, robustness                         [M3]
  export/    ONNX, int8, ncnn                                   [M5]
  guidance/  heatmap → bearing/size → track → commands (numpy)  [M6]
  onboard/   Pi 5 runtime + benchmark (no torch)                [M6]
configs/     Hydra configs
scripts/     entry points
tests/       CPU-only tests on synthetic data
```

## Licensing

No license has been chosen for this repository yet. Third-party material is tracked in `docs/THIRD_PARTY_NOTICES.md`. None of the code is copied from the CC BY-NC `facebookresearch/ijepa` or `facebookresearch/jepa` repositories. Dataset licenses are tracked per image by the data pipeline (see `docs/DATASETS.md`).
