# Talosaur Vision (JEPA)

Self-supervised vision for **Talosaur**, a low-cost AUV that finds, follows, and films marine animals in dim, murky water.

- **Design and plan:** [`docs/PLAN.md`](docs/PLAN.md)
- **Dataset sources and licenses:** [`docs/DATASETS.md`](docs/DATASETS.md)

| Milestone | Status |
|---|---|
| M0 scaffold (package, CI, synthetic data) | done |
| M1 data pipeline | done (run it on your machine) |
| M2 I-JEPA pretraining | done (run it on your GPUs) |
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

## M1: Build the dataset

Everything below runs on the training machine; this repo never stores images. The data root (`--root`, default `data/`) ends up as:

```
data/raw/<source>/          downloads (safe to delete after ingest)
data/images/<source>/       resized JPEGs (short side 288 px for video frames, 384 px for labelled stills)
data/interim/<source>/      records.parquet + manifest.json (license, attribution, options)
data/index/<name>.parquet   the curated index used by training and evaluation
```

**0. Dry run with synthetic data (2 minutes)**

```bash
python scripts/data/fetch.py synthetic --root data --accept-license synthetic
python scripts/data/build_dataset.py --config configs/curate/debug.yaml --root data
```

**1. Fetch sources.** Every fetcher prints the license and refuses to download until you pass the matching `--accept-license` id. Read [`docs/DATASETS.md`](docs/DATASETS.md) first.

```bash
# FathomNet: images download one URL at a time (max 5 in parallel, as FathomNet's own downloader does).
# Start with a subset, e.g. midwater/deep MBARI imagery:
python scripts/data/fetch.py fathomnet --root data --accept-license fathomnet-tou --owner-codes MBARI --min-depth 50 --max-images 150000
# Re-running the same command resumes; failed images are retried.

# NOAA Okeanos Explorer video: export segment URLs from the NCEI Video Portal into a CSV
# (template: data_sources/examples/noaa_manifest_template.csv). Frames are extracted at 1 fps,
# de-interlaced, overlays cropped, near-duplicates dropped; each video is deleted afterwards.
python scripts/data/fetch.py noaa_oer --root data --accept-license noaa-public-domain --manifest my_dives.csv

python scripts/data/fetch.py deepfish --root data --accept-license CC-BY-4.0            # 7.1 GB tar, SHA-256 checked
python scripts/data/fetch.py kakadu   --root data --accept-license CC-BY-4.0            # via the Zenodo API
python scripts/data/fetch.py brackish --root data --accept-license CC-BY-SA-4.0 --from-dir ~/Downloads/brackish   # from Kaggle
python scripts/data/fetch.py river_herring --root data --accept-license CDLA-Permissive-1.0 \
    --metadata <COCO-camera-traps JSON from the LILA page> --images-dir <downloaded images> --max-empty 20000
python scripts/data/fetch.py own      --root data --accept-license own --from-dir ~/talosaur_dives   # your footage
```

**2. Curate and report.**

```bash
python scripts/data/build_dataset.py --config configs/curate/v1.yaml --root data --workers 100
# -> data/index/underwater_v1.parquet and reports/dataset/underwater_v1/report.md (+ figures/)
```

The report covers:
- per-source, license, and split counts;
- the light × clarity grid;
- evaluation-slice sizes;
- label counts;
- the **fraction of animals smaller than one 16-px patch** at 112/160/224/208×112;
- metric distributions;
- contact sheets of the dark/murky/clear slices and of frames removed as empty water. Check those sheets.

**3. Calibrate dark/murky/clear on your data (about 30 minutes of labelling).**

```bash
python scripts/data/label_frames.py page --index data/index/underwater_v1.parquet --root data --n 300 --out labels.html
# open labels.html; keys 1/2/3 = dark/dim/bright, q/w/e = murky/moderate/clear; click "Download labels"
python scripts/data/label_frames.py calibrate --labels labels.json --index data/index/underwater_v1.parquet \
    --out configs/curate/thresholds.yaml
# set `thresholds_file: configs/curate/thresholds.yaml` in configs/curate/v1.yaml, then rebuild (metrics are cached)
```

What curation does:
- **Near-duplicates.** Perceptual hash plus a standardised thumbnail plus a mean-colour check. A training image that duplicates a val/test image is dropped (no leakage), and within a split the best-labelled copy is kept.
- **Empty open water.** Unlabelled training frames with no structure are downsampled, keeping more of them in murky or dark conditions, where faint animals hide.
- **Splits** are grouped by dive, video, habitat, or deployment.
- **Sampling weights** balance the light × clarity buckets, with a cap on each source's share.

## M2: I-JEPA pretraining

Hardware assumed: **4× RTX 2080 Super (8 GB), 112 cores**. Turing has no bf16 hardware, so the engine uses **fp16 autocast + GradScaler**. All configs live in `configs/` (Hydra); any value can be overridden on the command line.

**0. Smoke test (CPU, about 1 minute, no data needed)**

```bash
python scripts/train.py experiment=debug_cpu
```

**1. Measure your real throughput first.** This replaces the estimates in `docs/PLAN.md` §4.7.

```bash
python scripts/bench_train_throughput.py --model vit_tiny  --size 224 --batch-sizes 128 192 256 --gpus 4
python scripts/bench_train_throughput.py --model vit_small --size 224 --batch-sizes 64 96 128 --gpus 4
python scripts/bench_train_throughput.py --index data/index/underwater_v1.parquet --root data --workers 24   # + data loading
```

Use the largest batch that doesn't OOM as `train.batch_size` (per GPU).

**2. Preview the underwater degradation on your images**

```bash
python scripts/viz_degrade.py --index data/index/underwater_v1.parquet --root data --out degrade.png
```

**3. Train.** Either one run on all 4 GPUs, or the first ablation round with one run per GPU:

```bash
torchrun --standalone --nproc_per_node=4 scripts/train.py experiment=ijepa_tiny_224 degrade=context_only
EXP=ijepa_tiny_224 bash scripts/launch_ablation.sh      # E1 none | E2 context_only | E3 shared | E5 ImageNet init
torchrun --standalone --nproc_per_node=4 scripts/train.py experiment=ijepa_small_224 degrade=<best>   # E6
python scripts/train.py experiment=ijepa_tiny_multires degrade=<best>                              # E4
```

Each run writes these files to `runs/<run_name>/`:
- `metrics.jsonl`, and `tb/` for TensorBoard;
- `latest.pt`, which gives automatic resume: re-running the same command continues;
- `ckpt_epXXXX.pt`;
- `encoder_target.pt`, the file evaluation and export use;
- `config.yaml` and `env.json`, recording the resolved config, git SHA, and versions.

What to watch in TensorBoard (`tensorboard --logdir runs`):

| Metric | Healthy | Worry if |
|---|---|---|
| `train/loss` | falls, then flattens | NaN, or `train/nonfinite_losses` > 0 repeatedly |
| `target/patch_rankme`, `target/patch_std_norm` | stay well above ~10% of the embedding dim / ~0.3 | drop towards 0 (collapse; a warning is logged) |
| `probe/target/frame_auroc` (+ `_dark`, `_murky`, `_clear`) | rises over epochs | flat at ~0.5 |
| `probe/target/patch_auroc`, `probe/target/centroid_err_deg_median` | AUROC up, error (degrees) down | not improving while the loss falls |
| `train/img_per_s`, `train/data_time_frac` | stable; data time < 10% | data time high: raise `train.num_workers` |

Implementation details:
- **Masking follows the official I-JEPA configs**: 4 targets at scale 0.15–0.2 and aspect 0.75–1.5; 1 context block at 0.85–1.0; `min_keep` 10.
- **Masking fixes the reference collator's edge bias.** With the official behaviour at 224 px, 14% of patches (the whole last row and column) are never targets. Context coverage also drops from 79% at the top-left to 2% at the bottom-right. Set `mask.official_compat=true` to reproduce the official behaviour for comparison.
- **Learning rate:** peak lr = `optim.lr` × global batch / 2048. Warmup is 10 epochs. Weight decay rises 0.04 → 0.4 and EMA momentum 0.996 → 1.0, as in I-JEPA.
- **Degradation modes:**
  - `none` is the baseline;
  - `shared` degrades context and target identically (plain augmentation);
  - `context_only` degrades the context and keeps the target clean (the robustness objective). In this mode both the context and target encoders are probed;
  - `independent` degrades them separately.

**Send me back** `metrics.jsonl`, `config.yaml`, and `env.json` from each run, plus the throughput JSON. That's enough for me to tune the next round.

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
