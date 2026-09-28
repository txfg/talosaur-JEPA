# Talosaur Vision (JEPA)

Self-supervised vision for **Talosaur**, a low-cost AUV that finds, follows, and films marine animals in dim, murky water.

- **Design and plan:** [`docs/PLAN.md`](docs/PLAN.md)
- **Dataset sources and licenses:** [`docs/DATASETS.md`](docs/DATASETS.md)
- **Raspberry Pi 5 setup, benchmark and onboard runtime:** [`docs/PI5.md`](docs/PI5.md)

| Milestone | Status |
|---|---|
| M0 scaffold (package, CI, synthetic data) | done |
| M1 data pipeline | done (run it on your machine) |
| M2 I-JEPA pretraining | done (run it on your GPUs) |
| M3 probes, heads, and condition slices | done |
| M4 baselines | done |
| M5 ONNX export and int8 | done |
| M6 Pi 5 benchmark and onboard runtime | done (run the benchmark on your Pi) |

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

The onboard runtime needs only `numpy`, `pyyaml`, and `onnxruntime` (or `ncnn`). **Never install torch on the Pi.**

```bash
bash scripts/pi/setup_pi5.sh      # apt picamera2 + ffmpeg, venv with --system-site-packages, pip install -e .[pi]
```

Full instructions are in [`docs/PI5.md`](docs/PI5.md).

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

## M3 + M4: Evaluation and baselines

Frozen-feature probes on the curated index, reported **separately on the dark / murky / clear slices** with bootstrap 95% CIs:

```bash
# edit configs/eval/default.yaml: point the jepa entries at your runs/<name>/encoder_target.pt
python scripts/eval.py --config configs/eval/default.yaml
python scripts/eval.py --config configs/eval/default.yaml --only jepa_tiny_ctx_target dinov2_vits14 --sizes 160x160 112x208
```

**Frame probe** ("animal present"): logistic regression on mean+max pooled tokens.
- Data:
  - DeepFish, River Herring and Brackish frames with frame labels;
  - FathomNet as **presence crops**: a crop is positive if it contains most of an animal box and negative if it touches no box.
- Weight decay is chosen on val.
- Metrics: AUROC, AP, and **TPR at 5% FPR** (the search-mode false-alarm budget), overall, per slice and per source.

**Patch probe** (heatmap): logistic regression on patch tokens.
- Positives are patches with at least 30% box/mask coverage.
- Negatives are used only where they are trustworthy: masks, exhaustive boxes, and empty frames.
- Metrics: patch AUROC/AP and best IoU, plus the steering metrics:
  - **centroid error in degrees** (as seen by the Camera Module 3 Wide, 102°×67°);
  - apparent-size error;
  - peak hit rate;
  - "found" fraction.

**Baselines**, run through the same harness: DINOv2 ViT-S/14, ImageNet ViT-Ti (AugReg), MobileNetV3-Large, and a random ViT-Ti as the lower bound. Every model is evaluated on its own patch grid (/14, /16); the steering metrics don't depend on the grid.

**Underwater-C** (`robustness: true`) applies held-out synthetic degradations at severities 1–5 to test images and reuses the trained heads. It is secondary to the real slices.

**Output** in `reports/eval/<name>/`:
- `report.md` and `results.json`;
- `heads_<backbone>_<HxW>.pt`: the fitted probes, which **are** the deployable frame and heatmap heads used by the M5 export. Standardisation is folded into the weights.

## M5: Export and int8

```bash
python scripts/export.py --encoder runs/ijepa_tiny_224_context_only_s0/encoder_target.pt \
    --heads-dir reports/eval/v1 --heads-backbone jepa_tiny_ctx_target \
    --index data/index/underwater_v1.parquet --root data \
    --sizes 112x112 160x160 224x224 112x208 --out exports/tiny_ctx
```

For each input size this produces (all under `exports/tiny_ctx/`):
- **ONNX fp32**: encoder + frame head + heatmap head in one graph. Input is RGB in [0, 1]; normalisation happens inside the graph. Checked against PyTorch.
- **Static int8**: QDQ, S8S8, per-channel weights. Only MatMul/Gemm/Conv are quantised; LayerNorm, Softmax and GELU stay in float. Calibration uses 300 images **stratified across dark/murky/clear**.
- **Dynamic int8**: the fallback.
- **ncnn fp16**: via PNNX. For this conversion, attention is re-expressed as `nn.MultiheadAttention`, giving identical outputs, so PNNX maps it onto ncnn's fused MultiHeadAttention layer.
- `manifest.json`: tells the Pi runtime the input size, grid and output meanings.
- `parity.md`: frame AUROC, patch AUROC, centroid error, heatmap correlation, file size and latency for torch fp32 vs each exported model. Anything outside the tolerances (|Δ AUROC| ≤ 0.01, Δ centroid ≤ 1°) is marked **NO**.

If static int8 fails parity, try these in order:
1. `--calib-method percentile`;
2. the dynamic model;
3. keep the first/last layers in float (`talosaur.export.quantize.first_and_last_nodes`);
4. deploy ncnn fp16. The Pi's A76 cores do fp16 arithmetic natively.

ncnn int8 needs ncnn's `ncnn2table`/`ncnn2int8` tools; see `docs/PI5.md`.

## M6: Raspberry Pi 5 benchmark and onboard runtime

The Pi side does not need a trained model: speed and memory don't depend on the weights. Start now:

```bash
# training box: full-size random-weight exports for the benchmark
python scripts/export.py --encoder random:vit_tiny --calib-random 32 \
    --sizes 112x112 160x160 224x224 112x208 --out exports/bench_tiny

# Pi: fps (model alone and full loop), latency, peak memory, temperature, throttling
python -m talosaur.onboard.benchmark --export-dir exports/bench_tiny \
    --runtimes ort_fp32 ort_int8 ort_dyn8 ncnn_fp16 --threads 2 3 4 --out reports/pi5/bench_idle
python -m talosaur.onboard.benchmark --export-dir exports/bench_tiny --runtimes ort_int8 ncnn_fp16 \
    --threads 2 3 4 --record-load picamera2 --out reports/pi5/bench_rec        # while recording 1080p30
```

The **onboard loop** (`python -m talosaur.onboard.app --config configs/onboard/pi5.yaml`) runs:
1. The camera's ISP-scaled 208×112 stream feeds the exported model.
2. **Guidance** turns the heatmap into a target: connected blobs with hysteresis, then a sub-patch centroid and apparent size.
3. The target becomes a bearing and elevation through a camera model: flat-port refraction, or an in-water calibration including lens distortion.
4. A Kalman tracker with outlier gating smooths it.
5. A state machine runs SEARCH → ACQUIRE → TRACK → FILM → LOST, plus RELEASE.
6. **One animal at a time.** Each animal is filmed for a time budget (`encounter.max_s`, default 60 s). Then the sub stops recording, backs off, turns away and swims on (RELEASE), and looks for a *different* animal.
   - Animals already filmed are recognised by appearance, not position: the model's patch features over the animal, compared with the ones it remembers.
   - A recognised animal is ignored for a cooldown. An animal that only swam out of view resumes its remaining time.
   - Every encounter is logged to `logs/encounters.jsonl` with its video files.
7. A controller issues normalised yaw-rate / heave / surge requests, with a deadband, rate limits and a hard stand-off.
8. **Recording** follows the state machine, with a pre-roll ring buffer so the approach is kept.
9. Telemetry and commands go out as JSONL and JSON over UDP, for an autopilot bridge. The autopilot choice is still open.

Two tools support it:
- **Toy model** (`python -m talosaur.onboard.toy_model`): a warm-colour detector in the export format, for checking the camera → guidance → recording → UDP chain in the pool before a trained model exists.
- **Replay** (`scripts/replay.py`): runs the identical pipeline on recorded video at the Pi's measured frame rate and writes an annotated video, telemetry and an encounter log. This is how guidance, including the "same animal" similarity threshold, is tuned without the vehicle.

Everything onboard is torch-free, and CI checks this on Python 3.13, the version Raspberry Pi OS Trixie ships. The picamera2 code is tested against a fake camera that follows the picamera2 0.3.37 API. **Nothing has run on a real Pi yet**: `docs/PI5.md` §14 lists what still needs hardware and §15 what to send back.

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
configs/     Hydra configs (+ configs/onboard/pi5.yaml for the vehicle)
scripts/     entry points (data/, train, eval, export, replay; pi/ setup + camera calibration)
tests/       CPU-only tests on synthetic data (onboard/guidance tests also run without torch)
```

## Licensing

No license has been chosen for this repository yet. Third-party material is tracked in `docs/THIRD_PARTY_NOTICES.md`. None of the code is copied from the CC BY-NC `facebookresearch/ijepa` or `facebookresearch/jepa` repositories. Dataset licenses are tracked per image by the data pipeline (see `docs/DATASETS.md`).
