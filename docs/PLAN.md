# Talosaur Vision: Plan and Project Structure

**Status:** proposal, awaiting approval. No pipeline code has been written yet.
**Date:** 2026-09-28 · **Scope:** Raspberry Pi 5 (1 GB, CPU only) + Camera Module 3 Wide onboard; RTX 2080-class GPUs for training. Jetson and later upgrades are out of scope for now.

---

## 0. Summary

This repo will hold one config-driven pipeline:

1. **Curated underwater dataset.** Frames and stills from verified sources are deduplicated, empty open water is downsampled, and every image is tagged with a light/turbidity bucket. Splits are grouped by dive, video, or deployment. Every image carries its license.
2. **I-JEPA pretraining of ViT-Tiny/16**, with ViT-Small/16 for comparison, on the 2080s. An optional underwater-degradation module can be switched on per run for ablations. Optional stages: V-JEPA video pretraining, and distillation from DINOv2.
3. **Frozen-feature evaluation.** A frame-level "animal present" probe and a per-patch heatmap probe, reported separately on **dark / murky / clear** subsets. Baselines: DINOv2 ViT-S/14, ImageNet ViT-Tiny, and MobileNetV3.
4. **Deployment.** Encoder and both heads are exported as one ONNX graph, int8 static-quantized with an underwater calibration set, and benchmarked on the Pi 5 with ONNX Runtime and ncnn. The Pi 5 runtime uses no PyTorch.
5. **Guidance and recording (expansion).** Heatmap → target bearing, elevation, and apparent size → filtered track → yaw/heave/surge commands. A SEARCH → ACQUIRE → TRACK → FILM → LOST state machine decides when to record, so CPU goes to video encoding only when there is something to film.

**What I can and can't do from here.** This sandbox has no GPU, no Pi, and its network policy blocks the dataset hosts (FathomNet, NOAA, Hugging Face, and others). I can build and unit-test every stage on CPU using synthetic data and tiny models. You run the real downloads, training, and Pi benchmarks with the scripts and checklists I provide, then send back the reports they generate so I can tune from them.

---

## 1. What the vehicle needs from vision

| Mode | Vision output | Latency/rate need | Failure that matters most |
|---|---|---|---|
| SEARCH | p(animal / interesting) per frame, novelty score | ≥ 1–3 Hz | false alarms on marine snow/backscatter; missed dim animals |
| ACQUIRE/TRACK | heatmap → centroid (bearing°, elevation°), apparent size, confidence | ≥ 3 Hz, stable | jitter, jumps between targets, lock onto lights/specks |
| FILM | "keep framing" + stand-off from apparent size | ≥ 3 Hz | approaching too close; losing target at edge of wide FoV |

The robustness question in the brief ("does it still work when dark, murky, lit by our own lamps?") is how every stage below is evaluated.

---

## 2. Key design decisions

| Decision | Choice | Reason |
|---|---|---|
| Mixed precision on 2080s | **fp16 autocast + GradScaler**, not bf16 | Turing (sm_75) has fp16 tensor cores but no bf16 tensor cores. The official I-JEPA configs use bf16, so this is a deliberate deviation. |
| Attention kernel | PyTorch SDPA (memory-efficient backend); plain matmul path for ONNX export | FlashAttention-2 needs Ampere or newer. |
| Reference code | **Reimplement I-JEPA** following the paper and the official hyperparameters/collator. Adapt **V-JEPA 2 code (MIT)** with attribution where useful. **Copy no code** from `facebookresearch/ijepa` or `facebookresearch/jepa`. | Both of those repos are **CC BY-NC 4.0** (verified). Copying from them would make this repo non-commercial. The official code is used as a *reference*: a test checks that our mask collator matches the official block-size and keep-count statistics. |
| Position encoding | Fixed 2-D sin-cos, no CLS token (as in I-JEPA) | One trained encoder can run at 112/160/224 px and at non-square grids without retraining. |
| Onboard input shape | Square 112/160/224 (the benchmarks you asked for) **plus 16:9 grids**, e.g. 208×112 → 13×7 patches | The camera is 16:9 wide. A square crop throws away ~44% of the horizontal field of view, while 208×112 costs about the same as 160×160 (0.54 vs 0.59 GMACs for ViT-Ti). |
| Underwater degradation | Batched on the GPU, physically motivated, with modes **off / shared / context_only / independent** | `context_only` (degraded context, clean target) turns augmentation into a robustness objective: predict clean-scene features from a degraded view. It is a clean ablation axis. |
| Splits | Grouped by dive / video / deployment / site, plus cross-source near-duplicate removal | Frames from one dive in both train and test inflate every metric. FathomNet also hosts NOAA imagery, so the same scene can appear in two sources. |
| Config & logging | Hydra + YAML; TensorBoard by default, W&B optional; fixed seeds; resumable atomic checkpoints | Ablations become `-m degrade=off,context_only`. |
| Onboard runtime | numpy + onnxruntime (or ncnn) only; the ISP scales the camera's low-res stream | 1 GB of RAM. PyTorch alone would use a large fraction of it. The ISP does the resize for free. |
| Recording | Recording is a vision-driven decision (state machine), with a pre-roll buffer | The Pi 5 has no hardware H.264 encoder (to be confirmed, §10), so encoding competes with inference for CPU. |

---

## 3. Data

### 3.1 Sources

Full verification details, exact access commands, and license quotes are in [`docs/DATASETS.md`](DATASETS.md). Summary:

> _Pending: source-by-source verification (URLs, access method, license) is in progress and will be filled in here and in `docs/DATASETS.md` before approval is requested._

**License posture (your decision, §11).** The pipeline stores `license`, `source`, `attribution`, and `commercial_ok` for **every image**. Any training set can then be rebuilt with, for example, `curate.license_filter=commercial_ok`. Whether model weights trained on NC data may be used commercially is legally unsettled. If commercial use is a possibility, we keep a "clean" training variant from the start.

### 3.2 Curation pipeline (`talosaur.data`)

1. **Fetch.** One module per source. Each prints the source's license and requires `--accept-license <id>` before downloading. It writes a manifest (URL, checksum, license, attribution, retrieval date).
2. **Frame extraction from video** (ffmpeg).
   - Sample at 1–2 fps, with scene-change boosting.
   - Detect burned-in HUD/text overlays with a temporal median plus an edge-persistence mask, then crop or mask them.
   - Keep `(source, video_id, t)` for grouping and for later V-JEPA clips.
3. **Deduplication**, three levels:
   - (a) temporal: skip a frame if its pHash is ≤ 6 bits from the last kept frame and SSIM > 0.9;
   - (b) global: pHash buckets across all sources;
   - (c) optional embedding pass: DINOv2-S or MobileNetV3 features, FAISS cosine > 0.95.
   - Duplicates of any eval/test image are removed from the pretraining pool.
4. **Quality and condition metrics** (on a 256-px copy):
   - mean and 95th-percentile luminance;
   - RMS contrast;
   - underwater dark-channel haze (G/B channels);
   - colour-cast magnitude and hue (CIELAB a\*/b\*), and red-attenuation ratio;
   - Laplacian-variance blur;
   - noise σ (Immerkær);
   - UCIQE.
5. **Condition buckets.** Light: *dark / dim / bright*. Clarity: *murky / moderate / clear*. Thresholds come from dataset percentiles and are validated against about 300 hand-labelled images (a small labelling CLI is included). The eval slices *dark*, *murky*, and *clear* are defined from these buckets.
6. **Empty open-water detection.** Low gradient energy, low local variance, and no bright blobs mark a frame as empty. Keep a configurable fraction (default 10–15%) as negatives. Once we have a trained encoder, the detector can be refined with a probe instead of the heuristics.
7. **Balancing.** Per-image sampling weights ∝ 1 / bucket frequency (capped), plus per-source caps so one long NOAA dive can't dominate. Stored in the index; used by a weighted sampler.
8. **Grouped splits.** `train / val / test` by group id. Probe/eval test sets are frozen and hashed.
9. **Storage.** Short side resized to 288 px for pretraining and 512 px for eval images with boxes. JPEG q=90 plus a single **Parquet index** (path, source, license, group, split, size, metrics, bucket, weight, labels, boxes).
10. **Dataset report** (`reports/dataset/<name>.md` + JSON + plots):
    - counts per source, license, split, and bucket (3×3 light × clarity grid);
    - dedup and empty-water removal counts;
    - resolution and metric histograms per source;
    - label balance;
    - **box-size distribution in patches at 112/160/224 and 208×112.** This shows directly what fraction of animals are smaller than one 16-px patch at each deploy size, which is the main evidence for choosing 112 vs 160.

### 3.3 Your own pool/lake footage

It goes through the same video path with `source=talosaur_dives, license=own`. Labelling uses CVAT or Label Studio (boxes) or our frame-label CLI. Your lake footage becomes the most important **murky-freshwater test set**, because no public dataset matches it exactly.

---

## 4. Pretraining

### 4.1 I-JEPA for small ViTs (`talosaur.ssl.ijepa`)

- **Encoders.**
  - ViT-Ti/16: dim 192, depth 12, 3 heads, 5.5 M params.
  - ViT-S/16: dim 384, depth 12, 6 heads, 21.7 M params.
  - Both: 2-D sin-cos positions, no CLS token, rectangular grids supported.
- **Predictor.** Narrow ViT: default dim 128, depth 6 for Ti; dim 192, depth 6 for S. Ablate width and depth. (I-JEPA found a narrow predictor helps.)
- **Masking.** The official I-JEPA multi-block recipe:
  - 1 context block, scale 0.85–1.0;
  - 4 target blocks, scale 0.15–0.2, aspect ratio 0.75–1.5;
  - targets removed from the context; `min_keep` of 10.
  - Parameters are re-derived per grid so they work on 10×10 and 13×7 grids. The exact values will be checked against the official `src/masks/multiblock.py`.
- **Target and loss.** EMA target encoder (momentum 0.996 → 1.0), smooth-L1 on layer-normed target features at target positions.
- **Optimisation.** AdamW with cosine LR and warmup; weight decay 0.04 → 0.4 (I-JEPA schedule); LR scaled linearly from the reference batch.
- **Resolution.** Default pretrain at 224 (Ti and S) with probes at 112/160/224/208×112. A Ti variant uses multi-resolution batches (128–224) so small inputs are in-distribution.
- **Init options.** `init=scratch` (default), `init=imagenet` (timm ViT-Ti weights, then continued I-JEPA on underwater data), `init=distilled` (from §4.4). This is the practical "what gets the best Pi model" comparison.
- **Engineering.**
  - fp16 AMP + GradScaler;
  - DDP via `torchrun` on 1–N GPUs;
  - optional gradient checkpointing (ViT-S on 8 GB);
  - atomic `latest.pt` + periodic checkpoints holding model, EMA, predictor, optimizer, scaler, schedules, sampler and RNG state; auto-resume;
  - the run directory saves the resolved config, git SHA, and `pip freeze`.

### 4.2 Underwater degradation (`talosaur.augment.degrade`, flag `degrade=`)

A batched torch module that runs on the GPU after loading. Every op has a probability and a severity range and is seeded. It is loosely based on the revised underwater image-formation model: direct attenuation plus backscatter over a random smooth range field *z(x)*.

1. **Wavelength-dependent attenuation.** `J·exp(−β_c z)`. Presets: *ocean blue*, *coastal green*, *lake green-brown (CDOM)*.
2. **Backscatter veil / haze.** `B∞_c·(1 − exp(−β_B z))`.
3. **Artificial light.** Spotlight cones, inverse-square falloff, vignetting. A "no ambient light" deep mode where only lit regions are visible. Red-light illumination is an option.
4. **Low light.** Exposure 0.02–0.5×, gamma, 8-bit quantisation.
5. **Sensor noise.** Poisson–Gaussian in linear space, chroma noise, optional ISP-denoise smear.
6. **Backscatter specks / marine snow.** Density, size, brightness, defocus, and motion streaks, denser near the light. These are the main false-positive source for the heatmap, so they also appear in evaluation.
7. **Blur.** Defocus, motion, forward-scatter.
8. **Compression.** JPEG/H.264-like blockiness at low bitrate.

**Modes:**
- `off` — baseline.
- `shared` — same degraded image for context and target; plain augmentation.
- `context_only` — degraded context, clean target; the robustness objective.
- `independent` — independent degradations for context and target.

Details:
- Ops skip themselves on inputs that are already severe. For example, low light is not applied to images whose mean luminance is below 0.1.
- For video, the water parameters are consistent across a clip, while the particles move from frame to frame.
- `scripts/viz_degrade.py` writes sample grids so you can sanity-check realism.
- With `context_only`, the context and target encoders differ. **Both are evaluated.**

### 4.3 Optional: V-JEPA video stage (`talosaur.ssl.vjepa`)

- Clips of 16 frames at stride 2–6 from 30 fps source, 128–160 px, pre-extracted to low-res clip shards so decoding doesn't starve the GPU.
- 3-D multi-block tube masking with the V-JEPA defaults (8 short-range + 2 long-range blocks spanning time) and an L1 loss.
- Initialised from the I-JEPA checkpoint (patch embed inflated to tubelets).
- Tubelet 2 is the V-JEPA-2 convention, where an image is fed as a repeated frame. It enables an **optional onboard 2-frame mode**: (previous, current) frames at the same token count as one image. That gives the encoder motion cues at almost no extra Pi cost. Single-frame deployment stays the default.

### 4.4 Optional: distillation into ViT-Ti (`talosaur.ssl.distill`)

- Teachers, in order of how permissive their licence is:
  1. **DINOv2 ViT-S/14 or ViT-B/14** (Apache-2.0; default);
  2. DINOv3 small models (DINOv3 License — terms in §10);
  3. I-JEPA ViT-H/14 (CC BY-NC; research only). There are **no official I-JEPA or V-JEPA weights at Tiny/Small size**, so any JEPA teacher is huge.
- Patch-token distillation: student 224 px/16 and teacher 196 px/14 both give **14×14 grids**, so no interpolation is needed. Loss is cosine plus smooth-L1 through a linear projector, plus a pooled-token term.
- Runs to compare: scratch-JEPA, distill-only, distill → JEPA, and JEPA + λ·distill.
- Teacher features can be cached for fixed crops to save 2080 time.

### 4.5 Collapse monitoring and in-training probes

Every N steps on a fixed batch:
- per-dimension std of L2-normalised embeddings (healthy ≈ 1/√D);
- RankMe effective rank of pooled and patch embeddings;
- predictor-output std;
- target/context cosine;
- grad-norm, AMP scale, and EMA momentum.

Alarms: a warning (or optional stop) if rank drops below 10% of D or std collapses.

Every K epochs: a periodic linear probe on cached probe sets. Frame AUROC, patch AUROC, and centroid error are logged next to the loss, so we can see whether the representation is improving and not just the loss falling.

### 4.6 Experiment grid (initial)

| ID | Model | Res | Degrade | Init | Purpose |
|---|---|---|---|---|---|
| E1 | ViT-Ti/16 | 224 | off | scratch | baseline |
| E2 | ViT-Ti/16 | 224 | context_only | scratch | robustness objective |
| E3 | ViT-Ti/16 | 224 | shared | scratch | augmentation-only control |
| E4 | ViT-Ti/16 | 128–224 multi-res | best of E1–3 | scratch | small-input deployment |
| E5 | ViT-Ti/16 | 224 | best | imagenet | practical init |
| E6 | ViT-S/16 | 224 | best | scratch | size comparison |
| E7* | ViT-Ti/16 | 160 video | best | E-best | V-JEPA stage |
| E8* | ViT-Ti/16 | 224 | best | DINOv2 distill | distillation study |

\* optional stages.

### 4.7 Compute estimates (analytic; **will be replaced by measured throughput in M2**)

Encoder cost:

| Encoder | 112² | 160² | 224² | 208×112 |
|---|---|---|---|---|
| ViT-Ti/16 | 0.28 GMACs | 0.59 | 1.25 | 0.54 |
| ViT-S/16 | 1.08 | 2.25 | 4.57 | 2.04 |

I-JEPA training costs roughly 4.6 GFLOPs per image for Ti@160, 9.5 for Ti@224, and 25 for S@224 (forward + backward, including the EMA target and the 4-target predictor).

For 100 epochs over ~400k images on **one** RTX 2080, assuming 4–10 effective TFLOPS (10–25% of the fp16 tensor peak, typical for small ViTs):

| Run | Estimated time |
|---|---|
| Ti@160 | 5–13 h |
| Ti@224 | 11–26 h |
| S@224 | 28–70 h |

With multiple GPUs, divide by the GPU count.

The data loader has to deliver ~400–2,000 img/s. Hence pre-resized JPEGs, degradation on the GPU, and a throughput benchmark as step 1 of M2. The uncertainty on these numbers is about 2×.

**Storage (estimate).**
- Raw video is processed as a stream (download → extract → delete), but plan **≥ 300–500 GB of scratch space** if you process many NOAA dives.
- The curated pool at ~40 KB/image is ~20–40 GB.
- Checkpoints: ~0.1 GB each (Ti) and ~0.4 GB each (S), including optimizer state.

---

## 5. Evaluation (`talosaur.eval`)

1. **Frame probe (animal / no animal).** Logistic regression on pooled frozen features.
   - Data: DeepFish fish/no-fish frames; Brackish frames with and without annotations; FathomNet **crop-level presence** (crops containing an Animalia box vs. crops with no box overlap).
   - FathomNet annotation may be incomplete, so FathomNet negatives are reported separately.
   - Metrics: AUROC, AP, balanced accuracy, and **TPR at 5% FPR** (search-mode false-alarm budget).
2. **Patch probe (localisation).** Per-patch logistic regression.
   - Labels: boxes → patch masks (positive if ≥ 30% of the patch is covered; configurable), DeepFish segmentation masks, Brackish/OzFish boxes.
   - Metrics: patch AUROC/AP, IoU at the best threshold, plus **steering metrics** in image coordinates: centroid error in degrees (using the camera model), log apparent-size error, and hit-rate (heatmap peak inside a GT box).
   - The steering metrics make models with different grids (DINOv2 /14, MobileNet /16 or /32) directly comparable.
3. **Slices.** Every metric is reported on *all / dark / murky / clear*, with slice sizes and bootstrap 95% CIs. Slices come from §3.2.
4. **Synthetic robustness sweep ("Underwater-C").** Clear test images are degraded at severities 1–5 per degradation type. To reduce circularity with the training augmentation, the test uses held-out parameter ranges and a different particle and noise generator. It is still reported as **secondary** to the real dark/murky slices.
5. **Baselines, through the identical harness.**
   - DINOv2 ViT-S/14 (at 224 and at a grid-matched 196);
   - ImageNet ViT-Ti/16 (timm AugReg);
   - MobileNetV3-Large (torchvision; stride-16 feature map for patch probes).
   - The table also reports **params, GMACs, and measured Pi 5 fps**, because accuracy per Pi millisecond is the real trade-off.
6. **Frozen-probe → deployed-head parity.** The deployed heads *are* these probes (with an optional tiny MLP/3×3 smoothing), so evaluation numbers carry straight over to the Pi graph.

---

## 6. Heads, guidance, and recording (expansion)

**Heads** (trained on the frozen encoder; optionally fine-tuned at the end):
- *Frame head.* Mean+max pooled patch tokens → LayerNorm → Linear → p(animal). An optional second output, p(interesting), is added later from your labelled footage.
- *Heatmap head.* Per-patch LayerNorm → Linear → logits on the h×w grid, plus an optional 3×3 depthwise smoothing.
- *Novelty (unsupervised, optional).* Mahalanobis distance of the current pooled embedding from a running mean/covariance of recent frames. It flags "something new in view" even for animals missing from the training labels.

**Guidance (`talosaur.guidance`, numpy only):**
1. Threshold the heatmap, take connected components, and pick the target component: the one nearest the previous track, otherwise the one with the most mass.
2. Probability-weighted centroid (sub-patch precision) and apparent size (√area fraction).
3. Pixel → angle through a camera model:
   - Camera Module 3 Wide, in-air horizontal FoV ≈ 102° (to confirm);
   - a **flat port narrows this underwater** by refraction (roughly 70° HFOV); a dome port preserves it;
   - an in-water checkerboard calibration script is included.
4. Constant-velocity Kalman filter on (bearing, elevation, log size), with gating and dropout hold.
5. Controller:
   - yaw rate ∝ bearing error; heave/pitch ∝ elevation error;
   - surge from apparent size toward a stand-off target (default: the animal fills 20–25% of frame width);
   - rate limits and a hard minimum stand-off.
6. State machine: **SEARCH → ACQUIRE** (N of M frames above threshold) **→ TRACK → FILM → LOST** (hold/turn toward last bearing for T s) **→ SEARCH**. Vehicle-level safety stays with the autopilot.
7. Backends: JSONL log (default), JSON over UDP, and MAVLink (ArduSub-compatible) if that is your stack (§11).

**Recording.** The main stream is H.264-encoded only in TRACK/FILM, with a pre-roll circular buffer so the approach is captured. The low-res stream always feeds the model. The benchmark measures model fps *while recording*, which tells us the real sustainable rate.

**Replay tool.** `scripts/replay.py video.mp4` runs the full onboard loop on recorded footage on your desktop or on the Pi. It writes an annotated video (heatmap, centroid, track, state, commands) plus a command log. This is how guidance gets tuned without the vehicle.

---

## 7. Deployment on the Pi 5 (1 GB)

- **Export.** Encoder + frame head + heatmap head as **one ONNX graph** per fixed input size (112², 160², 224², 208×112; 16:9 variants on request). Opset ≥ 17, positional embeddings baked in, export-safe attention. Checked with the ORT transformer optimizer.
- **Int8 static quantisation** (onnxruntime).
  - QDQ format, per-channel weights, MatMul/Gemm/Conv quantised; LayerNorm, Softmax, and GELU stay in float at first.
  - Calibration: ~300–500 train-split images **stratified across dark/murky/clear**. Dark images have much smaller activations; leaving them out would mis-set ranges.
  - Compare MinMax, Percentile, and Entropy calibration, and S8S8 vs U8S8.
  - Parity report (fp32 torch vs fp32 ORT vs int8): Δ frame AUROC, Δ patch AUROC, Δ centroid error, heatmap correlation.
  - Fallbacks if int8 hurts: exclude sensitive layers (patch-embed, final heads), dynamic quantisation, then QAT.
- **ncnn.** PyTorch → PNNX → ncnn, with fp16 storage and arithmetic (Cortex-A76 supports ARMv8.2 FP16 — to confirm) and int8 where ncnn supports the ops (to confirm, §10).
- **Pi benchmark** (`python -m talosaur.onboard.benchmark`).
  - Sweeps runtime (ORT fp32/int8, ncnn fp16/int8) × size (112/160/224/208×112) × threads (1–4).
  - Reports warm-up-excluded latency (mean/p50/p90/p99), fps, **peak process RSS and system available memory**, CPU temperature, frequency, and `vcgencmd get_throttled` flags.
  - Includes a **10-minute sustained run** (thermal throttling inside a sealed hull is a real risk) and optional concurrent camera capture and H.264 recording load.
  - Outputs Markdown + CSV.
- **Back-of-envelope Pi 5 estimate** (unmeasured; assumes 15–30 effective GFLOPS fp32 with 4 threads):

  | Model / size | Estimated fps |
  |---|---|
  | ViT-Ti @160 | ~12–25 |
  | ViT-Ti @224 | ~6–12 |
  | ViT-S @224 | ~1.5–3 |

  The 3 fps target at 112–160 looks comfortable, which leaves CPU for encoding. The benchmark in M6 replaces these guesses.
- **Memory budget** (target: whole system < 600 MB, process peak < 350 MB):
  - Raspberry Pi OS Lite 64-bit, no desktop;
  - ViT-Ti int8 ≈ 6 MB of weights;
  - ORT session plus Python and numpy ≈ 100–150 MB;
  - camera buffers plus encoder ≈ 100–200 MB;
  - zram swap enabled as a safety net;
  - picamera2 comes from apt (`python3-picamera2`) into a `--system-site-packages` venv.

---

## 8. Project structure

```
talosaur-JEPA/
├── README.md                     # how to run every stage
├── pyproject.toml                # package `talosaur`; extras: train, data, eval, deploy, pi, dev
├── configs/                      # Hydra
│   ├── pretrain.yaml             # root: defaults list
│   ├── model/        vit_tiny.yaml, vit_small.yaml
│   ├── ssl/          ijepa.yaml, vjepa.yaml, distill.yaml
│   ├── data/         underwater_v1.yaml, video_v1.yaml, synthetic.yaml
│   ├── degrade/      off.yaml, shared.yaml, context_only.yaml, independent.yaml
│   ├── experiment/   debug_cpu.yaml, ijepa_tiny_224.yaml, ijepa_small_224.yaml, ...
│   ├── curate/       v1.yaml     # dedup / empty-water / balancing thresholds
│   ├── eval/         probes.yaml, baselines.yaml, robustness.yaml
│   └── export/       onnx.yaml, quant.yaml
├── data_sources/                 # one YAML per source: URL, access method, license, attribution, verified-on
├── src/talosaur/
│   ├── data/          sources/{fathomnet,noaa_oer,deepfish,ozfish,brackish,fish4knowledge,local}.py
│   │                  video.py quality.py dedup.py curate.py index.py stats.py datasets.py labels.py synthetic.py
│   ├── augment/       degrade.py transforms.py
│   ├── models/        vit.py predictor.py heads.py baselines.py
│   ├── ssl/           masks.py ijepa.py vjepa.py distill.py engine.py
│   ├── monitor/       collapse.py probe_hook.py
│   ├── eval/          features.py probes.py slices.py robustness.py report.py
│   ├── export/        onnx_export.py quantize.py ncnn_convert.py parity.py
│   ├── guidance/      heatmap.py camera_model.py tracker.py controller.py state_machine.py backends.py   # numpy only
│   └── onboard/       runtime.py camera.py app.py benchmark.py                                           # no torch
├── scripts/
│   ├── data/          fetch_<source>.py, build_dataset.py, dataset_report.py, label_frames.py
│   ├── train.py  eval.py  export.py  quantize.py  replay.py  viz_degrade.py  bench_train_throughput.py
│   └── pi/            setup_pi5.sh, calibrate_camera.py
├── tests/                        # pytest, CPU-only, synthetic data
├── reports/                      # generated dataset / eval / benchmark reports (small, committed)
├── docs/                         # PLAN.md, DATASETS.md, COMPUTE.md, PI5.md, RESULTS.md, THIRD_PARTY_NOTICES.md
└── .github/workflows/ci.yml      # ruff + pytest (CPU) on every push
```

`talosaur.guidance` and `talosaur.onboard` must import **without torch**, and a test enforces this.

---

## 9. Milestones

Each milestone ends with green CI, a README section, and a short "run this, send me that" checklist.

| # | Milestone | Built and tested here (CPU, synthetic) | You run | Done when |
|---|---|---|---|---|
| M0 | Scaffold | package, Hydra configs, CI, synthetic data generator, lint/test | `pip install -e .[train,dev]`, `pytest` | CI green; `train.py experiment=debug_cpu` runs 20 steps |
| M1 | Data pipeline | fetchers (license-gated), frame extraction, dedup, metrics, buckets, empty-water, balancing, grouped splits, Parquet index, report | fetch small samples → full build | `reports/dataset/*.md` reviewed; thresholds calibrated on ~300 labels |
| M2 | I-JEPA pretraining | ViT, predictor, masks, EMA, fp16 AMP, DDP, resume, degradation module + modes, collapse monitors, probe hook, throughput bench | throughput bench, then E1–E3 (+E6) | loss ↓, no collapse, probe AUROC ↑; measured img/s replaces §4.7 estimates |
| M3 | Probes & slices | feature cache, frame/patch probes, box/mask → patch labels, slices + CIs, steering metrics, Underwater-C, report | `eval.py` on checkpoints | results table per slice in `docs/RESULTS.md` |
| M4 | Baselines | DINOv2-S/14, ImageNet ViT-Ti, MobileNetV3 adapters → same harness; params/GMACs table | `eval.py eval=baselines` | comparison table incl. slices |
| M5 | Export & int8 | fused ONNX (all sizes), ORT optimisation, static int8 + stratified calibration, ncnn conversion, parity report | `export.py`, `quantize.py` | Δ metrics within tolerance (e.g. ΔAUROC < 0.01, Δcentroid < 1°) or documented fallback |
| M6 | Pi 5 benchmark + onboard runtime | benchmark script, onboard app (camera → model → guidance → record), replay tool, setup script; tested on x86 with a file source | `setup_pi5.sh`, benchmark, replay | fps/RSS/thermal table at 112/160/224/208×112; ≥ 3 fps at ≤ 160 with recording on; < 1 GB total |
| M7* | V-JEPA stage | clip extraction, tube masks, video training, 2-frame export option | E7 | Δ probes vs I-JEPA, esp. murky/dark |
| M8* | Distillation | teacher wrappers, grid-matched patch distill, combined objectives | E8 | scratch vs distilled comparison |

\* optional. M3 and M4 can overlap with M2 training runs.

---

## 10. Uncertainties

> _Pending: verification of reference-code licenses/weights, Pi 5 hardware facts, and dataset access is in progress._

---

## 11. Decisions needed from you (defaults in bold)

1. **GPUs.** How many, and which variant? A 2080 or 2080 Super has 8 GB; a 2080 Ti has 11 GB. How many CPU cores, how much RAM, and how much free SSD on that box? *Default: **1× 8 GB, ≥ 8 cores, ≥ 32 GB RAM, ≥ 500 GB**; the code scales to N GPUs with DDP.*
2. **License posture.** Research / non-commercial now, or must the model stay commercially usable? *Default: **research use allowed; every image is license-tracked so a commercial-clean model can be rebuilt**; never copy CC BY-NC code.*
3. **Autopilot interface.** ArduSub/MAVLink, a custom MCU over UART, or undecided? *Default: **abstract interface + JSONL/UDP backends**; MAVLink backend if you say so.*
4. **Lights.** White LEDs, red, or none yet? *Default: **augmentation covers both**.*
5. **Recording.** Pi camera recording (costs Pi CPU), or a separate action cam? *Default: **Pi records 720p only in TRACK/FILM with pre-roll**; the benchmark decides.*
6. **Housing port.** Flat or dome? *Default: **flat-port model + in-water calibration**.*
7. **Existing footage.** Do you have any pool/lake video yet? It becomes the murky-freshwater test set.
8. **Approval scope.** *Default: **build M0–M6, then check in before the optional M7–M8**.*
