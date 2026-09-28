# Raspberry Pi 5 onboard runtime (M6)

How to set up the Pi 5 (1 GB), benchmark the exported models on it, run the onboard vision loop
(camera → model → guidance → commands + recording, one animal at a time), and tune guidance on
recorded video.

Everything here was built and tested on x86. The picamera2 calls are tested against a stand-in
that follows the picamera2 0.3.37 API. **Nothing has run on a real Pi yet.** §14 lists what
still needs checking on hardware.

```mermaid
flowchart LR
  cam["Camera Module 3 Wide<br/>(IMX708)"] --> isp["Pi 5 ISP"]
  isp -- "lores 208×112 YUV420" --> rgb["YUV→RGB<br/>preprocess"]
  rgb --> model["ViT-Ti + heads<br/>ORT int8 / ncnn fp16"]
  model -- "frame logit, heatmap, patch tokens" --> guid["guidance<br/>target → bearing → Kalman → FSM → controller<br/>+ search planner"]
  guid -- "JSONL / UDP JSON" --> ap["autopilot bridge<br/>(your choice, pending)"]
  ap -- "depth, heading (UDP JSON)" --> guid
  guid -- "encounter events" --> rec["H.264 recorder<br/>(software, continuous segments)"]
  isp -- "main 1280×720 YUV420" --> rec
```

## 1. Before you have a trained model

You can do the whole Pi side now, while the GPUs train:

```bash
# On the training box: full-size ViT-Ti / ViT-S with random weights, all runtimes (timing and memory
# do not depend on the weights; --calib-random calibrates int8 on noise, so accuracy is meaningless)
python scripts/export.py --encoder random:vit_tiny --calib-random 32 \
    --sizes 112x112 160x160 224x224 112x208 --out exports/bench_tiny
python scripts/export.py --encoder random:vit_small --calib-random 32 --name vit_small \
    --sizes 112x208 160x160 --out exports/bench_small
```

The **toy model** is a colour-contrast "detector": it scores warm colours per 16×16 patch. It
exercises the camera → guidance → recording → UDP chain end to end. In the pool, move an orange
object in front of the camera and watch the state machine and commands respond.

```bash
pip install onnx                                   # only needed to build it
python -m talosaur.onboard.toy_model --out exports/toy --sizes 112x208
python -m talosaur.onboard.app --export-dir exports/toy --model toy_112x208 --runtime ort_fp32
```

## 2. Set up the Pi

1. Flash **Raspberry Pi OS Lite (64-bit)**, the current Debian Trixie release with Python 3.13. Use Lite: no desktop on a 1 GB board.
2. Connect the Camera Module 3 Wide and check it: `rpicam-hello --list-cameras` should list an `imx708_wide`.
3. Clone this repo on the Pi and run the setup script:

   ```bash
   git clone <your repo url> talosaur-JEPA && cd talosaur-JEPA
   bash scripts/pi/setup_pi5.sh            # add --ncnn to also install the ncnn runtime
   source ~/talosaur-venv/bin/activate
   ```

   The script installs `python3-picamera2` and `ffmpeg` from apt, and makes a venv with
   `--system-site-packages` so picamera2 (apt-only) is visible. It then installs `.[pi]`
   (numpy, pyyaml, onnxruntime) and **no torch**, and proves the onboard modules import with
   torch blocked.
4. Copy the export folder from the training box:
   `rsync -av exports/tiny_ctx <user>@<pi-host>:~/talosaur-JEPA/exports/`.
   The folder holds `manifest.json`, the `.onnx` files and `ncnn/`. The runtime reads everything
   it needs from `manifest.json`: input size, grid, output meanings, and which files exist for
   which runtime.

## 3. Benchmark: pick runtime, input size and threads

```bash
# every model x runtime x thread count (each configuration runs in a fresh process)
python -m talosaur.onboard.benchmark --export-dir exports/bench_tiny \
    --runtimes ort_fp32 ort_int8 ort_dyn8 ncnn_fp16 --threads 2 3 4 --out reports/pi5/bench_idle

# the same while the camera records 1080p30 H.264 through picamera2 (the real encoder load)
python -m talosaur.onboard.benchmark --export-dir exports/bench_tiny \
    --runtimes ort_int8 ncnn_fp16 --threads 2 3 4 --record-load picamera2 --out reports/pi5/bench_rec

# 10 minutes flat out: temperature, clock and throttling every 5 s (run it inside the closed hull too)
python -m talosaur.onboard.benchmark --export-dir exports/bench_tiny --models vit_tiny_112x208 \
    --runtimes ort_int8 --threads 1 --iters 10 \
    --sustained-s 600 --sustained-model vit_tiny_112x208 --sustained-runtime ort_int8 --sustained-threads 3 \
    --record-load picamera2 --out reports/pi5/bench_sustained
```

`--record-load x264` substitutes an ffmpeg/libx264 encode when no camera is attached.

Each run writes `.md`, `.csv` and `.json`. Columns:
- **fps (model):** inference alone.
- **fps (loop):** the whole per-frame path: lores YUV → RGB, preprocessing, model, and guidance (blob finding, Kalman, state machine, controller, novelty).
- **ms p50/p90/p99:** latency percentiles.
- **peak RSS MB:** peak memory of that process alone.
- **sys avail MB:** `MemAvailable` from the system.
- **temp °C:** CPU temperature.
- **throttled:** decoded `vcgencmd get_throttled` flags.

**Choosing:**
1. Take the **smallest input** that the M3 evaluation says is accurate enough. 208×112 keeps the full 16:9 field of view.
2. Pick the runtime/thread count with the best **fps (loop)** in the *recording* run. It must reach at least **3 fps with margin**; aim for ≥ 5, since tracking gets smoother with rate.
3. Check that the sustained run shows **no throttling flags**. If it does, fix the cooling before trading accuracy for speed.
4. Leave one core free for the camera, encoder and control: 3 threads is usually right when recording.
5. Put the choice into `configs/onboard/pi5.yaml` (`model.name`, `runtime`, `threads`).

**Reference (not the Pi):** random-weight exports on a 4-vCPU Xeon VM @ 2.1 GHz. The Pi's A76
cores will be several times slower, so read the fps only as relative. Memory is the useful part:
it should carry over to aarch64 roughly.

| model @ 112×208 | runtime | peak RSS MB | fps, 1 thread | fps, 4 threads |
|---|---|---|---|---|
| ViT-Ti | ORT fp32 | 118 | 89 | 136 |
| ViT-Ti | ORT int8 (static) | 96 | 105 | 118 |
| ViT-Ti | ORT int8 (dynamic) | 95 | 125 | 163 |
| ViT-Ti | ncnn fp16 | 84 | 69 | 126 |
| ViT-S | ORT int8 (static) | 128 | 48 | 69 |
| ViT-S | ORT fp32 | 222 | 23 | 58 |

On x86, static int8 is not faster than fp32 at 4 threads. On the A76, which has int8 dot-product
instructions but no i8mm, it may be; that is what the Pi run decides. ncnn fp16 uses the A76's
native fp16 arithmetic, which the x86 run cannot show.

## 4. Run the onboard loop

```bash
# dry run on recorded video (no camera, no recording) - same code path as the vehicle
python -m talosaur.onboard.app --config configs/onboard/pi5.yaml --source video --video dive.mp4

# live, with the camera
python -m talosaur.onboard.app --config configs/onboard/pi5.yaml

# overrides: --export-dir, --model, --runtime, --threads, --no-record, --max-frames, --source synthetic
```

Every `stats_every_s` it logs fps, inference latency, state, memory and temperature, and
publishes the same as a `kind: "stats"` message.

**Start on boot** with a systemd unit, `/etc/systemd/system/talosaur.service`. Replace `<user>`;
recent Raspberry Pi OS images have no default `pi` user.

```ini
[Unit]
Description=Talosaur onboard vision
After=network-online.target

[Service]
User=<user>
WorkingDirectory=/home/<user>/talosaur-JEPA
ExecStart=/home/<user>/talosaur-venv/bin/python -m talosaur.onboard.app --config configs/onboard/pi5.yaml
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

Enable it with `sudo systemctl enable --now talosaur`. Watch it with `journalctl -u talosaur -f`.
SIGTERM stops the loop cleanly: the recording is closed and the camera released.

## 5. Output: telemetry and vehicle commands

The autopilot link is still undecided, so the loop publishes one JSON message per frame. It goes
to a JSONL log (`logs/guidance.jsonl`) and to UDP (`127.0.0.1:14600`), both set under `backends:`
in the config. A bridge process turns the commands into whatever your vehicle speaks.
`make_backend` raises `NotImplementedError` for `mavlink` until the stack is chosen.

```json
{"t": 12.4, "kind": "guidance", "frame": 187, "infer_ms": 61.2, "state": "TRACK",
 "frame_prob": 0.93, "novelty": 1.1, "recording": true, "events": [],
 "target": {"found": true, "cx": 0.62, "cy": 0.44, "size": 0.18, "mass": 3.1, "peak": 0.97, "n_blobs": 1, "yaw": 8.9, "pitch": 3.1},
 "track": {"active": true, "confirmed": true, "yaw": 8.5, "pitch": 2.9, "size": 0.18, "yaw_rate": 4.2, "pitch_rate": -0.3, "confidence": 1.0, "hits": 23, "misses": 0},
 "cmd": {"yaw_rate": 0.25, "heave": 0.03, "surge": 0.12, "heading_deg": null, "depth_m": null, "light": 0.3}}
```

**Conventions.**
- `target.cx/cy` are normalised image coordinates, with (0, 0) at the top left.
- `yaw` > 0: the target is right of centre. `pitch` > 0: above centre. Both are degrees in the water.
- `size` is √(area fraction) of the animal's blob.
- `cmd` rates are normalised requests in [-1, 1], always set:
  - `yaw_rate` > 0 = turn right;
  - `heave` > 0 = ascend;
  - `surge` > 0 = forward, < 0 = back off.
- `cmd` setpoints are optional, for an autopilot with heading and depth hold:
  - `heading_deg`: hold this heading. Set on search legs, and while turning away in RELEASE.
  - `depth_m`: go to and hold this depth. Set while searching.
  - A bridge that cannot use them ignores them: the rates already steer toward them whenever
    navigation input (below) is available.
- `cmd.light` is the lamp level, 0 to 1 (docs/SEARCH.md §4): off while searching if the camera can
  see by ambient light, dim if it cannot, and the `track` level close to an animal. `null` means
  guidance leaves the lights to the vehicle (`lights.control: false`). `lights` shows the decision:
  `dark`, the measured ambient brightness (`ambient`, 0–1, measured with the lamp off), and
  whether a lamp-off check is running.
- `state` is one of SEARCH, ACQUIRE, TRACK, FILM, LOST and RELEASE (moving on from an animal, §12).
- `events` carries `encounter_start`, `encounter_end`, `start_recording`, `stop_recording` and `state:<NAME>`, in that order within a frame. The recording events mark where an encounter clip begins and ends; only events mode cuts files at them.
- `recording` says whether video is being written. It is always true in continuous mode unless the disk guard had to stop.
- `encounter` is the animal being filmed:
  - `id` and `engaged_s`;
  - `remaining_s` under the hard cap;
  - `good_s`: seconds well framed;
  - `marginal_rate` and `long_run_rate`: what staying earns now against what the mission earns on
    average. The animal is left when the first falls below the second (§12).
- `reid` gives the target's appearance similarity to animals already filmed (`sim`), how many filmed animals in view were skipped, and how many are remembered.
- `nav` echoes the navigation input in use (below). `search`, during SEARCH only, shows the search
  mode (`profile`, `extensive` or `intensive`), the depth band being worked, the leg heading, and the
  band with the best detection rate so far.
- At the end of each encounter, a `kind: "encounter"` message summarises it (§12).

**Navigation input (optional, strongly recommended).** The search planner (docs/SEARCH.md) needs
depth and heading from the vehicle. The autopilot bridge sends them as small JSON datagrams to UDP
port 14601, at 5–50 Hz:

```json
{"depth_m": 152.3, "heading_deg": 41.0, "yaw_rate_dps": -2.5, "altitude_m": null}
```

- Enable it with `nav: {kind: udp, port: 14601}` in `pi5.yaml`.
- Any field may be missing or `null`. Heading may be magnetic or gyro-integrated; it only has to be
  consistent during the dive.
- A sample older than 1.5 s (`nav.valid_for_s`) is treated as missing.
- Without navigation input, search falls back to a scan-then-hop pattern with no depth control.
- `kind: mavlink` raises `NotImplementedError` until the autopilot is chosen.

Test the link from a laptop on the same network. The telemetry's `nav` field shows the sample for
1.5 s:

```bash
python3 -c 'import json, socket; socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(
    json.dumps({"depth_m": 120, "heading_deg": 90}).encode(), ("<pi-address>", 14601))'
```

**Safety stays with the autopilot.** Depth and altitude limits, obstacle avoidance, leak and
battery failsafes all belong there. Vision only sends requests, already clipped, slew-limited and
with a hard stand-off. `search.min_depth_m` / `max_depth_m` only bound the depths the planner asks
for; they are not a safety limit. The bridge should treat a message older than ~0.5 s (by receive
time) as "all zero", so a crashed or stalled vision process never leaves a stale command active.

## 6. Recording

The Pi 5 has **no hardware H.264 encoder**. picamera2's `H264Encoder` is the software libav/x264
encoder there, and it competes with inference for the four cores.

**Continuous (default, `recording.mode: continuous`).** The main stream is recorded for the whole
run. The state machine never switches it off.
- **Segments.** A new file starts every `segment_s` (5 min), switched at a keyframe so no frame is
  lost: `recordings/talosaur_<session>_00001.ts`, `_00002.ts`, and so on.
- **Crash-safe format.** Segments are MPEG-TS. A file cut short by a power loss or crash still
  plays up to that point, unlike an unfinished MP4. VLC plays `.ts` directly. To convert without
  re-encoding: `ffmpeg -i x.ts -c copy x.mp4`.
- **Segment index.** `talosaur_<session>_segments.jsonl` lists every segment with its start and
  end on the app clock, its wall-clock start, and whether it shows an animal.
- **Encounter log.** Each line of `logs/encounters.jsonl` lists the files that show that animal and
  the offset into the first one: `"recordings": [{"file": "..._00007.ts", "offset_s": 212.4}]`.
- **Storage.** 6 Mbit/s is 2.7 GB per hour. The app logs the free space and the hours it allows at
  start, and adds `disk_free_mb` to its stats messages. For long dives, use a large, fast card or an
  NVMe SSD on the Pi 5's PCIe port.
- **Low disk.** Below `min_free_mb` (2 GB), `low_disk: delete_empty` deletes the oldest finished
  segments that show **no** animal. It never deletes animal footage or anything since the current
  encounter began. `low_disk: stop` stops recording instead. Either way, recording stops below
  `min_free_mb / 4` so the Pi's own filesystem never fills up.

**Encounters only (`recording.mode: events`).** Files only around encounters: from entering TRACK,
through FILM and LOST, until `postroll_s` after leaving them. This saves storage. With
`preroll_s: 0` the encoder runs only while recording, which also saves CPU during SEARCH.

| setting | effect |
|---|---|
| `recording.mode: continuous` (default) | the encoder runs all the time; everything is kept |
| `recording.mode: events`, `preroll_s: 5` | the encoder runs all the time into a 5 s ring buffer; only encounters are saved, with the approach |
| `recording.mode: events`, `preroll_s: 0` | the encoder runs only while recording; loses the approach, saves CPU during SEARCH |
| `source.main_size: [1280, 720]` (default) vs `[1920, 1080]` | 720p costs roughly half the encode CPU of 1080p |
| a separate action camera | no Pi CPU at all; vision still decides where to point the vehicle |

Events-mode files are raw H.264 (`talosaur_YYYYmmdd_HHMMSS_NNN.h264`). Wrap one without
re-encoding: `ffmpeg -framerate 15 -i talosaur_X.h264 -c copy talosaur_X.mp4`. The encoder writes a
keyframe every second, so segment switches, and the pre-roll in events mode, are accurate to 1 s.

## 7. Memory budget (1 GB)

| item | expected |
|---|---|
| OS (Lite, no desktop), services | measure with `free -m` on your image |
| vision process: Python + numpy + onnxruntime + ViT-Ti int8 | ~100 MB peak RSS (x86 reference above; the benchmark measures the Pi) |
| camera buffers (4 × 1280×720 YUV420 + lores) | ~6 MB of CMA memory, outside the process RSS |
| software H.264 encoder | measure: compare `free -m` with `recording.preroll_s` 0 vs 5 |

ViT-S int8 (~130 MB) also fits. Keep the zram swap that Raspberry Pi OS enables by default as a
safety net, but a healthy run should never touch it. The benchmark's `sys avail MB` column
shows the headroom.

## 8. Thermals

The Pi 5 firmware lowers the CPU clock when the SoC gets hot, and inference then slows down
without any error. A sealed hull has no airflow. Use the active cooler, or better, a heat path to
the hull wall (for example an aluminium plate to the end cap). Run the sustained benchmark inside
the closed hull, in water at your operating temperature. The timeline shows temperature, clock and
throttle flags every 5 s. The app also logs temperature and `throttled` in its stats messages.

## 9. Camera settings underwater (`source.controls` in `pi5.yaml`)

- **Focus: manual.** Autofocus hunts in murky water. Set `AfMode: 0` and `LensPosition` in dioptres (1/m).
  - Behind a flat port, objects look ~25% closer (virtual image at d/1.33).
  - To focus on animals ~d m away in water, set `LensPosition ≈ 1.33 / d`. For example, 0.9 for 1.5 m.
  - The config default is 1.0, about 1.3 m in water. Confirm in the pool.
- **Exposure: `AeExposureMode: 1` (short).** Less motion blur at the price of more gain and noise, which the model was trained to tolerate (the degradation augmentation).
  - The frame rate (`fps: 15`) caps the exposure at ~66 ms.
- **White balance.** AWB can be fooled by a scene that is all blue-green. If colours pump, fix the gains after a pool test with `AwbEnable: false` and `ColourGains: [r, b]`.
- **HDR** (`source.hdr`): `sensor` (IMX708 on-sensor HDR, limits the sensor mode to 2304×1296), or `isp` / `night` (Pi 5 ISP modes). **Untested**: compare against `off` on the same pool scene before using one.
- **Stream shape.** The model's input comes from the ISP's lores stream, scaled from the same full-field crop as the recording stream. A small aspect mismatch (208×112 vs 16:9) stretches it by ~4%. That is harmless: the camera model works in normalised coordinates.

## 10. Camera model and underwater calibration

Guidance converts the target's image position into bearing and elevation (`guidance.camera`):

| `port` | model |
|---|---|
| `flat` (default) | nominal 102°×67° in-air field of view, then Snell refraction at a flat port (n = 1.333): ~71°×49° in water |
| `dome` / `air` | nominal field of view unchanged (a well-centred dome keeps it) |
| `calibrated` | intrinsics **and lens distortion** measured in water through the actual port |

The nominal models ignore the wide lens's barrel distortion, which grows toward the edges. Once
the housing is final, calibrate in water:

```bash
sudo apt install python3-opencv
python scripts/pi/calibrate_camera.py capture --out calib_imgs --n 30    # move a printed checkerboard around the whole view, 0.5-2 m
python scripts/pi/calibrate_camera.py solve --images calib_imgs --board 9x6 --square 0.025
```

Copy `fx, fy, cx, cy, dist` from `camera_calibration.yaml` into `guidance.camera` and set
`port: calibrated`. The values are normalised by image size, so they apply to the lores stream
too, since it shows the same field of view.

## 11. Tuning guidance on recorded video

```bash
python scripts/replay.py --video dive.mp4 --export-dir exports/tiny_ctx --model vit_tiny_112x208 \
    --runtime ort_int8 --pi-fps 6 --out replay_dive.mp4      # --pi-fps = the loop fps you measured
```

Replay runs the exact onboard pipeline on your desktop or on the Pi. It drops frames to the Pi's
rate, so the tracker sees what the vehicle would, and writes:
- an annotated video: heatmap, detected centroid, filtered track, state, the animal being filmed and its time left, the appearance similarity to animals already filmed, command bars, and a red dot while recording;
- `replay_dive.jsonl` with the per-frame telemetry;
- `replay_dive.encounters.jsonl` with one line per animal (§12).

Knobs in `pi5.yaml` → `guidance:`:

| knob | raise it when | lower it when |
|---|---|---|
| `heat_thr` (seed) | specks and backscatter start tracks | small / distant animals are missed |
| `heat_thr_low` (extent) | blobs merge with the background | apparent size is underestimated |
| `min_mass` | single bright specks pass | one-patch animals are rejected |
| `fsm.acquire_n` / `acquire_m` | false acquisitions | slow to react |
| `fsm.film_center_deg`, `film_size`, `film_hold_s` | FILM flickers | FILM never reached |
| `controller.target_size` / `standoff_size` | it stays too far / too close | — |
| `controller.yaw_kp`, `yaw_kd`, `slew_per_s` | sluggish turning | oscillation around the target |
| `encounter.same_sim` | different animals are taken for one already filmed | the same animal is filmed again as "new" |
| `encounter.max_s` | too little footage per animal | it lingers on one animal |

Keep the vehicle's own turn rate in mind. Replay is open loop: the recorded camera does not turn
toward the target, so a target crossing the frame never looks centred for long.

## 12. One animal at a time: when to move on, not filming the same fish twice

The sub decides for each animal when more footage of it is worth less than going to find another.
Then it stops documenting that animal, moves away, and looks for a *different* one. Settings are
under `guidance.encounter` in `pi5.yaml`; the reasoning and evidence are in docs/SEARCH.md §3.

1. **Encounter.** Locking onto an animal (ACQUIRE → TRACK) starts an encounter with a new id. In the
   default continuous mode the video is already running; in events mode a clip starts.
2. **When to leave** (`rule: mvt`, the default). Time in TRACK, FILM and LOST counts. The sub leaves
   for one of three reasons:
   - `enough`: the animal has been well framed long enough that more footage is worth less than
     searching on. How long that is depends on how easy animals have been to find so far, and on
     how different this one is from those already filmed.
   - `no_shot`: no good shot within `giveup_s` (30 s), or it was never well framed and the rule
     gave up on it sooner.
   - `budget`: the hard cap `max_s` (180 s) is reached.

   Nothing ends an encounter before `min_s` (10 s). `rule: fixed` keeps only the hard cap, as
   before this change.

   **The sub never chases.** If the animal swims away (its apparent size shrinks fast for
   `flee_s` while the sub is not backing off), the sub stops approaching and only turns to keep it
   in view. If it gets away, the encounter ends as `fled`, and that animal is left alone for the
   cooldown.
3. **RELEASE.** On leaving, the state machine enters RELEASE:
   - the video keeps running in continuous mode, and the encounter log marks where this animal is
     in it; in events mode the clip stops after its post-roll;
   - the sub backs off (`controller.release_backoff_s`);
   - it turns away from the side the animal was on: `release_turn_deg` (120°) with a heading input,
     otherwise for `release_turn_s`;
   - it swims on for the rest of `fsm.release_s` (default 12 s);
   - detections are ignored meanwhile; then SEARCH resumes with a tight local search, because
     animals come in patches (docs/SEARCH.md §2).
4. **Recognising animals already filmed, by appearance rather than position.** Underwater position
   is unreliable, so it isn't used. The model exports its per-patch features (`patch_tokens`).
   The sub averages them over the animal's heatmap blob, subtracts the average background (water)
   features, and compares the result with each remembered animal by cosine similarity:
   - **Already left on purpose** (similarity ≥ `same_sim`, and the encounter ended for one of the
     four reasons above): the animal is ignored for `cooldown_s` (default 5 min), even while it
     stays in view. If another animal is in view at the same time, that one is chosen instead.
   - **Only lost** (it swam out of view first): it is **resumed** with the footage it already has,
     so its value keeps diminishing where it left off. A fish that keeps coming and going is still
     filmed for at most `max_s` in total.
   - **New-looking animals are worth more.** The less an animal resembles any filmed so far on
     this run, the higher its footage is valued (up to 2× with `novelty_bonus: 1.0`), so it gets
     more time.
5. **Encounter log.** Every encounter is written to `logs/encounters.jsonl`, and also sent as a
   `kind: "encounter"` message on the backends. Each line records:
   - the animal's id;
   - why it ended (`enough`, `no_shot`, `budget`, `fled`, `lost` or `shutdown`);
   - total time on that animal (`engaged_s`), time well framed (`good_s`) and time in FILM;
   - the footage value it was credited with, and its novelty weight;
   - the best-framed moment (`best_t`);
   - the video files, with the offset into the first one.

   ```json
   {"kind": "encounter", "id": 3, "reason": "enough", "engaged_s": 71.3, "good_s": 52.0,
    "value": 1.21, "novelty_weight": 1.47, "film_s": 44.8, "resumed": true, "best_t": 431.7,
    "recordings": [{"file": "recordings/talosaur_20261003_101512_00002.ts", "offset_s": 131.2}]}
   ```

**Limits and calibration.**
- **Look-alikes.** Appearance separates animals that *look* different. Two fish of the same species
  and size (a school) look the same, so after filming one the sub leaves the whole school alone for
  `cooldown_s`.
- **Threshold.** `same_sim: 0.8` is a starting guess; the right value depends on the trained model.
  To set it:
  1. Run replay on footage where the same animal comes back, and on footage where a different one
     appears. Replay logs every similarity (`reid.sim`, and `sim=` on the overlay).
  2. Set `same_sim` between the two groups of values.

  On the toy model, a returning fish scores ~1.0 and a differently coloured fish 0.67.
- **Manoeuvre.** Without a heading input, the move-on turn is timed (seconds of turning, not
  degrees) until the vehicle's turn rate is calibrated.
- **Older exports.** Exports made before this change have no patch tokens. The sub still moves on
  from each animal but cannot recognise animals or weight novel ones, and the app warns at start. Re-export with
  `scripts/export.py`. The parity report's `token cos` column shows that int8 keeps these features.
- **No limit.** `max_s: 0` removes the hard cap; the leave rule still decides. To follow one animal
  indefinitely, set `rule: fixed` and `max_s: 0`.

## 13. Optional: ncnn int8

This path is **untested here**; ORT int8 and ncnn fp16 are the tested ones. You need ncnn's
`ncnn2table` and `ncnn2int8` tools, from a release matching your `ncnn` pip version or built from
source. In current ncnn, `ncnn2int8` quantizes the Gemm (MLP) weights itself and requires a
calibration table for MultiHeadAttention.

```bash
# on the training box: save the stratified calibration images as .npy
python scripts/export.py ... --ncnn-calib          # -> exports/<name>/ncnn/calib_<model>/calib_list.txt
cd exports/<name>/ncnn
ncnn2table vit_tiny_112x208.ncnn.param vit_tiny_112x208.ncnn.bin calib_vit_tiny_112x208/calib_list.txt \
    vit_tiny_112x208.table shape=[208,112,3] method=kl type=1        # shape is [w,h,c]; type=1 = npy input
ncnn2int8 vit_tiny_112x208.ncnn.param vit_tiny_112x208.ncnn.bin \
    vit_tiny_112x208_int8.ncnn.param vit_tiny_112x208_int8.ncnn.bin vit_tiny_112x208.table
```

Then add `"ncnn_int8_param": "ncnn/vit_tiny_112x208_int8.ncnn.param"` and
`"ncnn_int8_bin": "ncnn/vit_tiny_112x208_int8.ncnn.bin"` to that model's `files` in
`manifest.json`, and benchmark `--runtimes ncnn_int8`. Before deploying it, check its accuracy
with replay on labelled clips, or with `scripts/eval.py`.

## 14. Not yet verified on hardware

1. **All Pi numbers**: fps, memory, thermals, and the CPU cost of recording while running inference.
2. **picamera2 behaviour.** Checked only against the 0.3.37 API with a fake camera:
   - lores YUV420 capture and the configured sizes;
   - `CircularOutput` pre-roll with the libav encoder;
   - `iperiod` / `framerate` on the encoder;
   - the HDR modes.
3. The ncnn int8 conversion (§13).
4. The in-water calibration workflow and the flat-port focus rule of thumb.
5. Camera controls in the dark: gain limits and noise at depth.
6. **Recognising animals with the trained model** (§12). Tested only with synthetic features and the toy model's colours. How well JEPA patch tokens separate real animals, and the right `same_sim`, must come from your footage.
7. **Search and leave-rule settings** (docs/SEARCH.md). Tested only in the simulator, whose animal
   densities, detection ranges and reactions to the vehicle are assumptions. In particular:
   - what the camera detects with the lights off at your depths (`lights.search: 0.0` relies on it);
   - the vehicle's real speed per unit of `surge` (`search.speed_mps_per_unit`), for the coverage map;
   - how well the autopilot holds the heading and depth setpoints.

## 15. What to send back

- `reports/pi5/bench_idle.md`, `bench_rec.md` and `bench_sustained.md`, plus their `.json` files.
- `free -m` output with the app idle in SEARCH, and while recording.
- One short pool clip with the toy or trained model, plus its `logs/guidance.jsonl` and `logs/encounters.jsonl`. Replay it on the desktop to tune.
- From the first dives: `logs/guidance.jsonl` and `logs/encounters.jsonl` with navigation input on.
  The finds per depth band and the encounter values replace the simulator's assumptions
  (docs/SEARCH.md §7). Summarise a dive with:

  ```bash
  python -m talosaur.onboard.dive_report logs/guidance.jsonl logs/encounters.jsonl \
      --config configs/onboard/pi5.yaml --out reports/dives/first_dive.md
  ```

  The report gives the time per state, the search modes, finds per minute of search in each depth
  band, and each encounter's reason and duration. It also shows the appearance similarities (for
  `same_sim`) and the lamp's decisions with the ambient brightness it measured (for `dark_luma`).
