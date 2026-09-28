#!/usr/bin/env python
"""Run the complete onboard loop on recorded video and write an annotated video + telemetry.

  python scripts/replay.py --video dive.mp4 --export-dir exports/tiny_ctx --model vit_tiny_112x208 \
      --runtime ort_int8 --pi-fps 8 --out replay_dive.mp4

``--pi-fps`` drops frames to the frame rate you measured on the Pi, so the tracker and state
machine see what the vehicle would see. Overlay: heatmap, target centroid, track, state, command
bars and a red dot while "recording".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from talosaur.guidance.heatmap import sigmoid
from talosaur.guidance.novelty import NoveltyDetector
from talosaur.guidance.pipeline import Guidance, GuidanceConfig
from talosaur.onboard.camera import VideoFileSource
from talosaur.onboard.runtime import load_runner, preprocess, resize_bilinear_u8, resize_u8
from talosaur.utils.io import load_yaml

STATE_COLORS = {
    "SEARCH": (160, 160, 160),
    "ACQUIRE": (255, 200, 0),
    "TRACK": (0, 200, 255),
    "FILM": (0, 255, 120),
    "LOST": (255, 80, 80),
}


def annotate(frame: np.ndarray, heat_logit: np.ndarray, tele: dict, cam) -> np.ndarray:
    from PIL import Image, ImageDraw

    H, W = frame.shape[:2]
    prob = sigmoid(heat_logit)
    hm = (
        resize_bilinear_u8((prob[..., None] * 255).repeat(3, -1).astype(np.uint8), H, W)[..., 0].astype(
            np.float32
        )
        / 255
    )
    overlay = frame.astype(np.float32)
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    a = (0.45 * hm)[..., None]
    overlay = overlay * (1 - a) + red * a
    img = Image.fromarray(overlay.clip(0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    col = STATE_COLORS.get(tele["state"], (255, 255, 255))
    tg = tele["target"]
    if tg["found"]:
        x, y = tg["cx"] * W, tg["cy"] * H
        d.line([(x - 12, y), (x + 12, y)], fill=(255, 255, 0), width=2)
        d.line([(x, y - 12), (x, y + 12)], fill=(255, 255, 0), width=2)
    tr = tele["track"]
    if tr["active"]:
        # the filtered track, projected back into the image
        u, v = cam.project(tr["yaw"], tr["pitch"])
        tx, ty = u * W, v * H
        r = max(8, tr["size"] * min(W, H) / 2)
        d.ellipse([tx - r, ty - r, tx + r, ty + r], outline=col, width=3)
    d.rectangle([0, 0, W, 26], fill=(0, 0, 0))
    txt = f"{tele['state']:<7} p(animal)={tele['frame_prob']:.2f}"
    if tele.get("novelty") is not None:
        txt += f" novelty={tele['novelty']:.1f}"
    if tr["active"]:
        txt += f" yaw={tr['yaw']:+.0f} pitch={tr['pitch']:+.0f} size={tr['size']:.2f}"
    d.text((8, 6), txt, fill=col)
    cmd = tele["cmd"]
    for i, (k, v) in enumerate((("yaw", cmd["yaw_rate"]), ("heave", cmd["heave"]), ("surge", cmd["surge"]))):
        y0 = H - 18 * (3 - i)
        d.text((8, y0), k, fill=(255, 255, 255))
        cx0 = 70
        d.rectangle([cx0, y0 + 4, cx0 + 100, y0 + 12], outline=(200, 200, 200))
        x0, x1 = sorted((cx0 + 50, cx0 + 50 + 50 * max(-1.0, min(1.0, v))))  # bar grows left for negative
        d.rectangle([x0, y0 + 4, x1, y0 + 12], fill=(0, 200, 255))
    if tele["recording"]:
        d.ellipse([W - 24, 6, W - 10, 20], fill=(255, 0, 0))
    return np.asarray(img)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--runtime", default="ort_int8")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument(
        "--config", default="configs/onboard/pi5.yaml", help="guidance settings are taken from here"
    )
    ap.add_argument("--pi-fps", type=float, default=8.0)
    ap.add_argument("--width", type=int, default=960, help="output video width")
    ap.add_argument("--out", default="replay.mp4")
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args(argv)
    import av

    runner = load_runner(a.export_dir, a.model, a.runtime, a.threads)
    H, W = runner.spec.input_hw
    gcfg = GuidanceConfig.from_dict(
        (load_yaml(a.config) or {}).get("guidance") if Path(a.config).exists() else None
    )
    guid = Guidance(gcfg)
    src = VideoFileSource(a.video, size=(W, H), keep_full=True, max_fps=a.pi_fps)
    out = Path(a.out)
    tele_path = out.with_suffix(".jsonl")
    container = None
    stream = None
    n = 0
    states: dict[str, int] = {}
    with open(tele_path, "w") as tf:
        while True:
            fr = src.read()
            if fr is None or (a.max_frames and n >= a.max_frames):
                break
            f, h, e = runner(preprocess(fr.rgb, (H, W)))
            if guid.novelty is None and gcfg.novelty:
                guid.novelty = NoveltyDetector(int(np.asarray(e).size))
            _, tele, _ = guid.step(fr.t, f, h, e)
            tf.write(json.dumps(tele) + "\n")
            states[tele["state"]] = states.get(tele["state"], 0) + 1
            full = fr.full
            oh = int(round(full.shape[0] * a.width / full.shape[1])) // 2 * 2
            vis = annotate(resize_u8(full, oh, a.width), h, tele, gcfg.camera)
            if container is None:
                container = av.open(str(out), mode="w")
                stream = container.add_stream("libx264", rate=int(round(a.pi_fps)))
                stream.width, stream.height = vis.shape[1], vis.shape[0]
                stream.pix_fmt = "yuv420p"
            for pkt in stream.encode(av.VideoFrame.from_ndarray(vis, format="rgb24")):
                container.mux(pkt)
            n += 1
    if container is not None:
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()
    src.close()
    print(json.dumps({"frames": n, "states": states, "video": str(out), "telemetry": str(tele_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
