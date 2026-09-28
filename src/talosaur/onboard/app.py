"""Talosaur onboard vision loop (Raspberry Pi 5): camera -> model -> guidance -> commands + recording.

  python -m talosaur.onboard.app --config configs/onboard/pi5.yaml
  python -m talosaur.onboard.app --config configs/onboard/pi5.yaml --source video --video dive.mp4   # dry run
  python -m talosaur.onboard.app --config configs/onboard/pi5.yaml --export-dir exports/toy --model toy_112x208 \
      --runtime ort_fp32          # camera/recording/UDP check with the toy colour model (talosaur.onboard.toy_model)

Never imports torch. Telemetry (state, target, track, command, latency) goes to the configured
backends (JSONL log and/or JSON over UDP); recording follows the state machine.
"""

from __future__ import annotations

import argparse
import signal
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from talosaur.guidance.backends import make_backend
from talosaur.guidance.novelty import NoveltyDetector
from talosaur.guidance.pipeline import Guidance, GuidanceConfig
from talosaur.onboard import sysinfo
from talosaur.onboard.camera import Picamera2Source, SyntheticSource, VideoFileSource
from talosaur.onboard.recorder import NullRecorder, Picamera2Recorder
from talosaur.onboard.runtime import load_runner, preprocess
from talosaur.utils.io import load_yaml
from talosaur.utils.log import get_logger

log = get_logger("onboard")


def build(cfg: dict, source_override: str | None = None, video: str | None = None):
    m = cfg["model"]
    runner = load_runner(m["export_dir"], m["name"], m.get("runtime", "ort_int8"), int(m.get("threads", 3)))
    H, W = runner.spec.input_hw
    sc = cfg.get("source", {})
    src_kind = source_override or sc.get("kind", "picamera2")
    if src_kind == "picamera2":
        source = Picamera2Source(
            (W, H),
            tuple(sc.get("main_size", (1280, 720))),
            float(sc.get("fps", 15)),
            sc.get("controls"),
            sc.get("hdr", "off"),
        )
    elif src_kind == "video":
        source = VideoFileSource(video or sc["video"], size=(W, H), max_fps=sc.get("max_fps"))
    elif src_kind == "synthetic":
        source = SyntheticSource((W, H), int(sc.get("n_frames", 300)), float(sc.get("fps", 10)))
    else:
        raise ValueError(f"unknown source {src_kind!r}")
    rc = cfg.get("recording", {})
    recorder = NullRecorder()
    if src_kind == "picamera2" and rc.get("enabled", True):
        try:
            recorder = Picamera2Recorder(
                source.cam,
                rc.get("out_dir", "recordings"),
                int(rc.get("bitrate", 6_000_000)),
                float(rc.get("preroll_s", 5.0)),
                float(sc.get("fps", 15)),
            )
        except Exception:
            source.close()  # release the camera, or the next start fails with "device busy"
            raise
    guidance = Guidance(GuidanceConfig.from_dict(cfg.get("guidance")))  # novelty sized on the first frame
    backend = make_backend(cfg.get("backends", [{"kind": "jsonl", "path": "logs/guidance.jsonl"}]))
    return runner, source, recorder, guidance, backend


def run(
    cfg: dict, source_override: str | None = None, video: str | None = None, max_frames: int | None = None
) -> dict:
    runner, source, recorder, guidance, backend = build(cfg, source_override, video)
    H, W = runner.spec.input_hw
    stop = {"flag": False}

    def _sig(*_):
        stop["flag"] = True

    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for s in (signal.SIGINT, signal.SIGTERM):
            old_handlers[s] = signal.signal(s, _sig)
    lat: deque[float] = deque(maxlen=200)
    loop_t: deque[float] = deque(maxlen=200)
    n = 0
    t_stats = time.monotonic()
    stats_every = float(cfg.get("stats_every_s", 10.0))
    states: dict[str, int] = {}
    n_recordings = 0
    try:
        while not stop["flag"]:
            t_loop = time.perf_counter()
            fr = source.read()
            if fr is None:
                break
            t0 = time.perf_counter()
            f, h, e = runner(preprocess(fr.rgb, (H, W)))
            t_inf = time.perf_counter() - t0
            if guidance.novelty is None and guidance.cfg.novelty:
                guidance.novelty = NoveltyDetector(int(np.asarray(e).size))
            cmd, tele, events = guidance.step(fr.t, f, h, e)
            for ev in events:
                if ev == "start_recording":
                    recorder.start(fr.t)
                    n_recordings += 1
                elif ev == "stop_recording":
                    recorder.stop(fr.t)
            lat.append(t_inf * 1000)
            loop_t.append(time.perf_counter() - t_loop)
            tele["kind"] = "guidance"
            tele["frame"] = fr.index
            tele["infer_ms"] = round(t_inf * 1000, 2)
            backend.publish(tele)
            states[tele["state"]] = states.get(tele["state"], 0) + 1
            n += 1
            if time.monotonic() - t_stats > stats_every:
                t_stats = time.monotonic()
                snap = sysinfo.snapshot()
                fps = len(loop_t) / max(1e-6, sum(loop_t))
                log.info(
                    f"fps {fps:.1f} | infer p50 {np.percentile(lat, 50):.1f} ms p95 {np.percentile(lat, 95):.1f} | "
                    f"state {tele['state']} | rss {snap['rss_mb']} MB avail {snap['mem_available_mb']} MB | {snap['cpu_temp_c']:.1f} C"
                )
                backend.publish(
                    {"t": fr.t, "kind": "stats", "fps": fps, **snap, "throttled": sysinfo.throttled()}
                )
            if max_frames and n >= max_frames:
                break
    finally:
        recorder.close()
        source.close()
        backend.close()
        for s, hnd in old_handlers.items():
            signal.signal(s, hnd)
    summary = {
        "frames": n,
        "states": states,
        "recordings": n_recordings,
        "infer_ms_p50": float(np.percentile(lat, 50)) if lat else float("nan"),
        "loop_fps": len(loop_t) / max(1e-6, sum(loop_t)) if loop_t else float("nan"),
        "peak_rss_mb": sysinfo.peak_rss_mb(),
    }
    log.info(f"done: {summary}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/onboard/pi5.yaml")
    ap.add_argument("--source", default=None, choices=["picamera2", "video", "synthetic"])
    ap.add_argument("--video", default=None)
    ap.add_argument("--export-dir", default=None, help="override model.export_dir")
    ap.add_argument("--model", default=None, help="override model.name (a manifest key)")
    ap.add_argument("--runtime", default=None, help="override model.runtime")
    ap.add_argument("--threads", type=int, default=None, help="override model.threads")
    ap.add_argument("--no-record", action="store_true", help="never record (recording.enabled=false)")
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args(argv)
    cfg = load_yaml(a.config)
    for key, val in (
        ("export_dir", a.export_dir),
        ("name", a.model),
        ("runtime", a.runtime),
        ("threads", a.threads),
    ):
        if val is not None:
            cfg["model"][key] = val
    if a.no_record:
        cfg.setdefault("recording", {})["enabled"] = False
    Path("logs").mkdir(exist_ok=True)
    run(cfg, a.source, a.video, a.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
