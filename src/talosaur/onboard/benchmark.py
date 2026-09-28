"""Raspberry Pi 5 benchmark: fps, latency and memory for every exported model x runtime x
thread count, optionally while H.264 recording loads the CPU, plus a sustained thermal run.

  python -m talosaur.onboard.benchmark --export-dir exports/tiny_ctx
  python -m talosaur.onboard.benchmark --export-dir exports/tiny_ctx --runtimes ort_int8 ncnn_fp16 --threads 2 3 4 \
      --record-load x264 --sustained-s 600 --sustained-model vit_tiny_112x208

Each configuration runs in a fresh subprocess, so the peak RSS is that configuration's alone.
Writes <out>.json, <out>.csv and <out>.md (default reports/pi5/bench).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from talosaur.onboard import sysinfo
from talosaur.onboard.runtime import RUNTIME_FILES, available_runtimes

RUNTIMES = list(RUNTIME_FILES)


def _single(spec: dict) -> dict:
    """Run inside a subprocess: load one model and time it (``iters`` runs, or ``seconds`` of
    sustained running with a temperature / clock / throttling sample every ``sample_s``)."""
    from talosaur.onboard.runtime import load_runner

    rss0 = sysinfo.rss_mb()
    t0 = time.perf_counter()
    runner = load_runner(spec["export_dir"], spec["model"], spec["runtime"], spec["threads"])
    load_s = time.perf_counter() - t0
    H, W = runner.spec.input_hw
    x = np.random.default_rng(0).random((3, H, W), dtype=np.float32)
    for _ in range(spec["warmup"]):
        runner(x)
    seconds = float(spec.get("seconds") or 0)
    sample_s = float(spec.get("sample_s", 5.0))
    lat: list[float] = []
    timeline = []
    start = time.perf_counter()
    next_sample = start + sample_s
    while True:
        t = time.perf_counter()
        runner(x)
        now = time.perf_counter()
        lat.append((now - t) * 1000)
        if seconds:
            if now >= next_sample:
                next_sample += sample_s
                recent = lat[-50:]
                timeline.append(
                    {
                        "t": round(now - start, 1),
                        "fps": 1000 / float(np.mean(recent)),
                        "temp_c": sysinfo.cpu_temp_c(),
                        "freq_mhz": sysinfo.cpu_freq_mhz(),
                        "throttled": sysinfo.throttled().get("flags"),
                    }
                )
            if now - start >= seconds:
                break
        elif len(lat) >= spec["iters"]:
            break
    a = np.asarray(lat)
    loop_ms = _time_full_loop(runner, int(spec.get("loop_iters", 0)))
    return {
        **{k: spec[k] for k in ("model", "runtime", "threads")},
        "loop_fps": float(1000 / loop_ms) if loop_ms else None,
        "loop_ms_p50": loop_ms,
        "size": spec["model"].rsplit("_", 1)[-1],
        "iters": int(a.size),
        "fps": float(1000 / a.mean()),
        "ms_mean": float(a.mean()),
        "ms_p50": float(np.percentile(a, 50)),
        "ms_p90": float(np.percentile(a, 90)),
        "ms_p99": float(np.percentile(a, 99)),
        "load_s": load_s,
        "rss_before_mb": rss0,
        "rss_after_mb": sysinfo.rss_mb(),
        "peak_rss_mb": sysinfo.peak_rss_mb(),
        "mem_available_mb": sysinfo.mem_available_mb(),
        "cpu_temp_c": sysinfo.cpu_temp_c(),
        "cpu_freq_mhz": sysinfo.cpu_freq_mhz(),
        "throttled": sysinfo.throttled().get("flags"),
        "timeline": timeline,
    }


def _time_full_loop(runner, iters: int) -> float | None:
    """Median ms of the whole per-frame path the app runs: lores YUV420 -> RGB, preprocess, model,
    guidance (targets, appearance memory, tracker, state machine, controller, novelty)."""
    if iters <= 0:
        return None
    from talosaur.guidance.novelty import NoveltyDetector
    from talosaur.guidance.pipeline import Guidance
    from talosaur.onboard.camera import yuv420_to_rgb
    from talosaur.onboard.runtime import preprocess

    H, W = runner.spec.input_hw
    stride = (W + 63) // 64 * 64
    rng = np.random.default_rng(1)
    yuv = rng.integers(0, 256, (H * 3 // 2, stride), dtype=np.uint8)
    guid = Guidance()
    ms = []
    for i in range(iters + 3):
        t = time.perf_counter()
        out = runner.run(preprocess(yuv420_to_rgb(yuv, W, H), (H, W)))
        if guid.novelty is None:
            guid.novelty = NoveltyDetector(int(np.asarray(out.emb).size))
        guid.step(i * 0.1, out.frame, out.heat, out.emb, out.tokens)
        if i >= 3:
            ms.append((time.perf_counter() - t) * 1000)
    return float(np.median(ms))


class RecordingLoad:
    """Emulates the CPU cost of recording while benchmarking."""

    def __init__(self, kind: str, size=(1920, 1080), fps: int = 30):
        self.kind, self.proc, self.cam = kind, None, None
        if kind == "none":
            return
        if kind == "picamera2":
            from picamera2 import Picamera2
            from picamera2.encoders import H264Encoder
            from picamera2.outputs import FileOutput

            self.cam = Picamera2()
            self.cam.configure(
                self.cam.create_video_configuration(
                    main={"size": size, "format": "YUV420"}, controls={"FrameRate": fps}
                )
            )
            self.cam.start_recording(H264Encoder(bitrate=10_000_000), FileOutput(os.devnull))
        elif kind == "x264":
            if not shutil.which("ffmpeg"):
                raise RuntimeError("--record-load x264 needs the ffmpeg CLI (sudo apt install ffmpeg)")
            cmd = ["ffmpeg", "-loglevel", "error", "-re", "-f", "lavfi", "-i", f"testsrc2=size={size[0]}x{size[1]}:rate={fps}",
                   "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-f", "null", "-"]  # fmt: skip
            self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(2.0)
        else:
            raise ValueError(kind)

    def stop(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        if self.cam is not None:
            self.cam.stop_recording()
            self.cam.close()


def run_config(spec: dict) -> dict:
    out = subprocess.run(
        [sys.executable, "-m", "talosaur.onboard.benchmark", "--single", json.dumps(spec)],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return {**{k: spec[k] for k in ("model", "runtime", "threads")}, "error": out.stderr.strip()[-500:]}
    return json.loads(out.stdout.strip().splitlines()[-1])


def _fmt(v, spec: str = ".1f") -> str:
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else format(v, spec)


def to_markdown(rows: list[dict], meta: dict) -> str:
    cols = [
        "model",
        "runtime",
        "threads",
        "fps (model)",
        "fps (loop)",
        "ms p50",
        "ms p90",
        "ms p99",
        "peak RSS MB",
        "sys avail MB",
        "temp °C",
        "throttled",
    ]
    lines = [
        "# Raspberry Pi 5 benchmark",
        "",
        f"{meta.get('device', '')} | total RAM {_fmt(meta.get('mem_total_mb'), '.0f')} MB | "
        f"recording load: {meta.get('record_load')} | {meta.get('date')}",
        "",
        "fps (model) = inference alone; fps (loop) = camera YUV -> RGB, preprocessing, model and guidance.",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "---|" * len(cols),
    ]
    for r in rows:
        if "error" in r:
            cells = [
                r["model"],
                r["runtime"],
                str(r["threads"]),
                "error: " + r["error"].replace("|", "/").splitlines()[-1][:80],
            ]
            cells += [""] * (len(cols) - len(cells))
        else:
            cells = [
                r["model"],
                r["runtime"],
                str(r["threads"]),
                _fmt(r["fps"]),
                _fmt(r.get("loop_fps")),
                _fmt(r["ms_p50"]),
                _fmt(r["ms_p90"]),
                _fmt(r["ms_p99"]),
                _fmt(r["peak_rss_mb"], ".0f"),
                _fmt(r["mem_available_mb"], ".0f"),
                _fmt(r["cpu_temp_c"]),
                ",".join(r["throttled"] or []) or "-",
            ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--single", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--export-dir", default=None)
    ap.add_argument("--models", nargs="*", default=None, help="manifest keys (default: all)")
    ap.add_argument(
        "--runtimes", nargs="*", default=["ort_fp32", "ort_int8", "ort_dyn8", "ncnn_fp16"], choices=RUNTIMES
    )
    ap.add_argument("--threads", nargs="*", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--loop-iters", type=int, default=30, help="also time the full per-frame loop (0 = skip)")
    ap.add_argument("--record-load", default="none", choices=["none", "picamera2", "x264"])
    ap.add_argument("--sustained-s", type=int, default=0, help="also run one config for N seconds (thermals)")
    ap.add_argument("--sustained-model", default=None)
    ap.add_argument("--sustained-runtime", default="ort_int8", choices=RUNTIMES)
    ap.add_argument("--sustained-threads", type=int, default=3)
    ap.add_argument(
        "--sample-s", type=float, default=5.0, help="sustained run: seconds between thermal samples"
    )
    ap.add_argument("--out", default="reports/pi5/bench")
    a = ap.parse_args(argv)
    if a.single:
        print(json.dumps(_single(json.loads(a.single))))
        return 0
    if not a.export_dir:
        ap.error("--export-dir is required")
    manifest = json.loads((Path(a.export_dir) / "manifest.json").read_text())
    models = a.models or list(manifest["models"])
    unknown = [
        m for m in models + ([a.sustained_model] if a.sustained_model else []) if m not in manifest["models"]
    ]
    if unknown:
        ap.error(f"not in {a.export_dir}/manifest.json: {unknown}; available: {list(manifest['models'])}")
    rows = []
    load = RecordingLoad(a.record_load)
    try:
        for m in models:
            for rt in a.runtimes:
                if rt not in available_runtimes(manifest["models"][m]):
                    print(f"{m:>22} {rt:>10}: not in this export, skipped", flush=True)
                    continue
                for th in a.threads:
                    spec = {
                        "export_dir": a.export_dir,
                        "model": m,
                        "runtime": rt,
                        "threads": th,
                        "iters": a.iters,
                        "warmup": a.warmup,
                        "loop_iters": a.loop_iters,
                    }
                    r = run_config(spec)
                    rows.append(r)
                    msg = r.get("error") or (
                        f"{r['fps']:.1f} fps model-only, {_fmt(r['loop_fps'])} fps full loop, "
                        f"p50 {r['ms_p50']:.1f} ms, peak RSS {r['peak_rss_mb']:.0f} MB"
                    )
                    print(f"{m:>22} {rt:>10} x{th}: {msg}", flush=True)
        sustained = None
        if a.sustained_s:
            sm = a.sustained_model or models[0]
            spec = {
                "export_dir": a.export_dir,
                "model": sm,
                "runtime": a.sustained_runtime,
                "threads": a.sustained_threads,
                "iters": 0,
                "warmup": a.warmup,
                "seconds": a.sustained_s,
                "sample_s": a.sample_s,
            }
            print(
                f"sustained run: {sm} {a.sustained_runtime} x{a.sustained_threads} for {a.sustained_s} s ...",
                flush=True,
            )
            sustained = run_config(spec)
    finally:
        load.stop()
    meta = {
        "device": open("/proc/device-tree/model").read().strip("\x00")
        if os.path.exists("/proc/device-tree/model")
        else os.uname().machine,
        "mem_total_mb": sysinfo.mem_total_mb(),
        "record_load": a.record_load,
        "date": time.strftime("%Y-%m-%d %H:%M"),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(
        json.dumps({"meta": meta, "results": rows, "sustained": sustained}, indent=1)
    )
    keys = [
        "model",
        "size",
        "runtime",
        "threads",
        "fps",
        "loop_fps",
        "ms_p50",
        "ms_p90",
        "ms_p99",
        "peak_rss_mb",
        "mem_available_mb",
        "cpu_temp_c",
    ]
    with open(str(out) + ".csv", "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            if "error" not in r:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    md = to_markdown(rows, meta)
    if sustained and "timeline" in sustained:
        md += "\n## Sustained run\n\n| t (s) | fps | temp °C | MHz | throttled |\n|---|---|---|---|---|\n"
        md += (
            "\n".join(
                f"| {s['t']} | {_fmt(s['fps'])} | {_fmt(s['temp_c'])} | {_fmt(s['freq_mhz'], '.0f')} | {','.join(s['throttled'] or []) or '-'} |"
                for s in sustained["timeline"]
            )
            + "\n"
        )
    Path(str(out) + ".md").write_text(md)
    print(f"\nwrote {out}.json / .csv / .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
