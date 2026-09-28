"""M6 onboard tests: camera conversion, sources, runtime loading, the toy model, the app loop, the
benchmark and replay. Torch-free; picamera2 is replaced by a fake so the camera/recorder call
sequence is checked without a Pi."""

from __future__ import annotations

import json
import runpy
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from conftest import HAS_AV, HAS_ORT, HAS_PIL

from talosaur.onboard import sysinfo
from talosaur.onboard.camera import Picamera2Source, SyntheticSource, yuv420_to_rgb
from talosaur.onboard.recorder import NullRecorder, Picamera2Recorder
from talosaur.onboard.runtime import (
    ModelSpec,
    available_runtimes,
    load_runner,
    preprocess,
    resize_bilinear_u8,
    resize_u8,
)

REPO = Path(__file__).resolve().parents[1]
HAS_ONNX = True
try:
    import onnx  # noqa: F401
except ImportError:
    HAS_ONNX = False
requires_toy = pytest.mark.skipif(
    not (HAS_ONNX and HAS_ORT), reason="needs onnx (to build the toy model) + onnxruntime"
)


@pytest.fixture(scope="module")
def toy_export(tmp_path_factory):
    if not (HAS_ONNX and HAS_ORT):
        pytest.skip("needs onnx + onnxruntime")
    from talosaur.onboard.toy_model import write_toy_export

    return write_toy_export(tmp_path_factory.mktemp("toy"), sizes=[(112, 208), (64, 64)])


def _rgb_to_i420(rgb: np.ndarray, stride: int) -> np.ndarray:
    """BT.601 full-range RGB -> planar I420 laid out like picamera2's (h * 3 // 2, stride) array."""
    h, w, _ = rgb.shape
    f = rgb.astype(np.float64)
    y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    u = -0.168736 * f[..., 0] - 0.331264 * f[..., 1] + 0.5 * f[..., 2] + 128
    v = 0.5 * f[..., 0] - 0.418688 * f[..., 1] - 0.081312 * f[..., 2] + 128
    u = u.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    v = v.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
    out = np.zeros((h * 3 // 2, stride), np.uint8)
    out[:h, :w] = np.clip(np.round(y), 0, 255)
    chroma = np.zeros((2, h // 2, stride // 2))
    chroma[0, :, : w // 2] = u
    chroma[1, :, : w // 2] = v
    out[h:] = np.clip(np.round(chroma), 0, 255).astype(np.uint8).reshape(h // 2, stride)
    return out


def test_yuv420_to_rgb_roundtrip_with_stride():
    rng = np.random.default_rng(0)
    h, w = 112, 208
    rgb = rng.integers(0, 256, (h // 2, w // 2, 3)).repeat(2, 0).repeat(2, 1).astype(np.uint8)  # 2x2 blocks
    rgb = np.clip(rgb, 20, 235)  # stay inside the gamut so clipping does not hide errors
    back = yuv420_to_rgb(_rgb_to_i420(rgb, stride=256), w, h)
    assert back.shape == (h, w, 3) and back.dtype == np.uint8
    assert np.abs(back.astype(int) - rgb.astype(int)).max() <= 3


def test_preprocess_and_resize(monkeypatch):
    img = np.full((720, 1280, 3), 200, np.uint8)
    for with_cv2 in (True, False):
        if not with_cv2:
            monkeypatch.setitem(sys.modules, "cv2", None)  # the Pi usually has no OpenCV
        x = preprocess(img, (112, 208))
        assert x.shape == (3, 112, 208) and x.dtype == np.float32
        assert np.allclose(x, 200 / 255, atol=1e-6)
    grad = np.tile(np.arange(64, dtype=np.uint8)[None, :, None], (32, 1, 3))
    small = resize_bilinear_u8(grad, 16, 32)
    assert small.shape == (16, 32, 3) and np.all(np.diff(small[0, :, 0].astype(int)) >= 0)
    assert resize_bilinear_u8(grad, 32, 64) is grad


def test_resize_u8_antialiases_and_keeps_centre():
    stripes = np.zeros((720, 1280, 3), np.uint8)
    stripes[:, ::2] = 255  # 1-px stripes: pure aliasing bait
    small = resize_u8(stripes, 112, 208)
    assert small.shape == (112, 208, 3)
    assert np.abs(small.astype(int) - 128).max() <= 2  # averages to grey instead of aliasing
    dot = np.zeros((1080, 1920, 3), np.uint8)
    dot[530:550, 950:970] = 255  # centred square
    ys, xs = np.nonzero(resize_u8(dot, 112, 208)[..., 0] > 40)
    assert (ys.mean() + 0.5) / 112 == pytest.approx(0.5, abs=0.01)
    assert (xs.mean() + 0.5) / 208 == pytest.approx(0.5, abs=0.01)
    assert resize_u8(np.full((10, 10), 7, np.uint8), 5, 5).shape == (5, 5)  # grayscale too


def test_sysinfo_probes_degrade_gracefully():
    snap = sysinfo.snapshot()
    assert set(snap) >= {"rss_mb", "peak_rss_mb", "mem_available_mb", "cpu_temp_c", "cpu_freq_mhz"}
    assert sysinfo.peak_rss_mb() > 0
    assert isinstance(sysinfo.cpu_temp_c(), float)  # nan off the Pi
    assert "available" in sysinfo.throttled()
    assert isinstance(sysinfo.is_raspberry_pi(), bool)


def test_synthetic_source_alternates_presence():
    src = SyntheticSource((208, 112), n_frames=90, fps=10)
    frames = []
    while (fr := src.read()) is not None:
        frames.append(fr)
    assert len(frames) == 90 and frames[0].rgb.shape == (112, 208, 3)
    warm = [int((fr.rgb[..., 0] > 150).sum()) for fr in frames]
    assert min(warm[:40]) > 50 and max(warm[40:80]) == 0 and min(warm[80:]) > 50
    assert frames[10].t == pytest.approx(1.0)


# ----------------------------------------------------------------------------- runtime


def _manifest(tmp_path, files):
    entry = {
        "input": {"shape": [1, 3, 112, 208]},
        "outputs": {"frame_logit": {"meaning": ["animal"]}},
        "grid": [7, 13],
        "files": files,
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"models": {"m_112x208": entry}}))
    return entry


def test_model_spec_and_runtime_errors(tmp_path):
    entry = _manifest(tmp_path, {"onnx_fp32": "m.onnx", "ncnn_param": "m.param", "ncnn_bin": "m.bin"})
    spec = ModelSpec.from_manifest(entry)
    assert spec.input_hw == (112, 208) and spec.grid == (7, 13) and spec.frame_outputs == ["animal"]
    assert available_runtimes(entry) == ["ort_fp32", "ncnn_fp16", "ncnn_fp32"]
    with pytest.raises(FileNotFoundError, match="available"):
        load_runner(tmp_path, "m_112x208", "ort_int8")
    with pytest.raises(ValueError, match="unknown runtime"):
        load_runner(tmp_path, "m_112x208", "tensorrt")


@requires_toy
def test_toy_model_outputs_and_detects_warm_object(toy_export):
    r = load_runner(toy_export, "toy_112x208", "ort_fp32", threads=1)
    assert r.spec.input_hw == (112, 208) and r.spec.grid == (7, 13)
    water = np.zeros((112, 208, 3), np.uint8)
    water[..., 1], water[..., 2] = 60, 90
    f, h, e = r(preprocess(water, (112, 208)))
    assert f.shape == (1,) and h.shape == (7, 13) and e.shape == (3,)
    assert h.max() < 0 and f[0] < 0
    fish = water.copy()
    fish[48:64, 160:192] = (230, 150, 40)  # rows 3, cols 10-11 in patch units
    f, h, _ = r(preprocess(fish, (112, 208)))
    assert f[0] > 0 and np.unravel_index(h.argmax(), h.shape)[0] == 3
    assert set(np.argwhere(h > 0)[:, 1]) == {10, 11}


# ----------------------------------------------------------------------------- app


def _app_cfg(export_dir, tmp_path, n_frames=120):
    from talosaur.utils.io import load_yaml

    cfg = load_yaml(REPO / "configs" / "onboard" / "pi5.yaml")
    cfg["model"].update(export_dir=str(export_dir), name="toy_112x208", runtime="ort_fp32", threads=1)
    cfg["source"] = {"kind": "synthetic", "n_frames": n_frames, "fps": 10}
    cfg["backends"] = [{"kind": "jsonl", "path": str(tmp_path / "tele.jsonl")}]
    cfg["stats_every_s"] = 0.0  # also exercise the periodic stats message
    return cfg


@requires_toy
def test_app_runs_the_full_loop_on_synthetic_frames(toy_export, tmp_path):
    from talosaur.onboard.app import run

    summary = run(_app_cfg(toy_export, tmp_path))
    assert summary["frames"] == 120
    assert summary["states"].get("TRACK", 0) > 30 and summary["recordings"] >= 1
    rows = [json.loads(line) for line in (tmp_path / "tele.jsonl").read_text().splitlines()]
    guid = [r for r in rows if r["kind"] == "guidance"]
    assert len(guid) == 120 and any(r["kind"] == "stats" for r in rows)
    tracking = [r for r in guid if r["state"] == "TRACK"]
    # the synthetic fish swims left -> right: the yaw command follows it
    assert tracking[0]["cmd"]["yaw_rate"] < 0 < tracking[25]["cmd"]["yaw_rate"]
    for r in guid:
        assert all(abs(v) <= 1.0 for v in r["cmd"].values())
    assert any("start_recording" in r["events"] for r in guid)


@requires_toy
def test_app_cli_overrides(toy_export, tmp_path, monkeypatch):
    from talosaur.onboard import app
    from talosaur.utils.io import save_yaml

    cfg = _app_cfg("does/not/exist", tmp_path)
    cfg["model"]["name"] = "nope"
    save_yaml(cfg, tmp_path / "cfg.yaml")
    monkeypatch.chdir(tmp_path)
    args = [
        "--config",
        "cfg.yaml",
        "--export-dir",
        str(toy_export),
        "--model",
        "toy_64x64",
        "--runtime",
        "ort_fp32",
    ]
    assert app.main(args + ["--source", "synthetic", "--max-frames", "15", "--no-record"]) == 0
    rows = (tmp_path / "tele.jsonl").read_text().splitlines()
    assert sum(json.loads(r)["kind"] == "guidance" for r in rows) == 15


# ----------------------------------------------------------------------------- fake picamera2


class _FakeCam:
    def __init__(self):
        self.calls, self.cfg = [], None

    def create_video_configuration(self, main, lores=None, controls=None, buffer_count=4):
        return {"main": dict(main), "lores": dict(lores) if lores else None, "controls": dict(controls or {})}

    def configure(self, cfg):
        self.cfg = cfg
        lw, lh = cfg["lores"]["size"]
        cfg["lores"]["size"] = (lw - lw % 64 if lw > 64 else lw, lh)  # pretend libcamera aligned the width
        self.calls.append("configure")

    def camera_configuration(self):
        return self.cfg

    def start(self):
        self.calls.append("start")

    def capture_array(self, name):
        w, h = self.cfg["lores"]["size"]
        stride = (w + 63) // 64 * 64 + 64
        return _rgb_to_i420(np.full((h, w, 3), (30, 90, 120), np.uint8), stride)

    def start_encoder(self, encoder, output):
        self.calls.append(("start_encoder", type(output).__name__))

    def stop_encoder(self):
        self.calls.append("stop_encoder")

    def stop(self):
        self.calls.append("stop")

    def close(self):
        self.calls.append("close")


@pytest.fixture
def fake_picamera2(monkeypatch):
    pkg = types.ModuleType("picamera2")
    enc = types.ModuleType("picamera2.encoders")
    outs = types.ModuleType("picamera2.outputs")
    made = {}

    def picamera2_factory():
        made["cam"] = _FakeCam()
        return made["cam"]

    class H264Encoder:
        def __init__(self, bitrate=None, repeat=True, iperiod=None, framerate=None):
            self.bitrate, self.iperiod, self.framerate = bitrate, iperiod, framerate
            made["encoder"] = self

    class FileOutput:
        def __init__(self, file=None):
            self.file = file

    class CircularOutput(FileOutput):
        def __init__(self, file=None, buffersize=150):
            super().__init__(file)
            self.buffersize, self.fileoutput, self.log = buffersize, None, []

        def start(self):
            self.log.append(("start", self.fileoutput))

        def stop(self):
            self.log.append(("stop", self.fileoutput))

    pkg.Picamera2 = picamera2_factory
    enc.H264Encoder = H264Encoder
    outs.FileOutput, outs.CircularOutput = FileOutput, CircularOutput
    pkg.encoders, pkg.outputs = enc, outs
    for name, mod in (("picamera2", pkg), ("picamera2.encoders", enc), ("picamera2.outputs", outs)):
        monkeypatch.setitem(sys.modules, name, mod)
    return made


def test_picamera2_source_uses_configured_lores_size(fake_picamera2):
    src = Picamera2Source((208, 112), (1280, 720), fps=15, controls={"AfMode": 0}, hdr="off")
    cam = fake_picamera2["cam"]
    assert cam.cfg["lores"]["format"] == "YUV420" and cam.cfg["controls"] == {"FrameRate": 15.0, "AfMode": 0}
    fr = src.read()
    assert fr.rgb.shape == (112, 192, 3)  # the adjusted width, not the requested 208
    assert np.abs(fr.rgb.astype(int) - np.array([30, 90, 120])).max() <= 3
    assert preprocess(fr.rgb, (112, 208)).shape == (3, 112, 208)
    src.close()
    assert cam.calls[-2:] == ["stop", "close"]
    with pytest.raises(ValueError):
        Picamera2Source((208, 112), hdr="sometimes")


def test_picamera2_recorder_preroll_sequence(fake_picamera2, tmp_path):
    Picamera2Source((208, 112), (1280, 720), fps=15)
    cam = fake_picamera2["cam"]
    rec = Picamera2Recorder(cam, tmp_path / "rec", bitrate=4_000_000, preroll_s=5, fps=15)
    enc = fake_picamera2["encoder"]
    assert (enc.iperiod, enc.framerate, enc.bitrate) == (15, 15, 4_000_000)
    assert ("start_encoder", "CircularOutput") in cam.calls  # encoding into the ring buffer already
    assert rec.output.buffersize == 75
    rec.start(1.0)
    rec.start(1.5)  # idempotent
    rec.stop(9.0)
    rec.close()
    starts = [e for e in rec.output.log if e[0] == "start"]
    assert len(starts) == 1 and starts[0][1].endswith(".h264")
    assert cam.calls[-1] == "stop_encoder"


def test_picamera2_recorder_without_preroll(fake_picamera2, tmp_path):
    Picamera2Source((208, 112))
    cam = fake_picamera2["cam"]
    rec = Picamera2Recorder(cam, tmp_path / "rec", preroll_s=0)
    assert not any(isinstance(c, tuple) for c in cam.calls)  # no encoder until recording starts
    rec.start(0.0)
    assert ("start_encoder", "FileOutput") in cam.calls
    rec.stop(1.0)
    assert cam.calls[-1] == "stop_encoder"
    n = NullRecorder()
    n.start(0.0)
    assert n.recording
    n.close()


# ----------------------------------------------------------------------------- benchmark and replay


@requires_toy
def test_benchmark_writes_reports(toy_export, tmp_path):
    from talosaur.onboard.benchmark import main

    out = tmp_path / "bench"
    args = ["--export-dir", str(toy_export), "--models", "toy_64x64", "--runtimes", "ort_fp32", "ort_int8"]
    args += ["--threads", "1", "--iters", "5", "--warmup", "1", "--loop-iters", "5", "--out", str(out)]
    args += [
        "--sustained-s",
        "1",
        "--sample-s",
        "0.3",
        "--sustained-runtime",
        "ort_fp32",
        "--sustained-threads",
        "1",
    ]
    assert main(args) == 0
    res = json.loads(Path(str(out) + ".json").read_text())
    (row,) = res["results"]  # ort_int8 is not in the toy export: skipped
    assert row["runtime"] == "ort_fp32" and row["fps"] > 0 and row["loop_fps"] > 0 and row["iters"] == 5
    assert res["sustained"]["timeline"] and res["sustained"]["iters"] > 5
    md = Path(str(out) + ".md").read_text()
    assert "| toy_64x64 | ort_fp32 | 1 |" in md and "## Sustained run" in md
    assert Path(str(out) + ".csv").read_text().count("\n") == 2


def test_benchmark_markdown_error_row():
    from talosaur.onboard.benchmark import to_markdown

    md = to_markdown(
        [{"model": "m", "runtime": "ncnn_fp16", "threads": 2, "error": "Traceback\nImportError: ncnn"}], {}
    )
    row = md.strip().splitlines()[-1]
    assert "ImportError: ncnn" in row and row.count("|") == md.splitlines()[6].count("|")


def _write_test_video(path: Path, n: int = 40) -> Path:
    import av

    src = SyntheticSource((320, 176), n_frames=n, fps=10)
    with av.open(str(path), mode="w") as c:
        s = c.add_stream("libx264", rate=10)
        s.width, s.height, s.pix_fmt = 320, 176, "yuv420p"
        while (fr := src.read()) is not None:
            for pkt in s.encode(av.VideoFrame.from_ndarray(fr.rgb, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)
    return path


@requires_toy
@pytest.mark.skipif(not (HAS_AV and HAS_PIL), reason="needs PyAV + Pillow")
def test_replay_writes_annotated_video_and_telemetry(toy_export, tmp_path, monkeypatch):
    from talosaur.onboard.camera import VideoFileSource

    video = _write_test_video(tmp_path / "dive.mp4")
    src = VideoFileSource(video, size=(208, 112), keep_full=True, max_fps=5)
    got = []
    while (fr := src.read()) is not None:
        got.append(fr)
    src.close()
    assert len(got) == 20 and got[0].rgb.shape == (112, 208, 3) and got[0].full.shape == (176, 320, 3)

    out = tmp_path / "replay.mp4"
    argv = ["replay.py", "--video", str(video), "--export-dir", str(toy_export), "--model", "toy_112x208"]
    argv += ["--runtime", "ort_fp32", "--threads", "1", "--pi-fps", "10", "--width", "320", "--out", str(out)]
    argv += ["--config", str(REPO / "configs" / "onboard" / "pi5.yaml")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as ex:
        runpy.run_path(str(REPO / "scripts" / "replay.py"), run_name="__main__")
    assert ex.value.code == 0
    tele = [json.loads(line) for line in out.with_suffix(".jsonl").read_text().splitlines()]
    assert len(tele) == 40 and {"SEARCH", "TRACK"} <= {t["state"] for t in tele}
    import av

    with av.open(str(out)) as c:
        frames = list(c.decode(video=0))
    assert len(frames) == 40 and (frames[0].width, frames[0].height) == (320, 176)
