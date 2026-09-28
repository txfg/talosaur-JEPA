"""Model runners for the Raspberry Pi 5 (numpy + onnxruntime or ncnn; never torch).

Every runner maps one RGB image (3, H, W) float32 in [0, 1] to
``(frame_logit (K,), heatmap_logit (h, w), embedding (D,))``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ModelSpec:
    input_hw: tuple[int, int]
    grid: tuple[int, int]
    frame_outputs: list[str]
    backbone: str = ""

    @classmethod
    def from_manifest(cls, entry: dict) -> ModelSpec:
        shape = entry["input"]["shape"]
        return cls(
            (int(shape[2]), int(shape[3])),
            tuple(entry["grid"]),
            list(entry["outputs"]["frame_logit"]["meaning"]),
            entry.get("backbone", ""),
        )


class OrtRunner:
    def __init__(self, onnx_path: str | Path, spec: ModelSpec, threads: int = 4):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = max(1, threads)
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.enable_mem_pattern = True
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.spec = spec

    def __call__(self, x: np.ndarray):
        f, h, e = self.sess.run(self.output_names, {self.input_name: x[None].astype(np.float32, copy=False)})
        return f[0], h[0], e[0]


class NcnnRunner:
    def __init__(
        self, param: str | Path, binf: str | Path, spec: ModelSpec, threads: int = 4, fp16: bool = True
    ):
        import ncnn

        self.ncnn = ncnn
        self.net = ncnn.Net()
        self.net.opt.num_threads = max(1, threads)
        self.net.opt.use_fp16_packed = fp16
        self.net.opt.use_fp16_storage = fp16
        self.net.opt.use_fp16_arithmetic = fp16
        self.net.load_param(str(param))
        self.net.load_model(str(binf))
        self.spec = spec

    def __call__(self, x: np.ndarray):
        arr = np.ascontiguousarray(x, dtype=np.float32)  # must outlive the extractor (no copy in ncnn.Mat)
        mat = self.ncnn.Mat(arr)
        ex = self.net.create_extractor()
        ex.input("in0", mat)
        outs = []
        for name in ("out0", "out1", "out2"):
            ret, m = ex.extract(name)
            if ret != 0:
                raise RuntimeError(f"ncnn extract {name} failed ({ret})")
            outs.append(np.array(m, copy=True))
        del mat, arr
        f, h, e = outs
        return f.reshape(-1), h.reshape(self.spec.grid), e.reshape(-1)


def load_runner(export_dir: str | Path, model: str, runtime: str = "ort_int8", threads: int = 4):
    """``model`` is a manifest key such as ``vit_tiny_112x208``; ``runtime`` one of
    ``ort_fp32``, ``ort_int8``, ``ort_dyn8``, ``ncnn_fp16``, ``ncnn_fp32``, ``ncnn_int8``."""
    export_dir = Path(export_dir)
    manifest = json.loads((export_dir / "manifest.json").read_text())
    entry = manifest["models"][model]
    spec = ModelSpec.from_manifest(entry)
    files = entry["files"]
    if runtime not in RUNTIME_FILES:
        raise ValueError(f"unknown runtime {runtime!r}; choose from {sorted(RUNTIME_FILES)}")
    missing = [k for k in RUNTIME_FILES[runtime] if k not in files]
    if missing:
        have = sorted(r for r, keys in RUNTIME_FILES.items() if all(k in files for k in keys))
        raise FileNotFoundError(
            f"{model}: the export has no {missing[0]} for runtime {runtime!r}; available: {have}"
        )
    if runtime.startswith("ort"):
        return OrtRunner(export_dir / files[RUNTIME_FILES[runtime][0]], spec, threads)
    param, binf = RUNTIME_FILES[runtime]
    return NcnnRunner(
        export_dir / files[param], export_dir / files[binf], spec, threads, fp16=runtime == "ncnn_fp16"
    )


# manifest file keys each runtime needs
RUNTIME_FILES = {
    "ort_fp32": ("onnx_fp32",),
    "ort_int8": ("onnx_int8",),
    "ort_dyn8": ("onnx_dyn8",),
    "ncnn_fp16": ("ncnn_param", "ncnn_bin"),
    "ncnn_fp32": ("ncnn_param", "ncnn_bin"),
    "ncnn_int8": ("ncnn_int8_param", "ncnn_int8_bin"),
}


def available_runtimes(entry: dict) -> list[str]:
    """Runtimes a manifest entry has files for."""
    return [r for r, keys in RUNTIME_FILES.items() if all(k in entry["files"] for k in keys)]


def resize_bilinear_u8(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Plain bilinear resize (pixel-centre aligned, rounded) - for upsampling or small changes."""
    H, W = img.shape[:2]
    if (H, W) == (h, w):
        return img
    ys = (np.arange(h) + 0.5) * H / h - 0.5
    xs = (np.arange(w) + 0.5) * W / w - 0.5
    y0 = np.clip(np.floor(ys).astype(np.int32), 0, H - 1)
    x0 = np.clip(np.floor(xs).astype(np.int32), 0, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    wy = np.clip(ys - y0, 0, 1)[:, None, None].astype(np.float32)
    wx = np.clip(xs - x0, 0, 1)[None, :, None].astype(np.float32)
    im = img.astype(np.float32)
    if im.ndim == 2:
        im = im[..., None]
    top = im[y0][:, x0] * (1 - wx) + im[y0][:, x1] * wx
    bot = im[y1][:, x0] * (1 - wx) + im[y1][:, x1] * wx
    out = np.clip(top * (1 - wy) + bot * wy + 0.5, 0, 255).astype(np.uint8)
    return out.reshape((h, w) + img.shape[2:])


def resize_u8(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Anti-aliased resize without OpenCV: integer box (area) reduction, then bilinear.

    Plain bilinear aliases badly when shrinking a 1280x720 frame to 208x112; the Pi camera's
    ISP (and OpenCV's INTER_AREA, PIL) filter instead, so replayed video should too. The box
    stage trims at most (factor - 1) pixels, split evenly between the edges so the image centre
    (and thus every bearing) stays put.
    """
    H, W = img.shape[:2]
    fy, fx = max(1, H // h), max(1, W // w)
    if fy > 1 or fx > 1:
        Hc, Wc = H // fy * fy, W // fx * fx
        oy, ox = (H - Hc) // 2, (W - Wc) // 2
        crop = img[oy : oy + Hc, ox : ox + Wc].astype(np.float32)
        crop = crop.reshape(Hc // fy, fy, Wc // fx, fx, *img.shape[2:]).mean(axis=(1, 3))
        img = np.clip(crop + 0.5, 0, 255).astype(np.uint8)
    return resize_bilinear_u8(img, h, w)


def preprocess(rgb: np.ndarray, input_hw: tuple[int, int]) -> np.ndarray:
    """uint8 RGB (H, W, 3) -> float32 (3, h, w) in [0, 1] at the model's input size."""
    h, w = input_hw
    if rgb.shape[:2] != (h, w):
        try:
            import cv2

            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        except ImportError:
            rgb = resize_u8(rgb, h, w)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1), dtype=np.float32) * (1.0 / 255.0)
