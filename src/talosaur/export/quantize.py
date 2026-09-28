"""int8 quantisation with ONNX Runtime.

* **static** (default): QDQ format, S8S8 (ORT's recommended default), per-channel weights,
  only MatMul/Gemm/Conv quantised (LayerNorm, Softmax, GELU and adds stay in float: ViTs are
  sensitive there). Calibration uses underwater images **stratified over dark / murky / clear**,
  because dark frames have much smaller activations than bright ones.
* **dynamic**: weights int8, activations quantised on the fly (ORT's recommendation for
  transformers); the fallback if static calibration hurts accuracy.

On the Pi 5 (Cortex-A76) ORT's int8 kernels use the ARMv8.2 dot-product instructions; the i8mm
(SMMLA) kernels need ARMv8.6 and are not available on this CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from talosaur.utils.log import get_logger

log = get_logger("quant")


def stratified_calibration_rows(df, n: int = 300, split: str = "train", seed: int = 0):
    """Pick ``n`` rows spread evenly over the dark / murky / clear / other slices (and sources)."""
    sub = df[df["split"] == split] if (df["split"] == split).any() else df
    slices = [s for s in ("dark", "murky", "clear", "other") if (sub["slice"] == s).any()]
    per = max(1, n // max(1, len(slices)))
    parts = []
    for s in slices:
        g = sub[sub["slice"] == s]
        parts.append(g.sample(min(per, len(g)), random_state=seed))
    import pandas as pd

    out = pd.concat(parts) if parts else sub.iloc[:0]
    if len(out) < n:
        rest = sub.drop(out.index, errors="ignore")
        out = pd.concat([out, rest.sample(min(n - len(out), len(rest)), random_state=seed)])
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_images(paths, hw, root: str | Path = ".") -> np.ndarray:
    """(N, 3, H, W) float32 RGB in [0, 1], resized exactly like the Pi runtime does."""
    from PIL import Image

    H, W = hw
    out = []
    for p in paths:
        with Image.open(Path(root) / p) as im:
            im = im.convert("RGB").resize((W, H), Image.Resampling.BILINEAR)
            out.append(np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    return np.stack(out)


class ArrayCalibrationReader:
    """onnxruntime CalibrationDataReader over an in-memory (N, 3, H, W) float32 array."""

    def __init__(self, images: np.ndarray, input_name: str = "image"):
        self.images, self.input_name, self.i = images, input_name, 0

    def get_next(self):
        if self.i >= len(self.images):
            return None
        x = self.images[self.i : self.i + 1]
        self.i += 1
        return {self.input_name: x}

    def rewind(self) -> None:
        self.i = 0


def _make_reader(images, input_name):
    from onnxruntime.quantization import CalibrationDataReader

    class _R(ArrayCalibrationReader, CalibrationDataReader):
        pass

    return _R(images, input_name)


def quantize_static_int8(
    fp32_path: str | Path,
    out_path: str | Path,
    calib_images: np.ndarray,
    method: str = "minmax",
    per_channel: bool = True,
    op_types: tuple[str, ...] = ("MatMul", "Gemm", "Conv"),
    preprocess: bool = True,
    exclude_nodes: list[str] | None = None,
) -> Path:
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quant_pre_process,
        quantize_static,
    )

    fp32_path, out_path = Path(fp32_path), Path(out_path)
    src = strip_value_info(fp32_path, out_path.with_name(out_path.stem + "_clean.onnx"))
    if preprocess:
        pre = out_path.with_name(out_path.stem + "_pre.onnx")
        quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
        Path(src).unlink(missing_ok=True)
        src = pre
    methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    extra = {"CalibMovingAverage": True} if method == "minmax" else {}
    if method == "percentile":
        extra["CalibPercentile"] = 99.99
    quantize_static(
        str(src),
        str(out_path),
        _make_reader(calib_images, "image"),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        op_types_to_quantize=list(op_types),
        nodes_to_exclude=exclude_nodes or [],
        calibrate_method=methods[method],
        extra_options=extra,
    )
    Path(src).unlink(missing_ok=True)
    return out_path


def strip_value_info(src: str | Path, dst: str | Path) -> Path:
    """Drop intermediate shape annotations (the dynamo exporter can leave some that ORT's
    quantiser's shape inference rejects); they are re-inferred."""
    import onnx

    m = onnx.load(str(src))
    m.graph.ClearField("value_info")
    onnx.save(m, str(dst))
    return Path(dst)


def quantize_dynamic_int8(
    fp32_path: str | Path, out_path: str | Path, op_types: tuple[str, ...] = ("MatMul", "Gemm")
) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_path = Path(out_path)
    clean = strip_value_info(fp32_path, out_path.with_name(out_path.stem + "_clean.onnx"))
    try:
        quantize_dynamic(
            str(clean),
            str(out_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=list(op_types),
            per_channel=True,
        )
    finally:
        clean.unlink(missing_ok=True)
    return out_path


def first_and_last_nodes(path: str | Path, n: int = 1) -> list[str]:
    """Names of the patch-embedding conv and the head Gemm/MatMuls (candidates to keep in fp32)."""
    import onnx

    g = onnx.load(str(path)).graph
    conv = [nd.name for nd in g.node if nd.op_type == "Conv"][:n]
    heads = [nd.name for nd in g.node if nd.op_type in ("Gemm", "MatMul")][-2 * n :]
    return conv + heads
