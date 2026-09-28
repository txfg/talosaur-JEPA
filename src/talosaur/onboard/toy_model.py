"""A tiny colour-contrast "model" in the exported-model format, for dry runs and tests.

It is *not* a detector. Per 16x16 patch it scores warm colours against blue-green water
(``scale * (mean(R - 0.5 B) - offset)``) and pools that into the same four outputs as a real
export (frame logit, heatmap logit, embedding, patch tokens = mean RGB per patch, so objects of
different colours count as different animals). Uses:

* check the Pi's camera -> model -> guidance -> recording -> UDP chain before a trained model
  exists (in the pool, move an orange object in front of the camera and watch the vehicle
  commands and recording events);
* the torch-free onboard tests.

Building the file needs the ``onnx`` package (``pip install onnx``); running it needs only
onnxruntime.

  python -m talosaur.onboard.toy_model --out exports/toy --sizes 112x208 160x160
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def build_toy_onnx(
    path: str | Path,
    input_hw: tuple[int, int] = (112, 208),
    patch: int = 16,
    weights: tuple[float, float, float] = (1.0, 0.0, -0.5),
    scale: float = 10.0,
    offset: float = 0.15,
) -> Path:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    H, W = input_hw
    if H % patch or W % patch:
        raise ValueError(f"input {input_hw} must be a multiple of the patch size {patch}")
    gh, gw = H // patch, W // patch
    w = np.zeros((1, 3, patch, patch), np.float32)
    for c in range(3):
        w[0, c] = scale * weights[c] / (patch * patch)
    inits = [
        numpy_helper.from_array(w, "w"),
        numpy_helper.from_array(np.array([-scale * offset], np.float32), "b"),
        numpy_helper.from_array(np.array([1, gh, gw], np.int64), "heat_shape"),
        numpy_helper.from_array(np.array([1, 3, gh * gw], np.int64), "tok_shape"),
    ]
    nodes = [
        helper.make_node(
            "Conv", ["image", "w", "b"], ["heat4"], kernel_shape=[patch, patch], strides=[patch, patch]
        ),
        helper.make_node("ReduceMax", ["heat4"], ["frame_logit"], axes=[2, 3], keepdims=0),
        helper.make_node("Reshape", ["heat4", "heat_shape"], ["heatmap_logit"]),
        helper.make_node("GlobalAveragePool", ["image"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["embedding"], axis=1),
        # "patch tokens": mean RGB of every patch, (1, N, 3) row-major like the real export
        helper.make_node(
            "AveragePool", ["image"], ["patch_rgb"], kernel_shape=[patch, patch], strides=[patch, patch]
        ),
        helper.make_node("Reshape", ["patch_rgb", "tok_shape"], ["tok_cn"]),
        helper.make_node("Transpose", ["tok_cn"], ["patch_tokens"], perm=[0, 2, 1]),
    ]
    graph = helper.make_graph(
        nodes,
        "talosaur_toy",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, H, W])],
        [
            helper.make_tensor_value_info("frame_logit", TensorProto.FLOAT, [1, 1]),
            helper.make_tensor_value_info("heatmap_logit", TensorProto.FLOAT, [1, gh, gw]),
            helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 3]),
            helper.make_tensor_value_info("patch_tokens", TensorProto.FLOAT, [1, gh * gw, 3]),
        ],
        inits,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="talosaur-toy"
    )
    model.ir_version = 8  # readable by every onnxruntime >= 1.14
    onnx.checker.check_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path


def toy_manifest_entry(input_hw: tuple[int, int], files: dict[str, str], patch: int = 16) -> dict:
    """Same layout as talosaur.export.onnx_export.manifest_entry."""
    H, W = input_hw
    return {
        "input": {
            "name": "image",
            "shape": [1, 3, H, W],
            "layout": "NCHW",
            "dtype": "float32",
            "range": "RGB in [0, 1]",
        },
        "outputs": {
            "frame_logit": {"shape": [1, 1], "meaning": ["animal"]},
            "heatmap_logit": {
                "shape": [1, H // patch, W // patch],
                "meaning": "toy: warm-colour contrast per patch",
            },
            "embedding": {"shape": [1, 3], "meaning": "toy: mean RGB"},
            "patch_tokens": {
                "shape": [1, (H // patch) * (W // patch), 3],
                "meaning": "toy: mean RGB per patch",
            },
        },
        "grid": [H // patch, W // patch],
        "patch": patch,
        "backbone": "toy_colour",
        "files": files,
    }


def write_toy_export(out_dir: str | Path, sizes=((112, 208),), name: str = "toy") -> Path:
    """Write ``<out_dir>/manifest.json`` + one fp32 ONNX per size; returns ``out_dir``."""
    out = Path(out_dir)
    manifest = {
        "created": "toy",
        "encoder": "toy_colour (no encoder; see talosaur.onboard.toy_model)",
        "models": {},
    }
    for hw in sizes:
        model = f"{name}_{hw[0]}x{hw[1]}"
        onnx_path = build_toy_onnx(out / f"{model}_fp32.onnx", tuple(hw))
        manifest["models"][model] = toy_manifest_entry(tuple(hw), {"onnx_fp32": onnx_path.name})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="exports/toy")
    ap.add_argument("--sizes", nargs="+", default=["112x208"])
    a = ap.parse_args(argv)
    sizes = [tuple(int(v) for v in s.split("x")) for s in a.sizes]
    out = write_toy_export(a.out, sizes)
    print(f"wrote {out}/manifest.json ({', '.join(a.sizes)}); run with --runtime ort_fp32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
