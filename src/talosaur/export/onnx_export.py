"""Build the fused onboard network (encoder + heads) and export it to ONNX at fixed input sizes.

The exported graph takes ``image``: float32 (1, 3, H, W), RGB in [0, 1] (normalisation is inside
the graph) and returns ``frame_logit`` (1, K), ``heatmap_logit`` (1, h, w), ``embedding`` (1, D)
and ``patch_tokens`` (1, h*w, D). A ``manifest.json`` next to the models tells the Pi runtime how
to feed and read them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from talosaur.models.heads import FrameHead, HeatmapHead, TalosaurNet
from talosaur.utils.log import get_logger

log = get_logger("export")

INPUT_NAME = "image"
OUTPUT_NAMES = ["frame_logit", "heatmap_logit", "embedding", "patch_tokens"]


def load_encoder(encoder_ckpt: str | Path, which: str = "target"):
    """``encoder_ckpt``: a training run's ``encoder_target.pt``, or ``random:<preset>`` (e.g.
    ``random:vit_tiny``) for an untrained encoder - speed and memory on the Pi do not depend on the
    weights, so the Pi benchmark can run before any training has finished."""
    from talosaur.eval.backbones import JepaBackbone

    if str(encoder_ckpt).startswith("random:"):
        from talosaur.models.vit import build_vit

        torch.manual_seed(0)
        bb = JepaBackbone(encoder=build_vit(str(encoder_ckpt).split(":", 1)[1]))
    else:
        bb = JepaBackbone(str(encoder_ckpt), which=which)
    enc = bb.encoder
    enc.set_attn_impl("math")  # explicit matmul/softmax: ONNX- and PNNX-friendly
    return enc, bb.info.mean, bb.info.std


def build_net(
    encoder_ckpt: str | Path, heads_path: str | Path | None, input_hw, which: str = "target"
) -> TalosaurNet:
    """TalosaurNet from an ``encoder_target.pt`` and an eval ``heads_*.pt`` (random heads if None)."""
    enc, mean, std = load_encoder(encoder_ckpt, which)
    fh, hh = FrameHead(enc.embed_dim), HeatmapHead(enc.embed_dim)
    if heads_path is not None:
        h = torch.load(heads_path, map_location="cpu", weights_only=False)
        if int(h["dim"]) != enc.embed_dim:
            raise ValueError(f"heads were fitted for dim {h['dim']}, encoder has {enc.embed_dim}")
        fh = FrameHead(enc.embed_dim, h["frame_head"]["weight"].shape[0])
        fh.linear.load_state_dict(h["frame_head"])
        hh.linear.load_state_dict(h["heatmap_head"])
        mean, std = tuple(h.get("mean", mean)), tuple(h.get("std", std))
    return TalosaurNet(enc, fh, hh, mean, std, input_hw).eval()


def export_onnx(net: TalosaurNet, path: str | Path, opset: int = 18, dynamo: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.rand(1, 3, *net.input_hw)
    kw = dict(input_names=[INPUT_NAME], output_names=OUTPUT_NAMES, opset_version=opset)
    try:
        torch.onnx.export(net, (x,), str(path), dynamo=dynamo, **kw)
    except Exception as e:
        if not dynamo:
            raise
        log.warning(
            f"dynamo ONNX export failed ({type(e).__name__}); falling back to the TorchScript exporter"
        )
        kw["opset_version"] = min(opset, 17)
        torch.onnx.export(net, (x,), str(path), dynamo=False, **kw)
    _inline_external_data(path)
    return path


def _inline_external_data(path: Path) -> None:
    """The dynamo exporter may store weights in ``<model>.onnx.data``; re-save as one file
    (simpler to copy to the Pi; ViT-Ti/S are far below ONNX's 2 GB single-file limit)."""
    import onnx

    data = Path(str(path) + ".data")
    m = onnx.load(str(path), load_external_data=True)
    onnx.save_model(m, str(path), save_as_external_data=False)
    data.unlink(missing_ok=True)


def ort_session(path: str | Path, threads: int = 0):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def check_onnx_matches_torch(net: TalosaurNet, path: str | Path, n: int = 3, tol: float = 1e-3) -> float:
    """Max abs difference between PyTorch and ONNX Runtime fp32 outputs on random inputs."""
    sess = ort_session(path)
    worst = 0.0
    for i in range(n):
        x = torch.rand(1, 3, *net.input_hw, generator=torch.Generator().manual_seed(i))
        with torch.no_grad():
            ref = [t.numpy() for t in net(x)]
        out = sess.run(OUTPUT_NAMES, {INPUT_NAME: x.numpy()})
        worst = max(worst, *(float(np.abs(a - b).max()) for a, b in zip(out, ref)))
    if worst > tol:
        raise AssertionError(f"ONNX output differs from PyTorch by {worst:.2e} (> {tol})")
    return worst


def manifest_entry(net: TalosaurNet, files: dict[str, str], backbone: str, frame_outputs=("animal",)) -> dict:
    return {
        "input": {
            "name": INPUT_NAME,
            "shape": [1, 3, *net.input_hw],
            "layout": "NCHW",
            "dtype": "float32",
            "range": "RGB in [0, 1]",
        },
        "outputs": {
            "frame_logit": {"shape": [1, net.frame_head.linear.out_features], "meaning": list(frame_outputs)},
            "heatmap_logit": {
                "shape": [1, *net.grid_hw],
                "meaning": "per-patch animal-likeness (sigmoid -> probability)",
            },
            "embedding": {
                "shape": [1, net.encoder.embed_dim],
                "meaning": "mean-pooled features (novelty detection)",
            },
            "patch_tokens": {
                "shape": [1, net.grid_hw[0] * net.grid_hw[1], net.encoder.embed_dim],
                "meaning": "final patch features, row-major over the grid (recognising animals already filmed)",
            },
        },
        "grid": list(net.grid_hw),
        "patch": net.encoder.patch_size,
        "backbone": backbone,
        "files": files,
    }
