"""M5 tests: ONNX export parity, int8 quantisation, ncnn conversion, parity report, export script."""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import requires_onnx

torch = pytest.importorskip("torch")


def _net(hw=(48, 96), depth=2):
    from talosaur.models.heads import FrameHead, HeatmapHead, TalosaurNet
    from talosaur.models.vit import build_vit

    torch.manual_seed(0)
    enc = build_vit("vit_tiny", depth=depth).eval()
    enc.set_attn_impl("math")
    return TalosaurNet(
        enc, FrameHead(192), HeatmapHead(192), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), hw
    ).eval()


@requires_onnx
def test_onnx_export_matches_torch_single_file(tmp_path):
    from talosaur.export.onnx_export import check_onnx_matches_torch, export_onnx

    net = _net()
    p = export_onnx(net, tmp_path / "m.onnx")
    assert p.exists() and not (tmp_path / "m.onnx.data").exists()
    assert check_onnx_matches_torch(net, p) < 1e-3


@requires_onnx
def test_quantize_static_and_dynamic(tmp_path):
    from talosaur.export.onnx_export import export_onnx
    from talosaur.export.quantize import quantize_dynamic_int8, quantize_static_int8
    from talosaur.onboard.runtime import ModelSpec, OrtRunner

    net = _net()
    fp32 = export_onnx(net, tmp_path / "m.onnx")
    rng = np.random.default_rng(0)
    calib = rng.random((8, 3, 48, 96), dtype=np.float32)
    s8 = quantize_static_int8(fp32, tmp_path / "m_int8.onnx", calib)
    d8 = quantize_dynamic_int8(fp32, tmp_path / "m_dyn8.onnx")
    import onnx

    ops = {n.op_type for n in onnx.load(str(s8)).graph.node}
    assert "QuantizeLinear" in ops and "DequantizeLinear" in ops
    assert "LayerNormalization" in ops  # stays in float
    spec = ModelSpec((48, 96), (3, 6), ["animal"])
    x = calib[0]
    with torch.no_grad():
        ref = net(torch.from_numpy(x[None]))[1].numpy()[0]
    for path in (s8, d8):
        f, h, e = OrtRunner(path, spec, threads=1)(x)
        assert f.shape == (1,) and h.shape == (3, 6) and e.shape == (192,)
        assert np.corrcoef(h.ravel(), ref.ravel())[0, 1] > 0.9


def test_ncnn_conversion_with_fused_attention(tmp_path):
    pytest.importorskip("pnnx")
    pytest.importorskip("ncnn")
    from talosaur.export.ncnn_convert import export_ncnn
    from talosaur.onboard.runtime import ModelSpec, NcnnRunner

    net = _net((48, 96), depth=2)
    param, binf = export_ncnn(net, tmp_path, "m", fp16=True)
    assert param.read_text().count("MultiHeadAttention") == 2
    x = np.random.default_rng(1).random((3, 48, 96), dtype=np.float32)
    with torch.no_grad():
        ref = [t.numpy()[0] for t in net(torch.from_numpy(x[None]))]
    out = NcnnRunner(param, binf, ModelSpec((48, 96), (3, 6), ["animal"]), threads=1, fp16=False)(x)
    for a, b in zip(out, ref):
        assert np.abs(a.reshape(b.shape) - b).max() / (np.abs(b).max() + 1e-9) < 5e-3


def test_torch_mha_swap_is_exact():
    from talosaur.models.vit import with_torch_mha

    net = _net()
    alt = with_torch_mha(net)
    x = torch.rand(2, 3, 48, 96)
    with torch.no_grad():
        for a, b in zip(net(x), alt(x)):
            assert torch.allclose(a, b, atol=1e-5)


@requires_onnx
@pytest.mark.slow
def test_export_script_end_to_end(synthetic_index, tmp_path):
    pytest.importorskip("pnnx")
    from talosaur.models.vit import build_vit

    root, idx = synthetic_index
    enc = build_vit("vit_tiny", depth=2)
    ck = tmp_path / "enc.pt"
    torch.save(
        {
            "model_config": {"name": "vit_tiny", **enc.cfg.to_dict()},
            "target_encoder": enc.state_dict(),
            "encoder": enc.state_dict(),
        },
        ck,
    )
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "export_script", pathlib.Path(__file__).parents[1] / "scripts" / "export.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "exp"
    code = mod.main(
        [
            "--encoder",
            str(ck),
            "--index",
            str(idx),
            "--root",
            str(root),
            "--sizes",
            "48x96",
            "--out",
            str(out),
            "--calib",
            "16",
            "--parity-images",
            "16",
            "--ncnn-calib",
        ]
    )
    assert code in (0, 3)
    calib = (out / "ncnn" / "calib_vit_tiny_48x96" / "calib_list.txt").read_text().split()
    assert len(calib) == 16 and np.load(calib[0]).shape == (3, 48, 96)
    man = json.loads((out / "manifest.json").read_text())
    entry = man["models"]["vit_tiny_48x96"]
    assert entry["grid"] == [3, 6] and entry["input"]["shape"] == [1, 3, 48, 96]
    for k in ("onnx_fp32", "onnx_int8", "onnx_dyn8", "ncnn_param", "ncnn_bin"):
        assert (out / entry["files"][k]).exists()
    assert "ort_int8" in (out / "parity.md").read_text()
    from talosaur.onboard.runtime import load_runner

    r = load_runner(out, "vit_tiny_48x96", "ort_int8", threads=1)
    f, h, e = r(np.zeros((3, 48, 96), dtype=np.float32))
    assert h.shape == (3, 6)


def _export_script():
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "export_script", pathlib.Path(__file__).parents[1] / "scripts" / "export.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@requires_onnx
def test_random_encoder_benchmark_export(tmp_path):
    """The Pi benchmark must not wait for training: full-depth random ViT-Ti, int8 on noise."""
    out = tmp_path / "bench"
    code = _export_script().main(
        [
            "--encoder",
            "random:vit_tiny",
            "--calib-random",
            "4",
            "--sizes",
            "32x64",
            "--no-ncnn",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    man = json.loads((out / "manifest.json").read_text())
    assert "benchmark-only" in man["note"]
    entry = man["models"]["vit_tiny_32x64"]
    assert set(entry["files"]) == {"onnx_fp32", "onnx_int8", "onnx_dyn8"}
    from talosaur.onboard.runtime import load_runner

    f, h, e = load_runner(out, "vit_tiny_32x64", "ort_int8", threads=1)(np.zeros((3, 32, 64), np.float32))
    assert h.shape == (2, 4) and e.shape == (192,)
