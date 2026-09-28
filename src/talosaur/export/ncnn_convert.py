"""PyTorch -> PNNX -> ncnn conversion (fp16 storage/arithmetic by default on the Pi's A76 cores).

ncnn int8 needs the ``ncnn2table`` / ``ncnn2int8`` command-line tools, which are built from the
ncnn source tree (they are not in the ``ncnn`` pip wheel). :func:`write_ncnn_calibration_list`
prepares the calibration image list for ``ncnn2table``; see docs/PI5.md.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import torch

from talosaur.utils.log import get_logger

log = get_logger("ncnn")


def export_ncnn(net, out_dir: str | Path, name: str, fp16: bool = True) -> tuple[Path, Path]:
    import pnnx

    from talosaur.models.vit import with_torch_mha

    net = with_torch_mha(net)  # PNNX -> ncnn fused MultiHeadAttention (see TorchMHA)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    x = torch.rand(1, 3, *net.input_hw)
    work = out_dir / f"_{name}_pnnx"
    work.mkdir(exist_ok=True)
    param, binf = out_dir / f"{name}.ncnn.param", out_dir / f"{name}.ncnn.bin"
    cwd = os.getcwd()
    try:  # pnnx writes its intermediate files next to the .pt path
        os.chdir(work)
        pnnx.export(net, f"{name}.pt", inputs=(x,), ncnnparam=str(param), ncnnbin=str(binf), fp16=fp16)
    finally:
        os.chdir(cwd)
        shutil.rmtree(work, ignore_errors=True)
    text = param.read_text()
    if "MultiHeadAttention" not in text:
        log.warning("PNNX did not fuse attention into ncnn MultiHeadAttention layers; ncnn will be slower")
    return param, binf


def ncnn_infer(
    param: str | Path, binf: str | Path, x: np.ndarray, threads: int = 4, fp16: bool = True
) -> list[np.ndarray]:
    """Run one (3, H, W) float32 image; returns [frame_logit, heatmap_logit, embedding]."""
    import ncnn

    net = ncnn.Net()
    net.opt.num_threads = threads
    net.opt.use_fp16_packed = fp16
    net.opt.use_fp16_storage = fp16
    net.opt.use_fp16_arithmetic = fp16
    net.load_param(str(param))
    net.load_model(str(binf))
    ex = net.create_extractor()
    # ncnn.Mat wraps the numpy buffer without copying: keep ``arr`` alive until extraction is
    # done (passing a temporary such as ``x.astype(...)`` would leave ncnn reading freed memory).
    arr = np.ascontiguousarray(x, dtype=np.float32)
    mat = ncnn.Mat(arr)
    ex.input("in0", mat)
    outs = []
    for name in ("out0", "out1", "out2"):
        ret, m = ex.extract(name)
        if ret != 0:
            raise RuntimeError(f"ncnn extract {name} failed ({ret})")
        outs.append(np.array(m, copy=True))
    del mat, arr
    return outs


def write_ncnn_calibration_list(images: np.ndarray, out_dir: str | Path) -> Path:
    """Save calibration images as .npy files + a list file for ``ncnn2table`` (type=npy)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, im in enumerate(images):
        p = out_dir / f"calib_{i:04d}.npy"
        np.save(p, im.astype(np.float32))
        paths.append(str(p.resolve()))
    lst = out_dir / "calib_list.txt"
    lst.write_text("\n".join(paths) + "\n")
    return lst
