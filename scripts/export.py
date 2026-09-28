#!/usr/bin/env python
"""Export encoder + heads for the Pi: ONNX fp32 / int8 (static + dynamic) and ncnn, per input
size, with a parity report on labelled test images.

  python scripts/export.py --encoder runs/ijepa_tiny_224_context_only_s0/encoder_target.pt \
      --heads-dir reports/eval/v1 --heads-backbone jepa_tiny_ctx_target \
      --index data/index/underwater_v1.parquet --root data --sizes 112x112 160x160 224x224 112x208 \
      --out exports/tiny_ctx

Writes exports/<name>/{manifest.json, <model>_{fp32,int8,dyn8}.onnx, ncnn/<model>.ncnn.{param,bin},
parity.md, parity.json}. Copy the folder to the Pi and run the benchmark (docs/PI5.md).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from talosaur.data.index import read_table
from talosaur.export.onnx_export import build_net, check_onnx_matches_torch, export_onnx, manifest_entry
from talosaur.export.parity import labelled_batch, parity_markdown, run_parity
from talosaur.export.quantize import (
    load_images,
    quantize_dynamic_int8,
    quantize_static_int8,
    stratified_calibration_rows,
)
from talosaur.onboard.runtime import ModelSpec, NcnnRunner, OrtRunner
from talosaur.utils.log import get_logger

log = get_logger("export")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", required=True, help="runs/<run>/encoder_target.pt")
    ap.add_argument("--which", default="target", choices=["target", "context"])
    ap.add_argument("--heads-dir", default=None, help="reports/eval/<name> with heads_<backbone>_<HxW>.pt")
    ap.add_argument("--heads-backbone", default=None, help="backbone name used in scripts/eval.py")
    ap.add_argument("--index", default=None, help="curated index for calibration + parity")
    ap.add_argument("--root", default="data")
    ap.add_argument("--sizes", nargs="+", default=["112x112", "160x160", "224x224", "112x208"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="vit_tiny")
    ap.add_argument("--calib", type=int, default=300, help="calibration images (stratified dark/murky/clear)")
    ap.add_argument("--calib-method", default="minmax", choices=["minmax", "percentile", "entropy"])
    ap.add_argument("--parity-images", type=int, default=400)
    ap.add_argument("--no-ncnn", action="store_true")
    ap.add_argument("--tol-auroc", type=float, default=0.01)
    ap.add_argument("--tol-centroid-deg", type=float, default=1.0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = read_table(a.index) if a.index else None
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "encoder": str(a.encoder),
        "models": {},
    }
    parity_all, md = {}, ["# Export parity", ""]
    worst_ok = True
    for size in a.sizes:
        hw = tuple(int(v) for v in size.split("x"))
        heads = None
        if a.heads_dir and a.heads_backbone:
            cand = Path(a.heads_dir) / f"heads_{a.heads_backbone}_{hw[0]}x{hw[1]}.pt"
            heads = cand if cand.exists() else None
            if heads is None:
                log.warning(f"no heads for {size} ({cand}); exporting with untrained heads")
        net = build_net(a.encoder, heads, hw, a.which)
        model = f"{a.name}_{hw[0]}x{hw[1]}"
        fp32 = export_onnx(net, out / f"{model}_fp32.onnx")
        diff = check_onnx_matches_torch(net, fp32)
        log.info(f"{model}: ONNX fp32 matches torch (max abs diff {diff:.1e})")
        files = {"onnx_fp32": fp32.name}
        spec = None
        if df is not None:
            cal = stratified_calibration_rows(df, a.calib)
            imgs = load_images(cal["path"], hw, a.root)
            files["onnx_int8"] = quantize_static_int8(
                fp32, out / f"{model}_int8.onnx", imgs, method=a.calib_method
            ).name
        files["onnx_dyn8"] = quantize_dynamic_int8(fp32, out / f"{model}_dyn8.onnx").name
        if not a.no_ncnn:
            from talosaur.export.ncnn_convert import export_ncnn

            param, binf = export_ncnn(net, out / "ncnn", model, fp16=True)
            files["ncnn_param"] = f"ncnn/{param.name}"
            files["ncnn_bin"] = f"ncnn/{binf.name}"
        entry = manifest_entry(net, files, a.name)
        manifest["models"][model] = entry
        spec = ModelSpec.from_manifest(entry)
        if df is not None:
            test = df[df["split"] == "test"]
            test = test[(test["frame_label"] != -1) | test["boxes"].map(len).gt(0)]
            if len(test):
                x, lab = labelled_batch(test, a.root, hw, net.encoder.patch_size, a.parity_images)
                runners = {
                    "ort_fp32": OrtRunner(out / files["onnx_fp32"], spec),
                    "ort_dyn8": OrtRunner(out / files["onnx_dyn8"], spec),
                }
                if "onnx_int8" in files:
                    runners["ort_int8"] = OrtRunner(out / files["onnx_int8"], spec)
                if "ncnn_param" in files:
                    runners["ncnn_fp16"] = NcnnRunner(
                        out / files["ncnn_param"], out / files["ncnn_bin"], spec, fp16=True
                    )
                sizes = {
                    "ort_fp32": (out / files["onnx_fp32"]).stat().st_size / 1e6,
                    "ort_dyn8": (out / files["onnx_dyn8"]).stat().st_size / 1e6,
                }
                if "onnx_int8" in files:
                    sizes["ort_int8"] = (out / files["onnx_int8"]).stat().st_size / 1e6
                if "ncnn_bin" in files:
                    sizes["ncnn_fp16"] = (out / files["ncnn_bin"]).stat().st_size / 1e6
                res = run_parity(net, runners, x, lab, sizes_mb=sizes)
                parity_all[model] = res
                tol = {"auroc": a.tol_auroc, "centroid_deg": a.tol_centroid_deg}
                table = parity_markdown(res, tol)
                worst_ok &= "| NO |" not in table
                md += [f"## {model} ({res['n_images']} test images)", "", table, ""]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if parity_all:
        (out / "parity.json").write_text(json.dumps(parity_all, indent=1, default=float))
        md += [
            f"Tolerances: |Δ frame AUROC| <= {a.tol_auroc}, Δ centroid error <= {a.tol_centroid_deg}°. If int8 fails them:",
            "try `--calib-method percentile`, use the dynamic model (ort_dyn8), exclude the first/last layers",
            "(talosaur.export.quantize.first_and_last_nodes), or deploy ncnn fp16.",
        ]
        (out / "parity.md").write_text("\n".join(md))
    log.info(f"exported {list(manifest['models'])} -> {out}")
    if not worst_ok:
        log.warning("some quantised models exceed the parity tolerances; see parity.md")
    return 0 if worst_ok else 3


if __name__ == "__main__":
    sys.exit(main())
