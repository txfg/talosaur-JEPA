"""The Raspberry Pi runtime must never need torch. These tests run in a torch-free CI job, and
locally they block `import torch` in a subprocess to prove the onboard modules don't use it."""

import json
import subprocess
import sys
import textwrap

import pytest

ONBOARD_MODULES = [
    "talosaur",
    "talosaur.guidance.backends",
    "talosaur.guidance.camera_model",
    "talosaur.guidance.controller",
    "talosaur.guidance.encounters",
    "talosaur.guidance.heatmap",
    "talosaur.guidance.lights",
    "talosaur.guidance.nav",
    "talosaur.guidance.novelty",
    "talosaur.guidance.pipeline",
    "talosaur.guidance.search",
    "talosaur.guidance.state_machine",
    "talosaur.guidance.tracker",
    "talosaur.sim.camera",
    "talosaur.sim.run",
    "talosaur.sim.vehicle",
    "talosaur.sim.world",
    "talosaur.onboard.app",
    "talosaur.onboard.benchmark",
    "talosaur.onboard.camera",
    "talosaur.onboard.nav_input",
    "talosaur.onboard.recorder",
    "talosaur.onboard.runtime",
    "talosaur.onboard.sysinfo",
    "talosaur.onboard.toy_model",
]


def _run_without_torch(code: str) -> subprocess.CompletedProcess:
    prelude = "import sys\nsys.modules['torch'] = None  # any `import torch` now raises ImportError\n"
    return subprocess.run(
        [sys.executable, "-c", prelude + textwrap.dedent(code)], capture_output=True, text=True
    )


def test_onboard_modules_import_without_torch():
    code = "\n".join(f"import {m}" for m in ONBOARD_MODULES) + "\nprint('ok')\n"
    r = _run_without_torch(code)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("ok")


def test_onboard_loop_and_benchmark_run_without_torch(tmp_path):
    pytest.importorskip("onnx", reason="the toy model is built with the onnx package")
    pytest.importorskip("onnxruntime")
    from talosaur.onboard.toy_model import write_toy_export

    export = write_toy_export(tmp_path / "toy")
    code = f"""
        import json
        from talosaur.onboard.app import run
        from talosaur.onboard.benchmark import _single
        cfg = {{
            "model": {{"export_dir": {str(export)!r}, "name": "toy_112x208", "runtime": "ort_fp32", "threads": 1}},
            "source": {{"kind": "synthetic", "n_frames": 60, "fps": 10}},
            "backends": [{{"kind": "jsonl", "path": {str(tmp_path / "tele.jsonl")!r}}}],
            "encounter_log": {str(tmp_path / "encounters.jsonl")!r},
        }}
        summary = run(cfg)
        bench = _single({{"export_dir": {str(export)!r}, "model": "toy_112x208", "runtime": "ort_fp32",
                          "threads": 1, "iters": 5, "warmup": 1, "loop_iters": 5}})
        assert "torch" not in [m.split(".")[0] for m in sys.modules if sys.modules[m] is not None]
        print(json.dumps({{"summary": summary, "fps": bench["fps"]}}))
    """
    r = _run_without_torch(code)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["summary"]["frames"] == 60 and "TRACK" in out["summary"]["states"]
    assert out["fps"] > 0
