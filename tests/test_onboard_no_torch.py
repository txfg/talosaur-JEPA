"""The Raspberry Pi runtime must never need torch. These tests run in a torch-free CI job, and
locally they block `import torch` in a subprocess to prove the onboard modules don't use it."""

import subprocess
import sys
import textwrap

ONBOARD_MODULES = ["talosaur"]


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
