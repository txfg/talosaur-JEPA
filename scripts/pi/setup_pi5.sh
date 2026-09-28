#!/usr/bin/env bash
# Set up the Talosaur onboard runtime on a Raspberry Pi 5 (1 GB) running Raspberry Pi OS Lite (64-bit).
# Usage: bash scripts/pi/setup_pi5.sh [--ncnn]
set -euo pipefail
WITH_NCNN=0; [[ "${1:-}" == "--ncnn" ]] && WITH_NCNN=1

echo "== system packages (picamera2 comes from apt, not pip)"
sudo apt update
sudo apt install -y python3-picamera2 python3-venv python3-pip ffmpeg

echo "== venv that can see the apt-installed picamera2"
python3 -m venv --system-site-packages ~/talosaur-venv
source ~/talosaur-venv/bin/activate
pip install --upgrade pip
pip install -e ".[pi]"          # numpy, pyyaml, onnxruntime; NO torch on the Pi
if [[ $WITH_NCNN == 1 ]]; then pip install ncnn; fi

echo "== sanity checks"
python - <<'PY'
import sys
sys.modules["torch"] = None     # prove the onboard code never needs torch
import numpy, onnxruntime
import talosaur.onboard.app, talosaur.guidance.pipeline
print("onnxruntime", onnxruntime.__version__, "| numpy", numpy.__version__, "| onboard imports OK without torch")
PY
free -m
swapon --show || true
echo "Tip: keep Raspberry Pi OS Lite (no desktop) on the 1 GB board; zram swap is the default on recent images."
echo "Next: copy exports/<name>/ from the training box, then run"
echo "  python -m talosaur.onboard.benchmark --export-dir exports/<name>"
