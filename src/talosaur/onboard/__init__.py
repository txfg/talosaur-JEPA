"""Raspberry Pi 5 onboard runtime and benchmark. Must never import torch."""

import os

# The guidance maths uses tiny matrices; OpenBLAS worker threads would only compete with the
# inference threads for the Pi's four cores. Takes effect when numpy has not been imported yet,
# which is the case for `python -m talosaur.onboard.app` and `python -m talosaur.onboard.benchmark`.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
