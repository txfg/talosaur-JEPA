"""Talosaur vision: JEPA self-supervised perception for a low-cost AUV.

Sub-packages and their dependency footprint:

* ``talosaur.guidance`` and ``talosaur.onboard`` need only numpy (+ onnxruntime/ncnn at runtime).
  They must never import torch: they run on a Raspberry Pi 5 with 1 GB of RAM.
* ``talosaur.data`` needs the ``data`` extra (Pillow, pandas, pyarrow, scipy, PyAV).
* ``talosaur.models``, ``talosaur.ssl``, ``talosaur.augment``, ``talosaur.monitor`` and
  ``talosaur.eval`` need the ``train`` extra (torch).
* ``talosaur.export`` needs the ``deploy`` extra (onnx, onnxruntime).
"""

__version__ = "0.1.0"
