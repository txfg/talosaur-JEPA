"""Metric logging: JSONL (always, easy to send back for analysis), TensorBoard, optional W&B."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class MetricWriter:
    def __init__(
        self, out_dir: str | Path, tensorboard: bool = True, wandb: dict | None = None, enabled: bool = True
    ):
        self.enabled = enabled
        self.out = Path(out_dir)
        self.tb = None
        self.wb = None
        if not enabled:
            return
        self.out.mkdir(parents=True, exist_ok=True)
        self.jsonl = open(self.out / "metrics.jsonl", "a", buffering=1)
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(str(self.out / "tb"))
            except Exception:  # tensorboard not installed
                self.tb = None
        if wandb and wandb.get("enabled"):
            import wandb as _wandb

            self.wb = _wandb.init(
                project=wandb.get("project", "talosaur-jepa"),
                name=wandb.get("name"),
                config=wandb.get("config"),
                dir=str(self.out),
                resume="allow",
            )

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if not self.enabled or not metrics:
            return
        clean = {k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()}
        self.jsonl.write(json.dumps({"step": step, "time": time.time(), **clean}) + "\n")
        if self.tb is not None:
            for k, v in clean.items():
                if isinstance(v, (int, float)) and v == v:
                    self.tb.add_scalar(k, v, step)
        if self.wb is not None:
            self.wb.log(clean, step=step)

    def close(self) -> None:
        if not self.enabled:
            return
        self.jsonl.close()
        if self.tb is not None:
            self.tb.close()
        if self.wb is not None:
            self.wb.finish()
