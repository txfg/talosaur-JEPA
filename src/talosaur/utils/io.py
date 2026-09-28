"""Atomic file writes and small serialisation helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml


def _atomic_write(path: str | os.PathLike, writer: Callable[[str], None]) -> Path:
    """Write via a temp file in the same directory, then ``os.replace`` (atomic on POSIX).

    A crash mid-write can never leave a truncated checkpoint/manifest behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        writer(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> Path:
    def w(tmp: str) -> None:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

    return _atomic_write(path, w)


def atomic_write_text(path: str | os.PathLike, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_torch_save(obj: Any, path: str | os.PathLike) -> Path:
    import torch

    return _atomic_write(path, lambda tmp: torch.save(obj, tmp))


def save_json(obj: Any, path: str | os.PathLike, indent: int = 2) -> Path:
    return atomic_write_text(
        path, json.dumps(obj, indent=indent, sort_keys=False, default=_json_default) + "\n"
    )


def load_json(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_yaml(obj: Any, path: str | os.PathLike) -> Path:
    return atomic_write_text(path, yaml.safe_dump(obj, sort_keys=False))


def load_yaml(path: str | os.PathLike) -> Any:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _json_default(o: Any) -> Any:
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")
