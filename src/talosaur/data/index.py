"""Parquet read/write for record tables and the curated index."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def write_table(df, path: str | os.PathLike) -> Path:
    """Atomically write a DataFrame to Parquet (list columns such as ``boxes`` are preserved)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    if "phash" in df.columns:  # uint64 -> int64 bit pattern (Parquet has no uint64 in all readers)
        df["phash"] = np.asarray(df["phash"].to_numpy(), dtype=np.uint64).view(np.int64)
    table = pa.Table.from_pandas(df, preserve_index=False)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, path)
    return path


def read_table(path: str | os.PathLike):
    import pandas as pd

    df = pd.read_parquet(path)
    if "phash" in df.columns:
        df["phash"] = np.asarray(df["phash"].to_numpy(), dtype=np.int64).view(np.uint64)
    for col in ("boxes", "box_labels", "box_is_animal"):
        if col in df.columns:
            df[col] = df[col].map(_to_list)
    return df


def _to_list(v):
    if v is None:
        return []
    if isinstance(v, np.ndarray):
        return [(_to_list(x) if isinstance(x, np.ndarray) else x) for x in v.tolist()]
    return list(v)


def concat_record_tables(paths: list[str | os.PathLike]):
    import pandas as pd

    frames = [read_table(p) for p in paths]
    frames = [f for f in frames if len(f)]
    if not frames:
        raise ValueError(f"no records found in {paths}")
    df = pd.concat(frames, ignore_index=True)
    dup = df["image_id"].duplicated()
    if dup.any():
        raise ValueError(f"duplicate image_id across sources, e.g. {df.loc[dup, 'image_id'].iloc[0]}")
    return df
