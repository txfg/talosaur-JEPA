"""Shared machinery for dataset sources: license gate, source registry YAML, downloads, manifests.

Data root layout (``--root``)::

    raw/<source>/...            downloaded archives (safe to delete after ingest)
    images/<source>/...         resized JPEGs used for training / evaluation
    masks/<source>/...          binary animal masks (where available)
    interim/<source>/records.parquet + manifest.json
    index/<name>.parquet        curated index (scripts/data/build_dataset.py)
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from talosaur.data.licenses import license_info
from talosaur.utils.io import load_yaml, save_json
from talosaur.utils.log import get_logger

log = get_logger("data.sources")

USER_AGENT = "talosaur-data/0.1 (+https://github.com/txfg/talosaur-JEPA)"


class LicenseNotAccepted(RuntimeError):
    pass


def sources_dir() -> Path:
    env = os.environ.get("TALOSAUR_SOURCES_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[4] / "data_sources"


@dataclass
class SourceInfo:
    name: str
    title: str
    license_id: str
    accept_id: str
    homepage: str = ""
    attribution: str = ""
    citation: str = ""
    ml_training_clause: bool = False
    license_note: str = ""
    access: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, name: str, directory: Path | None = None) -> SourceInfo:
        path = (directory or sources_dir()) / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"no source registry entry {path}")
        d = load_yaml(path)
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


def require_license_acceptance(info: SourceInfo, accept: str | None) -> None:
    """Print the terms and refuse to proceed unless ``accept == info.accept_id``."""
    lic = license_info(info.license_id)
    banner = [
        f"Source: {info.title} ({info.name})",
        f"License: {lic.name}" + (f"  {lic.url}" if lic.url else ""),
    ]
    if info.license_note:
        banner.append(f"Notes: {info.license_note.strip()}")
    if info.attribution:
        banner.append(f"Attribution: {info.attribution.strip()}")
    for line in banner:
        log.info(line)
    if accept != info.accept_id:
        raise LicenseNotAccepted(
            f"Re-run with --accept-license {info.accept_id} after reading the terms above "
            f"(and docs/DATASETS.md). Nothing was downloaded."
        )


# --------------------------------------------------------------------------- downloads


def _hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(
    url: str,
    dest: str | os.PathLike,
    sha256: str | None = None,
    md5: str | None = None,
    retries: int = 5,
    timeout: float = 60.0,
    session=None,
) -> Path:
    """Download with resume (HTTP Range), retries with exponential backoff and checksum check.

    ``file://`` URLs and plain local paths are copied (used by tests and manual downloads).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and _checksum_ok(dest, sha256, md5):
        return dest
    local = url[7:] if url.startswith("file://") else (url if os.path.exists(url) else None)
    if local:
        shutil.copyfile(local, dest)
    else:
        import requests

        sess = session or requests.Session()
        part = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(retries):
            try:
                have = part.stat().st_size if part.exists() else 0
                headers = {"User-Agent": USER_AGENT}
                if have:
                    headers["Range"] = f"bytes={have}-"
                with sess.get(url, stream=True, timeout=timeout, headers=headers) as r:
                    if r.status_code == 416:  # already complete
                        break
                    r.raise_for_status()
                    mode = "ab" if have and r.status_code == 206 else "wb"
                    with open(part, mode) as f:
                        for chunk in r.iter_content(chunk_size=1 << 20):
                            f.write(chunk)
                break
            except Exception as e:  # network errors: back off and resume
                if attempt == retries - 1:
                    raise
                wait = 2 ** (attempt + 1)
                log.warning(f"download failed ({e}); retry {attempt + 1}/{retries - 1} in {wait}s")
                time.sleep(wait)
        os.replace(part, dest)
    if not _checksum_ok(dest, sha256, md5):
        raise OSError(f"checksum mismatch for {dest}")
    return dest


def _checksum_ok(path: Path, sha256: str | None, md5: str | None) -> bool:
    if sha256 and _hash_file(path, "sha256") != sha256.lower():
        return False
    if md5 and _hash_file(path, "md5") != md5.lower():
        return False
    return True


def safe_extract(archive: str | os.PathLike, dest: str | os.PathLike) -> Path:
    """Extract tar/zip refusing absolute paths and ``..`` traversal."""
    archive, dest = Path(archive), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()

    def check(name: str) -> None:
        target = (dest / name).resolve()
        if not str(target).startswith(str(root)):
            raise OSError(f"unsafe path in archive: {name}")

    if zipfile.is_zipfile(archive):
        try:  # adds Deflate64 (compress_type 9) to zipfile; the Kakadu zip uses it
            import zipfile_deflate64  # noqa: F401
        except ImportError:
            pass
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                check(n)
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                check(m.name)
                if m.issym() or m.islnk():
                    raise OSError(f"links not allowed in archive: {m.name}")
            if hasattr(tarfile, "data_filter"):  # Python >= 3.12: also strip unsafe metadata
                t.extractall(dest, filter="data")
            else:
                t.extractall(dest)
    return dest


# --------------------------------------------------------------------------- manifests


def write_manifest(root: Path, info: SourceInfo, extra: dict[str, Any]) -> Path:
    manifest = {
        "source": info.name,
        "title": info.title,
        "license_id": info.license_id,
        "license_note": info.license_note,
        "attribution": info.attribution,
        "citation": info.citation,
        "accepted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **extra,
    }
    return save_json(manifest, Path(root) / "interim" / info.name / "manifest.json")


def records_path(root: Path, source: str) -> Path:
    return Path(root) / "interim" / source / "records.parquet"
