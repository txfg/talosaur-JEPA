"""Zenodo record download (used for the Kakadu freshwater-fish dataset, record 7250921).

Uses the public REST API ``GET https://zenodo.org/api/records/<id>``; each file entry carries a
key, size, ``md5:<hex>`` checksum and links. Both the InvenioRDM (2023+) and legacy link layouts
are handled.
"""

from __future__ import annotations

from pathlib import Path

from talosaur.data.sources.base import USER_AGENT, download
from talosaur.utils.log import get_logger

log = get_logger("data.zenodo")
API = "https://zenodo.org/api/records/{record}"


def list_files(record: int | str, session=None) -> list[dict]:
    import requests

    sess = session or requests.Session()
    r = sess.get(API.format(record=record), timeout=60, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    files = r.json().get("files", [])
    out = []
    for f in files:
        key = f.get("key") or f.get("filename")
        links = f.get("links", {})
        # InvenioRDM: links.content is the binary; legacy API: links.self / links.download is.
        url = links.get("content") or links.get("download") or links.get("self")
        if not url:
            url = f"https://zenodo.org/records/{record}/files/{key}?download=1"
        md5 = None
        ck = f.get("checksum") or ""
        if ck.startswith("md5:"):
            md5 = ck[4:]
        out.append({"key": key, "url": url, "md5": md5, "size": f.get("size")})
    return out


def download_record(record: int | str, dest: Path, only: list[str] | None = None) -> list[Path]:
    dest = Path(dest)
    paths = []
    for f in list_files(record):
        if only and not any(s in f["key"] for s in only):
            continue
        log.info(f"zenodo {record}: {f['key']} ({(f['size'] or 0) / 1e9:.2f} GB)")
        paths.append(download(f["url"], dest / f["key"], md5=f["md5"]))
    return paths
