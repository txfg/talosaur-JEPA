"""Pack a source's stored images and masks into HDF5 files, and read them back transparently.

Hundreds of thousands of ~50 KB JPEGs are slow to list, copy and back up, and each wastes part
of a filesystem block. ``scripts/data/pack_h5.py`` moves ``images/<source>/`` and
``masks/<source>/`` into ``h5/<source>/part-NNN.h5``. Records keep their original relative
paths: every reader opens images through :func:`open_image`, which returns the loose file when
it exists and otherwise the bytes from the source's pack.

Each part file holds one contiguous byte blob plus a small index::

    blob     uint8 [total bytes]  the stored files back to back, byte-for-byte (JPEG / PNG);
                                  no HDF5 filter, since JPEG and PNG bytes don't compress further
    key      str   [n]            data-root-relative path, e.g. "images/deepfish/cls/x.jpg"
    offset   int64 [n]            start of each file in ``blob``
    length   int64 [n]
    mtime    int64 [n]            original st_mtime (keeps the curation metrics cache valid)
    crc32    uint32 [n]

A pack run writes a new part to ``part-NNN.h5.tmp`` and renames it when complete, so a crash
never damages earlier parts; later parts win if a key appears twice.
"""

from __future__ import annotations

import io
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

PACK_DIR = "h5"
PACKED_KINDS = ("images", "masks")


class SourcePack:
    """Read-only view of all ``h5/<source>/part-*.h5`` files; safe across fork (reopens per pid)."""

    def __init__(self, directory: str | os.PathLike):
        import h5py

        self.dir = Path(directory)
        self.parts = sorted(self.dir.glob("part-*.h5"))
        self.index: dict[
            str, tuple[int, int, int, int, int]
        ] = {}  # key -> (part, offset, length, mtime, crc)
        for pi, p in enumerate(self.parts):
            with h5py.File(p, "r") as f:
                keys = f["key"].asstr()[:]
                cols = [f[c][:].tolist() for c in ("offset", "length", "mtime", "crc32")]
            for k, *v in zip(keys, *cols):
                self.index[k] = (pi, *v)
        self._files: dict[int, object] = {}
        self._pid: int | None = None

    def __contains__(self, key: str) -> bool:
        return key in self.index

    def __len__(self) -> int:
        return len(self.index)

    def _blob(self, part: int):
        import h5py

        if self._pid != os.getpid():  # handles inherited over fork are not safe to use
            self._files, self._pid = {}, os.getpid()
        f = self._files.get(part)
        if f is None:
            f = self._files[part] = h5py.File(self.parts[part], "r")
        return f["blob"]

    def read(self, key: str) -> bytes:
        part, off, n, _, _ = self.index[key]
        return self._blob(part)[off : off + n].tobytes()

    def stat(self, key: str) -> tuple[int, int]:
        _, _, n, t, _ = self.index[key]
        return n, t

    def crc32(self, key: str) -> int:
        return self.index[key][4]


_PACKS: dict[str, SourcePack | None] = {}


def _pack_for(root: Path, source: str, refresh: bool = False) -> SourcePack | None:
    """The source's pack, loaded once per process; ``refresh`` reloads it if parts were added."""
    d = root / PACK_DIR / source
    k = str(d)
    if k in _PACKS and refresh:
        have = _PACKS[k].parts if _PACKS[k] is not None else []
        if sorted(d.glob("part-*.h5")) != have:
            del _PACKS[k]
    if k not in _PACKS:
        _PACKS[k] = SourcePack(d) if any(d.glob("part-*.h5")) else None
    return _PACKS[k]


def clear_cache() -> None:
    """Forget loaded pack indexes (call after packing more files in the same process)."""
    _PACKS.clear()


def _locate(path: str | os.PathLike) -> tuple[SourcePack, str] | None:
    """Find the pack holding ``<root>/(images|masks)/<source>/...``, trying each candidate root."""
    parts = Path(path).parts
    for refresh in (False, True):  # a miss may mean a part was packed since we loaded the index
        for i in range(len(parts) - 3, -1, -1):
            if parts[i] in PACKED_KINDS:
                root = Path(*parts[:i]) if i else Path(".")
                pack = _pack_for(root, parts[i + 1], refresh=refresh)
                key = "/".join(parts[i:])
                if pack is not None and key in pack:
                    return pack, key
    return None


def exists(path: str | os.PathLike) -> bool:
    """True if the file exists loose on disk or inside its source's pack."""
    return os.path.exists(path) or _locate(path) is not None


def open_image(path: str | os.PathLike):
    """A BytesIO for ``PIL.Image.open``: the loose file's bytes if it is on disk, else the pack's.

    The loose file is read right away, so a pack run deleting it concurrently can't leave a
    path that no longer opens.
    """
    try:
        with open(path, "rb") as f:
            return io.BytesIO(f.read())
    except FileNotFoundError:
        pass
    hit = _locate(path)
    if hit is None:
        raise FileNotFoundError(path)
    pack, key = hit
    return io.BytesIO(pack.read(key))


def stat_image(path: str | os.PathLike) -> tuple[int, int]:
    """``(size, int(mtime))`` of a loose or packed file."""
    try:
        st = os.stat(path)
        return st.st_size, int(st.st_mtime)
    except FileNotFoundError:
        pass
    hit = _locate(path)
    if hit is None:
        raise FileNotFoundError(path)
    pack, key = hit
    return pack.stat(key)


# --------------------------------------------------------------------------- packing


def loose_files(root: Path, source: str) -> list[str]:
    """Data-root-relative paths of every stored file of ``source`` still on disk."""
    out = []
    for kind in PACKED_KINDS:
        d = root / kind / source
        if d.is_dir():
            out += [
                p.relative_to(root).as_posix()
                for p in d.rglob("*")
                if p.is_file() and not p.name.endswith(".tmp")
            ]
    return sorted(out)


def _same(pack: SourcePack | None, root: Path, key: str) -> bool:
    """True if ``key`` is in ``pack`` with the same bytes as the loose file."""
    if pack is None or key not in pack:
        return False
    data = (root / key).read_bytes()
    return pack.stat(key)[0] == len(data) and pack.crc32(key) == zlib.crc32(data)


def pack_source(
    root: str | os.PathLike, source: str, delete: bool = False, workers: int = 16, log=None
) -> dict[str, int]:
    """Write ``source``'s loose files that aren't packed yet (or changed) into a new part.

    With ``delete``, a loose file is removed only once a verified part holds identical bytes.
    """
    import h5py

    root = Path(root)
    say = log.info if log else print
    d = root / PACK_DIR / source
    d.mkdir(parents=True, exist_ok=True)
    existing = SourcePack(d) if any(d.glob("part-*.h5")) else None
    files = loose_files(root, source)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        packed = list(ex.map(lambda k: _same(existing, root, k), files))
    todo = [k for k, done in zip(files, packed) if not done]
    stats = {
        "loose": len(files),
        "already_packed": len(files) - len(todo),
        "packed": 0,
        "bytes": 0,
        "deleted": 0,
    }

    if todo:
        sts = [(root / k).stat() for k in todo]
        sizes = np.array([st.st_size for st in sts], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        mtimes = np.array([int(st.st_mtime) for st in sts], dtype=np.int64)
        crcs = np.zeros(len(todo), dtype=np.uint32)
        final = d / f"part-{len(list(d.glob('part-*.h5'))):03d}.h5"
        tmp = final.with_suffix(".h5.tmp")
        total = int(sizes.sum())
        say(f"{source}: packing {len(todo)} files ({total / 1e9:.2f} GB) -> {final}")

        with h5py.File(tmp, "w") as f, ThreadPoolExecutor(max_workers=workers) as ex:
            blob = f.create_dataset("blob", shape=(total,), dtype=np.uint8)
            buf, pos = bytearray(), 0
            for i, data in enumerate(ex.map(lambda k: (root / k).read_bytes(), todo)):  # ordered
                if len(data) != sizes[i]:
                    raise OSError(f"{todo[i]} changed size while packing")
                crcs[i] = zlib.crc32(data)
                buf += data
                if len(buf) >= 256 << 20 or i == len(todo) - 1:
                    blob[pos : pos + len(buf)] = np.frombuffer(buf, dtype=np.uint8)
                    pos += len(buf)
                    buf = bytearray()
            f.create_dataset("key", data=np.array(todo, dtype=object), dtype=h5py.string_dtype())
            f.create_dataset("offset", data=offsets)
            f.create_dataset("length", data=sizes)
            f.create_dataset("mtime", data=mtimes)
            f.create_dataset("crc32", data=crcs)
            f.attrs["source"] = source
        verify_part(tmp)
        os.replace(tmp, final)
        stats["packed"], stats["bytes"] = len(todo), total
        clear_cache()

    if delete:
        pack = SourcePack(d)
        for k in files:  # every loose file is now in a verified part with identical bytes
            if k in pack:
                (root / k).unlink()
                stats["deleted"] += 1
        for kind in PACKED_KINDS:
            _remove_empty_dirs(root / kind / source)
        clear_cache()
    return stats


def verify_part(path: str | os.PathLike, window: int = 256 << 20) -> int:
    """Re-read a part file and check every entry's CRC; returns the number of entries."""
    import h5py

    with h5py.File(path, "r") as f:
        blob = f["blob"]
        off, length, crc = f["offset"][:], f["length"][:], f["crc32"][:]
        i, n = 0, len(off)
        while i < n:  # entries are contiguous, so read them in ~window-sized slabs
            j = i + 1
            while j < n and off[j] + length[j] - off[i] <= window:
                j += 1
            slab = blob[off[i] : off[j - 1] + length[j - 1]].tobytes()
            for o, m, c in zip(off[i:j] - off[i], length[i:j], crc[i:j]):
                if zlib.crc32(slab[o : o + m]) != int(c):
                    raise OSError(f"CRC mismatch in {path} at offset {off[i] + o}")
            i = j
    return n


def _remove_empty_dirs(d: Path) -> None:
    if not d.is_dir():
        return
    for sub in sorted((p for p in d.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if not any(sub.iterdir()):
            sub.rmdir()
    if not any(d.iterdir()):
        d.rmdir()
