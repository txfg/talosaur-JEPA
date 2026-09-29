"""HDF5 image packs: packing is lossless, readers fall back to packs, re-packing is incremental."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import requires_av, requires_pandas, requires_pil

h5py = pytest.importorskip("h5py")


@requires_pil
def test_pack_roundtrip_delete_and_incremental(tmp_path):
    from PIL import Image

    from talosaur.data import h5store
    from talosaur.data.imageio import image_size, load_rgb, save_jpeg

    rng = np.random.default_rng(0)
    rels = [f"images/src/a/{i}.jpg" for i in range(5)] + ["masks/src/a/0.png"]
    for r in rels[:5]:
        save_jpeg(Image.fromarray(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)), tmp_path / r)
    (tmp_path / rels[5]).parent.mkdir(parents=True)
    Image.fromarray(np.eye(8, dtype=np.uint8) * 255).save(tmp_path / rels[5])
    before = {r: (tmp_path / r).read_bytes() for r in rels}
    pixels = load_rgb(tmp_path / rels[0])
    size0 = h5store.stat_image(tmp_path / rels[0])

    st = h5store.pack_source(tmp_path, "src", delete=True)
    assert st["packed"] == 6 and st["deleted"] == 6
    assert not (tmp_path / "images" / "src").exists() and not (tmp_path / "masks" / "src").exists()
    for r in rels:
        assert h5store.exists(tmp_path / r)
        assert h5store.open_image(tmp_path / r).read() == before[r]  # byte-identical
    assert np.array_equal(load_rgb(tmp_path / rels[0]), pixels)
    assert image_size(tmp_path / rels[1]) == (32, 24)
    assert h5store.stat_image(tmp_path / rels[0]) == size0  # metrics-cache keys are unchanged
    assert not h5store.exists(tmp_path / "images/src/missing.jpg")
    with pytest.raises(FileNotFoundError):
        h5store.open_image(tmp_path / "images/src/missing.jpg")

    # a new file goes into a second part; already-packed files are not written again
    save_jpeg(
        Image.fromarray(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)), tmp_path / "images/src/b/new.jpg"
    )
    st2 = h5store.pack_source(tmp_path, "src")
    assert st2["packed"] == 1 and st2["already_packed"] == 0
    assert sorted(p.name for p in (tmp_path / "h5" / "src").iterdir()) == ["part-000.h5", "part-001.h5"]
    st3 = h5store.pack_source(tmp_path, "src")
    assert st3["packed"] == 0 and st3["already_packed"] == 1
    assert h5store.verify_part(tmp_path / "h5" / "src" / "part-000.h5") == 6


@requires_pandas
@requires_pil
@requires_av
@pytest.mark.slow
def test_curation_and_datasets_read_from_packs(tmp_path):
    from talosaur.data import h5store
    from talosaur.data.curate import CurateConfig, run_curation
    from talosaur.data.sources import run_fetch

    run_fetch("synthetic", tmp_path, "synthetic", n_images=20, n_videos=1)
    h5store.pack_source(tmp_path, "synthetic", delete=True)
    assert not list((tmp_path / "images").rglob("*.jpg"))
    # re-running the fetch treats packed stills as done instead of re-creating them
    # (video frame extraction always rewrites its frames; re-packing then finds them unchanged)
    run_fetch("synthetic", tmp_path, "synthetic", n_images=20, n_videos=1)
    assert not [p for p in (tmp_path / "images").rglob("*.jpg") if "clip" not in p.name]
    assert h5store.pack_source(tmp_path, "synthetic", delete=True)["packed"] == 0

    cfg = CurateConfig(
        name="t", root=str(tmp_path), sources=["synthetic"], workers=1, val_frac=0.2, test_frac=0.2
    )
    df, _, _ = run_curation(cfg)
    assert len(df) > 0 and df["lum_mean"].notna().all()

    torch = pytest.importorskip("torch")
    from talosaur.data.torch_datasets import LabeledFrames, PretrainImages

    x = PretrainImages(df, tmp_path, size=(32, 32))[0]
    assert isinstance(x, torch.Tensor) and tuple(x.shape) == (3, 32, 32)
    item = LabeledFrames(df, tmp_path, (32, 32), 16)[0]
    assert tuple(item["image"].shape) == (3, 32, 32)


@requires_pil
def test_reader_sees_parts_packed_after_it_loaded_the_index(tmp_path):
    from PIL import Image

    from talosaur.data import h5store
    from talosaur.data.imageio import load_rgb, save_jpeg

    rng = np.random.default_rng(0)
    for name in ("a", "b"):
        save_jpeg(Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)), tmp_path / f"images/s/{name}.jpg")
    h5store.pack_source(tmp_path, "s", delete=True)
    load_rgb(tmp_path / "images/s/a.jpg")  # this process now holds part-000's index
    save_jpeg(Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)), tmp_path / "images/s/c.jpg")
    load_rgb(tmp_path / "images/s/c.jpg")  # loose
    # another process packs c into part-001 and deletes it, without clearing our cache
    packed = h5store._PACKS.copy()
    h5store.pack_source(tmp_path, "s", delete=True)
    h5store._PACKS.update(packed)
    assert load_rgb(tmp_path / "images/s/c.jpg").shape == (8, 8, 3)
