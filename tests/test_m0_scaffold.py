import json
import os

import numpy as np
from conftest import requires_av, requires_pil

from talosaur.data import synthetic as syn
from talosaur.utils.io import atomic_write_text, load_json, save_json, sha256_file
from talosaur.utils.seed import derive_seed


def test_derive_seed_is_stable_and_distinct():
    a = derive_seed(0, "masks", 10)
    assert a == derive_seed(0, "masks", 10)
    assert a != derive_seed(0, "masks", 11)
    assert a != derive_seed(1, "masks", 10)
    assert 0 <= a < 2**63


def test_atomic_json_roundtrip(tmp_path):
    p = save_json({"a": np.int64(3), "b": np.float32(0.5), "c": np.arange(3)}, tmp_path / "x" / "m.json")
    assert load_json(p) == {"a": 3, "b": 0.5, "c": [0, 1, 2]}
    assert not [f for f in os.listdir(p.parent) if f.endswith(".tmp")]
    atomic_write_text(tmp_path / "t.txt", "hello")
    assert len(sha256_file(tmp_path / "t.txt")) == 64


def test_scene_ground_truth_is_consistent(rng):
    for cond in syn.CONDITIONS:
        sc = syn.make_scene(rng, 128, 160, condition=cond, n_animals=2)
        assert sc.image.shape == (128, 160, 3) and sc.image.dtype == np.uint8
        assert sc.boxes.shape[1] == 4
        assert np.all(sc.boxes >= 0) and np.all(sc.boxes <= 1)
        assert np.all(sc.boxes[:, 2] > sc.boxes[:, 0]) and np.all(sc.boxes[:, 3] > sc.boxes[:, 1])
        # every box contains mask pixels
        for x0, y0, x1, y1 in sc.boxes:
            sub = sc.mask[int(y0 * 128) : int(np.ceil(y1 * 128)), int(x0 * 160) : int(np.ceil(x1 * 160))]
            assert sub.any()
    empty = syn.make_scene(rng, 64, 64, n_animals=0)
    assert not empty.has_animal and not empty.mask.any()


def test_conditions_differ_in_expected_direction():
    rng = np.random.default_rng(0)
    lum = {
        c: np.mean([syn.make_scene(rng, 96, 96, condition=c).image.mean() for _ in range(8)])
        for c in syn.CONDITIONS
    }
    std = {
        c: np.mean([syn.make_scene(rng, 96, 96, condition=c).image.std() for _ in range(8)])
        for c in syn.CONDITIONS
    }
    assert lum["dark"] < 0.4 * lum["clear"]
    assert std["murky"] < std["clear"]


def test_video_frames_empty_span_and_overlay(rng):
    frames, boxes = syn.make_video_frames(
        rng, n_frames=20, h=72, w=128, empty_span=(0.4, 0.6), overlay_text=True
    )
    assert frames.shape == (20, 72, 128, 3)
    counts = [len(b) for b in boxes]
    assert 0 in counts and max(counts) == 1
    # HUD band is static and saturated
    assert (frames[:, :4, :, :] == 255).any(axis=(1, 2, 3)).all()


@requires_pil
def test_write_image_dataset(tmp_path):
    ann = syn.write_image_dataset(tmp_path / "ds", n=9, seed=1, size=(64, 80))
    coco = json.loads(ann.read_text())
    assert len(coco["images"]) == 9
    assert {im["condition"] for im in coco["images"]} == set(syn.CONDITIONS)
    assert (tmp_path / "ds" / coco["images"][0]["file_name"]).exists()


@requires_av
def test_write_video_roundtrip(tmp_path, rng):
    import av

    frames, _ = syn.make_video_frames(rng, n_frames=12, h=64, w=96)
    p = syn.write_video(tmp_path / "v.mp4", frames, fps=10)
    with av.open(str(p)) as c:
        n = sum(1 for _ in c.decode(video=0))
    assert n == 12
