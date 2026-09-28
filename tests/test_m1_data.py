"""M1 data pipeline tests (CPU, synthetic / mocked sources only)."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest
from conftest import requires_av, requires_pandas, requires_pil

from talosaur.data import synthetic as syn
from talosaur.data.dedup import (
    DedupConfig,
    hamming,
    is_near_duplicate,
    near_duplicate_clusters,
    phash64,
    thumbnail,
)
from talosaur.data.labels import (
    box_centroid_and_size,
    boxes_to_coverage,
    mask_to_coverage,
    patch_targets,
    sample_presence_crops,
)
from talosaur.data.licenses import license_info, normalize_license, train_commercial_ok

# --------------------------------------------------------------------------- licenses


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://creativecommons.org/licenses/by-nc-nd/4.0/", "CC-BY-NC-ND-4.0"),
        ("CC BY-NC-ND 4.0", "CC-BY-NC-ND-4.0"),
        ("CC-BY", "CC-BY-4.0"),
        ("CC0", "CC0-1.0"),
        ("https://creativecommons.org/publicdomain/zero/1.0/", "CC0-1.0"),
        ("Attribution-NonCommercial 4.0 International", "CC-BY-NC-4.0"),
        ("Attribution-ShareAlike 4.0 International", "CC-BY-SA-4.0"),
        ("CC BY 3.0 AU", "CC-BY-3.0-AU"),
        ("CDLA-Permissive-1.0", "CDLA-Permissive-1.0"),
        (None, "unknown"),
        ("some bespoke terms", "unknown"),
    ],
)
def test_normalize_license(raw, expected):
    assert normalize_license(raw) == expected


def test_commercial_flags():
    assert not license_info("CC-BY-NC-ND-4.0").commercial_ok
    assert train_commercial_ok("CC-BY-4.0")
    assert not train_commercial_ok("CC-BY-NC-4.0")
    assert train_commercial_ok("CC-BY-NC-ND-4.0", source_ml_training_clause=True)
    assert not train_commercial_ok("unknown")


# --------------------------------------------------------------------------- dedup


def test_phash_stable_under_noise_and_brightness(rng):
    sc = syn.make_scene(rng, 160, 240, condition="clear", n_animals=2)
    img = sc.image.astype(np.float32)
    noisy = np.clip(img * 0.9 + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
    assert hamming(phash64(sc.image), phash64(noisy))[0] <= 6
    other = syn.make_scene(rng, 160, 240, condition="clear", n_animals=2)
    assert hamming(phash64(sc.image), phash64(other.image))[0] > 10


def test_small_object_keeps_frames_distinct(rng):
    frames, boxes = syn.make_video_frames(rng, n_frames=40, h=144, w=256, empty_span=(0.0, 0.5))
    empty = next(f for f, b in zip(frames, boxes) if len(b) == 0)
    with_fish = next(f for f, b in zip(frames, boxes) if len(b) == 1 and 0.3 < b[0][0] < 0.7)
    assert not is_near_duplicate(phash64(empty), thumbnail(empty), phash64(with_fish), thumbnail(with_fish))


def test_near_duplicate_clusters(rng):
    base = [syn.make_scene(rng, 96, 128, condition="clear", n_animals=1).image for _ in range(6)]
    imgs, truth = [], []
    for k, im in enumerate(base):
        for _ in range(3):  # three jittered copies of each base image
            j = np.clip(im.astype(np.int16) + rng.integers(-3, 4, im.shape), 0, 255).astype(np.uint8)
            imgs.append(j)
            truth.append(k)
    hashes = np.array([phash64(i) for i in imgs], dtype=np.uint64)
    thumbs = np.stack([thumbnail(i) for i in imgs])
    cl = near_duplicate_clusters(hashes, thumbs, DedupConfig(seed=1))
    truth = np.array(truth)
    for k in range(6):
        assert len(set(cl[truth == k])) == 1  # copies merged
    assert len(set(cl)) == 6  # distinct scenes kept apart


def test_dark_and_murky_frames_with_same_layout_are_not_duplicates():
    rng = np.random.default_rng(5)
    frames_d, _ = syn.make_video_frames(np.random.default_rng(7), n_frames=3, condition="dark", animal=False)
    frames_m, _ = syn.make_video_frames(np.random.default_rng(7), n_frames=3, condition="murky", animal=False)
    d, m = frames_d[0], frames_m[0]
    assert not is_near_duplicate(phash64(d), thumbnail(d), phash64(m), thumbnail(m))
    _ = rng


# --------------------------------------------------------------------------- quality


def test_condition_buckets_on_synthetic():
    from talosaur.data.quality import BucketThresholds, clarity_bucket, compute_metrics, light_bucket

    rng = np.random.default_rng(0)
    t = BucketThresholds()
    dark = [compute_metrics(syn.make_scene(rng, 144, 256, condition="dark").image) for _ in range(10)]
    murky = [compute_metrics(syn.make_scene(rng, 144, 256, condition="murky").image) for _ in range(10)]
    clear = [compute_metrics(syn.make_scene(rng, 144, 256, condition="clear").image) for _ in range(10)]
    assert all(light_bucket(m["lum_mean"], t) == "dark" for m in dark)
    assert sum(clarity_bucket(m["clarity"], t) == "murky" for m in murky) >= 9
    assert sum(clarity_bucket(m["clarity"], t) == "clear" for m in clear) >= 7
    assert np.median([m["clarity"] for m in clear]) > np.median([m["clarity"] for m in murky]) + 0.3


def test_fit_thresholds_from_labels():
    from talosaur.data.quality import fit_thresholds_from_labels

    rng = np.random.default_rng(0)
    lum = np.r_[rng.uniform(0.0, 0.05, 30), rng.uniform(0.15, 0.2, 30), rng.uniform(0.4, 0.6, 30)]
    cla = np.r_[rng.uniform(0.0, 0.2, 30), rng.uniform(0.4, 0.5, 30), rng.uniform(0.7, 0.9, 30)]
    ll = ["dark"] * 30 + ["dim"] * 30 + ["bright"] * 30
    cl = ["murky"] * 30 + ["moderate"] * 30 + ["clear"] * 30
    t, agree = fit_thresholds_from_labels({"lum_mean": lum, "clarity": cla}, ll, cl)
    assert agree["light_agreement"] == 1.0 and agree["clarity_agreement"] == 1.0
    assert 0.05 <= t.dark_lum <= 0.15 and 0.2 <= t.dim_lum <= 0.4


# --------------------------------------------------------------------------- labels


def test_boxes_to_coverage_and_targets():
    cov = boxes_to_coverage([[0.0, 0.0, 0.5, 0.5]], (4, 4))
    assert cov.shape == (4, 4)
    assert np.allclose(cov[:2, :2], 1.0) and np.allclose(cov[2:, :], 0.0)
    cov2 = boxes_to_coverage([[0.1, 0.1, 0.2, 0.2]], (2, 2))  # 1% of the image
    assert 0.0 < cov2[0, 0] < 0.1
    t = patch_targets(np.array([[0.0, 0.1], [0.5, 1.0]]), pos_thr=0.3)
    assert t.tolist() == [[0, -1], [1, 1]]


def test_mask_to_coverage_matches_area():
    m = np.zeros((64, 64), dtype=bool)
    m[:32, :32] = True
    cov = mask_to_coverage(m, (2, 2))
    assert cov[0, 0] > 0.9 and cov[1, 1] < 0.1


def test_centroid_and_presence_crops(rng):
    c = box_centroid_and_size([[0.0, 0.0, 0.2, 0.2], [0.8, 0.8, 1.0, 1.0]])
    assert np.allclose(c[:2], (0.5, 0.5)) and 0.28 < c[2] < 0.29
    crops = sample_presence_crops([[0.4, 0.4, 0.6, 0.6]], [True], rng, n=20)
    labels = {lab for _, lab in crops}
    assert labels <= {0, 1} and len(crops) > 5


# --------------------------------------------------------------------------- video


@requires_av
@requires_pil
def test_extract_frames_overlay_and_static_dedup(tmp_path, rng):
    from talosaur.data.video import ExtractConfig, extract_frames

    frames, _ = syn.make_video_frames(
        rng, n_frames=60, h=144, w=256, overlay_text=True, static_span=(0.3, 0.7)
    )
    p = syn.write_video(tmp_path / "v.mp4", frames, fps=10)
    res = extract_frames(p, tmp_path / "out", "vid", ExtractConfig(fps=2.0, short_side=96))
    st = res["stats"]
    assert res["overlay"]["detected"] and res["overlay"]["top"] > 0
    assert st["sampled"] == 12 and st["dropped_near_duplicates"] >= 2
    assert st["kept"] == len(res["frames"]) == st["sampled"] - st["dropped_near_duplicates"]
    from PIL import Image

    im = Image.open(tmp_path / "out" / res["frames"][0]["path"])
    assert min(im.size) == 96


@requires_av
@requires_pil
def test_forced_deinterlace_restores_aspect(tmp_path, rng):
    from talosaur.data.video import ExtractConfig, extract_frames

    frames, _ = syn.make_video_frames(rng, n_frames=10, h=144, w=256)
    p = syn.write_video(tmp_path / "v.mp4", frames, fps=10)
    res = extract_frames(
        p,
        tmp_path / "o",
        "v",
        ExtractConfig(fps=5, short_side=72, deinterlace="on", overlay="off", dedup=False),
    )
    f = res["frames"][0]
    assert abs(f["width"] / f["height"] - 256 / 144) < 0.05


# --------------------------------------------------------------------------- sources


@requires_pandas
@requires_pil
def test_license_gate_blocks_download(tmp_path):
    from talosaur.data.sources import LicenseNotAccepted, run_fetch

    with pytest.raises(LicenseNotAccepted):
        run_fetch("synthetic", tmp_path, accept=None)
    assert not (tmp_path / "images").exists()


@requires_pandas
@requires_pil
def test_coco_ingest(tmp_path):
    from talosaur.data.sources.base import SourceInfo
    from talosaur.data.sources.generic import ingest_coco

    ann = syn.write_image_dataset(tmp_path / "raw", n=12, seed=3, size=(120, 160))
    info = SourceInfo(name="kakadu", title="t", license_id="CC-BY-4.0", accept_id="CC-BY-4.0")
    recs = ingest_coco(info, tmp_path / "root", ann, short_side=96, boxes_exhaustive=True)
    assert len(recs) == 12
    coco = json.loads(ann.read_text())
    n_boxes = sum(len(r.boxes) for r in recs)
    assert n_boxes == len(coco["annotations"])
    r = next(r for r in recs if r.boxes)
    assert r.frame_label == 1 and r.train_commercial_ok and r.mask_path
    assert all(0 <= v <= 1 for b in r.boxes for v in b)
    assert (tmp_path / "root" / r.path).exists()
    assert any(rr.frame_label == 0 for rr in recs)


@requires_pandas
@requires_pil
def test_deepfish_ingest_layout(tmp_path):
    from PIL import Image

    from talosaur.data.sources.base import SourceInfo
    from talosaur.data.sources.deepfish import ingest_deepfish

    base = tmp_path / "DeepFish"
    rng = np.random.default_rng(0)

    def img(p):
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(syn.make_scene(rng, 108, 192, n_animals=1).image).save(p)

    (base / "Classification").mkdir(parents=True)
    (base / "Classification" / "train.csv").write_text(
        "ID,labels\n7117/valid/7117_a,1\n9894/empty/9894_b,0\n"
    )
    img(base / "Classification" / "7117/valid/7117_a.jpg")
    img(base / "Classification" / "9894/empty/9894_b.jpg")
    (base / "Segmentation").mkdir()
    (base / "Segmentation" / "test.csv").write_text("ID,labels\nvalid/7398_c,1\n")
    img(base / "Segmentation" / "images" / "valid/7398_c.jpg")
    m = np.zeros((108, 192), dtype=np.uint8)
    m[30:60, 40:100] = 255
    (base / "Segmentation" / "masks" / "valid").mkdir(parents=True)
    Image.fromarray(m).save(base / "Segmentation" / "masks" / "valid/7398_c.png")
    (base / "Localization").mkdir()
    (base / "Localization" / "val.csv").write_text("ID,labels,counts\nvalid/7117_d,1,2\n")
    img(base / "Localization" / "images" / "valid/7117_d.jpg")
    pts = np.zeros((108, 192), dtype=np.uint8)
    pts[10, 20] = pts[50, 150] = 1
    (base / "Localization" / "masks" / "valid").mkdir(parents=True)
    Image.fromarray(pts).save(base / "Localization" / "masks" / "valid/7117_d.png")

    info = SourceInfo(name="deepfish", title="DeepFish", license_id="CC-BY-4.0", accept_id="CC-BY-4.0")
    recs = {r.image_id: r for r in ingest_deepfish(info, tmp_path / "root", tmp_path, short_side=96)}
    assert recs["deepfish:cls:7117/valid/7117_a"].frame_label == 1
    assert recs["deepfish:cls:9894/empty/9894_b"].frame_label == 0
    assert recs["deepfish:cls:7117/valid/7117_a"].group_id == "deepfish:7117"
    seg = recs["deepfish:seg:valid/7398_c"]
    assert seg.group_id == "deepfish:7398" and seg.mask_path and len(seg.boxes) == 1
    assert seg.split_hint == "test" and seg.boxes_exhaustive
    x0, y0, x1, y1 = seg.boxes[0]
    assert abs(x0 - 40 / 192) < 0.01 and abs(y1 - 60 / 108) < 0.01
    loc = recs["deepfish:loc:valid/7117_d"]
    assert len(loc.extra["points"]) == 2 and loc.frame_label == 1


def _fake_fathomnet(monkeypatch, images_list, png_bytes):
    """Install fake `fathomnet` modules and a fake HTTP session."""
    Box = types.SimpleNamespace

    def image(uuid, url, boxes):
        return types.SimpleNamespace(
            uuid=uuid,
            url=url,
            boundingBoxes=boxes,
            depthMeters=250.0,
            latitude=36.7,
            longitude=-122.0,
            imagingType="ROV",
            timestamp="2019-06-01T10:00:00Z",
            contributorsEmail="a@b.org",
        )

    ims = [image(u, url, [Box(**b) for b in bxs]) for u, url, bxs in images_list]
    pkg = types.ModuleType("fathomnet")
    api = types.ModuleType("fathomnet.api")
    images = types.ModuleType("fathomnet.api.images")
    uploads = types.ModuleType("fathomnet.api.imagesetuploads")
    worms = types.ModuleType("fathomnet.api.worms")
    dto = types.ModuleType("fathomnet.dto")
    images.find_by_concept = lambda c: ims
    images.find_all = lambda pageable: ims if pageable.number == 0 else []
    images.find = lambda cons: ims
    uploads.find_by_image_uuid = lambda u: [
        types.SimpleNamespace(
            darwinCore=types.SimpleNamespace(
                license="https://creativecommons.org/licenses/by-nc-nd/4.0/" if u == "u1" else "CC0",
                ownerInstitutionCode="MBARI",
                rightsHolder="MBARI",
            )
        )
    ]
    worms.get_ancestors_names = lambda c: ["object", "Animalia", "Cnidaria"] if c == "Nanomia" else ["object"]
    dto.Pageable = lambda size=None, number=None: types.SimpleNamespace(size=size, number=number)
    dto.GeoImageConstraints = lambda **kw: types.SimpleNamespace(**kw)
    api.images, api.imagesetuploads, api.worms = images, uploads, worms
    pkg.api, pkg.dto = api, dto
    for name, mod in {
        "fathomnet": pkg,
        "fathomnet.api": api,
        "fathomnet.api.images": images,
        "fathomnet.api.imagesetuploads": uploads,
        "fathomnet.api.worms": worms,
        "fathomnet.dto": dto,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    import requests

    class Resp:
        status_code = 200
        content = png_bytes

        def raise_for_status(self):
            return None

    monkeypatch.setattr(requests.Session, "get", lambda self, url, **kw: Resp())


@requires_pandas
@requires_pil
def test_fathomnet_fetch_mocked(tmp_path, monkeypatch):
    import io

    from PIL import Image

    from talosaur.data.sources.base import SourceInfo
    from talosaur.data.sources.fathomnet import fetch_fathomnet

    buf = io.BytesIO()
    Image.fromarray(syn.make_scene(np.random.default_rng(0), 400, 600).image).save(buf, format="PNG")
    boxes1 = [
        dict(x=60, y=40, width=120, height=80, concept="Nanomia", rejected=False),
        dict(x=300, y=200, width=50, height=50, concept="Trash", rejected=False),
    ]
    _fake_fathomnet(
        monkeypatch,
        [
            ("u1", "https://x.org/EX1708_VID_20170921T005000Z_ROVHD_frame_001678.png", boxes1),
            ("u2", "https://x.org/D1234_20190601T100000.png", []),
        ],
        buf.getvalue(),
    )
    info = SourceInfo.load("fathomnet")
    recs = {r.image_id: r for r in fetch_fathomnet(info, tmp_path, max_images=10, short_side=200, workers=2)}
    r1, r2 = recs["fathomnet:u1"], recs["fathomnet:u2"]
    assert r1.license == "CC-BY-NC-ND-4.0" and r1.train_commercial_ok  # ToS ML-training clause
    assert r2.license == "CC0-1.0"
    assert r1.box_is_animal == [True, False] and r1.frame_label == 1
    assert r2.frame_label == -1 and not r2.boxes_exhaustive
    assert r1.group_id == "fathomnet:EX1708_VID_20170921T005000Z_ROVHD"
    assert np.allclose(r1.boxes[0], [0.1, 0.1, 0.3, 0.3], atol=1e-3)
    assert min(r1.width, r1.height) == 200
    # resumable: a second call re-uses records.jsonl and downloads nothing new
    recs2 = fetch_fathomnet(info, tmp_path, max_images=10, short_side=200)
    assert len(recs2) == 2


@requires_pandas
@requires_pil
@requires_av
def test_noaa_manifest_filters_non_ex(tmp_path, rng):
    from talosaur.data.sources.base import SourceInfo
    from talosaur.data.sources.noaa_oer import fetch_noaa_oer, parse_name

    assert parse_name("EX1708_DIVE16_20170922T195500Z_CPHD.mov") == {
        "cruise": "EX1708",
        "dive": "DIVE16",
        "date": "20170922",
        "stem": "EX1708_DIVE16_20170922T195500Z_CPHD",
    }
    frames, _ = syn.make_video_frames(rng, n_frames=30, h=144, w=256)
    ex = syn.write_video(tmp_path / "src" / "EX1711_DIVE03_20171201T120000Z_ROVHD.mp4", frames, fps=10)
    nau = syn.write_video(tmp_path / "src" / "NA090_dive1.mp4", frames, fps=10)
    manifest = tmp_path / "m.csv"
    manifest.write_text(f"url,notes\n{ex},gulf\n{nau},nautilus\n")
    info = SourceInfo.load("noaa_oer")
    recs = fetch_noaa_oer(info, tmp_path / "root", manifest, fps=1.0, short_side=72)
    assert recs and all(r.group_id == "noaa_oer:EX1711:DIVE03" for r in recs)
    assert all(r.license == "public-domain-us-gov" for r in recs)
    assert ex.exists()  # local sources are never deleted


@requires_pandas
@requires_pil
def test_camera_traps_ingest(tmp_path, rng):
    from PIL import Image

    from talosaur.data.sources.base import SourceInfo
    from talosaur.data.sources.generic import ingest_camera_traps

    imgs, anns = [], []
    for i in range(10):
        Image.fromarray(syn.make_scene(rng, 64, 96).image).save(tmp_path / f"f{i}.jpg")
        imgs.append(
            {
                "id": i,
                "file_name": f"f{i}.jpg",
                "location": "weir1" if i < 5 else "weir2",
                "datetime": "2022-05-01 21:10:00",
                "width": 96,
                "height": 64,
            }
        )
        anns.append({"id": i, "image_id": i, "category_id": 0 if i % 2 else 1})
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "images": imgs,
                "annotations": anns,
                "categories": [{"id": 0, "name": "empty"}, {"id": 1, "name": "herring"}],
            }
        )
    )
    info = SourceInfo(name="river_herring", title="t", license_id="CDLA-Permissive-1.0", accept_id="x")
    recs = ingest_camera_traps(info, tmp_path / "root", meta, images_dir=tmp_path, short_side=48, max_empty=3)
    assert sum(r.frame_label == 0 for r in recs) == 3 and sum(r.frame_label == 1 for r in recs) == 5
    assert {r.group_id for r in recs} == {"river_herring:weir1:2022-05-01", "river_herring:weir2:2022-05-01"}


# --------------------------------------------------------------------------- curation end-to-end


@requires_pandas
@requires_pil
@requires_av
@pytest.mark.slow
def test_curation_end_to_end(tmp_path):
    from talosaur.data.curate import CurateConfig, run_curation
    from talosaur.data.index import read_table
    from talosaur.data.sources import run_fetch
    from talosaur.data.stats import write_report

    run_fetch("synthetic", tmp_path, "synthetic", n_images=45, n_videos=2)
    cfg = CurateConfig(
        name="t",
        root=str(tmp_path),
        sources=["synthetic"],
        workers=1,
        val_frac=0.2,
        test_frac=0.2,
        empty_water={"enabled": True, "keep_frac": {"default": 0.0}},
    )
    df, removed, log = run_curation(cfg)
    assert df.groupby("group_id")["split"].nunique().max() == 1  # grouped splits
    assert (df.loc[df["split"] == "train", "weight"] > 0).all()
    assert (df.loc[df["split"] != "train", "weight"] == 0).all()
    assert set(df["slice"]) <= {"dark", "murky", "clear", "other"}
    assert (
        not df["is_empty"][df["split"] == "train"].any() or (df["labeled"] | (df["frame_label"] == 1)).any()
    )
    idx = read_table(tmp_path / "index" / "t.parquet")
    assert len(idx) == len(df) and idx["phash"].dtype == np.uint64
    report = write_report(df, removed, log, tmp_path, tmp_path / "rep", figures=True)
    text = report.read_text()
    for section in ("Images per source", "Licenses", "Condition buckets", "Labels", "Animal box size"):
        assert section in text
    assert (tmp_path / "rep" / "figures").exists()


@requires_pandas
@requires_pil
def test_dedup_leakage_rule_drops_train_copy_of_test_image(tmp_path):
    import pandas as pd

    from talosaur.data.curate import CurateConfig, apply_dedup

    rng = np.random.default_rng(0)
    rows = []
    base = syn.make_scene(rng, 96, 128, n_animals=1).image
    other = syn.make_scene(rng, 96, 128, n_animals=1).image
    for i, (im, split) in enumerate([(base, "test"), (base, "train"), (other, "train")]):
        rows.append(
            {
                "image_id": f"i{i}",
                "split": split,
                "phash": phash64(im),
                "thumb": thumbnail(im).tobytes(),
                "mask_path": None,
                "boxes": [],
                "frame_label": -1,
                "width": 128,
                "height": 96,
            }
        )
    df = pd.DataFrame(rows)
    df["phash"] = df["phash"].astype(np.uint64)
    keep, reason = apply_dedup(df, CurateConfig())
    assert keep.tolist() == [True, False, True]
    assert reason[1] == "dup_of_eval"
