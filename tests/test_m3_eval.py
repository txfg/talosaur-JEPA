"""M3/M4 tests: heads, backbones, evaluation harness, report."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_talosaur_net_shapes_and_heads_from_probes():
    from talosaur.models.heads import FrameHead, HeatmapHead, TalosaurNet, heads_from_probes
    from talosaur.models.vit import build_vit

    enc = build_vit("vit_tiny", depth=2)
    net = TalosaurNet(enc, FrameHead(192, 2), HeatmapHead(192), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (112, 208))
    f, h, e, tok = net(torch.rand(3, 3, 112, 208))
    assert f.shape == (3, 2) and h.shape == (3, 7, 13) and e.shape == (3, 192)
    assert tok.shape == (3, 91, 192) and torch.allclose(tok.mean(dim=1), e)
    fl, pl = torch.nn.Linear(384, 1), torch.nn.Linear(192, 1)
    fh, hh = heads_from_probes(192, fl, pl)
    assert torch.equal(fh.linear.weight, fl.weight) and torch.equal(hh.linear.bias, pl.bias)


@pytest.mark.parametrize(
    "spec,grid,dim",
    [
        ({"kind": "random_vit_tiny"}, (7, 13), 192),
        ({"kind": "timm", "timm_name": "vit_tiny_patch16_224.augreg_in21k_ft_in1k"}, (7, 13), 192),
        ({"kind": "dinov2_vits14"}, (8, 15), 384),
        ({"kind": "mobilenetv3_large"}, (7, 13), None),
    ],
)
def test_backbones_patch_grid(spec, grid, dim):
    pytest.importorskip("timm")
    from talosaur.eval.backbones import build_backbone

    bb = build_backbone(spec, pretrained=False).eval()
    hw = bb.input_size((112, 208))
    with torch.no_grad():
        t = bb(torch.zeros(2, 3, *hw))
    assert t.shape[:3] == (2, *grid)
    assert t.shape[3] == bb.info.dim and (dim is None or bb.info.dim == dim)


def test_jepa_backbone_loads_checkpoint(tmp_path):
    from talosaur.eval.backbones import JepaBackbone
    from talosaur.models.vit import build_vit

    enc = build_vit("vit_tiny", depth=2)
    torch.save(
        {
            "model_config": {"name": "vit_tiny", **enc.cfg.to_dict()},
            "target_encoder": enc.state_dict(),
            "encoder": enc.state_dict(),
            "mean": [0.5] * 3,
            "std": [0.5] * 3,
        },
        tmp_path / "e.pt",
    )
    bb = JepaBackbone(str(tmp_path / "e.pt"), name="j")
    assert bb.info.mean == (0.5, 0.5, 0.5) and len(bb.encoder.blocks) == 2


def test_labeled_frames_crop_mode(synthetic_index):
    from talosaur.data.index import read_table
    from talosaur.data.torch_datasets import LabeledFrames

    root, idx = synthetic_index
    df = read_table(idx)
    row = df[df["boxes"].map(len) > 0].iloc[:1].copy()
    b = row.iloc[0]["boxes"][0]
    row["crop"] = [[max(0, b[0] - 0.05), max(0, b[1] - 0.05), min(1, b[2] + 0.05), min(1, b[3] + 0.05)]]
    item = LabeledFrames(row, root, (64, 64))[0]
    assert item["image"].shape == (3, 64, 64)
    assert item["coverage"].max() > 0.5  # the animal fills most of the crop
    assert not bool(item["patch_valid"]) or row.iloc[0]["boxes_exhaustive"]


@pytest.mark.slow
def test_harness_end_to_end(synthetic_index, tmp_path):
    from talosaur.data.index import read_table
    from talosaur.eval.backbones import build_backbone
    from talosaur.eval.harness import EvalConfig, evaluate_backbone
    from talosaur.eval.report import write_report

    root, idx = synthetic_index
    df = read_table(idx)
    cfg = EvalConfig(
        batch_size=32,
        workers=0,
        bootstrap=20,
        robustness_ops=("haze", "particles"),
        robustness_levels=(1, 5),
        robustness_max_images=24,
    )
    bb = build_backbone({"kind": "random_vit_tiny", "name": "rand"})
    res = evaluate_backbone(bb, df, root, (64, 64), cfg, torch.device("cpu"), tmp_path, robustness=True)
    assert res["frame"]["all"]["auroc"]["value"] > 0.5
    assert 0 <= res["patch"]["all"]["patch_auroc"] <= 1
    assert "centroid_err_deg_median" in res["patch"]["all"]
    assert set(res["robustness"]) == {"haze/1", "haze/5", "particles/1", "particles/5"}
    heads = torch.load(tmp_path / "heads_rand_64x64.pt", weights_only=False)
    assert heads["frame_head"]["weight"].shape == (1, 384) and heads["heatmap_head"]["weight"].shape == (
        1,
        192,
    )
    rep = write_report(
        [res], tmp_path, {"rand_64x64": {"params_m": 5.5, "gmacs": 0.09}}, {"rand_64x64": 42.0}
    )
    text = rep.read_text()
    assert "Frame probe" in text and "Underwater-C" in text and "42.000" in text
    assert json.loads((tmp_path / "results.json").read_text())["results"][0]["backbone"] == "rand"


def test_count_params_and_gmacs_vit_tiny():
    from talosaur.eval.backbones import build_backbone, count_params_and_gmacs

    p, g = count_params_and_gmacs(build_backbone({"kind": "random_vit_tiny"}), (224, 224))
    assert 5.3 < p < 5.8
    assert 1.1 < g < 1.35  # analytic estimate in docs/PLAN.md: 1.25 GMACs
    _ = np


def test_frame_rows_adds_presence_crops_for_box_only_sources():
    pd = pytest.importorskip("pandas")
    from talosaur.eval.harness import EvalConfig, frame_rows

    rows = [
        {"split": "train", "source": "fathomnet", "frame_label": -1, "boxes": [[0.05, 0.05, 0.2, 0.2]],
         "box_is_animal": [True]},
        {"split": "train", "source": "deepfish", "frame_label": 0, "boxes": [], "box_is_animal": []},
    ]
    df = pd.DataFrame(rows)
    out = frame_rows(df, EvalConfig(crops_per_image=4), "train")
    crops = out[out["source"] == "fathomnet_crops"]
    assert len(crops) > 0 and set(crops["frame_label"]) <= {0, 1}
    assert (out["source"] == "deepfish").sum() == 1
    assert frame_rows(df, EvalConfig(crops_per_image=4), "train")["crop"].tolist() == out["crop"].tolist()
