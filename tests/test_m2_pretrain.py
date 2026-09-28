"""M2 tests: ViT, masks, predictor, I-JEPA step, degradation, samplers, monitors, probes, engine."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import requires_torch

torch = pytest.importorskip("torch")


# --------------------------------------------------------------------------- models


def test_sincos_shapes_and_distinct_positions():
    from talosaur.models.pos_embed import sincos_2d, sincos_3d

    pe = sincos_2d(192, 7, 13)
    assert pe.shape == (91, 192)
    assert np.unique(pe.round(5), axis=0).shape[0] == 91
    assert sincos_3d(192, 2, 7, 13).shape == (182, 192)


@pytest.mark.parametrize("hw", [(112, 112), (160, 160), (224, 224), (112, 208)])
def test_vit_rectangular_grids(hw):
    from talosaur.models.vit import build_vit

    m = build_vit("vit_tiny").eval()
    x = torch.randn(2, 3, *hw)
    with torch.no_grad():
        t, grid = m.forward_grid(x)
    assert grid == (hw[0] // 16, hw[1] // 16)
    assert t.shape == (2, grid[0], grid[1], 192)


def test_sdpa_and_math_attention_agree():
    from talosaur.models.vit import build_vit

    m = build_vit("vit_tiny").eval()
    x = torch.randn(2, 3, 64, 96)
    with torch.no_grad():
        a = m(x)
        m.set_attn_impl("math")
        b = m(x)
    assert torch.allclose(a, b, atol=1e-5)


def test_masked_forward_gathers_tokens():
    from talosaur.models.vit import apply_masks, build_vit

    m = build_vit("vit_tiny", depth=1).eval()
    x = torch.randn(2, 3, 64, 64)
    idx = torch.tensor([[0, 5, 9], [1, 2, 15]])
    t = torch.arange(2 * 16 * 4, dtype=torch.float32).view(2, 16, 4)
    g = apply_masks(t, [idx])
    assert torch.equal(g[1, :, 0], t[1, [1, 2, 15], 0])
    with torch.no_grad():
        assert m(x, [idx]).shape == (2, 3, 192)


def test_param_counts():
    from talosaur.models.vit import build_vit

    n = sum(p.numel() for p in build_vit("vit_tiny").parameters())
    assert 5.3e6 < n < 5.8e6
    n = sum(p.numel() for p in build_vit("vit_small").parameters())
    assert 21e6 < n < 22.5e6


# --------------------------------------------------------------------------- masks


@pytest.mark.parametrize("grid", [(14, 14), (10, 10), (7, 13)])
def test_masks_disjoint_sizes_and_edges(grid):
    from talosaur.ssl.masks import MultiBlockMasks, coverage_stats

    m = MultiBlockMasks()
    enc, preds = m(32, grid, torch.Generator().manual_seed(0))
    assert len(enc) == 1 and len(preds) == 4
    assert enc[0].shape[1] >= 10
    for b in range(32):
        e = set(enc[0][b].tolist())
        assert len(e) == enc[0].shape[1]
        for p in preds:
            assert not e & set(p[b].tolist())
    tgt, ctx = coverage_stats(m, grid, n_batches=20, batch_size=32)
    assert (tgt > 0).all(), "every patch must sometimes be a target"
    assert tgt[-1].mean() > 0.05 and tgt[:, -1].mean() > 0.05


def test_official_compat_reproduces_edge_bias():
    from talosaur.ssl.masks import MultiBlockMasks, coverage_stats

    tgt, ctx = coverage_stats(MultiBlockMasks(official_compat=True), (14, 14), n_batches=20, batch_size=32)
    assert tgt[-1].sum() == 0 and tgt[:, -1].sum() == 0
    assert ctx[:3, :3].mean() > 5 * ctx[-3:, -3:].mean()


def test_masks_deterministic():
    from talosaur.ssl.masks import MultiBlockMasks

    m = MultiBlockMasks()
    a = m(8, (14, 14), torch.Generator().manual_seed(3))
    b = m(8, (14, 14), torch.Generator().manual_seed(3))
    assert all(torch.equal(x, y) for x, y in zip(a[1], b[1])) and torch.equal(a[0][0], b[0][0])


# --------------------------------------------------------------------------- I-JEPA step


def _tiny_ijepa():
    from talosaur.models.predictor import Predictor
    from talosaur.models.vit import build_vit
    from talosaur.ssl.ijepa import IJEPA

    enc = build_vit("vit_tiny", depth=2, embed_dim=64, num_heads=2)
    return IJEPA(enc, Predictor(64, 32, 2, 2))


def test_ijepa_loss_shapes_and_overfit():
    from talosaur.ssl.masks import MultiBlockMasks

    torch.manual_seed(0)
    model = _tiny_ijepa()
    x = torch.randn(4, 3, 64, 64)
    enc, preds = MultiBlockMasks(min_keep=4)(4, (4, 4), torch.Generator().manual_seed(0))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
    losses = []
    for _ in range(30):
        loss, stats = model(x, x, enc, preds)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
    assert all(p.grad is None for p in model.target_encoder.parameters())
    assert stats["pred_std"] > 0


def test_momentum_update():
    model = _tiny_ijepa()
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(1.0)
    before = [p.clone() for p in model.target_encoder.parameters()]
    model.momentum_update(0.9)
    for b, t, e in zip(before, model.target_encoder.parameters(), model.encoder.parameters()):
        assert torch.allclose(t, 0.9 * b + 0.1 * e, atol=1e-6)


def test_schedules():
    from talosaur.ssl import schedules as s

    assert s.warmup_cosine(0, 100, 10, 1.0, 0.2, 0.0) == pytest.approx(0.2)
    assert s.warmup_cosine(10, 100, 10, 1.0, 0.2, 0.0) == pytest.approx(1.0)
    assert s.warmup_cosine(100, 100, 10, 1.0, 0.2, 0.0) == pytest.approx(0.0, abs=1e-9)
    assert s.weight_decay(0, 100, 0.04, 0.4) == pytest.approx(0.04)
    assert s.weight_decay(100, 100, 0.04, 0.4) == pytest.approx(0.4)
    assert s.linear(50, 100, 0.996, 1.0) == pytest.approx(0.998)


# --------------------------------------------------------------------------- degradation


def _batch(n=4, h=64, w=96):
    from talosaur.data.synthetic import make_scene

    rng = np.random.default_rng(0)
    imgs = [make_scene(rng, h, w, condition="clear", n_animals=1).image for _ in range(n)]
    return torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255


def test_degrade_range_determinism_and_modes():
    from talosaur.augment.degrade import DegradeConfig, UnderwaterDegradation, make_views

    x = _batch()
    d = UnderwaterDegradation(DegradeConfig(p=1.0))
    a = d(x, torch.Generator().manual_seed(5))
    b = d(x, torch.Generator().manual_seed(5))
    assert torch.equal(a, b) and a.shape == x.shape
    assert a.min() >= 0 and a.max() <= 1 and not torch.equal(a, x)
    c0, t0 = make_views(x, "none", d)
    assert c0 is x and t0 is x
    c1, t1 = make_views(x, "context_only", d, torch.Generator().manual_seed(1))
    assert t1 is x and not torch.equal(c1, x)
    c2, t2 = make_views(x, "shared", d, torch.Generator().manual_seed(1))
    assert c2 is t2
    c3, t3 = make_views(x, "independent", d, torch.Generator().manual_seed(1))
    assert not torch.equal(c3, t3)
    with pytest.raises(ValueError):
        make_views(x, "bogus", d)


def test_degrade_low_light_darkens_and_skips_dark_inputs():
    from talosaur.augment.degrade import DegradeConfig, UnderwaterDegradation

    x = _batch()
    d = UnderwaterDegradation(DegradeConfig(p=1.0))
    ops = {"low_light": True}
    y = d(x, torch.Generator().manual_seed(0), ops=ops, severity=torch.ones(len(x)))
    assert y.mean() < 0.6 * x.mean()
    dark = x * 0.05
    yd = d(dark, torch.Generator().manual_seed(0), ops=ops, severity=torch.ones(len(x)))
    # already-dark input: low light is skipped, only the 8-bit round trip remains
    assert torch.allclose(yd, torch.round(dark * 255) / 255, atol=1.5 / 255)


def test_jpeg_like_high_quality_is_near_identity():
    from talosaur.augment.degrade import jpeg_like

    x = _batch(2, 64, 64)
    y = jpeg_like(x, torch.tensor([95.0, 95.0]))
    assert (y - x).abs().mean() < 0.02
    z = jpeg_like(x, torch.tensor([10.0, 10.0]))
    assert (z - x).abs().mean() > (y - x).abs().mean()


def test_particles_and_eval_variant_run():
    from talosaur.augment.degrade import DegradeConfig, UnderwaterDegradation

    x = _batch()
    for variant in ("train", "eval"):
        d = UnderwaterDegradation(DegradeConfig(p=1.0, variant=variant))
        y = d(
            x,
            torch.Generator().manual_seed(2),
            ops={"particles": True, "light": True},
            severity=torch.ones(len(x)),
        )
        assert (y > x + 0.1).any()


# --------------------------------------------------------------------------- data / samplers


def test_distributed_weighted_sampler_shards_and_resumes():
    from talosaur.data.torch_datasets import DistributedWeightedSampler

    w = np.r_[np.ones(50), 10 * np.ones(50)]
    s0 = DistributedWeightedSampler(w, 100, rank=0, world=2, seed=1)
    s1 = DistributedWeightedSampler(w, 100, rank=1, world=2, seed=1)
    a, b = list(s0), list(s1)
    assert len(a) == len(b) == 50
    heavy = np.mean([i >= 50 for i in a + b])
    assert heavy > 0.8
    s0.set_start(10)
    assert list(s0) == a[10:]
    s0.set_epoch(1)
    s0.set_start(0)
    assert list(s0) != a


def test_multires_batch_sampler_resume_alignment():
    from talosaur.data.torch_datasets import DistributedWeightedSampler, MultiResBatchSampler

    s = DistributedWeightedSampler(np.ones(64), 64, seed=0)
    bs = MultiResBatchSampler(s, 8, [(64, 64), (48, 96)], seed=0)
    bs.set_epoch(0)
    full = list(bs)
    assert len(full) == 8 and {b[0][1] for b in full} <= {(64, 64), (48, 96)}
    bs.set_epoch(0)
    bs.set_start(3)
    assert list(bs) == full[3:]


def test_pretrain_images_from_files(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image

    from talosaur.data.torch_datasets import PretrainImages

    rng = np.random.default_rng(0)
    for i in range(3):
        Image.fromarray(rng.integers(0, 255, (90, 160, 3), dtype=np.uint8)).save(tmp_path / f"{i}.jpg")
    import pandas as pd

    ds = PretrainImages(pd.DataFrame({"path": [f"{i}.jpg" for i in range(3)]}), tmp_path, (64, 64))
    assert ds[0].shape == (3, 64, 64) and ds[0].dtype == torch.uint8
    assert ds[(1, (48, 96))].shape == (3, 48, 96)


# --------------------------------------------------------------------------- monitors / probes / metrics


def test_collapse_metrics():
    from talosaur.monitor.collapse import collapse_report, embedding_std, is_collapsed, rankme

    torch.manual_seed(0)
    good = torch.randn(512, 64)
    bad = torch.randn(512, 1) @ torch.randn(1, 64) + 1e-4 * torch.randn(
        512, 64
    )  # rank-1 (dimensional collapse)
    const = torch.randn(1, 64).expand(512, 64) + 1e-3 * torch.randn(512, 64)  # complete collapse
    assert rankme(good) > 40 and rankme(bad) < 2
    assert embedding_std(good) > 0.8 and embedding_std(const) < 0.1
    rep = collapse_report(bad.view(8, 64, 64), prefix="t/")
    assert is_collapsed(rep, prefix="t/")
    assert not is_collapsed(collapse_report(good.view(8, 64, 64), prefix="t/"), prefix="t/")


def test_metrics_known_values():
    from talosaur.eval.metrics import angular_error_deg, average_precision, roc_auc, soft_centroid, tpr_at_fpr

    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx(0.75)
    assert roc_auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert average_precision(y, np.array([0.1, 0.4, 0.35, 0.8])) == pytest.approx((1 + 2 / 3) / 2)
    assert (
        tpr_at_fpr(np.r_[np.zeros(100), np.ones(10)], np.r_[np.linspace(0, 1, 100), np.full(10, 2.0)], 0.05)
        == 1.0
    )
    hm = np.zeros((4, 4))
    hm[1, 2] = 1.0
    cx, cy, _ = soft_centroid(hm)
    assert (cx, cy) == pytest.approx((0.625, 0.375))
    assert angular_error_deg((0.6, 0.5), (0.5, 0.5)) == pytest.approx(10.2)


def test_logreg_probe_separates():
    from talosaur.eval.linear import fit_logreg
    from talosaur.eval.metrics import roc_auc

    torch.manual_seed(0)
    X = torch.randn(400, 16)
    y = (X[:, 0] + 0.3 * X[:, 1] > 0).long()
    lin = fit_logreg(X * 5 + 3, y)
    s = lin(X * 5 + 3).squeeze(-1).detach().numpy()
    assert roc_auc(y.numpy(), s) > 0.98


def test_probe_hook_synthetic_runs():
    from talosaur.models.vit import build_vit
    from talosaur.monitor.probe_hook import ProbeHook

    hook = ProbeHook.synthetic(40, 24, (64, 64), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    enc = build_vit("vit_tiny", depth=2)
    out = hook.run(enc, torch.device("cpu"))
    for k in ("probe/frame_auroc", "probe/patch_auroc", "probe/centroid_err_deg_median"):
        assert k in out
    assert 0.0 <= out["probe/patch_auroc"] <= 1.0


# --------------------------------------------------------------------------- engine smoke


@requires_torch
@pytest.mark.slow
def test_engine_debug_run_and_resume(tmp_path):
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    from talosaur.ssl.engine import Trainer

    cfg_dir = str((__import__("pathlib").Path(__file__).parents[1] / "configs").resolve())
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        cfg = compose(
            config_name="pretrain",
            overrides=[
                "experiment=debug_cpu",
                f"out_dir={tmp_path}",
                "train.max_steps=6",
                "degrade=context_only",
            ],
        )
    last = Trainer(cfg).train()
    assert (tmp_path / "latest.pt").exists() and (tmp_path / "encoder_target.pt").exists()
    assert "probe/target/patch_auroc" in last and "probe/context/patch_auroc" in last
    sd = torch.load(tmp_path / "latest.pt", weights_only=False)
    assert sd["global_step"] == 6
    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        cfg = compose(
            config_name="pretrain",
            overrides=["experiment=debug_cpu", f"out_dir={tmp_path}", "train.max_steps=9"],
        )
    t = Trainer(cfg)
    assert t.global_step == 6
    t.train()
    assert torch.load(tmp_path / "latest.pt", weights_only=False)["global_step"] == 9
    lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
    assert any('"train/loss"' in ln for ln in lines) and any('"target/patch_rankme"' in ln for ln in lines)
