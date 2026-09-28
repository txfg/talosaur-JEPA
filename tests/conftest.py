import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


def _has(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


HAS_TORCH = _has("torch")
HAS_PIL = _has("PIL")
HAS_AV = _has("av")
HAS_PANDAS = _has("pandas") and _has("pyarrow")
HAS_ORT = _has("onnxruntime")
HAS_ONNX = _has("onnx")

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
requires_pil = pytest.mark.skipif(not HAS_PIL, reason="Pillow not installed")
requires_av = pytest.mark.skipif(not HAS_AV, reason="PyAV not installed")
requires_pandas = pytest.mark.skipif(not HAS_PANDAS, reason="pandas/pyarrow not installed")
requires_ort = pytest.mark.skipif(not HAS_ORT, reason="onnxruntime not installed")
requires_onnx = pytest.mark.skipif(
    not (HAS_ONNX and HAS_ORT and HAS_TORCH), reason="onnx/onnxruntime/torch missing"
)


@pytest.fixture(scope="session")
def synthetic_index(tmp_path_factory):
    """A curated synthetic index (fetch + curation), built once per test session.

    Returns (root, index_path)."""
    if not (HAS_PANDAS and HAS_PIL and HAS_AV):
        pytest.skip("needs pandas/pyarrow, Pillow and PyAV")
    from talosaur.data.curate import CurateConfig, run_curation
    from talosaur.data.sources import run_fetch

    root = tmp_path_factory.mktemp("synthetic_data")
    run_fetch("synthetic", root, "synthetic", n_images=120, n_videos=2)
    cfg = CurateConfig(
        name="syn", root=str(root), sources=["synthetic"], workers=1, val_frac=0.2, test_frac=0.25
    )
    run_curation(cfg)
    return root, root / "index" / "syn.parquet"
