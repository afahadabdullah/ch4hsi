"""Model / dataset tests (skipped where torch is not installed). Runs on NCCS via `pytest -q tests`."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
torch = pytest.importorskip("torch")

from ch4hsi.config import load_config  # noqa: E402
from ch4hsi.models.unet import UNet  # noqa: E402


def test_unet_shapes():
    m = UNet(18, base=8, depth=3)
    for hw in (64, 72):
        y = m(torch.randn(2, 18, hw, hw))
        assert y.shape == (2, 1, hw, hw)


def test_dataset_and_one_step(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_pipeline_numpy import build_pipeline  # synthetic scenes -> preprocess -> split

    import ch4hsi.dataset as dsm
    from ch4hsi.train import masked_loss
    from ch4hsi.utils import load_json

    _, root = build_pipeline(tmp_path)
    cfg = load_config(None, [f"paths.data_root={root}", "train.tile=32", "train.tiles_per_epoch=16", "train.val_tiles=8",
                             "train.synth_aug.enabled=true", "train.synth_aug.prob=1.0"])
    norm = load_json(root / "splits" / "norm.json")
    dirs = [str(p) for p in (root / "scenes").iterdir()]
    ds = dsm.TileDataset(dirs, norm, cfg.train, "train", seed=0)
    x, y, v = ds[0]
    assert x.shape == (18, 32, 32) and y.shape == (1, 32, 32) and v.dtype == torch.bool
    assert torch.isfinite(x).all()
    model = UNet(18, base=8, depth=3)
    xb = torch.stack([ds[i][0] for i in range(4)])
    yb = torch.stack([ds[i][1] for i in range(4)])
    vb = torch.stack([ds[i][2] for i in range(4)])
    loss, _, _ = masked_loss(model(xb), yb, vb, 5.0, 1.0)
    loss.backward()
    assert np.isfinite(float(loss))
