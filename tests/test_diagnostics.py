"""Evaluation core + diagnostic suite on a tiny in-memory synthetic dataset (no torch / netCDF4 / rasterio).

The model score comes from the per-pixel logistic-regression baseline, which exercises exactly the code
path `ch4hsi evaluate` / `ch4hsi diagnose` use with the U-Net.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ch4hsi import demo  # noqa: E402
from ch4hsi.baselines import PixelLogReg  # noqa: E402
from ch4hsi.config import run_dir  # noqa: E402
from ch4hsi.diagnostics import diagnose  # noqa: E402
from ch4hsi.evaluate import bootstrap_ci, evaluate_scenes  # noqa: E402
from ch4hsi.plotting import wilson  # noqa: E402
from ch4hsi.utils import load_json  # noqa: E402


def test_wilson_and_bootstrap():
    lo, hi = wilson(np.array([0, 5, 10]), np.array([10, 10, 10]))
    assert np.all(lo <= np.array([0, 0.5, 1.0])) and np.all(hi >= np.array([0, 0.5, 1.0]))
    assert 0 < lo[1] < 0.5 < hi[1] < 1
    rng = np.random.default_rng(0)
    sdf = pd.DataFrame({f"m_{k}": rng.integers(0, 50, 20) for k in
                        ("tp", "fp", "fn", "gt_total", "gt_detected", "pred_total", "pred_tp")})
    ci = bootstrap_ci(sdf, "m", n_boot=200)
    assert set(ci) >= {"f1", "iou", "precision", "recall"}
    assert all(a <= b for a, b in ci.values())


def test_evaluate_and_diagnose(tmp_path=None):
    root = Path(tmp_path or tempfile.mkdtemp())
    ss = demo.build(root, n_pos=6, n_neg=3, rows=120, cols=110, seed=1, q_range=(800, 5000),
                    extra_overrides=["split.method=random", "split.fractions=[0.34,0.33,0.33]", "eval.bootstrap=50"])
    cfg = ss.cfg
    sp = load_json(Path(cfg.paths.splits) / "splits.json")["splits"]
    norm = load_json(Path(cfg.paths.splits) / "norm.json")
    lr = PixelLogReg(norm).fit([Path(cfg.paths.scenes) / s for s in sp["train"]], n_per_scene=4000)
    out = run_dir(cfg)

    def score(sc):
        return lr.predict(sc["features"], sc["valid"])

    res = evaluate_scenes(cfg, score, out, label="Pixel logistic regression")
    for m in ("model", "mf", "mf_fixed"):
        assert {"f1", "iou", "ap", "plume_recall", "ci90"} <= set(res[m])
    assert res["model_label"] == "Pixel logistic regression"
    for f in ("thresholds.json", "metrics_test.json", "per_scene_test.csv", "per_plume_test.csv", "pr_hist.npz",
              "reliability.npz"):
        assert (out / f).exists(), f
    sdf = pd.read_csv(out / "per_scene_test.csv")
    assert {"model_tp", "mf_fp", "mf_fixed_fn", "model_iou"} <= set(sdf.columns)
    # the pooled counts in the csv reproduce the headline metrics
    tp, fp = sdf.model_tp.sum(), sdf.model_fp.sum()
    assert abs(res["model"]["precision"] - tp / max(tp + fp, 1)) < 1e-9

    summary = diagnose(cfg, out, score_fn=score, watermark="test")
    d = out / "diagnostics"
    for name in ("02_threshold_sweep", "03_roc", "04_score_distributions", "05_reliability", "10_noise",
                 "12_split_map"):
        assert (d / f"{name}.png").exists(), name
    assert (d / "DIAGNOSTICS.md").exists() and (d / "diagnostics_summary.json").exists()
    assert any((d / "maps").glob("*.png"))
    assert 0 <= summary["calibration"]["ece"] <= 1
    print("evaluate + diagnose ok:", {k: round(v, 3) for k, v in res["model"].items() if isinstance(v, float)})


if __name__ == "__main__":
    test_wilson_and_bootstrap()
    test_evaluate_and_diagnose()
    print("ok")
