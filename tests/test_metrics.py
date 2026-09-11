import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ch4hsi.metrics import (HistPR, auroc, object_counts, pixel_counts, pod_with_ci, prf,  # noqa: E402
                            remove_small)


def test_histpr_perfect_and_random():
    rng = np.random.default_rng(0)
    gt = rng.random((100, 100)) < 0.1
    valid = np.ones_like(gt)
    h = HistPR(0, 1, 1000)
    h.update(gt.astype(float) * 0.9 + 0.05, gt, valid)
    assert h.average_precision() > 0.99 and h.best_f1()["f1"] > 0.99
    h2 = HistPR(0, 1, 1000)
    h2.update(rng.random(gt.shape), gt, valid)
    assert abs(h2.average_precision() - gt.mean()) < 0.03


def test_pixel_and_object():
    gt = np.zeros((50, 50), int)
    gt[5:10, 5:10] = 1
    gt[30:35, 30:35] = 2
    pred = np.zeros((50, 50), bool)
    pred[6:9, 6:9] = True           # hits plume 1
    pred[40:45, 5:10] = True        # false alarm
    pred[20, 20] = True             # speck (removed)
    pred = remove_small(pred, 4)
    valid = np.ones_like(pred)
    tp, fp, fn = pixel_counts(pred, gt > 0, valid)
    assert (tp, fp, fn) == (9, 25, 41)
    m = prf(tp, fp, fn)
    assert abs(m["iou"] - 9 / 75) < 1e-9
    o = object_counts(pred, gt, valid)
    assert o["gt_total"] == 2 and o["gt_detected"] == 1 and o["pred_total"] == 2 and o["pred_tp"] == 1


def test_auroc():
    assert auroc([3, 4, 5], [0, 1, 2]) == 1.0
    assert auroc([1, 1], [1, 1]) == 0.5


def test_pod_fit():
    rng = np.random.default_rng(1)
    q = np.repeat([100, 200, 300, 500, 750, 1000, 1500, 2000, 3000], 40)
    true_a, true_b = np.log10(600), 0.12
    p = 1 / (1 + np.exp(-(np.log10(q) - true_a) / true_b))
    det = rng.random(len(q)) < p
    out = pod_with_ci(q, det, n_boot=50)
    assert 450 < out["mdl50"] < 800, out
    assert out["mdl90"] > out["mdl50"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
