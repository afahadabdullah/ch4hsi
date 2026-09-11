"""Detection metrics: pixel PR / IoU, plume(object)-level matching, scene AUROC, POD curves."""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize

EIGHT = np.ones((3, 3), bool)


class HistPR:
    """Streaming precision-recall over a fixed score grid (memory-constant over many scenes)."""

    def __init__(self, lo=0.0, hi=1.0, n_bins=2000):
        self.edges = np.linspace(lo, hi, n_bins + 1)
        self.pos = np.zeros(n_bins + 1, np.int64)   # last bin: >= hi
        self.neg = np.zeros(n_bins + 1, np.int64)

    def update(self, score, gt, valid):
        s = score[valid]
        g = gt[valid].astype(bool)
        idx = np.clip(np.searchsorted(self.edges, s, side="right") - 1, 0, len(self.pos) - 1)
        self.pos += np.bincount(idx[g], minlength=len(self.pos))
        self.neg += np.bincount(idx[~g], minlength=len(self.neg))

    def curve(self):
        """precision, recall, thresholds for 'score >= threshold'."""
        tp = np.cumsum(self.pos[::-1])[::-1]
        fp = np.cumsum(self.neg[::-1])[::-1]
        P = self.pos.sum()
        thr = np.append(self.edges[:-1], self.edges[-1])
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(tp + fp > 0, tp / (tp + fp), 1.0)
            rec = tp / max(P, 1)
        return prec, rec, thr

    def average_precision(self):
        prec, rec, _ = self.curve()
        order = np.argsort(rec)
        r, p = rec[order], prec[order]
        # step-wise interpolation (sklearn-like): sum over recall increments of precision
        p_interp = np.maximum.accumulate(p[::-1])[::-1]
        return float(np.sum(np.diff(np.concatenate([[0], r])) * p_interp))

    def best_f1(self):
        prec, rec, thr = self.curve()
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec + 1e-12), 0)
        i = int(np.argmax(f1))
        return dict(threshold=float(thr[i]), f1=float(f1[i]), precision=float(prec[i]), recall=float(rec[i]))

    def state(self):
        return dict(edges=self.edges, pos=self.pos, neg=self.neg)


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if (tp + fp) and (tp + fn) and (p + r) else float("nan")
    iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
    return dict(precision=p, recall=r, f1=f1, iou=iou)


def remove_small(binary, min_px):
    if min_px <= 1:
        return binary
    lab, n = ndimage.label(binary, structure=EIGHT)
    if n == 0:
        return binary
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def pixel_counts(pred, gt, valid):
    p, g = pred & valid, gt.astype(bool) & valid
    tp = int((p & g).sum())
    return tp, int((p & ~g).sum()), int((~p & g).sum())


def object_counts(pred, gt_ids, valid, min_overlap=1):
    """Plume-level matching. gt_ids: int map (0 = background, k = plume complex k).

    A GT plume is detected if >= min_overlap predicted pixels fall inside it; a predicted component
    is a true positive if it touches any GT plume, otherwise a false alarm.
    """
    pred = pred & valid
    gt_ids = np.where(valid, gt_ids, 0)
    ids = np.unique(gt_ids[gt_ids > 0])
    hits = ndimage.sum_labels(pred, gt_ids, index=ids) if len(ids) else np.array([])
    detected = hits >= min_overlap
    lab, n = ndimage.label(pred, structure=EIGHT)
    if n:
        touch = ndimage.maximum(gt_ids > 0, lab, index=np.arange(1, n + 1)).astype(bool)
    else:
        touch = np.array([], bool)
    return dict(gt_total=int(len(ids)), gt_detected=int(detected.sum()), pred_total=int(n),
                pred_tp=int(touch.sum()), per_gt=dict(zip(ids.tolist(), detected.tolist())))


def auroc(scores_pos, scores_neg):
    sp, sn = np.asarray(scores_pos, float), np.asarray(scores_neg, float)
    if len(sp) == 0 or len(sn) == 0:
        return float("nan")
    allv = np.concatenate([sp, sn])
    ranks = _rankdata(allv)
    return float((ranks[: len(sp)].sum() - len(sp) * (len(sp) + 1) / 2) / (len(sp) * len(sn)))


def _rankdata(a):
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


# ---------------------------------------------------------------- probability of detection
def _nll(params, x, y):
    a, logb = params
    z = (x - a) / np.exp(logb)
    p = 1 / (1 + np.exp(-np.clip(z, -50, 50)))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_pod(values, detected):
    """Logistic POD in log10(value). Returns dict(a, b) with POD = sigmoid((log10 v - a) / b)."""
    x = np.log10(np.asarray(values, float))
    y = np.asarray(detected, float)
    a0 = np.median(x)
    res = minimize(_nll, x0=[a0, np.log(0.2)], args=(x, y), method="Nelder-Mead",
                   options=dict(xatol=1e-4, fatol=1e-6, maxiter=4000))
    return dict(a=float(res.x[0]), b=float(np.exp(res.x[1])), converged=bool(res.success))


def pod_quantile(fit, p):
    return float(10 ** (fit["a"] + fit["b"] * np.log(p / (1 - p))))


def pod_with_ci(values, detected, levels=(0.5, 0.9), n_boot=500, seed=0):
    values, detected = np.asarray(values, float), np.asarray(detected, float)
    fit = fit_pod(values, detected)
    out = {f"mdl{int(l * 100)}": pod_quantile(fit, l) for l in levels}
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in out}
    for _ in range(n_boot):
        ix = rng.integers(0, len(values), len(values))
        if detected[ix].min() == detected[ix].max():
            continue
        f = fit_pod(values[ix], detected[ix])
        for l in levels:
            boots[f"mdl{int(l * 100)}"].append(pod_quantile(f, l))
    for k, v in boots.items():
        if v:
            out[k + "_ci"] = [float(np.percentile(v, 5)), float(np.percentile(v, 95))]
    out["fit"] = fit
    return out
