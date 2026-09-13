#!/usr/bin/env python
"""How much detection performance is sitting in the post-processing? (no GPU, no re-preprocessing)

    python scripts/detection_sweep.py -c configs/x86_v100.yaml --split test

At ~1σ per-pixel SNR a plume is only detectable by pooling pixels: a coherent N-pixel complex gives
about √N σ. This sweeps the three knobs that do that pooling, straight off the stored `mf.npy`, and
reports a free-response ROC (plume recall vs false alarms per 1000 km² on plume-free scenes):

    normalisation   raw MF (ppm·m, one global threshold) vs per-scene z-score
                    z = (mf − scene background median) / robust σ of that scene, which makes one
                    threshold mean the same thing in a quiet scene and a noisy one
    smoothing       Gaussian σ in pixels before thresholding (the matched spatial filter)
    min size        connected component size required to call a detection

Writes detection_sweep.csv + detection_sweep.png next to the scenes directory. Nothing here changes
the pipeline: it tells you which settings are worth putting into configs (eval.mf_smooth_sigma_px,
eval.min_component_px) and whether a per-scene normalised MF channel is worth a re-preprocess.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ch4hsi import plotting as P  # noqa: E402
from ch4hsi.config import load_config  # noqa: E402
from ch4hsi.metrics import EIGHT  # noqa: E402
from ch4hsi.utils import get_logger, load_json  # noqa: E402

log = get_logger("detection_sweep")
PIXEL_KM2 = 0.0036          # 60 m pixels


def smooth(x, valid, sigma):
    """Normalised-convolution smoothing so invalid pixels do not bleed in."""
    if sigma <= 0:
        return np.where(valid, x, 0).astype(np.float32)
    num = ndimage.gaussian_filter(np.where(valid, x, 0).astype(np.float32), sigma)
    den = ndimage.gaussian_filter(valid.astype(np.float32), sigma)
    out = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0)
    return np.where(valid, out, 0).astype(np.float32)


def scene_scores(d: Path, sigmas, normalise: str):
    """Score maps for one scene, one per smoothing scale."""
    mf = np.nan_to_num(np.load(d / "mf.npy"), nan=0.0).astype(np.float32)
    valid = np.load(d / "valid.npy")
    mask = (np.load(d / "mask.npy") > 0) & valid
    ids = np.load(d / "plume_id.npy") if (d / "plume_id.npy").exists() else mask.astype(np.int16)
    if normalise == "zscore":
        bg = mf[valid & ~mask]
        med = float(np.median(bg)) if bg.size else 0.0
        sig = float(1.4826 * np.median(np.abs(bg - med))) if bg.size else 1.0
        mf = (mf - med) / max(sig, 1e-3)
    return {s: smooth(mf, valid, s) for s in sigmas}, valid, mask, ids


def evaluate(scores, valid, mask, ids, thresholds, min_sizes):
    """Plume-level hits / predicted components / pixel counts for each (threshold, min size)."""
    rows = []
    gt_ids = np.unique(ids[ids > 0])
    for t in thresholds:
        hot = (scores >= t) & valid
        if not hot.any():
            rows += [dict(threshold=t, min_size=ms, gt_total=len(gt_ids), gt_hit=0, pred_total=0, pred_tp=0,
                          tp=0, fp=0, fn=int(mask.sum()), valid_px=int(valid.sum())) for ms in min_sizes]
            continue
        lab, n = ndimage.label(hot, structure=EIGHT)
        sizes = np.bincount(lab.ravel())
        touch = ndimage.maximum(mask, lab, index=np.arange(1, n + 1)).astype(bool) if n else np.zeros(0, bool)
        for ms in min_sizes:
            keep = sizes >= ms
            keep[0] = False
            pred = keep[lab]
            comp_keep = keep[1:]
            hits = ndimage.sum_labels(pred, ids, index=gt_ids) if len(gt_ids) else np.array([])
            rows.append(dict(threshold=t, min_size=ms, gt_total=len(gt_ids), gt_hit=int((hits >= 1).sum()),
                             pred_total=int(comp_keep.sum()), pred_tp=int((comp_keep & touch).sum()),
                             tp=int((pred & mask).sum()), fp=int((pred & ~mask).sum()),
                             fn=int((~pred & mask).sum()), valid_px=int(valid.sum())))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", default=None)
    ap.add_argument("--set", "-s", action="append", default=[])
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--smooth", type=float, nargs="*", default=[0, 1, 2, 4, 8], help="Gaussian σ in pixels")
    ap.add_argument("--min-sizes", type=int, nargs="*", default=[4, 16, 64])
    ap.add_argument("--normalise", nargs="*", default=["raw", "zscore"], choices=["raw", "zscore"])
    ap.add_argument("--n-thresholds", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.set)
    scene_dir = Path(cfg.paths.scenes)
    sp = Path(cfg.paths.splits) / "splits.json"
    ids_all = (sum(load_json(sp)["splits"].values(), []) if a.split == "all" else load_json(sp)["splits"][a.split]) \
        if sp.exists() else sorted(p.parent.name for p in scene_dir.glob("*/meta.json"))
    dirs = [scene_dir / s for s in ids_all if (scene_dir / s / "mf.npy").exists()][: a.scenes]
    if not dirs:
        raise SystemExit(f"no preprocessed scenes for split {a.split} under {scene_dir}")
    log.info("%d scenes | smoothing %s | min sizes %s | %s", len(dirs), a.smooth, a.min_sizes, a.normalise)

    out_rows = []
    for norm in a.normalise:
        # one threshold grid per (normalisation, smoothing): smoothing changes the scale of the scores,
        # so a grid taken from one scale would not cover the useful range of the others
        rng = np.random.default_rng(0)
        pool = {s: [] for s in a.smooth}
        for d in dirs[: min(6, len(dirs))]:
            sc, valid, _, _ = scene_scores(d, a.smooth, norm)
            for s, score in sc.items():
                v = score[valid]
                pool[s].append(rng.choice(v, size=min(50000, v.size), replace=False))
        thr = {s: np.unique(np.quantile(np.concatenate(v), np.linspace(0.90, 0.99995, a.n_thresholds)))
               for s, v in pool.items()}
        for s in a.smooth:
            log.info("[%s σ=%g] thresholds %s", norm, s, np.round(thr[s], 2).tolist())
        for i, d in enumerate(dirs, 1):
            meta = load_json(d / "meta.json")
            sc, valid, mask, ids = scene_scores(d, a.smooth, norm)
            for s, score in sc.items():
                for r in evaluate(score, valid, mask, ids, thr[s], a.min_sizes):
                    r.update(normalise=norm, smooth_px=s, scene_id=meta["scene_id"], role=meta["role"])
                    out_rows.append(r)
            if i % 10 == 0:
                log.info("[%s] %d/%d scenes", norm, i, len(dirs))

    df = pd.DataFrame(out_rows)
    g = df.groupby(["normalise", "smooth_px", "min_size", "threshold"], as_index=False).agg(
        gt_total=("gt_total", "sum"), gt_hit=("gt_hit", "sum"), tp=("tp", "sum"), fp=("fp", "sum"),
        fn=("fn", "sum"), pred_total=("pred_total", "sum"), pred_tp=("pred_tp", "sum"),
        fa_components=("pred_total", lambda s: s[df.loc[s.index, "role"] == "neg"].sum()),
        neg_px=("valid_px", lambda s: s[df.loc[s.index, "role"] == "neg"].sum()))
    g["plume_recall"] = g.gt_hit / g.gt_total.clip(lower=1)
    g["plume_precision"] = g.pred_tp / g.pred_total.clip(lower=1)
    g["pixel_f1"] = 2 * g.tp / (2 * g.tp + g.fp + g.fn).clip(lower=1)
    g["pixel_iou"] = g.tp / (g.tp + g.fp + g.fn).clip(lower=1)
    g["fa_per_1000km2"] = 1000 * g.fa_components / (g.neg_px * PIXEL_KM2).clip(lower=1e-9)

    out = Path(a.out) if a.out else scene_dir.parent / "detection_sweep"
    g.to_csv(out.with_suffix(".csv"), index=False)
    _plot(g, out.with_suffix(".png"), a.split, len(dirs))

    # the operating point each configuration can reach at a tolerable false-alarm rate
    log.info("best plume recall at <= 1 false alarm per 1000 km² (current config = raw, σ=1, min 4 px):")
    ok = g[g.fa_per_1000km2 <= 1.0]
    for (norm, s, ms), sub in ok.groupby(["normalise", "smooth_px", "min_size"]):
        b = sub.loc[sub.plume_recall.idxmax()]
        log.info("  %-6s σ=%-4.1f min=%-3d -> plume recall %.2f (precision %.2f), pixel F1 %.3f, FA %.2f/1000km²",
                 norm, s, ms, b.plume_recall, b.plume_precision, b.pixel_f1, b.fa_per_1000km2)
    log.info("wrote %s and %s", out.with_suffix(".csv"), out.with_suffix(".png"))
    return 0


def _plot(g, path, split, n_scenes):
    plt = P.apply_style()
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2))
    smooths = sorted(g.smooth_px.unique())
    colors = [P.C_MODEL, P.C_MF, P.C_MF_FIXED, P.C_EXTRA, "#8a8985"]
    best_ms = sorted(g.min_size.unique())[len(g.min_size.unique()) // 2]
    for ax, norm in ((axs[0], "raw"), (axs[1], "zscore")):
        sub0 = g[(g.normalise == norm) & (g.min_size == best_ms)]
        if sub0.empty:
            ax.text(0.5, 0.5, f"{norm}: not run", ha="center", color=P.MUTED, transform=ax.transAxes)
            continue
        for s, c in zip(smooths, colors):
            sub = sub0[sub0.smooth_px == s].sort_values("fa_per_1000km2")
            ax.plot(sub.fa_per_1000km2.clip(lower=1e-3), sub.plume_recall, "o-", color=c, ms=4,
                    label=f"σ = {s:g} px")
        ax.set_xscale("log")
        ax.set(xlabel="false alarms per 1000 km² (plume-free scenes)", ylabel="plume recall", ylim=(-0.02, 1.02),
               title=f"{'MF, ppm·m' if norm == 'raw' else 'MF, per-scene z-score'} (min {best_ms} px)")
        ax.legend(ncol=2)
    for s, c in zip(smooths, colors):
        for norm, ls in (("raw", "-"), ("zscore", "--")):
            sub = g[(g.normalise == norm) & (g.smooth_px == s)]
            if not sub.empty:
                axs[2].plot(sub.groupby("min_size").pixel_f1.max().index,
                            sub.groupby("min_size").pixel_f1.max().values, ls, color=c, marker="o", ms=4,
                            label=f"{norm} σ={s:g}")
    axs[2].set_xscale("log")
    axs[2].set(xlabel="minimum component size (pixels)", ylabel="best pixel F1", title="Pixel F1 vs size filter")
    axs[2].legend(fontsize=6.5, ncol=2)
    fig.suptitle(f"Detection post-processing sweep — {n_scenes} {split} scenes, matched-filter map only",
                 x=0.01, ha="left", fontsize=11, weight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
