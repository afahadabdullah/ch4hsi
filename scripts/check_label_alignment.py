#!/usr/bin/env python
"""Are the plume-complex labels actually on top of the matched-filter enhancements?

    python scripts/check_label_alignment.py -c configs/x86_v100.yaml [--split test] [--scenes 40]

For every labelled scene it compares the MF enhancement inside the labels with the background, and
cross-correlates the label mask against the MF map over a ±`--max-shift` pixel window. Three outcomes:

  excess at shift (0, 0)          labels sit on the enhancements. Read `snr_inside` for whether the
                                  signal is usable: > ~2 is comfortable, < ~1 means the plumes are at or
                                  below the MF noise and pixel-level metrics will look terrible even
                                  though nothing is misplaced (compare σ against `ch4hsi mf-check`).
  excess at shift (dy, dx) ≠ 0    systematic geolocation offset (label reprojection / GLT / CRS).
  no excess at any shift           labels do not correspond to these granules at all (wrong scene↔plume
                                  matching or wrong granule).

Writes <run or scenes dir>/label_alignment.{png,json} and prints a verdict. Read it together with
`ch4hsi mf-check`, which compares our MF against the operational L2B CH4ENH: that separates "our MF is
wrong" from "the labels are in the wrong place".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ch4hsi import plotting as P  # noqa: E402
from ch4hsi.config import load_config  # noqa: E402
from ch4hsi.utils import dump_json, get_logger, load_json  # noqa: E402

log = get_logger("label_alignment")


def scene_stats(d: Path, max_shift: int):
    """MF inside vs outside the labels, and the label-mask × MF cross-correlation surface."""
    mask = np.load(d / "mask.npy") > 0
    valid = np.load(d / "valid.npy")
    if not (mask & valid).any():
        return None
    mf = np.nan_to_num(np.load(d / "mf.npy"), nan=0.0).astype(np.float32)
    meta = load_json(d / "meta.json")
    sigma = float(meta.get("sigma_median_ppmm") or np.nan)
    inside, outside = mf[mask & valid], mf[~mask & valid]
    # robust background scale: the MF background has a heavy tail, so std() overstates the noise
    bg_med = float(np.median(outside))
    bg_sigma = float(1.4826 * np.median(np.abs(outside - bg_med))) or float(np.std(outside))
    m = (mask & valid).astype(np.float32)
    x = np.where(valid, mf, 0.0)
    corr = fftconvolve(x, m[::-1, ::-1], mode="same") / max(m.sum(), 1)     # mean MF under the shifted mask
    cy, cx = corr.shape[0] // 2, corr.shape[1] // 2
    s = max_shift
    win = corr[max(0, cy - s):cy + s + 1, max(0, cx - s):cx + s + 1]
    dy, dx = np.unravel_index(np.argmax(win), win.shape)
    off = (int(dy - min(s, cy)), int(dx - min(s, cx)))
    return dict(scene_id=meta["scene_id"], n_label_px=int(mask.sum()), sigma_ppmm=sigma,
                inside_median=float(np.median(inside)), inside_p90=float(np.percentile(inside, 90)),
                outside_median=bg_med, outside_p90=float(np.percentile(outside, 90)),
                outside_std=float(np.std(outside)), bg_sigma_robust=bg_sigma,
                excess_ppmm=float(np.median(inside) - bg_med),
                snr_inside=float((np.median(inside) - bg_med) / max(bg_sigma, 1e-6)),
                frac_label_px_gt_2sigma=float(np.mean(inside > bg_med + 2 * bg_sigma)),
                z_inside=float((np.median(inside) - bg_med) / max(np.std(outside), 1e-6)),
                best_shift_y=off[0], best_shift_x=off[1],
                corr_at_zero=float(corr[cy, cx]), corr_at_best=float(win[dy, dx]), window=win)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", default=None)
    ap.add_argument("--set", "-s", action="append", default=[])
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--scenes", type=int, default=40, help="max labelled scenes to check")
    ap.add_argument("--max-shift", type=int, default=25, help="cross-correlation search radius (pixels)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    cfg = load_config(a.config, a.set)
    scene_dir = Path(cfg.paths.scenes)
    sp = Path(cfg.paths.splits) / "splits.json"
    ids = (sum(load_json(sp)["splits"].values(), []) if a.split == "all" else load_json(sp)["splits"][a.split]) \
        if sp.exists() else sorted(p.parent.name for p in scene_dir.glob("*/meta.json"))
    rows = []
    for sid in ids:
        d = scene_dir / sid
        if not (d / "mask.npy").exists():
            continue
        r = scene_stats(d, a.max_shift)
        if r:
            rows.append(r)
            log.info("%s  label px %6d  MF inside %7.0f vs outside %7.0f ppm·m (SNR %.2f)  best shift (%+d, %+d)",
                     r["scene_id"], r["n_label_px"], r["inside_median"], r["outside_median"], r["snr_inside"],
                     r["best_shift_y"], r["best_shift_x"])
        if len(rows) >= a.scenes:
            break
    if not rows:
        raise SystemExit(f"no labelled scenes found in {scene_dir} for split {a.split}")

    snr = np.array([r["snr_inside"] for r in rows])
    exc = np.array([r["excess_ppmm"] for r in rows])
    sy = np.array([r["best_shift_y"] for r in rows])
    sx = np.array([r["best_shift_x"] for r in rows])
    at_zero = float(np.mean((sy == 0) & (sx == 0)))
    gain = np.array([r["corr_at_best"] / max(r["corr_at_zero"], 1e-6) for r in rows])
    aligned = at_zero > 0.5 or (abs(np.median(sy)) <= 1 and abs(np.median(sx)) <= 1)
    has_signal = np.median(exc) > 0 and np.mean(exc > 0) > 0.8        # labels are brighter than background
    if has_signal and aligned:
        verdict = ("labels sit on the enhancements" if np.median(snr) >= 2 else
                   f"labels sit on the enhancements, but weakly: median excess {np.median(exc):.0f} ppm·m is only "
                   f"{np.median(snr):.1f}x the background scatter — pixel metrics will be poor for noise reasons, "
                   f"not label reasons")
    elif has_signal:
        verdict = (f"systematic offset: the labels are brighter than background but the correlation peaks at "
                   f"({int(np.median(sy)):+d}, {int(np.median(sx)):+d}) px, not (0, 0)")
    else:
        verdict = "no MF excess under the labels at any shift — labels and imagery do not correspond"
    summary = dict(n_scenes=len(rows), split=a.split,
                   median_excess_ppmm=float(np.median(exc)), median_snr_inside=float(np.median(snr)),
                   frac_scenes_positive_excess=float(np.mean(exc > 0)), frac_scenes_snr_gt_2=float(np.mean(snr > 2)),
                   median_bg_sigma_ppmm=float(np.median([r["bg_sigma_robust"] for r in rows])),
                   median_frac_label_px_gt_2sigma=float(np.median([r["frac_label_px_gt_2sigma"] for r in rows])),
                   median_shift=[int(np.median(sy)), int(np.median(sx))],
                   frac_best_shift_zero=at_zero, median_corr_gain_at_best=float(np.median(gain)),
                   median_sigma_ppmm=float(np.nanmedian([r["sigma_ppmm"] for r in rows])), verdict=verdict,
                   scenes=[{k: v for k, v in r.items() if k != "window"} for r in rows])

    out = Path(a.out) if a.out else scene_dir.parent / "label_alignment"
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(summary, out.with_suffix(".json"))
    _plot(rows, summary, out.with_suffix(".png"), a.max_shift)
    log.info("median excess under labels %.0f ppm·m = %.2f x background scatter (robust σ %.0f ppm·m; MF σ %.0f) | "
             "best shift (%+d, %+d) in %.0f%% of scenes", summary["median_excess_ppmm"], summary["median_snr_inside"],
             summary["median_bg_sigma_ppmm"], summary["median_sigma_ppmm"], *summary["median_shift"], 100 * at_zero)
    log.info("VERDICT: %s", verdict)
    log.info("wrote %s and %s", out.with_suffix(".json"), out.with_suffix(".png"))
    return 0


def _plot(rows, summary, path, max_shift):
    plt = P.apply_style()
    stack = np.stack([r["window"] for r in rows if r["window"].shape == rows[0]["window"].shape])
    mean_win = stack.mean(0)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    ins = [r["inside_median"] for r in rows]
    out = [r["outside_median"] for r in rows]
    axs[0].scatter(out, ins, s=26, color=P.C_MODEL, edgecolor=P.SURFACE, lw=0.7)
    lim = [min(out + ins), max(out + ins)]
    axs[0].plot(lim, lim, color=P.GRID, lw=1.2)
    axs[0].set(xlabel="median MF outside labels (ppm·m)", ylabel="median MF inside labels (ppm·m)",
               title="Per scene: labelled vs background")
    axs[1].hist([r["snr_inside"] for r in rows], bins=20, color=P.C_MODEL)
    axs[1].axvline(0, color=P.MUTED, lw=1)
    axs[1].axvline(2, color=P.C_MF, lw=1.2, ls="--")
    axs[1].set(xlabel="(median inside − background) / robust σ of background", ylabel="scenes",
               title=f"Label SNR (median {summary['median_snr_inside']:.2f}; dashed = usable)")
    ext = [-max_shift, max_shift, max_shift, -max_shift]
    im = axs[2].imshow(mean_win, cmap=P.ppmm_cmap(), extent=ext)
    fig.colorbar(im, ax=axs[2], fraction=0.046, label="mean MF under shifted labels (ppm·m)")
    axs[2].plot(0, 0, "+", color=P.C_MODEL, ms=12, mew=2)
    dy, dx = summary["median_shift"]
    axs[2].plot(dx, dy, "x", color=P.C_MF, ms=10, mew=2)
    axs[2].set(xlabel="label shift, columns (px)", ylabel="label shift, rows (px)",
               title="Cross-correlation (+ = no shift, × = best)")
    axs[2].grid(False)
    fig.suptitle(f"Label / imagery alignment — {summary['n_scenes']} {summary['split']} scenes: "
                 f"{summary['verdict']}", x=0.01, ha="left", fontsize=11, weight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
