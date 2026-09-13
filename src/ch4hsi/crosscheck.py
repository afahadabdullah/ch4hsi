"""Cross-check our column-wise MF against the official EMIT L2B CH4 enhancement (EMITL2BCH4ENH).

`ch4hsi fetch-enh` downloads the operational enhancement for some plume scenes; this stage compares it
with ours on the same grid and answers the question the label-alignment check leaves open: when
plume pixels barely rise above the background, is that the scene or is it our retrieval?

Per scene it reports the correlation (raw and lightly smoothed), the robust background noise of both
products (MAD-based, over pixels outside the labels), the median enhancement inside the labels, and
the resulting in-plume SNR. `noise_ratio = sigma_ours / sigma_ref` much above 1 means our MF is the
weak link; ~1 with a low SNR in both means the plumes really are near the noise floor for EMIT.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Cfg
from .io.emit import scene_id_from_name
from .utils import dump_json, get_logger

log = get_logger(__name__)


def robust_sigma(x: np.ndarray) -> float:
    """MAD-based noise scale; the MF background has a heavy tail that inflates std()."""
    x = x[np.isfinite(x)]
    if x.size < 10:
        return float("nan")
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return float(1.4826 * mad) if mad > 0 else float(np.std(x))


def _smooth(a, sigma=1.0):
    from scipy import ndimage
    return ndimage.gaussian_filter(np.nan_to_num(a, nan=0.0).astype(np.float32), sigma)


def compare(ours: np.ndarray, ref: np.ndarray, valid: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Pure-array comparison of two enhancement maps (ppm·m) on the same grid."""
    ok = valid & np.isfinite(ours) & np.isfinite(ref)
    if ok.sum() < 100:
        return dict(n_px=int(ok.sum()), error="too few comparable pixels")
    a, b = ours[ok].astype(np.float64), ref[ok].astype(np.float64)
    lab = (mask.astype(bool) & ok) if mask is not None else np.zeros_like(ok)
    bg = ok & ~lab
    out = dict(n_px=int(ok.sum()), n_label_px=int(lab.sum()),
               pearson_r=float(np.corrcoef(a, b)[0, 1]),
               pearson_r_smoothed=float(np.corrcoef(_smooth(np.where(valid, ours, 0))[ok],
                                                    _smooth(np.where(valid, ref, 0))[ok])[0, 1]),
               sigma_ours=robust_sigma(ours[bg]), sigma_ref=robust_sigma(ref[bg]),
               median_bg_ours=float(np.median(ours[bg])), median_bg_ref=float(np.median(ref[bg])))
    out["noise_ratio"] = out["sigma_ours"] / out["sigma_ref"] if out["sigma_ref"] else float("nan")
    hi = b > np.percentile(b, 99.5)
    out["slope_top05pct"] = float(np.polyfit(b[hi], a[hi], 1)[0]) if hi.sum() > 10 else float("nan")
    if lab.sum() > 20:
        for nm, arr, sig, bgm in (("ours", ours, out["sigma_ours"], out["median_bg_ours"]),
                                  ("ref", ref, out["sigma_ref"], out["median_bg_ref"])):
            med = float(np.median(arr[lab]))
            out[f"label_median_{nm}"] = med
            out[f"label_snr_{nm}"] = float((med - bgm) / sig) if sig and np.isfinite(sig) else float("nan")
    return out


def run(cfg: Cfg):
    import rasterio
    enh_dir = Path(cfg.paths.raw) / "enh"
    files = sorted(p for p in enh_dir.glob("*") if p.suffix.lower() in (".tif", ".tiff"))
    log.info("%d enhancement file(s) under %s", len(files), enh_dir)
    if not files:
        others = sorted(p.name for p in enh_dir.glob("*"))[:5] if enh_dir.exists() else []
        log.warning("nothing to compare: run `ch4hsi fetch-enh` first (dir exists: %s, e.g. %s)",
                    enh_dir.exists(), others or "empty")
        return
    rows, skipped = [], {"no_scene": 0, "unparsed": 0, "grid": 0}
    for tif in files:
        try:
            sid = scene_id_from_name(tif.name)
        except ValueError:
            skipped["unparsed"] += 1
            continue
        d = Path(cfg.paths.scenes) / sid
        if not (d / "mf.npy").exists():
            skipped["no_scene"] += 1
            continue
        ours, valid = np.load(d / "mf.npy"), np.load(d / "valid.npy")
        mask = np.load(d / "mask.npy") if (d / "mask.npy").exists() else None
        with rasterio.open(tif) as src:
            ref = src.read(1).astype(np.float32)
            nod = src.nodata if src.nodata is not None else -9999.0
        ref = np.where(ref == nod, np.nan, ref)
        if ref.shape != ours.shape:
            log.warning("%s: grid mismatch, ours %s vs L2B %s — skipping", sid, ours.shape, ref.shape)
            skipped["grid"] += 1
            continue
        r = compare(ours, ref, valid, mask)
        r["scene_id"] = sid
        rows.append(r)
        log.info("%s r=%.3f (smoothed %.3f) slope=%.2f | noise ours/ref = %.0f/%.0f ppm·m (%.1fx) | "
                 "in-plume SNR ours %.2f vs ref %.2f", sid, r.get("pearson_r", float("nan")),
                 r.get("pearson_r_smoothed", float("nan")), r.get("slope_top05pct", float("nan")),
                 r.get("sigma_ours", float("nan")), r.get("sigma_ref", float("nan")),
                 r.get("noise_ratio", float("nan")), r.get("label_snr_ours", float("nan")),
                 r.get("label_snr_ref", float("nan")))
    if not rows:
        log.warning("no scene matched an enhancement file (%s). Scene ids come from the file name; check "
                    "that %s holds *CH4ENH* GeoTIFFs for scenes that were preprocessed.", skipped, enh_dir)
        return
    med = {k: float(np.nanmedian([r.get(k, np.nan) for r in rows]))
           for k in ("pearson_r", "pearson_r_smoothed", "slope_top05pct", "sigma_ours", "sigma_ref",
                     "noise_ratio", "label_snr_ours", "label_snr_ref")}
    summary = dict(n_scenes=len(rows), skipped=skipped, medians=med, scenes=rows)
    nr = med["noise_ratio"]
    summary["verdict"] = (
        "our MF is much noisier than the operational product — fix the retrieval before the model" if nr > 1.5 else
        "our MF matches the operational noise level; the plumes are simply near EMIT's detection floor"
        if med["label_snr_ref"] < 2 else
        "our MF is comparable to the operational product and the plumes are detectable in it")
    dump_json(summary, Path(cfg.paths.catalog) / "mf_crosscheck.json")
    log.info("median over %d scenes: r=%.3f | noise ours/ref = %.0f/%.0f ppm·m (%.1fx) | in-plume SNR %.2f vs %.2f",
             len(rows), med["pearson_r"], med["sigma_ours"], med["sigma_ref"], nr, med["label_snr_ours"],
             med["label_snr_ref"])
    log.info("VERDICT: %s", summary["verdict"])
    _plot(rows, summary, Path(cfg.paths.catalog) / "mf_crosscheck.png")


def _plot(rows, summary, path):
    from . import plotting as P
    plt = P.apply_style()
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4))
    r = [x.get("pearson_r", np.nan) for x in rows]
    axs[0].hist(r, bins=20, color=P.C_MODEL)
    axs[0].set(xlabel="Pearson r vs L2B CH4ENH", ylabel="scenes",
               title=f"Agreement (median {summary['medians']['pearson_r']:.2f})")
    so, sr = [x.get("sigma_ours", np.nan) for x in rows], [x.get("sigma_ref", np.nan) for x in rows]
    axs[1].scatter(sr, so, s=26, color=P.C_MODEL, edgecolor=P.SURFACE, lw=0.7)
    lim = [0, float(np.nanmax(so + sr)) * 1.05]
    axs[1].plot(lim, lim, color=P.GRID, lw=1.2)
    axs[1].set(xlabel="operational L2B noise σ (ppm·m)", ylabel="our MF noise σ (ppm·m)", xlim=lim, ylim=lim,
               title=f"Background noise ({summary['medians']['noise_ratio']:.1f}× operational)")
    for k, c, lab in (("label_snr_ours", P.C_MODEL, "our MF"), ("label_snr_ref", P.C_MF, "operational L2B")):
        v = [x.get(k, np.nan) for x in rows]
        if np.isfinite(v).any():
            axs[2].hist(np.asarray(v, float)[np.isfinite(v)], bins=15, histtype="step", lw=2, color=c, label=lab)
    axs[2].axvline(2, color=P.MUTED, lw=1, ls="--")
    axs[2].set(xlabel="in-plume SNR (median label enhancement / background σ)", ylabel="scenes",
               title="Is the labelled plume above the noise?")
    axs[2].legend()
    fig.suptitle(f"MF cross-check vs EMIT L2B CH4ENH — {summary['verdict']}", x=0.01, ha="left", fontsize=11,
                 weight="semibold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("wrote %s", path)
