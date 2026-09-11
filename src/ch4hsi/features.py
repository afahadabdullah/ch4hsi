"""Per-pixel input features. All are defined in a sensor-agnostic way (physical units or
self-normalised radiance), so an EMIT-trained model can be applied to AVIRIS-NG features.

  mf          matched-filter enhancement (ppm·m)
  mf_snr      mf / sigma_column (dimensionless)
  swir_resid  n bins of (L / mu_column - 1) across the retrieval window (CH4 absorption = dips)
  albedo      mean window radiance / scene median (surface brightness; drives MF false positives)
  rgb         visible radiance / scene p99 (surface context: roads, roofs, water, clouds)
"""
from __future__ import annotations

import numpy as np

from .io.glt import glt_index, ortho
from .physics.matched_filter import columnwise_mf


def channel_names(feature_list, n_swir_bins):
    names = []
    for f in feature_list:
        if f == "swir_resid":
            names += [f"swir_resid_{i}" for i in range(n_swir_bins)]
        elif f == "rgb":
            names += ["rgb_r", "rgb_g", "rgb_b"]
        elif f in ("mf", "mf_snr", "albedo"):
            names.append(f)
        else:
            raise ValueError(f"unknown feature {f}")
    return names


def band_selector(phys, pre):
    lo, hi = phys.mf_window_nm

    def sel(wl, fwhm, good):
        m = (wl >= lo) & (wl <= hi) & good
        for a, b in phys.exclude_nm:
            m &= ~((wl >= a) & (wl <= b))
        rgb = np.array([int(np.argmin(np.abs(wl - x))) for x in pre.rgb_nm])
        return {"win": np.where(m)[0], "rgb": rgb}

    return sel


def build_features(rad_win, rad_rgb, valid, mf, sigma, mu, feature_list, n_bins):
    rows, cols, b = rad_win.shape
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        if "mf" in feature_list:
            out["mf"] = mf[..., None]
        if "mf_snr" in feature_list:
            out["mf_snr"] = (mf / sigma[None, :])[..., None]
        if "swir_resid" in feature_list:
            resid = rad_win / mu[None, :, :] - 1.0
            bins = np.array_split(np.arange(b), n_bins)
            out["swir_resid"] = np.stack([resid[..., ix].mean(axis=-1) for ix in bins], axis=-1)
        if "albedo" in feature_list:
            a = rad_win.mean(axis=-1)
            med = np.nanmedian(a[valid]) if valid.any() else 1.0
            out["albedo"] = (a / med)[..., None]
        if "rgb" in feature_list:
            rgb = rad_rgb.astype(np.float32)
            p99 = np.array([np.nanpercentile(rgb[..., i][valid], 99) if valid.any() else 1.0 for i in range(3)])
            out["rgb"] = np.clip(rgb / np.maximum(p99, 1e-6)[None, None, :], 0, 3)
    stack = np.concatenate([out[f].astype(np.float32) for f in feature_list], axis=-1)
    stack[~valid] = np.nan
    return stack


def scene_products(rad_win, rad_rgb, k, glt_x, glt_y, phys, pre, feature_list, valid=None, extra_sensor=None):
    """Full sensor->ortho product chain for one scene (also reused by the MDL injection loop).

    Returns dict with ortho arrays: features (C,H,W) float32, valid (H,W) bool, mf (H,W), sigma (H,W),
    plus any `extra_sensor` arrays (rows, cols) mapped to ortho.
    """
    from .io.emit import valid_mask
    if valid is None:
        valid = valid_mask(rad_win) & valid_mask(rad_rgb)
    mf, sigma, mu = columnwise_mf(rad_win, k, valid, column_group=phys.column_group, shrinkage=phys.shrinkage,
                                  two_pass=phys.two_pass, outlier_sigma=phys.outlier_sigma,
                                  albedo_correction=phys.albedo_correction)
    feats = build_features(rad_win, rad_rgb, valid, mf, sigma, mu, feature_list, pre.n_swir_bins)
    rows, cols = mf.shape
    aux = [mf, np.broadcast_to(sigma[None, :], (rows, cols)), valid.astype(np.float32)]
    names = ["mf", "sigma", "valid"]
    for n, a in (extra_sensor or {}).items():
        aux.append(a.astype(np.float32))
        names.append(n)
    cache = glt_index(glt_x, glt_y)
    F = ortho(feats, glt_x, glt_y, _cache=cache)
    A = ortho(np.stack(aux, axis=-1), glt_x, glt_y, _cache=cache)
    res = {n: A[..., i] for i, n in enumerate(names)}
    v = (np.nan_to_num(res.pop("valid"), nan=0) > 0.5) & np.all(np.isfinite(F), axis=-1)
    res["features"] = np.moveaxis(F, -1, 0)
    res["valid"] = v
    res["mu_window_mean"] = float(np.nanmean(mu))
    return res
