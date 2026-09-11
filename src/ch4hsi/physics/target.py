"""CH4 unit-absorption spectrum k(lambda) = d ln L / d(ppm·m).

The high-resolution LUT (radiance for 0..16000 ppm·m CH4 enhancements) is the one shipped with
mag1c (Foote et al. 2020, BSD-3; `pip install mag1c` puts ch4.hdr/ch4.lut in the package dir).
The convolution + log-linear fit follows mag1c's `generate_template_from_bands`, but we keep the
slope in physical units (per ppm·m) so that a matched filter with target t = mu * k returns
ppm·m directly and synthetic plumes can be injected as L' = L * exp(k * dX).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np

from ..io.envi import open_envi

LUT_CONCENTRATIONS = np.array([0, 500, 1000, 2000, 4000, 8000, 16000], dtype=np.float64)  # ppm·m


def find_lut_dir(spec: str = "mag1c") -> Path:
    env = os.environ.get("CH4HSI_LUT_DIR")
    if env:
        return Path(env)
    if spec not in ("mag1c", "", None):
        return Path(spec)
    s = importlib.util.find_spec("mag1c")  # does not import mag1c (and therefore not torch)
    if s is None or not s.submodule_search_locations:
        raise FileNotFoundError(
            "CH4 LUT not found. `pip install mag1c` (ships ch4.hdr/ch4.lut) or set physics.lut to a "
            "directory containing ch4.hdr + ch4.lut, or CH4HSI_LUT_DIR.")
    return Path(list(s.submodule_search_locations)[0])


def load_lut(spec: str = "mag1c"):
    """Return (wavelengths_nm [nw], radiance [n_conc, nw], concentrations [n_conc])."""
    if spec == "synthetic":
        return synthetic_lut()
    d = find_lut_dir(spec)
    arr, hdr, wl, _ = open_envi(d / "ch4.hdr", d / "ch4.lut", mmap=False)
    rads = np.asarray(arr, dtype=np.float64).reshape(-1, arr.shape[-1])
    if rads.shape[0] != len(LUT_CONCENTRATIONS):
        raise ValueError(f"unexpected LUT shape {rads.shape}")
    return wl, rads, LUT_CONCENTRATIONS


def synthetic_lut():
    """Toy LUT with CH4-like absorption near 2200-2400 nm. For tests / smoke runs ONLY."""
    wl = np.arange(2000.0, 2600.0, 0.5)
    tau = np.zeros_like(wl)
    for c, w, a in [(2210, 6, 0.4), (2250, 8, 0.3), (2290, 5, 0.7), (2320, 7, 1.0), (2360, 6, 0.8),
                    (2375, 4, 0.5), (2420, 8, 0.4), (2460, 10, 0.3)]:
        tau += a * np.exp(-0.5 * ((wl - c) / w) ** 2)
    base = 1.0 + 0.1 * np.sin(wl / 50.0)
    k_true = -2.5e-5 * tau            # per ppm·m, peak ~ -2.5e-5 (EMIT-like magnitude)
    rads = np.stack([base * np.exp(k_true * c) for c in LUT_CONCENTRATIONS])
    return wl, rads, LUT_CONCENTRATIONS


def unit_absorption(centers_nm, fwhm_nm, lut=None, spec: str = "mag1c") -> np.ndarray:
    """k for each band (negative where CH4 absorbs), units: per ppm·m."""
    centers = np.asarray(centers_nm, dtype=np.float64)
    fwhm = np.asarray(fwhm_nm, dtype=np.float64)
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(fwhm))):
        raise ValueError("non-finite band centres / FWHM")
    wave, rads, conc = lut if lut is not None else load_lut(spec)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    resp = np.exp(-((wave[:, None] - centers[None, :]) ** 2) / (2 * sigma[None, :] ** 2))
    s = resp.sum(axis=0)
    resp = np.divide(resp, s, out=np.zeros_like(resp), where=s > 0)
    resampled = rads @ resp                                   # (n_conc, nb)
    lograd = np.log(np.clip(resampled, 1e-12, None))
    A = np.stack([np.ones_like(conc), conc]).T
    slope, *_ = np.linalg.lstsq(A, lograd, rcond=None)
    return slope[1]
