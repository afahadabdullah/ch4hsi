"""Synthetic methane plumes for minimum-detection-limit (MDL) experiments and augmentation.

Steady 2-D Gaussian plume (column-integrated) with Briggs rural class-D lateral spread:
    Omega(x, y) = Q / (sqrt(2 pi) sigma_y(x) U) * exp(-y^2 / (2 sigma_y^2))   [kg m-2]
    sigma_y(x)  = 0.08 x (1 + 1e-4 x)^-1/2
The cross-wind integral of each sub-pixel is done analytically (erf), so mass is conserved even
next to the source. The plume is truncated at x = U * age (snapshot of a plume released `age`
seconds ago). Optional multiplicative log-normal "turbulence" breaks the idealised shape.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import erf

from .units import kgm2_to_ppmm


def sigma_y(x):
    x = np.maximum(x, 1e-3)
    return 0.08 * x / np.sqrt(1.0 + 1e-4 * x)


def gaussian_plume_ppmm(shape, src_rc, q_kgph, u_ms, theta_rad, pixel_m=60.0, supersample=5,
                        max_age_s=900.0, turbulence=0.0, rng=None, pressure_pa=101325.0,
                        temperature_k=273.15) -> np.ndarray:
    """Column enhancement (ppm·m) on an (H, W) grid of square pixels; source at pixel (r, c).

    theta: wind direction (radians, 0 = +column direction, counter-clockwise).
    """
    H, W = shape
    rng = rng if rng is not None else np.random.default_rng()
    s = supersample
    d = pixel_m / s
    # only evaluate a window around the source for speed
    Lmax = u_ms * max_age_s
    rad_px = int(np.ceil(Lmax / pixel_m)) + 3
    r0, r1 = max(0, int(src_rc[0]) - rad_px), min(H, int(src_rc[0]) + rad_px + 1)
    c0, c1 = max(0, int(src_rc[1]) - rad_px), min(W, int(src_rc[1]) + rad_px + 1)
    out = np.zeros((H, W), np.float64)
    if r1 <= r0 or c1 <= c0:
        return out.astype(np.float32)
    rr = (np.arange((r1 - r0) * s) + 0.5) / s + r0           # sub-pixel centres in pixel units
    cc = (np.arange((c1 - c0) * s) + 0.5) / s + c0
    R, C = np.meshgrid(rr, cc, indexing="ij")
    sr, sc = src_rc[0] + 0.5, src_rc[1] + 0.5
    dx_m = (C - sc) * pixel_m
    dy_m = -(R - sr) * pixel_m                                # image rows increase downward
    x = dx_m * np.cos(theta_rad) + dy_m * np.sin(theta_rad)   # downwind
    y = -dx_m * np.sin(theta_rad) + dy_m * np.cos(theta_rad)  # crosswind
    q = q_kgph / 3600.0
    xs = np.maximum(x, d / 2)
    sy = sigma_y(xs)
    # mass per sub-cell = (Q/U) * dx * P(|y| within cell) ; column density = mass / d^2
    frac = 0.5 * (erf((y + d / 2) / (np.sqrt(2) * sy)) - erf((y - d / 2) / (np.sqrt(2) * sy)))
    mass = (q / u_ms) * d * frac
    mass = np.where((x > -d / 2) & (x < Lmax), mass, 0.0)
    col = mass / d ** 2                                       # kg m-2 on the sub-grid
    if turbulence > 0:
        noise = gaussian_filter(rng.standard_normal(col.shape), sigma=2 * s)
        noise /= noise.std() + 1e-12
        col = col * np.exp(turbulence * noise - 0.5 * turbulence ** 2)
    # aggregate sub-pixels (mean column density per pixel)
    colpx = col.reshape(r1 - r0, s, c1 - c0, s).mean(axis=(1, 3))
    out[r0:r1, c0:c1] = kgm2_to_ppmm(colpx, pressure_pa, temperature_k)
    return out.astype(np.float32)


def inject(rad: np.ndarray, ppmm_map: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Beer-Lambert injection L' = L * exp(k * dX) for the bands of `rad` (rows, cols, b)."""
    return rad * np.exp(ppmm_map[..., None].astype(np.float64) * k[None, None, :]).astype(rad.dtype)


def random_sources(valid: np.ndarray, n: int, min_sep: int, margin: int, rng) -> list[tuple[int, int]]:
    H, W = valid.shape
    cand = np.argwhere(valid[margin:H - margin, margin:W - margin]) + margin
    rng.shuffle(cand)
    chosen: list[tuple[int, int]] = []
    for r, c in cand:
        if all((r - a) ** 2 + (c - b) ** 2 >= min_sep ** 2 for a, b in chosen):
            chosen.append((int(r), int(c)))
            if len(chosen) == n:
                break
    return chosen
