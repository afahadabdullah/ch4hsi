"""Unit conversions and integrated-mass-enhancement (IME) flux estimates."""
from __future__ import annotations

import numpy as np

R_GAS = 8.314462618      # J mol-1 K-1
M_CH4 = 0.01604          # kg mol-1


def ppmm_to_kgm2(ppmm, pressure_pa=101325.0, temperature_k=273.15):
    """1 ppm·m = 1e-6 * n_air * M_CH4 kg m-2  (~7.16e-7 kg m-2 at 0 C / 1 atm)."""
    n_air = pressure_pa / (R_GAS * temperature_k)           # mol m-3
    return np.asarray(ppmm) * 1e-6 * n_air * M_CH4


def kgm2_to_ppmm(kgm2, pressure_pa=101325.0, temperature_k=273.15):
    return np.asarray(kgm2) / ppmm_to_kgm2(1.0, pressure_pa, temperature_k)


def ime_kg(ppmm_map, mask, pixel_area_m2, **kw) -> float:
    vals = np.where(mask & np.isfinite(ppmm_map), ppmm_map, 0.0)
    return float(ppmm_to_kgm2(vals.sum(), **kw) * pixel_area_m2)


def ime_flux_kgph(ime, plume_length_m, u_eff_ms) -> float:
    """Q = U_eff * IME / L  (Varon et al. 2018), returned in kg h-1."""
    return float(u_eff_ms * ime / max(plume_length_m, 1e-6) * 3600.0)


def analytic_flux_mdl(sigma_ppmm, n_pix, pixel_m, u_ms, k_sigma=3.0, **kw) -> float:
    """Back-of-envelope MDL: smallest plume of n_pix pixels each at k_sigma * sigma.

    IME = n_pix * A * k*sigma, L = sqrt(n_pix * A), Q = U * IME / L.
    """
    A = pixel_m ** 2
    ime = n_pix * A * float(ppmm_to_kgm2(k_sigma * sigma_ppmm, **kw))
    return ime_flux_kgph(ime, np.sqrt(n_pix * A), u_ms)
