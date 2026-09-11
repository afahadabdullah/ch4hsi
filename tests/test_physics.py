"""Physics / geometry unit tests (numpy + scipy only). Run: pytest tests/  or  python tests/test_physics.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ch4hsi.io.glt import ortho, sensor_from_ortho  # noqa: E402
from ch4hsi.physics.matched_filter import columnwise_mf  # noqa: E402
from ch4hsi.physics.plume_sim import gaussian_plume_ppmm, inject  # noqa: E402
from ch4hsi.physics.target import synthetic_lut, unit_absorption  # noqa: E402
from ch4hsi.physics.units import analytic_flux_mdl, kgm2_to_ppmm, ppmm_to_kgm2  # noqa: E402


def emit_like_bands():
    wl = np.arange(2122, 2488, 7.4)
    return wl, np.full_like(wl, 8.5)


def synthetic_scene(rows=300, cols=60, seed=0):
    rng = np.random.default_rng(seed)
    wl, fwhm = emit_like_bands()
    b = len(wl)
    # smooth surface spectra: albedo * (1 + low-order spectral shape) + per-column gain + noise
    base = 2.0 + 0.3 * np.cos((wl - 2100) / 120.0)
    alb = 0.6 + 0.4 * rng.random((rows, cols, 1))
    shape = 1 + 0.05 * rng.standard_normal((rows, cols, 3)) @ np.stack(
        [np.ones(b), (wl - wl.mean()) / 200, ((wl - wl.mean()) / 200) ** 2])
    gain = 1 + 0.02 * rng.standard_normal((1, cols, b))
    rad = alb * base[None, None, :] * shape * gain
    rad += 0.01 * rng.standard_normal(rad.shape)
    return rad.astype(np.float32), wl, fwhm


def test_units_roundtrip():
    assert abs(ppmm_to_kgm2(1.0) - 7.156e-7) < 5e-10
    assert np.isclose(kgm2_to_ppmm(ppmm_to_kgm2(1234.0)), 1234.0)


def test_unit_absorption_matches_injected_physics():
    wl, fwhm = emit_like_bands()
    k = unit_absorption(wl, fwhm, lut=synthetic_lut())
    assert k.shape == wl.shape and k.min() < -1e-6 and np.all(k < 1e-7)
    # strongest absorption of the toy LUT is near 2320-2360 nm
    assert 2300 < wl[np.argmin(k)] < 2380


def test_mf_recovers_enhancement():
    rad, wl, fwhm = synthetic_scene()
    k = unit_absorption(wl, fwhm, lut=synthetic_lut())
    truth = np.zeros(rad.shape[:2], np.float32)
    truth[100:110, 20:30] = 1500.0
    rad_p = inject(rad, truth, k)
    mf, sigma, mu = columnwise_mf(rad_p, k, shrinkage=1e-3, two_pass=True, albedo_correction=True)
    assert mf.shape == truth.shape and sigma.shape == (rad.shape[1],)
    plume = mf[100:110, 20:30].mean()
    bg = mf[truth == 0]
    assert abs(plume - 1500) < 0.25 * 1500, plume          # first-order MF bias is small at 1500 ppm·m
    assert abs(np.median(bg)) < 3 * np.median(sigma)
    # sigma is the noise-equivalent enhancement: empirical background spread should be comparable
    assert 0.3 < np.std(bg) / np.median(sigma) < 3.0


def test_mf_column_group_and_invalid():
    rad, wl, fwhm = synthetic_scene(cols=37)
    k = unit_absorption(wl, fwhm, lut=synthetic_lut())
    valid = np.ones(rad.shape[:2], bool)
    valid[:, 5] = False
    valid[10:20, :] = False
    mf, sigma, _ = columnwise_mf(rad, k, valid=valid, column_group=4)
    assert np.all(np.isnan(mf[:, 5])) and np.all(np.isnan(mf[10:20]))
    assert np.isfinite(mf[valid]).all()


def test_glt_roundtrip():
    rows, cols = 20, 15
    sensor = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
    H, W = 25, 18
    gx = np.zeros((H, W), int)
    gy = np.zeros((H, W), int)
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    gy[2:2 + rows, 1:1 + cols] = rr + 1        # 1-based
    gx[2:2 + rows, 1:1 + cols] = cc + 1
    o = ortho(sensor[..., None], gx, gy)[..., 0]
    assert np.isnan(o[0, 0]) and o[2, 1] == 0 and o[2 + rows - 1, cols] == sensor[-1, -1]
    back = sensor_from_ortho(o[..., None], gx, gy)[..., 0]
    assert np.array_equal(back, sensor)


def test_plume_mass_conservation():
    # mass on the grid must equal Q * age (while plume is inside the grid)
    q, u, age, px = 1000.0, 3.0, 600.0, 60.0
    m = gaussian_plume_ppmm((200, 200), (100, 100), q, u, 0.3, pixel_m=px, max_age_s=age)
    mass = ppmm_to_kgm2(m).sum() * px * px
    expected = q / 3600 * age
    assert abs(mass - expected) / expected < 0.03, (mass, expected)
    assert m.max() > 100  # 1 t/h at 60 m should give O(10^2-10^3) ppm·m near source


def test_analytic_mdl_order_of_magnitude():
    q = analytic_flux_mdl(sigma_ppmm=150, n_pix=9, pixel_m=60, u_ms=3)
    assert 100 < q < 5000


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
