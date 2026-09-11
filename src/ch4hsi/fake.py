"""Tiny synthetic EMIT-format dataset for end-to-end smoke tests (no internet, no Earthdata login).

Writes EMIT_L1B_RAD_*.nc files with the same variable/group layout as EMITL1BRAD, one GeoTIFF per
synthetic plume complex, and catalog/plumes.csv + catalog/scenes.csv, so every stage from
`preprocess` onwards runs exactly as on real data.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .config import Cfg
from .io.glt import ortho
from .physics.plume_sim import gaussian_plume_ppmm, inject
from .physics.target import unit_absorption
from .utils import get_logger

log = get_logger(__name__)
DX = 0.000542  # deg, ~60 m
SITES = [(-103.5, 32.0), (55.0, 39.0), (6.5, 31.5), (-108.0, 36.8), (112.0, 37.5), (52.0, 45.0), (24.0, 29.0), (-101.5, 31.0)]


def emit_bands():
    wl = np.linspace(381.0, 2493.0, 285)
    fwhm = np.full_like(wl, 8.5)
    good = ~(((wl > 1340) & (wl < 1450)) | ((wl > 1800) & (wl < 1960)))
    return wl, fwhm, good.astype(np.uint8)


def _surface(rows, cols, wl, rng):
    a = gaussian_filter(rng.standard_normal((rows, cols)), 6)
    alb = np.clip(0.25 + 0.12 * a / (a.std() + 1e-9), 0.03, 0.7)
    mix = np.clip(0.5 + 0.5 * gaussian_filter(rng.standard_normal((rows, cols)), 10) * 8, 0, 1)
    soil = 0.6 + 0.4 * (wl - 380) / 2100
    veg = np.where(wl < 700, 0.15, 0.8) * np.exp(-((wl - 1650) / 900) ** 2) + 0.1
    T = 5800.0
    planck = 1 / (wl * 1e-9) ** 5 / (np.exp(1.4388e-2 / (wl * 1e-9 * T)) - 1)
    planck /= planck.max()
    trans = np.ones_like(wl)
    for c, w, d in [(940, 25, 0.5), (1140, 30, 0.5), (1380, 40, 0.98), (1880, 60, 0.98), (2010, 15, 0.3), (2060, 15, 0.3)]:
        trans *= 1 - d * np.exp(-0.5 * ((wl - c) / w) ** 2)
    refl = alb[..., None] * (mix[..., None] * soil + (1 - mix[..., None]) * veg)
    gain = 1 + 0.01 * gaussian_filter(rng.standard_normal((1, cols, len(wl))), (0, 0, 3))
    L = 12.0 * refl * (planck * trans)[None, None, :] * gain
    L += rng.standard_normal(L.shape) * (0.003 * L + 0.0002)   # ~0.4-0.5% in the SWIR
    return L.astype(np.float32)


def _glt(rows, cols, angle_deg):
    th = np.deg2rad(angle_deg)
    c, s = np.cos(th), np.sin(th)
    H = int(np.ceil(abs(rows * c) + abs(cols * s))) + 4
    W = int(np.ceil(abs(rows * s) + abs(cols * c))) + 4
    ii, jj = np.meshgrid(np.arange(H) - H / 2 + 0.5, np.arange(W) - W / 2 + 0.5, indexing="ij")
    r = c * ii - s * jj + rows / 2
    q = s * ii + c * jj + cols / 2
    ri, qi = np.floor(r).astype(int), np.floor(q).astype(int)
    ok = (ri >= 0) & (ri < rows) & (qi >= 0) & (qi < cols)
    gy = np.where(ok, ri + 1, 0).astype(np.int32)
    gx = np.where(ok, qi + 1, 0).astype(np.int32)
    return gx, gy


def make_scene(rows, cols, k_full, n_plumes, rng, wl):
    L = _surface(rows, cols, wl, rng)
    maps = []
    for _ in range(n_plumes):
        src = (int(rng.integers(30, rows - 30)), int(rng.integers(30, cols - 30)))
        q = float(np.exp(rng.uniform(np.log(800), np.log(6000))))
        maps.append(gaussian_plume_ppmm((rows, cols), src, q, 3.0, float(rng.uniform(0, 2 * np.pi)),
                                        max_age_s=700, turbulence=0.3, rng=rng))
    if maps:
        L = inject(L, np.sum(maps, axis=0), k_full)
    return L, maps


def write_emit_nc(path, L, wl, fwhm, good, gx, gy, gt):
    import netCDF4
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("downtrack", L.shape[0])
        ds.createDimension("crosstrack", L.shape[1])
        ds.createDimension("bands", L.shape[2])
        v = ds.createVariable("radiance", "f4", ("downtrack", "crosstrack", "bands"), fill_value=-9999.0)
        v[:] = L
        g = ds.createGroup("sensor_band_parameters")
        for n, a, t in (("wavelengths", wl, "f4"), ("fwhm", fwhm, "f4"), ("good_wavelengths", good, "u1")):
            g.createVariable(n, t, ("bands",))[:] = a
        loc = ds.createGroup("location")
        loc.createDimension("ortho_y", gx.shape[0])
        loc.createDimension("ortho_x", gx.shape[1])
        loc.createVariable("glt_x", "i4", ("ortho_y", "ortho_x"))[:] = gx
        loc.createVariable("glt_y", "i4", ("ortho_y", "ortho_x"))[:] = gy
        ds.setncattr("geotransform", np.array(gt, dtype=np.float64))
        ds.setncattr("spatial_ref", "EPSG:4326")


def write_plume_tif(path, ppmm_ortho, gt, thr=150.0):
    import rasterio
    from affine import Affine
    m = ppmm_ortho >= thr
    if not m.any():
        return None
    rr, cc = np.nonzero(m)
    r0, r1, c0, c1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
    sub = np.where(m, ppmm_ortho, -9999.0)[r0:r1, c0:c1].astype(np.float32)
    tf = Affine.from_gdal(gt[0] + c0 * gt[1], gt[1], 0.0, gt[3] + r0 * gt[5], 0.0, gt[5])
    with rasterio.open(path, "w", driver="GTiff", height=sub.shape[0], width=sub.shape[1], count=1, dtype="float32",
                       crs="EPSG:4326", transform=tf, nodata=-9999.0) as dst:
        dst.write(sub, 1)
    return (gt[0] + c0 * gt[1], gt[3] + r1 * gt[5], gt[0] + c1 * gt[1], gt[3] + r0 * gt[5])


def run(cfg: Cfg, n_pos: int = 10, n_neg: int = 6, rows: int = 200, cols: int = 180, seed: int = 0):
    rng = np.random.default_rng(seed)
    raw, cat = Path(cfg.paths.raw), Path(cfg.paths.catalog)
    (cat / "plume_assets").mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    wl, fwhm, good = emit_bands()
    k_full = unit_absorption(wl, fwhm, spec=cfg.physics.lut)
    t0 = datetime(2024, 5, 1, 17, 0, 0, tzinfo=timezone.utc)
    plumes, scenes = [], []
    pid = 0
    for i in range(n_pos + n_neg):
        role = "pos" if i < n_pos else "neg"
        t = t0 + timedelta(days=3 * i, seconds=37 * i)
        ts = t.strftime("%Y%m%dT%H%M%S")
        sid = f"{ts}_{2400000 + i:07d}_{(i % 30) + 1:03d}"
        L, maps = make_scene(rows, cols, k_full, int(rng.integers(1, 4)) if role == "pos" else 0, rng, wl)
        gx, gy = _glt(rows, cols, angle_deg=float(rng.uniform(-15, 15)))
        lon, lat = SITES[i % len(SITES)]
        gt = (lon + rng.uniform(-0.3, 0.3), DX, 0.0, lat + rng.uniform(-0.3, 0.3), 0.0, -DX)
        name = f"EMIT_L1B_RAD_001_{sid}.nc"
        write_emit_nc(raw / name, L, wl, fwhm, good, gx, gy, gt)
        ids = []
        for m in maps:
            pid += 1
            pname = f"EMIT_L2B_CH4PLM_002_{ts}_{pid:06d}"
            bbox = write_plume_tif(cat / "plume_assets" / f"{pname}.tif", ortho(m[..., None], gx, gy)[..., 0], gt)
            if bbox is None:
                continue
            ids.append(pname)
            plumes.append(dict(plume_id=pname, source="fake", time=t.isoformat(), west=bbox[0], south=bbox[1],
                               east=bbox[2], north=bbox[3], tif_path=str(cat / "plume_assets" / f"{pname}.tif"),
                               json_path="", scene_tokens=sid))
        H, W = gx.shape
        scenes.append(dict(scene_id=sid, granule=name[:-3], time=t.isoformat(), west=gt[0], south=gt[3] + H * gt[5],
                           east=gt[0] + W * gt[1], north=gt[3], cloud=0, role=role, plume_ids=";".join(ids),
                           rad_url="", obs_url="", rad_file=name, rad_size=None))
        log.info("fake scene %s (%s, %d plumes)", sid, role, len(ids))
    pd.DataFrame(plumes).to_csv(cat / "plumes.csv", index=False)
    pd.DataFrame(scenes).to_csv(cat / "scenes.csv", index=False)
    log.info("fake dataset written under %s", cfg.paths.data_root)
