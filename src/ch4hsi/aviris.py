"""Optional / experimental: cross-sensor test on AVIRIS-NG (airborne, ~3-8 m GSD).

Because every model input is defined sensor-agnostically (ppm·m, MF SNR, self-normalised radiance),
the EMIT-trained model can be run on AVIRIS-NG after (optionally) block-averaging radiance to an
EMIT-like GSD. Labels are plume point locations (flight_line, lat, lon), e.g. from the ORNL DAAC
AVIRIS-NG plume datasets (doi:10.3334/ORNLDAAC/1727 or /2406) — build the CSV once by hand/script.

  ch4hsi aviris-fetch   -> downloads L1B radiance for the listed flight lines (AVIRIS-NG_L1B_radiance_2095)
  ch4hsi aviris-eval    -> MF + model detection rate at the labelled points, false alarms per km²
"""
from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Cfg, run_dir
from .utils import dump_json, get_logger, load_json

log = get_logger(__name__)


def fetch(cfg: Cfg):
    from .catalog import earthdata as ed
    A = cfg.aviris
    ed.login()
    fl = pd.read_csv(A.flightlines_csv).flight_line.astype(str).str.strip().unique()
    out = Path(cfg.paths.raw) / "aviris"
    for f in fl:
        grans = ed.search(short_name=A.short_name, granule_name=f"*{f}*", count=10)
        urls = [u for g in grans for u in ed.granule_links(g) if f in Path(u).name and "rdn" in Path(u).name]
        if not urls:
            log.warning("no L1B radiance found for %s", f)
            continue
        ed.download(urls, out / f, workers=cfg.download.workers)
        for p in (out / f).iterdir():
            if p.suffix in (".gz", ".tgz", ".tar"):
                with tarfile.open(p) as t:
                    t.extractall(out / f)
            elif p.suffix == ".zip":
                with zipfile.ZipFile(p) as z:
                    z.extractall(out / f)


def _map_info(hdr):
    mi = hdr.get("map info")
    if not mi:
        raise ValueError("ENVI header has no 'map info'")
    x0, y0, px, py = float(mi[3]), float(mi[4]), float(mi[5]), float(mi[6])
    rx, ry = float(mi[1]) - 1, float(mi[2]) - 1           # reference pixel (1-based in ENVI)
    zone, hemi = int(mi[7]), mi[8].strip().lower()
    epsg = (32600 if hemi.startswith("n") else 32700) + zone
    return (x0 - rx * px, px, y0 + ry * py, -py), epsg


def _coarsen_glt(gx, gy, f):
    rows, cols = np.abs(gy), np.abs(gx)
    ok = (rows > 0) & (cols > 0)
    gy2 = np.where(ok, (rows - 1) // f + 1, 0)[::f, ::f]
    gx2 = np.where(ok, (cols - 1) // f + 1, 0)[::f, ::f]
    return gx2, gy2


def _block_mean(a, f):
    r, c = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    a = a[:r, :c]
    return np.nanmean(a.reshape(r // f, f, c // f, f, *a.shape[2:]), axis=(1, 3))


def evaluate(cfg: Cfg):
    import torch
    from pyproj import Transformer
    from scipy import ndimage

    from .evaluate import load_model, mf_score, predict_scene
    from .features import scene_products
    from .io.envi import open_envi, read_header
    from .io.glt import sensor_from_ortho
    from .metrics import remove_small
    from .physics.target import load_lut, unit_absorption
    from .train import amp_dtype

    A = cfg.aviris
    out = run_dir(cfg)
    thr = load_json(out / "thresholds.json")
    norm = load_json(out / "norm.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(out, cfg, device)
    lut = load_lut(cfg.physics.lut)
    labels = pd.read_csv(A.flightlines_csv)
    rows = []
    for fl, pts in labels.groupby("flight_line"):
        d = Path(cfg.paths.raw) / "aviris" / str(fl)
        img_h = next(iter(sorted(d.rglob("*rdn*img.hdr"))), None)
        glt_h = next(iter(sorted(d.rglob("*rdn*glt.hdr"))), None)
        if img_h is None or glt_h is None:
            log.warning("%s: radiance or GLT header not found under %s", fl, d)
            continue
        arr, hdr, wl, fwhm = open_envi(img_h)
        glt, _, _, _ = open_envi(glt_h)
        ghdr = read_header(glt_h)
        bn = [b.lower() for b in ghdr.get("band names", [])]
        ix, iy = (0, 1) if not bn or "sample" in bn[0] or "x" == bn[0] else (1, 0)
        gx, gy = np.asarray(glt[..., ix]), np.asarray(glt[..., iy])
        lo, hi = cfg.physics.mf_window_nm
        win = np.where((wl >= lo) & (wl <= hi))[0]
        rgbi = np.array([int(np.argmin(np.abs(wl - x))) for x in cfg.preprocess.rgb_nm])
        sub = np.asarray(arr[:, :, np.concatenate([win, rgbi])], np.float32)
        sub[sub <= -9990] = np.nan
        raw = sensor_from_ortho(sub, gx, gy)
        gt, epsg = _map_info(hdr)
        gsd = gt[1]
        f = max(1, int(round(A.aggregate_to_m / gsd))) if A.aggregate_to_m else 1
        if f > 1:
            raw = _block_mean(raw, f)
            gx, gy = _coarsen_glt(gx, gy, f)
            gt = (gt[0], gt[1] * f, gt[2], gt[3] * f)
        raw = np.nan_to_num(raw, nan=-9999.0)
        k = unit_absorption(wl[win], fwhm[win], lut=lut, spec=cfg.physics.lut)
        prod = scene_products(raw[..., : len(win)], raw[..., len(win):], k, gx, gy, cfg.physics, cfg.preprocess,
                              cfg.features)
        prob = predict_scene(model, prod["features"], prod["valid"], norm, cfg.eval.infer_tile, cfg.eval.infer_stride,
                             device, amp_dtype(cfg.train.amp))
        scores = {"model": prob, "mf": mf_score(prod["mf"], prod["valid"], cfg.eval.mf_smooth_sigma_px)}
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        ex, ey = tr.transform(pts.lon.to_numpy(), pts.lat.to_numpy())
        pc = ((np.asarray(ex) - gt[0]) / gt[1]).astype(int)
        pr = ((np.asarray(ey) - gt[2]) / gt[3]).astype(int)
        rad_px = A.match_radius_m / abs(gt[1])
        area_km2 = prod["valid"].sum() * abs(gt[1] * gt[3]) / 1e6
        for m, key in (("model", "model"), ("mf", "mf"), ("mf_fixed", "mf")):
            det = remove_small((scores[key] >= thr[m]["threshold"]) & prod["valid"], cfg.eval.min_component_px)
            lab, n = ndimage.label(det, structure=np.ones((3, 3)))
            if n:
                idx = np.arange(1, n + 1)
                cen = np.array(ndimage.center_of_mass(det, lab, idx))
                reach = rad_px + np.sqrt(np.asarray(ndimage.sum(det, lab, idx)) / np.pi)   # + component radius
                near = np.hypot(cen[:, None, 0] - pr[None, :], cen[:, None, 1] - pc[None, :]) <= reach[:, None]
                hit, used = near.any(axis=0), near.any(axis=1)
            else:
                hit, used = np.zeros(len(pr), bool), np.zeros(0, bool)
            rows.append(dict(flight_line=fl, method=m, n_points=len(pts), detected=int(hit.sum()),
                             false_components=int((~used).sum()), area_km2=float(area_km2), agg_factor=f))
        log.info("%s: %s", fl, rows[-3:])
    df = pd.DataFrame(rows)
    df.to_csv(out / "aviris_eval.csv", index=False)
    summ = {m: dict(point_recall=float(g.detected.sum() / max(g.n_points.sum(), 1)),
                    false_alarms_per_100km2=float(100 * g.false_components.sum() / max(g.area_km2.sum(), 1e-9)))
            for m, g in df.groupby("method")} if len(df) else {}
    dump_json(summ, out / "aviris_summary.json")
    log.info("AVIRIS-NG summary: %s", summ)
