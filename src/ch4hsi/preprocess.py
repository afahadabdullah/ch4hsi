"""Stage 3: L1B radiance -> column-wise MF -> features -> orthorectify -> labels -> per-scene .npy stack.

Output: <scenes>/<scene_id>/
    features.npy  (C, H, W) float16      input channels, NaN outside the swath
    valid.npy     (H, W) bool
    mask.npy      (H, W) uint8           plume-complex label
    plume_id.npy  (H, W) int16           which plume complex (1..n), 0 = background
    mf.npy        (H, W) float32         matched filter, ppm·m (baseline + analysis)
    sigma.npy     (H, W) float16         column noise-equivalent enhancement, ppm·m
    meta.json
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Cfg
from .features import band_selector, channel_names, scene_products
from .io.emit import read_emit_l1b, scene_id_from_name
from .physics.target import load_lut, unit_absorption
from .utils import Timer, dump_json, get_logger, shard

log = get_logger(__name__)


def _plumes_for_scene(row, plumes: pd.DataFrame) -> list[dict]:
    v = row.get("plume_ids")
    ids = [s for s in v.split(";") if s] if isinstance(v, str) else []
    if not ids:
        return []
    sub = plumes[plumes.plume_id.isin(ids)].fillna({"tif_path": "", "json_path": ""})
    return sub[["plume_id", "tif_path", "json_path"]].to_dict("records")


def process_scene(l1b_path: str | Path, out_root: str | Path, cfg: Cfg, plumes: list[dict], role: str,
                  lut=None, extra_meta: dict | None = None) -> Path | None:
    from .labels import rasterize_plumes

    l1b_path = Path(l1b_path)
    phys, pre = cfg.physics, cfg.preprocess
    out = Path(out_root) / scene_id_from_name(l1b_path.name)
    if (out / "meta.json").exists() and not pre.overwrite:
        log.info("exists, skipping %s", out.name)
        return out
    with Timer(f"read {l1b_path.name}", log):
        sc = read_emit_l1b(l1b_path, band_selector(phys, pre))
    win = band_selector(phys, pre)(sc.wavelengths, sc.fwhm, sc.good)["win"]
    k = unit_absorption(sc.wavelengths[win], sc.fwhm[win], lut=lut, spec=phys.lut)
    with Timer(f"mf+features {sc.scene_id}", log):
        prod = scene_products(sc.radiance["win"], sc.radiance["rgb"], k, sc.glt_x, sc.glt_y, phys, pre, cfg.features)
    H, W = prod["valid"].shape
    mask, ids, ppmm, used = rasterize_plumes(plumes, (H, W), sc.geotransform, "EPSG:4326", cfg.labels.min_ppmm) \
        if plumes else (np.zeros((H, W), np.uint8), np.zeros((H, W), np.int32), None, [])
    mask &= prod["valid"].astype(np.uint8)

    tmp = out.with_name(out.name + ".partial")
    tmp.mkdir(parents=True, exist_ok=True)
    dt = np.dtype(pre.store_dtype)
    feats = np.nan_to_num(prod["features"], nan=0.0, posinf=0.0, neginf=0.0)
    np.save(tmp / "features.npy", np.clip(feats, -6e4, 6e4).astype(dt))
    np.save(tmp / "valid.npy", prod["valid"])
    np.save(tmp / "mask.npy", mask)
    np.save(tmp / "plume_id.npy", ids.astype(np.int16))
    np.save(tmp / "mf.npy", prod["mf"].astype(np.float32))
    np.save(tmp / "sigma.npy", np.nan_to_num(prod["sigma"], nan=0).astype(np.float16))
    gt = sc.geotransform
    meta = dict(scene_id=sc.scene_id, l1b=str(l1b_path), role=role, shape=[H, W], geotransform=list(gt),
                crs="EPSG:4326", channel_names=channel_names(cfg.features, pre.n_swir_bins),
                plume_ids=used, n_pos_px=int(mask.sum()), n_valid_px=int(prod["valid"].sum()),
                center_lon=gt[0] + gt[1] * W / 2, center_lat=gt[3] + gt[5] * H / 2,
                mf_window_nm=list(phys.mf_window_nm), n_window_bands=int(len(win)),
                k_bins=[float(k[ix].mean()) for ix in np.array_split(np.arange(len(win)), pre.n_swir_bins)],
                sigma_median_ppmm=float(np.nanmedian(prod["sigma"][prod["valid"]])) if prod["valid"].any() else None,
                mf_bg_std_ppmm=float(np.nanstd(prod["mf"][prod["valid"] & (mask == 0)])) if prod["valid"].any() else None)
    meta.update(extra_meta or {})
    dump_json(meta, tmp / "meta.json")
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    if pre.delete_raw:
        l1b_path.unlink(missing_ok=True)
    log.info("wrote %s  (%dx%d, %d plume px, sigma~%.0f ppm·m)", out, H, W, meta["n_pos_px"],
             meta["sigma_median_ppmm"] or float("nan"))
    return out


def run(cfg: Cfg, task_id=None, num_tasks=None):
    cat = Path(cfg.paths.catalog)
    scenes = pd.read_csv(cat / "scenes.csv", dtype={"plume_ids": str})
    plumes = pd.read_csv(cat / "plumes.csv", dtype={"plume_id": str})
    rows = shard(scenes.to_dict("records"), task_id, num_tasks)
    lut = load_lut(cfg.physics.lut)
    ok = fail = 0
    for row in rows:
        p = Path(cfg.paths.raw) / row["rad_file"]
        if not p.exists():
            log.warning("missing raw file %s (run the download stage)", p)
            fail += 1
            continue
        try:
            process_scene(p, cfg.paths.scenes, cfg, _plumes_for_scene(row, plumes), row["role"], lut=lut,
                          extra_meta={"time": row.get("time"), "granule": row.get("granule")})
            ok += 1
        except Exception as e:  # keep the array task going; failures are listed at the end
            log.exception("failed %s: %s", p.name, e)
            fail += 1
    log.info("preprocess done: %d ok, %d failed", ok, fail)
