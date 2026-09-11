"""Optional sanity check: compare our column-wise MF with the official EMIT L2B CH4 enhancement
(EMITL2BCH4ENH) on the same scenes. Expect strong spatial correlation and a slope near 1 inside
plumes; differences come from EMIT's scene-specific targets (water vapour, elevation, geometry).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import Cfg
from .io.emit import scene_id_from_name
from .utils import dump_json, get_logger

log = get_logger(__name__)


def run(cfg: Cfg):
    import rasterio
    enh_dir = Path(cfg.paths.raw) / "enh"
    rows = []
    for tif in sorted(enh_dir.glob("*CH4ENH*.tif")):
        sid = scene_id_from_name(tif.name)
        d = Path(cfg.paths.scenes) / sid
        if not (d / "mf.npy").exists():
            continue
        ours = np.load(d / "mf.npy")
        valid = np.load(d / "valid.npy")
        with rasterio.open(tif) as src:
            ref = src.read(1).astype(np.float32)
            nod = src.nodata
        if ref.shape != ours.shape:
            log.warning("%s: grid mismatch %s vs %s", sid, ref.shape, ours.shape)
            continue
        ok = valid & np.isfinite(ref) & (ref != (nod if nod is not None else -9999)) & np.isfinite(ours)
        a, b = ours[ok], ref[ok]
        hi = b > np.percentile(b, 99.5)
        slope = float(np.polyfit(b[hi], a[hi], 1)[0]) if hi.sum() > 10 else float("nan")
        rows.append(dict(scene_id=sid, pearson_r=float(np.corrcoef(a, b)[0, 1]), slope_top05pct=slope,
                         ours_std=float(a.std()), ref_std=float(b.std())))
        log.info("%s r=%.3f slope=%.2f std ours/ref=%.0f/%.0f", sid, *[rows[-1][k] for k in
                                                                     ("pearson_r", "slope_top05pct", "ours_std", "ref_std")])
    dump_json(rows, Path(cfg.paths.catalog) / "mf_crosscheck.json")
