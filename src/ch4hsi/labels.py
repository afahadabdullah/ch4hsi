"""Rasterise EMIT plume complexes (COG and/or GeoJSON outline) onto a scene's GLT map grid."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _affine(gt):
    from affine import Affine
    return Affine.from_gdal(*gt)


def rasterize_plumes(plumes: list[dict], dst_shape, dst_gt, dst_crs="EPSG:4326", min_ppmm=0.0):
    """plumes: [{'plume_id', 'tif_path', 'json_path'}]. Returns mask uint8, id map int32, ppmm float32."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    H, W = dst_shape
    mask = np.zeros((H, W), np.uint8)
    ids = np.zeros((H, W), np.int32)
    ppmm = np.full((H, W), np.nan, np.float32)
    dst_tf = _affine(dst_gt)
    used = []
    for i, p in enumerate(plumes, start=1):
        tif, js = p.get("tif_path"), p.get("json_path")
        inside = None
        if tif and Path(tif).exists():
            with rasterio.open(tif) as src:
                tmp = np.full((H, W), np.nan, np.float32)
                src_nodata = src.nodata if src.nodata is not None else -9999.0
                reproject(source=rasterio.band(src, 1), destination=tmp, src_transform=src.transform,
                          src_crs=src.crs, src_nodata=src_nodata, dst_transform=dst_tf, dst_crs=dst_crs,
                          dst_nodata=np.nan, resampling=Resampling.nearest)
            inside = np.isfinite(tmp) & (tmp > min_ppmm) & (tmp != src_nodata)
            ppmm = np.where(inside, np.fmax(ppmm, tmp), ppmm)
        elif js and Path(js).exists():
            from rasterio.features import rasterize
            with open(js) as f:
                gj = json.load(f)
            geoms = [ft["geometry"] for ft in gj.get("features", []) if ft.get("geometry")
                     and ft["geometry"]["type"] in ("Polygon", "MultiPolygon")]
            if geoms:
                inside = rasterize(geoms, out_shape=(H, W), transform=dst_tf, fill=0, default_value=1).astype(bool)
        if inside is not None and inside.any():
            mask |= inside.astype(np.uint8)
            ids[inside] = i
            used.append(p["plume_id"])
    return mask, ids, ppmm, used
