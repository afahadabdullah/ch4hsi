"""Geometric-lookup-table (GLT) orthorectification helpers.

EMIT (and AVIRIS-NG) GLTs store, for every output map pixel, the 1-based (x=crosstrack/sample,
y=downtrack/line) index of the raw sensor pixel; 0 marks no data and negative values mark
nearest-neighbour infill (we use the absolute value).
"""
from __future__ import annotations

import numpy as np


def glt_index(glt_x: np.ndarray, glt_y: np.ndarray):
    gx = np.nan_to_num(np.asarray(glt_x, dtype=np.float64), nan=0).astype(np.int64)
    gy = np.nan_to_num(np.asarray(glt_y, dtype=np.float64), nan=0).astype(np.int64)
    gx, gy = np.abs(gx), np.abs(gy)
    valid = (gx > 0) & (gy > 0)
    return gy - 1, gx - 1, valid


def ortho(sensor: np.ndarray, glt_x: np.ndarray, glt_y: np.ndarray, fill=np.nan, _cache=None) -> np.ndarray:
    """Map a sensor-geometry array (rows, cols, ...) to the GLT map grid (H, W, ...)."""
    rows, cols, valid = _cache if _cache is not None else glt_index(glt_x, glt_y)
    rows_ok = rows[valid].clip(0, sensor.shape[0] - 1)
    cols_ok = cols[valid].clip(0, sensor.shape[1] - 1)
    if isinstance(fill, float) and np.isnan(fill):
        out_dtype = np.promote_types(sensor.dtype, np.float32)
    else:
        out_dtype = sensor.dtype
    out = np.full(valid.shape + sensor.shape[2:], fill, dtype=out_dtype)
    out[valid] = sensor[rows_ok, cols_ok]
    return out


def sensor_from_ortho(ortho_arr: np.ndarray, glt_x, glt_y, fill=np.nan):
    """Inverse mapping (used for AVIRIS-NG, which is distributed orthorectified).

    Returns a raw-geometry cube (lines, samples, ...) filled from the ortho grid; raw pixels
    never referenced by the GLT stay at `fill`.
    """
    rows, cols, valid = glt_index(glt_x, glt_y)
    n_rows, n_cols = rows[valid].max() + 1, cols[valid].max() + 1
    out = np.full((n_rows, n_cols) + ortho_arr.shape[2:], fill,
                  dtype=np.promote_types(ortho_arr.dtype, np.float32))
    out[rows[valid], cols[valid]] = ortho_arr[valid]
    return out
