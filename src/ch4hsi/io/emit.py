"""EMIT L1B radiance reader (EMIT_L1B_RAD_*.nc).

Layout (EMITL1BRAD v001):
  /radiance                        (downtrack, crosstrack, bands) float32, uW cm-2 sr-1 nm-1, fill -9999
  /sensor_band_parameters/wavelengths, /fwhm, /good_wavelengths
  /location/glt_x, /glt_y          (ortho_y, ortho_x) 1-based sensor indices, 0 = no data
  global attrs: geotransform (GDAL 6-tuple, EPSG:4326), spatial_ref (WKT)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FILL = -9999.0
SCENE_RE = re.compile(r"(\d{8}T\d{6})_(\d{7})_(\d{3})")


def scene_id_from_name(name: str) -> str:
    m = SCENE_RE.search(Path(name).name)
    if not m:
        raise ValueError(f"cannot parse EMIT scene id from {name}")
    return "_".join(m.groups())


@dataclass
class EmitScene:
    scene_id: str
    wavelengths: np.ndarray
    fwhm: np.ndarray
    good: np.ndarray
    glt_x: np.ndarray
    glt_y: np.ndarray
    geotransform: tuple
    crs_wkt: str
    radiance: dict = field(default_factory=dict)   # name -> (rows, cols, nb) float32 subsets

    @property
    def ortho_shape(self):
        return self.glt_x.shape


def _read_nc4(path, band_sets):
    import netCDF4
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        sbp = ds.groups["sensor_band_parameters"]
        wl = np.asarray(sbp.variables["wavelengths"][:], dtype=np.float64)
        fwhm = np.asarray(sbp.variables["fwhm"][:], dtype=np.float64)
        good = (np.asarray(sbp.variables["good_wavelengths"][:]).astype(bool)
                if "good_wavelengths" in sbp.variables else np.ones_like(wl, bool))
        loc = ds.groups["location"]
        gx, gy = np.asarray(loc.variables["glt_x"][:]), np.asarray(loc.variables["glt_y"][:])
        gt = tuple(float(v) for v in np.atleast_1d(ds.getncattr("geotransform")))
        wkt = str(ds.getncattr("spatial_ref")) if "spatial_ref" in ds.ncattrs() else "EPSG:4326"
        rad_var = ds.variables["radiance"]
        full = np.asarray(rad_var[:], dtype=np.float32)
    return wl, fwhm, good, gx, gy, gt, wkt, full


def _read_h5(path, band_sets):
    import h5py
    with h5py.File(path, "r") as f:
        wl = f["sensor_band_parameters/wavelengths"][:].astype(np.float64)
        fwhm = f["sensor_band_parameters/fwhm"][:].astype(np.float64)
        good = (f["sensor_band_parameters/good_wavelengths"][:].astype(bool)
                if "good_wavelengths" in f["sensor_band_parameters"] else np.ones_like(wl, bool))
        gx, gy = f["location/glt_x"][:], f["location/glt_y"][:]
        gt = tuple(float(v) for v in np.atleast_1d(f.attrs["geotransform"]))
        wkt = f.attrs.get("spatial_ref", b"EPSG:4326")
        wkt = wkt.decode() if isinstance(wkt, bytes) else str(wkt)
        full = f["radiance"][:].astype(np.float32)
    return wl, fwhm, good, gx, gy, gt, wkt, full


def read_emit_l1b(path: str | Path, band_selector) -> EmitScene:
    """Read an EMIT L1B RAD file, keeping only the band subsets returned by `band_selector`.

    band_selector(wavelengths, fwhm, good) -> dict[name, index array]
    (the full 285-band cube is ~1.8 GB; only the subsets are kept in memory afterwards).
    """
    path = Path(path)
    try:
        wl, fwhm, good, gx, gy, gt, wkt, full = _read_nc4(path, None)
    except ImportError:
        wl, fwhm, good, gx, gy, gt, wkt, full = _read_h5(path, None)
    sets = band_selector(wl, fwhm, good)
    rad = {}
    for name, idx in sets.items():
        sub = full[:, :, idx]
        rad[name] = np.ascontiguousarray(sub)
    del full
    return EmitScene(scene_id=scene_id_from_name(path.name), wavelengths=wl, fwhm=fwhm, good=good,
                     glt_x=gx, glt_y=gy, geotransform=gt, crs_wkt=wkt, radiance=rad)


def valid_mask(rad: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(rad) & (rad > FILL + 1), axis=-1)
