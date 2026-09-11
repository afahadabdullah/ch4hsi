"""Dependency-free ENVI reader (used for the mag1c CH4 LUT and AVIRIS-NG radiance)."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ENVI_DTYPES = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4", 14: "i8", 15: "u8"}


def read_header(hdr_path: str | Path) -> dict:
    txt = Path(hdr_path).read_text()
    if not txt.lstrip().upper().startswith("ENVI"):
        raise ValueError(f"{hdr_path} is not an ENVI header")
    hdr: dict = {}
    for m in re.finditer(r"^\s*([^=\n]+?)\s*=\s*(\{[^}]*\}|[^\n]*)", txt, flags=re.M):
        key, val = m.group(1).strip().lower(), m.group(2).strip()
        if val.startswith("{"):
            items = [v.strip() for v in val[1:-1].replace("\n", " ").split(",")]
            hdr[key] = [v for v in items if v != ""]
        else:
            hdr[key] = val
    return hdr


def _floats(hdr, key):
    v = hdr.get(key)
    return None if v is None else np.array([float(x) for x in v])


def find_image(hdr_path: Path) -> Path:
    stem = hdr_path.with_suffix("")
    for cand in (stem, stem.with_suffix(".img"), stem.with_suffix(".bin"), stem.with_suffix(".dat"),
                 stem.with_suffix(".lut"), Path(str(stem) + ".img")):
        if cand.exists() and cand != hdr_path:
            return cand
    raise FileNotFoundError(f"no image file next to {hdr_path}")


def open_envi(hdr_path: str | Path, img_path: str | Path | None = None, mmap: bool = True):
    """Return (array[lines, samples, bands] (view, possibly memmap), header dict, wavelengths, fwhm)."""
    hdr_path = Path(hdr_path)
    hdr = read_header(hdr_path)
    img_path = Path(img_path) if img_path else find_image(hdr_path)
    lines, samples, bands = int(hdr["lines"]), int(hdr["samples"]), int(hdr["bands"])
    dt = np.dtype(ENVI_DTYPES[int(hdr["data type"])])
    dt = dt.newbyteorder(">" if int(hdr.get("byte order", 0)) == 1 else "<")
    offset = int(hdr.get("header offset", 0))
    il = hdr.get("interleave", "bsq").lower()
    shape = {"bsq": (bands, lines, samples), "bil": (lines, bands, samples), "bip": (lines, samples, bands)}[il]
    if mmap:
        raw = np.memmap(img_path, dtype=dt, mode="r", offset=offset, shape=shape)
    else:
        raw = np.fromfile(img_path, dtype=dt, offset=offset, count=int(np.prod(shape))).reshape(shape)
    arr = {"bsq": lambda a: a.transpose(1, 2, 0), "bil": lambda a: a.transpose(0, 2, 1), "bip": lambda a: a}[il](raw)
    wl = _floats(hdr, "wavelength")
    if wl is not None and hdr.get("wavelength units", "nm").lower().startswith("micro"):
        wl = wl * 1000.0
    fwhm = _floats(hdr, "fwhm")
    if fwhm is not None and wl is not None and np.nanmax(fwhm) < 1.0:
        fwhm = fwhm * 1000.0
    return arr, hdr, wl, fwhm
