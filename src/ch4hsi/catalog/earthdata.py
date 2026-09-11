"""NASA Earthdata (CMR + LP DAAC) helpers built on `earthaccess`.

Auth: put `machine urs.earthdata.nasa.gov login <user> password <pass>` in ~/.netrc (chmod 600),
or export EARTHDATA_USERNAME / EARTHDATA_PASSWORD.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..utils import get_logger

log = get_logger(__name__)
_SESSION = None


def login():
    import earthaccess
    for strategy in ("environment", "netrc"):
        try:
            auth = earthaccess.login(strategy=strategy)
            if auth is not None and getattr(auth, "authenticated", True):
                return auth
        except Exception:  # try next strategy
            pass
    raise RuntimeError("Earthdata login failed: create ~/.netrc or set EARTHDATA_USERNAME/PASSWORD")


def search(**kw):
    import earthaccess
    for attempt in range(5):
        try:
            return earthaccess.search_data(**kw)
        except Exception as e:
            wait = 5 * 2 ** attempt
            log.warning("CMR search failed (%s); retrying in %ds", e, wait)
            time.sleep(wait)
    raise RuntimeError(f"CMR search failed repeatedly: {kw}")


def parse_time(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = datetime.strptime(s[:15], "%Y%m%dT%H%M%S")
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def window(t: datetime, minutes: float):
    return ((t - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            (t + timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%SZ"))


def granule_bbox(umm: dict):
    try:
        geom = umm["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]
        if "GPolygons" in geom:
            pts = [p for poly in geom["GPolygons"] for p in poly["Boundary"]["Points"]]
            lons, lats = [p["Longitude"] for p in pts], [p["Latitude"] for p in pts]
            return min(lons), min(lats), max(lons), max(lats)
        if "BoundingRectangles" in geom:
            r = geom["BoundingRectangles"][0]
            return r["WestBoundingCoordinate"], r["SouthBoundingCoordinate"], r["EastBoundingCoordinate"], r["NorthBoundingCoordinate"]
    except (KeyError, IndexError, TypeError):
        pass
    return (None,) * 4


def granule_time(umm: dict) -> str | None:
    te = umm.get("TemporalExtent", {})
    if "RangeDateTime" in te:
        return te["RangeDateTime"].get("BeginningDateTime")
    return te.get("SingleDateTime")


def granule_sizes(umm: dict) -> dict:
    out = {}
    unit = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}
    for f in umm.get("DataGranule", {}).get("ArchiveAndDistributionInformation", []) or []:
        if "Name" in f and "Size" in f:
            out[f["Name"]] = int(float(f["Size"]) * unit.get(str(f.get("SizeUnit", "B")).upper(), 1))
    return out


def granule_links(g) -> list[str]:
    try:
        return list(g.data_links(access="external"))
    except TypeError:
        return list(g.data_links())


def _session():
    global _SESSION
    if _SESSION is None:
        import earthaccess
        _SESSION = earthaccess.get_requests_https_session()
    return _SESSION


def _fetch(url: str, dest: Path, expected: int | None, retries=4, public=False):
    import requests
    if dest.exists() and dest.stat().st_size > 0 and (not expected or dest.stat().st_size == expected):
        return dest, "exists"
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(retries):
        try:
            sess = requests if public else _session()
            with sess.get(url, stream=True, timeout=(30, 300)) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8 << 20):
                        f.write(chunk)
            if expected and tmp.stat().st_size != expected:
                raise IOError(f"size mismatch {tmp.stat().st_size} != {expected}")
            os.replace(tmp, dest)
            return dest, "ok"
        except Exception as e:
            log.warning("download %s failed (%s), attempt %d", dest.name, e, attempt + 1)
            time.sleep(10 * (attempt + 1))
    tmp.unlink(missing_ok=True)
    return dest, "failed"


def download(urls: list[str], out_dir: str | Path, sizes: dict | None = None, workers: int = 4, public=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = sizes or {}
    results = {"ok": 0, "exists": 0, "failed": 0}
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_fetch, u, out_dir / Path(u).name, sizes.get(Path(u).name), public=public): u for u in urls}
        for i, fu in enumerate(as_completed(futs), 1):
            dest, status = fu.result()
            results[status] += 1
            if status == "failed":
                failed.append(futs[fu])
            if i % 25 == 0 or i == len(futs):
                log.info("downloads %d/%d %s", i, len(futs), results)
    return results, failed
