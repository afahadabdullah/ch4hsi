"""Stage 1: build the plume-complex label catalogue -> <catalog>/plumes.csv (+ assets).

Sources
  lpdaac     EMITL2BCH4PLM (COG + GeoJSON per plume complex) via earthaccess (needs Earthdata login)
  ghgc_stac  US GHG Center STAC collection `emit-ch4plume-v1` (COGs, public; no scene names)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Cfg
from ..utils import get_logger
from . import earthdata as ed

log = get_logger(__name__)
TS_RE = re.compile(r"(\d{8}T\d{6})")
SCENE_TOKEN_RE = re.compile(r"(\d{8}T\d{6})_(\d{7})_(\d{3})")


def _find(props: dict, *needles):
    for k, v in props.items():
        kl = k.lower()
        if all(n in kl for n in needles):
            return v
    return None


def _geojson_bbox(gj: dict):
    xs, ys = [], []

    def walk(c):
        if isinstance(c, (list, tuple)) and c and isinstance(c[0], (int, float)):
            xs.append(c[0]), ys.append(c[1])
        elif isinstance(c, (list, tuple)):
            for cc in c:
                walk(cc)
    for ft in gj.get("features", []):
        if ft.get("geometry"):
            walk(ft["geometry"].get("coordinates", []))
    return (min(xs), min(ys), max(xs), max(ys)) if xs else (None,) * 4


def parse_plume_json(path: Path) -> dict:
    gj = json.loads(Path(path).read_text())
    props = {}
    for ft in gj.get("features", []):
        if ft.get("properties"):
            props = ft["properties"]
            break
    w, s, e, n = _geojson_bbox(gj)
    scenes = _find(props, "scene") or []
    if isinstance(scenes, str):
        scenes = re.split(r"[,;\s]+", scenes)
    tokens = sorted({"_".join(m.groups()) for sc in scenes for m in [SCENE_TOKEN_RE.search(str(sc))] if m})
    return dict(west=w, south=s, east=e, north=n, scene_tokens=";".join(tokens),
                max_ppmm=_find(props, "max", "concentration"),
                lat=_find(props, "latitude"), lon=_find(props, "longitude"),
                utc_time=_find(props, "utc", "time"), props=json.dumps(props))


def _tif_bbox(path):
    try:
        import rasterio
        with rasterio.open(path) as src:
            b = src.bounds
            return b.left, b.bottom, b.right, b.top
    except Exception:
        return (None,) * 4


def fetch_lpdaac(cfg: Cfg) -> pd.DataFrame:
    ed.login()
    L, E = cfg.labels, cfg.earthdata
    kw = dict(short_name=E.plume_short_name, version=E.plume_version, temporal=tuple(L.temporal),
              count=int(L.max_plumes) if L.max_plumes else -1)
    if L.bbox:
        kw["bounding_box"] = tuple(L.bbox)
    grans = ed.search(**kw)
    log.info("found %d plume-complex granules (%s v%s)", len(grans), E.plume_short_name, E.plume_version)
    assets = Path(cfg.paths.catalog) / "plume_assets"
    urls, sizes, recs = [], {}, []
    for g in grans:
        links = [u for u in ed.granule_links(g) if u.endswith((".tif", ".json"))]
        urls += links
        sizes.update(ed.granule_sizes(g["umm"]))
        tif = next((u for u in links if u.endswith(".tif")), None)
        js = next((u for u in links if u.endswith(".json")), None)
        gid = g["umm"].get("GranuleUR") or Path(tif or js).stem
        w, s, e, n = ed.granule_bbox(g["umm"])
        recs.append(dict(plume_id=gid, source="lpdaac", time=ed.granule_time(g["umm"]),
                         tif_path=str(assets / Path(tif).name) if tif else "",
                         json_path=str(assets / Path(js).name) if js else "",
                         g_west=w, g_south=s, g_east=e, g_north=n))
    res, failed = ed.download(urls, assets, sizes, workers=cfg.download.workers)
    log.info("plume assets: %s", res)
    rows = []
    for r in recs:
        info = parse_plume_json(Path(r["json_path"])) if r["json_path"] and Path(r["json_path"]).exists() else {}
        r.update(info)
        if r.get("west") is None:
            w, s, e, n = _tif_bbox(r["tif_path"]) if r["tif_path"] else (None,) * 4
            r.update(west=w if w is not None else r["g_west"], south=s if s is not None else r["g_south"],
                     east=e if e is not None else r["g_east"], north=n if n is not None else r["g_north"])
        m = TS_RE.search(r["plume_id"])
        if m:
            r["time"] = ed.parse_time(m.group(1)).isoformat()
        rows.append(r)
    return pd.DataFrame(rows)


def fetch_ghgc(cfg: Cfg) -> pd.DataFrame:
    import requests
    L = cfg.labels
    url = f"{L.ghgc_stac_url.rstrip('/')}/collections/{L.ghgc_collection}/items"
    params = {"limit": 250, "datetime": f"{L.temporal[0]}T00:00:00Z/{L.temporal[1]}T23:59:59Z"}
    if L.bbox:
        params["bbox"] = ",".join(str(v) for v in L.bbox)
    items, nxt = [], (url, params)
    while nxt:
        r = requests.get(nxt[0], params=nxt[1], timeout=60)
        r.raise_for_status()
        js = r.json()
        items += js.get("features", [])
        link = next((lk for lk in js.get("links", []) if lk.get("rel") == "next"), None)
        nxt = (link["href"], None) if link else None
        if L.max_plumes and len(items) >= int(L.max_plumes):
            items = items[: int(L.max_plumes)]
            break
    log.info("GHG Center STAC: %d plume items", len(items))
    assets = Path(cfg.paths.catalog) / "plume_assets"
    rows, urls = [], []
    for it in items:
        a = it["assets"].get(L.ghgc_asset) or next(iter(it["assets"].values()))
        href = a["href"]
        urls.append(href)
        w, s, e, n = it.get("bbox", [None] * 4)[:4]
        rows.append(dict(plume_id=it["id"], source="ghgc_stac", time=it["properties"].get("datetime"),
                         west=w, south=s, east=e, north=n, lat=(s + n) / 2 if s is not None else None,
                         lon=(w + e) / 2 if w is not None else None, scene_tokens="", max_ppmm=None,
                         tif_path=str(assets / Path(href.split("?")[0]).name), json_path="", props="{}"))
    ed.download(urls, assets, workers=cfg.download.workers, public=True)
    return pd.DataFrame(rows)


def run(cfg: Cfg):
    out = Path(cfg.paths.catalog)
    out.mkdir(parents=True, exist_ok=True)
    df = fetch_lpdaac(cfg) if cfg.labels.source == "lpdaac" else fetch_ghgc(cfg)
    df = df.dropna(subset=["west", "south", "east", "north", "time"]).drop_duplicates("plume_id")
    if df.empty:
        raise RuntimeError("no plume complexes found; check temporal range / credentials")
    df.to_csv(out / "plumes.csv", index=False)
    log.info("wrote %s (%d plume complexes, %s .. %s)", out / "plumes.csv", len(df), df.time.min(), df.time.max())
    if "max_ppmm" in df and df.max_ppmm.notna().any():
        log.info("max ppm·m quantiles: %s", np.nanpercentile(pd.to_numeric(df.max_ppmm, errors="coerce"), [10, 50, 90]))
