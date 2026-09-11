"""Stage 2a: resolve EMIT L1B scenes (positives: contain labelled plume complexes; negatives: plume-free
scenes over emission-prone regions) -> <catalog>/scenes.csv.
Stage 2b: download L1B radiance (Slurm array friendly).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Cfg
from ..io.emit import SCENE_RE
from ..utils import get_logger, shard
from . import earthdata as ed

log = get_logger(__name__)


def granule_record(g) -> dict | None:
    umm = g["umm"]
    ur = umm.get("GranuleUR") or g["meta"].get("native-id")
    links = ed.granule_links(g)
    rad = next((u for u in links if "_RAD_" in Path(u).name and u.endswith(".nc")), None)
    obs = next((u for u in links if "_OBS_" in Path(u).name and u.endswith(".nc")), None)
    if rad is None:
        return None
    m = SCENE_RE.search(Path(rad).name)
    w, s, e, n = ed.granule_bbox(umm)
    sizes = ed.granule_sizes(umm)
    return dict(scene_id="_".join(m.groups()) if m else ur, granule=ur, time=ed.granule_time(umm),
                west=w, south=s, east=e, north=n, cloud=umm.get("CloudCover"),
                rad_url=rad, obs_url=obs or "", rad_file=Path(rad).name,
                rad_size=sizes.get(Path(rad).name), obs_size=sizes.get(Path(obs).name) if obs else None)


def _overlaps(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def assign_plumes(scene: dict, plumes: pd.DataFrame, window_min: float) -> list[str]:
    t = pd.Timestamp(scene["time"])
    t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    sb = (scene["west"], scene["south"], scene["east"], scene["north"])
    if None in sb:
        return []
    dt = np.abs((plumes["tdt"] - t).dt.total_seconds().to_numpy()) <= window_min * 60
    out = []
    for r in plumes[dt].itertuples():
        if _overlaps(sb, (r.west, r.south, r.east, r.north)):
            out.append(r.plume_id)
    return out


def resolve_positive(cfg: Cfg, plumes: pd.DataFrame) -> dict:
    S, E = cfg.scenes, cfg.earthdata
    cache_p = Path(cfg.paths.catalog) / "plume_scene_cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    order = plumes.sample(frac=1.0, random_state=S.seed)
    scenes: dict[str, dict] = {}
    for i, p in enumerate(order.itertuples(), 1):
        if len(scenes) >= S.max_positive_scenes:
            break
        if p.plume_id not in cache:
            t = ed.parse_time(p.time)
            grans = ed.search(short_name=E.l1b_short_name, version=E.l1b_version, temporal=ed.window(t, S.time_window_min),
                              bounding_box=(p.west, p.south, p.east, p.north), count=20)
            cache[p.plume_id] = [r for r in (granule_record(g) for g in grans) if r]
            if i % 20 == 0:
                cache_p.write_text(json.dumps(cache))
                log.info("resolved %d plumes -> %d scenes", i, len(scenes))
        recs = cache[p.plume_id]
        tokens = set(str(getattr(p, "scene_tokens", "") or "").split(";")) - {""}
        if tokens:
            recs = [r for r in recs if r["scene_id"] in tokens] or recs
        for r in recs:
            scenes.setdefault(r["scene_id"], r)
    cache_p.write_text(json.dumps(cache))
    return scenes


def resolve_negative(cfg: Cfg, plumes: pd.DataFrame, n_target: int, exclude: set) -> dict:
    S, E, L = cfg.scenes, cfg.earthdata, cfg.labels
    rng = np.random.default_rng(S.seed + 1)
    t0, t1 = ed.parse_time(L.temporal[0]), ed.parse_time(L.temporal[1])
    regions = list(S.negative_regions.items())
    per_region = int(np.ceil(n_target / max(1, len(regions))))
    out: dict[str, dict] = {}
    for name, bbox in regions:
        cands = []
        # several random 45-day windows so we do not only get the first granules CMR returns
        for _ in range(6):
            start = t0 + (t1 - t0) * rng.random() * 0.95
            end = min(t1, start + pd.Timedelta(days=45))
            grans = ed.search(short_name=E.l1b_short_name, version=E.l1b_version,
                              temporal=(start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")),
                              bounding_box=tuple(bbox), count=max(10, S.negatives_per_region_query // 6))
            cands += [r for r in (granule_record(g) for g in grans) if r]
        ok = []
        for r in cands:
            if r["scene_id"] in exclude or r["scene_id"] in out:
                continue
            if r["cloud"] is not None and float(r["cloud"]) > S.max_cloud_cover:
                continue
            if assign_plumes(r, plumes, S.time_window_min):
                continue  # scene contains a catalogued plume -> not a negative
            ok.append(r)
        rng.shuffle(ok)
        for r in ok[:per_region]:
            r["region"] = name
            out[r["scene_id"]] = r
        log.info("negatives %-14s: %d candidates, %d kept", name, len(cands), min(len(ok), per_region))
    return out


def run_resolve(cfg: Cfg):
    ed.login()
    cat = Path(cfg.paths.catalog)
    plumes = pd.read_csv(cat / "plumes.csv", dtype={"plume_id": str, "scene_tokens": str})
    plumes["tdt"] = pd.to_datetime(plumes.time, utc=True, format="ISO8601")
    pos = resolve_positive(cfg, plumes)
    for r in pos.values():
        r["role"] = "pos"
        r["plume_ids"] = ";".join(assign_plumes(r, plumes, cfg.scenes.time_window_min))
    n_neg = int(round(cfg.scenes.negatives_per_positive * len(pos)))
    neg = resolve_negative(cfg, plumes, n_neg, set(pos)) if n_neg > 0 else {}
    for r in neg.values():
        r["role"] = "neg"
        r["plume_ids"] = ""
    df = pd.DataFrame(list(pos.values()) + list(neg.values()))
    df = df[~((df.role == "pos") & (df.plume_ids == ""))]
    df.to_csv(cat / "scenes.csv", index=False)
    gb = (df.rad_size.fillna(1.8e9).sum() / 1e9)
    log.info("wrote scenes.csv: %d positive, %d negative scenes (~%.0f GB of L1B radiance)",
             (df.role == "pos").sum(), (df.role == "neg").sum(), gb)


def run_download(cfg: Cfg, task_id=None, num_tasks=None):
    ed.login()
    df = pd.read_csv(Path(cfg.paths.catalog) / "scenes.csv")
    rows = shard(df.to_dict("records"), task_id, num_tasks)
    urls, sizes = [], {}
    for r in rows:
        if Path(cfg.paths.scenes, r["scene_id"], "meta.json").exists():
            continue  # already preprocessed
        urls.append(r["rad_url"])
        if pd.notna(r.get("rad_size")):
            sizes[r["rad_file"]] = int(r["rad_size"])
        if cfg.download.keep_obs and isinstance(r.get("obs_url"), str) and r["obs_url"]:
            urls.append(r["obs_url"])
    log.info("task: %d files to fetch", len(urls))
    res, failed = ed.download(urls, cfg.paths.raw, sizes, workers=cfg.download.workers)
    if failed:
        log.error("%d downloads failed, re-run this stage to retry: %s", len(failed), failed[:5])


def run_fetch_enh(cfg: Cfg, max_scenes: int = 30):
    """Optional: download the official EMIT L2B CH4 enhancement for a few positive scenes to
    cross-check our matched filter (see `ch4hsi mf-check`)."""
    ed.login()
    E = cfg.earthdata
    df = pd.read_csv(Path(cfg.paths.catalog) / "scenes.csv")
    df = df[df.role == "pos"].head(max_scenes)
    urls = []
    for r in df.itertuples():
        t = ed.parse_time(r.time)
        grans = ed.search(short_name=E.enh_short_name, version=E.enh_version, temporal=ed.window(t, 1),
                          bounding_box=(r.west, r.south, r.east, r.north), count=5)
        for g in grans:
            urls += [u for u in ed.granule_links(g) if "CH4ENH" in Path(u).name and r.scene_id in Path(u).name
                     and u.endswith(".tif")]
    ed.download(urls, Path(cfg.paths.raw) / "enh", workers=cfg.download.workers)
    log.info("fetched %d enhancement files at %s", len(urls), datetime.now())
