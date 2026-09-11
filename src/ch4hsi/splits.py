"""Stage 4: leakage-aware train/val/test split + per-channel normalisation statistics.

`geo_block` groups scenes into lat/lon blocks keyed by the plume location (positives) or the scene
centre (negatives), and assigns whole blocks to a split, so repeat overpasses of the same facility
never straddle train and test.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Cfg
from .utils import dump_json, get_logger, load_json

log = get_logger(__name__)


def scene_table(scenes_dir: str | Path) -> pd.DataFrame:
    rows = []
    for m in sorted(Path(scenes_dir).glob("*/meta.json")):
        meta = load_json(m)
        rows.append(dict(scene_id=meta["scene_id"], dir=str(m.parent), role=meta["role"],
                         n_pos_px=meta["n_pos_px"], lat=meta["center_lat"], lon=meta["center_lon"],
                         plume_ids=";".join(meta.get("plume_ids", []))))
    return pd.DataFrame(rows)


def _block_key(row, plumes, deg):
    lat, lon = row.lat, row.lon
    ids = [p for p in str(row.plume_ids).split(";") if p]
    if ids and plumes is not None:
        sub = plumes[plumes.plume_id.isin(ids)]
        if len(sub):
            lat = float(((sub.south + sub.north) / 2).iloc[0])
            lon = float(((sub.west + sub.east) / 2).iloc[0])
    return f"{int(np.floor(lat / deg))}_{int(np.floor(lon / deg))}"


def assign_splits(df: pd.DataFrame, plumes, cfg_split) -> dict:
    rng = np.random.default_rng(cfg_split.seed)
    names = ["train", "val", "test"]
    frac = np.asarray(cfg_split.fractions, float)
    frac = frac / frac.sum()
    if cfg_split.method == "random":
        df = df.sample(frac=1, random_state=cfg_split.seed)
        cuts = (np.cumsum(frac)[:-1] * len(df)).astype(int)
        parts = np.split(df.scene_id.to_numpy(), cuts)
        return {n: sorted(p.tolist()) for n, p in zip(names, parts)}
    df = df.copy()
    df["block"] = [_block_key(r, plumes, cfg_split.block_deg) for r in df.itertuples()]
    blocks = df.groupby("block").agg(n_pos=("role", lambda s: (s == "pos").sum()),
                                     n_neg=("role", lambda s: (s == "neg").sum())).reset_index()
    blocks = blocks.iloc[rng.permutation(len(blocks))]
    # big blocks first so the greedy balancing has room to correct
    blocks = blocks.sort_values("n_pos", ascending=False, kind="stable")
    T = np.array([blocks.n_pos.sum(), blocks.n_neg.sum()], float)
    cur = np.zeros((3, 2))
    assign = {}
    for b in blocks.itertuples():
        w = np.array([b.n_pos, b.n_neg], float)
        need = frac[:, None] * T[None, :] - cur          # remaining capacity per split
        score = (need * (w > 0)[None, :]).sum(axis=1)
        s = int(np.argmax(score))
        assign[b.block] = s
        cur[s] += w
    df["split"] = df.block.map(lambda k: names[assign[k]])
    return {n: sorted(df.scene_id[df.split == n].tolist()) for n in names}


def norm_stats(scene_dirs: list[str], n_per_scene: int, seed=0):
    rng = np.random.default_rng(seed)
    acc = []
    for d in scene_dirs:
        f = np.load(Path(d) / "features.npy", mmap_mode="r")
        v = np.load(Path(d) / "valid.npy")
        idx = np.flatnonzero(v.ravel())
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, size=min(n_per_scene, len(idx)), replace=False)
        rr, cc = np.unravel_index(np.sort(pick), v.shape)
        acc.append(np.asarray(f[:, rr, cc], np.float32).T)
    X = np.concatenate(acc)
    lo, hi = np.percentile(X, [0.5, 99.5], axis=0)
    Xc = np.clip(X, lo, hi)                 # robust to the heavy MF tail
    return dict(mean=Xc.mean(0).tolist(), std=np.maximum(Xc.std(0), 1e-6).tolist())


def run(cfg: Cfg):
    df = scene_table(cfg.paths.scenes)
    if df.empty:
        raise RuntimeError(f"no preprocessed scenes in {cfg.paths.scenes}")
    pl_path = Path(cfg.paths.catalog) / "plumes.csv"
    plumes = pd.read_csv(pl_path, dtype={"plume_id": str}) if pl_path.exists() else None
    splits = assign_splits(df, plumes, cfg.split)
    out = Path(cfg.paths.splits)
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for n, ids in splits.items():
        sub = df[df.scene_id.isin(ids)]
        summary[n] = dict(scenes=len(sub), pos=int((sub.role == "pos").sum()), neg=int((sub.role == "neg").sum()),
                          pos_px=int(sub.n_pos_px.sum()))
    log.info("split summary: %s", summary)
    train_dirs = df[df.scene_id.isin(splits["train"])].dir.tolist()
    stats = norm_stats(train_dirs, cfg.split.norm_samples_per_scene, cfg.split.seed)
    meta0 = load_json(Path(train_dirs[0]) / "meta.json")
    stats["channel_names"] = meta0["channel_names"]
    dump_json(dict(splits=splits, summary=summary), out / "splits.json")
    dump_json(stats, out / "norm.json")
    df.to_csv(out / "scene_table.csv", index=False)
    log.info("wrote %s and norm.json", out / "splits.json")
