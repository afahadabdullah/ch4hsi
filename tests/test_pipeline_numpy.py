"""Integration test of the numpy part of the pipeline on synthetic EMIT-like scenes:
preprocess (MF, features, GLT ortho, labels) -> split -> MF-baseline metrics.
netCDF / rasterio I/O are replaced by in-memory stand-ins so this runs without those packages;
the full I/O path is exercised by slurm/smoke.sbatch on NCCS.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ch4hsi.labels as labels_mod  # noqa: E402
import ch4hsi.preprocess as pp  # noqa: E402
from ch4hsi import fake, splits  # noqa: E402
from ch4hsi.config import load_config  # noqa: E402
from ch4hsi.evaluate import mf_score  # noqa: E402
from ch4hsi.io.emit import EmitScene  # noqa: E402
from ch4hsi.io.glt import ortho  # noqa: E402
from ch4hsi.metrics import HistPR, object_counts, pixel_counts, prf, remove_small  # noqa: E402
from ch4hsi.physics.target import unit_absorption  # noqa: E402


def build_pipeline(tmp: Path):
    """Preprocess 6 synthetic scenes into `tmp` and run the split stage. Returns (cfg, tmp)."""
    cfg = load_config(None, [f"paths.data_root={tmp}", "physics.lut=synthetic", "split.method=random",
                             "split.norm_samples_per_scene=500"])
    rng = np.random.default_rng(3)
    wl, fwhm, good = fake.emit_bands()
    k_full = unit_absorption(wl, fwhm, spec="synthetic")
    store = {}
    rows = []
    for i in range(6):
        role = "pos" if i < 4 else "neg"
        L, maps = fake.make_scene(120, 100, k_full, 2 if role == "pos" else 0, rng, wl)
        gx, gy = fake._glt(120, 100, angle_deg=10.0 * (i - 3))
        sid = f"20240501T17{i:02d}00_24000{i:02d}_001"
        store[sid] = (L, gx, gy, maps)
        rows.append(dict(sid=sid, role=role))

    def fake_read(path, selector):
        sid = Path(path).stem.replace("EMIT_L1B_RAD_001_", "")
        L, gx, gy, _ = store[sid]
        sets = selector(wl, fwhm, good.astype(bool))
        return EmitScene(scene_id=sid, wavelengths=wl, fwhm=fwhm, good=good.astype(bool), glt_x=gx, glt_y=gy,
                         geotransform=(0, 1, 0, 0, 0, -1), crs_wkt="EPSG:4326",
                         radiance={n: L[:, :, ix] for n, ix in sets.items()})

    def fake_rasterize(plumes, shape, gt, crs, min_ppmm):
        sid = plumes[0]["scene"]
        _, gx, gy, maps = store[sid]
        tot = ortho(np.sum(maps, 0)[..., None], gx, gy)[..., 0]
        ids_s = np.zeros(maps[0].shape, np.int32)
        for j, m in enumerate(maps, 1):
            ids_s[(m >= 150) & (ids_s == 0)] = j
        ids = np.nan_to_num(ortho(ids_s[..., None].astype(np.float32), gx, gy)[..., 0]).astype(np.int32)
        return (ids > 0).astype(np.uint8), ids, tot, [p["plume_id"] for p in plumes]

    orig = pp.read_emit_l1b, labels_mod.rasterize_plumes
    pp.read_emit_l1b, labels_mod.rasterize_plumes = fake_read, fake_rasterize
    try:
        for r in rows:
            pl = [dict(plume_id=f"p{r['sid']}_{j}", scene=r["sid"]) for j in range(2)] if r["role"] == "pos" else []
            out = pp.process_scene(f"/x/EMIT_L1B_RAD_001_{r['sid']}.nc", cfg.paths.scenes, cfg, pl, r["role"])
            assert (out / "features.npy").exists()
    finally:
        pp.read_emit_l1b, labels_mod.rasterize_plumes = orig
    splits.run(cfg)
    return cfg, tmp


def test_numpy_pipeline():
    cfg, tmp = build_pipeline(Path(tempfile.mkdtemp()))
    meta = pd.read_json(next(Path(cfg.paths.scenes).glob("*/meta.json")), typ="series")
    assert len(meta["channel_names"]) == 1 + 1 + 12 + 1 + 3
    assert len(meta["k_bins"]) == 12

    norm = pd.read_json(Path(cfg.paths.splits) / "norm.json", typ="series")
    assert len(norm["mean"]) == 18

    # MF baseline separates plume pixels from background
    h = HistPR(0, 5000, 500)
    tp = fp = fn = 0
    det = tot = 0
    for d in Path(cfg.paths.scenes).iterdir():
        mf, v = np.load(d / "mf.npy"), np.load(d / "valid.npy")
        mask, ids = np.load(d / "mask.npy"), np.load(d / "plume_id.npy")
        s = mf_score(mf, v, 1.0)
        h.update(s, mask > 0, v)
        pred = remove_small((s >= 400) & v, 4)
        a, b, c = pixel_counts(pred, mask > 0, v)
        tp, fp, fn = tp + a, fp + b, fn + c
        oc = object_counts(pred, ids, v)
        det, tot = det + oc["gt_detected"], tot + oc["gt_total"]
    m = prf(tp, fp, fn)
    print("MF baseline on synthetic scenes:", {k: round(x, 3) for k, x in m.items()}, "AP", round(h.average_precision(), 3),
          "plume recall", det, "/", tot)
    assert h.average_precision() > 0.3 and det / max(tot, 1) > 0.5


if __name__ == "__main__":
    test_numpy_pipeline()
    print("ok test_numpy_pipeline")
