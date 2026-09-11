"""In-memory synthetic dataset for demos, README figures and tests (no netCDF4 / rasterio / internet).

Same physics as `make-fake-data` (fake.py): synthetic EMIT-like radiance with 285 bands, Gaussian plumes
injected with Beer–Lambert, a rotated GLT and a real-looking geotransform over oil & gas regions. The
difference is that nothing is written as NetCDF/GeoTIFF: the L1B reader and the label rasteriser are
swapped for in-memory stand-ins, and the real `preprocess.process_scene` and `splits.run` do the rest.
The raw sensor-geometry cubes are kept in `SyntheticSet.store` so callers can re-inject plumes (MDL) or
draw sensor-geometry panels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import fake
from .config import Cfg, load_config
from .io.emit import EmitScene
from .io.glt import ortho
from .physics.plume_sim import gaussian_plume_ppmm, inject
from .physics.target import unit_absorption
from .utils import get_logger

log = get_logger(__name__)
LABEL_MIN_PPMM = 150.0          # same footprint threshold as fake.write_plume_tif


@dataclass
class SyntheticSet:
    cfg: Cfg
    wl: np.ndarray
    fwhm: np.ndarray
    good: np.ndarray
    k_full: np.ndarray
    store: dict = field(default_factory=dict)       # scene_id -> dict(L, gx, gy, maps, gt, role, q)

    @property
    def root(self) -> Path:
        return Path(self.cfg.paths.data_root)


def default_overrides(root, lut: str = "synthetic") -> list[str]:
    return [f"paths.data_root={root}", f"physics.lut={lut}", "run_name=synthetic_demo",
            "split.method=geo_block", "split.block_deg=1.0", "split.fractions=[0.5,0.25,0.25]",
            "split.norm_samples_per_scene=3000", "eval.n_example_figures=2", "eval.bootstrap=300",
            "eval.infer_tile=128", "eval.infer_stride=96", "mdl.n_scenes=4", "mdl.plumes_per_scene=4",
            "mdl.min_separation_px=45", "mdl.flux_kgph=[150,300,500,800,1200,2000,3500,6000]", "mdl.bootstrap=200"]


def build(root, n_pos: int = 16, n_neg: int = 8, rows: int = 220, cols: int = 200, seed: int = 0,
          lut: str = "synthetic", q_range=(300.0, 6000.0), extra_overrides: list[str] | None = None) -> SyntheticSet:
    import ch4hsi.labels as labels_mod
    import ch4hsi.preprocess as pp
    from . import splits

    cfg = load_config(None, default_overrides(root, lut) + list(extra_overrides or []))
    rng = np.random.default_rng(seed)
    wl, fwhm, good = fake.emit_bands()
    k_full = unit_absorption(wl, fwhm, spec=lut)
    ss = SyntheticSet(cfg, wl, fwhm, good.astype(bool), k_full)
    for i in range(n_pos + n_neg):
        role = "pos" if i < n_pos else "neg"
        L = fake._surface(rows, cols, wl, rng)
        maps, qs = [], []
        for _ in range(int(rng.integers(1, 4)) if role == "pos" else 0):
            src = (int(rng.integers(35, rows - 35)), int(rng.integers(35, cols - 35)))
            q = float(np.exp(rng.uniform(np.log(q_range[0]), np.log(q_range[1]))))
            maps.append(gaussian_plume_ppmm((rows, cols), src, q, float(rng.uniform(2.5, 4.0)),
                                            float(rng.uniform(0, 2 * np.pi)), max_age_s=700, turbulence=0.3, rng=rng))
            qs.append(q)
        if maps:
            L = inject(L, np.sum(maps, axis=0), k_full)
        gx, gy = fake._glt(rows, cols, angle_deg=float(rng.uniform(-14, 14)))
        lon, lat = fake.SITES[i % len(fake.SITES)]
        gt = (lon + rng.uniform(-0.3, 0.3), fake.DX, 0.0, lat + rng.uniform(-0.3, 0.3), 0.0, -fake.DX)
        sid = f"2024{5 + i // 28:02d}{1 + i % 28:02d}T17{i % 60:02d}00_{2400000 + i:07d}_{(i % 30) + 1:03d}"
        ss.store[sid] = dict(L=L, gx=gx, gy=gy, maps=maps, q=qs, gt=gt, role=role)

    def fake_read(path, selector):
        sid = Path(path).stem.replace("EMIT_L1B_RAD_001_", "")
        s = ss.store[sid]
        sets = selector(wl, fwhm, ss.good)
        return EmitScene(scene_id=sid, wavelengths=wl, fwhm=fwhm, good=ss.good, glt_x=s["gx"], glt_y=s["gy"],
                         geotransform=s["gt"], crs_wkt="EPSG:4326",
                         radiance={n: np.ascontiguousarray(s["L"][:, :, ix]) for n, ix in sets.items()})

    def fake_rasterize(plumes, shape, gt, crs, min_ppmm):
        s = ss.store[plumes[0]["scene"]]
        ids_s = np.zeros(s["maps"][0].shape, np.int32)
        for j, m in enumerate(s["maps"], 1):
            ids_s[(m >= LABEL_MIN_PPMM) & (ids_s == 0)] = j
        ids = np.nan_to_num(ortho(ids_s[..., None].astype(np.float32), s["gx"], s["gy"])[..., 0]).astype(np.int32)
        tot = ortho(np.sum(s["maps"], 0)[..., None], s["gx"], s["gy"])[..., 0]
        used = sorted({int(v) for v in np.unique(ids) if v > 0})
        return (ids > 0).astype(np.uint8), ids, tot, [plumes[j - 1]["plume_id"] for j in used]

    orig = pp.read_emit_l1b, labels_mod.rasterize_plumes
    pp.read_emit_l1b, labels_mod.rasterize_plumes = fake_read, fake_rasterize
    try:
        for sid, s in ss.store.items():
            pl = [dict(plume_id=f"{sid}_p{j}", scene=sid) for j in range(1, len(s["maps"]) + 1)]
            pp.process_scene(f"/synthetic/EMIT_L1B_RAD_001_{sid}.nc", cfg.paths.scenes, cfg, pl, s["role"],
                             extra_meta={"synthetic": True, "q_kgph": s["q"]})
    finally:
        pp.read_emit_l1b, labels_mod.rasterize_plumes = orig
    splits.run(cfg)
    log.info("synthetic dataset: %d plume + %d plume-free scenes under %s", n_pos, n_neg, root)
    return ss
