#!/usr/bin/env python
"""Figures for the README: workflow diagram (with real thumbnails at every stage) and example maps.

    python scripts/make_readme_figures.py                                    # synthetic (default if no run)
    python scripts/make_readme_figures.py --config configs/gh200.yaml --run unet_v1   # real data on Prism

Outputs (default docs/figures/):
    workflow.png        pipeline diagram; each box shows that stage's actual output for one scene
    scene_maps.png      georeferenced true colour | MF ppm·m | model probability | TP/FP/FN (model, MF)
                        for a plume scene and a plume-free scene
    physics_check.png   CH4 unit absorption k(λ), in-plume radiance ratio vs Beer–Lambert, MF vs truth
    mdl_injection.png   one background with plumes of increasing source rate, and the POD curve
    diag_*.png          three panels copied from the run's `ch4hsi diagnose` output

Data source
  real       preprocessed scenes of `--run` (test split), cached predictions from `evaluate`
             (runs/<run>/preds), thresholds, pr_hist.npz and MDL records; sensor-geometry panels read the
             scene's L1B file when it is still on disk (else ortho products are shown instead).
  synthetic  demo.build(): EMIT-like radiance with injected Gaussian plumes, run through the real
             preprocess / split / evaluate code. The model is the U-Net (quick training) when torch is
             importable, otherwise the per-pixel logistic-regression baseline — every panel says which.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ch4hsi import plotting as P  # noqa: E402
from ch4hsi.config import load_config, run_dir  # noqa: E402
from ch4hsi.evaluate import evaluate_scenes, load_scene, mf_score  # noqa: E402
from ch4hsi.features import band_selector, scene_products  # noqa: E402
from ch4hsi.metrics import HistPR, pod_with_ci, remove_small  # noqa: E402
from ch4hsi.physics.plume_sim import gaussian_plume_ppmm, inject, random_sources  # noqa: E402
from ch4hsi.utils import get_logger, load_json  # noqa: E402

log = get_logger("readme_figures")


# ====================================================================== data bundles
class Bundle(dict):
    __getattr__ = dict.get


def _has_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def synthetic_bundle(root: Path, model: str, seed: int) -> Bundle:
    from ch4hsi import demo
    from ch4hsi.baselines import PixelLogReg
    if root.exists():
        shutil.rmtree(root)
    ss = demo.build(root, seed=seed)
    cfg = ss.cfg
    out = run_dir(cfg)
    sp = load_json(Path(cfg.paths.splits) / "splits.json")["splits"]
    norm = load_json(Path(cfg.paths.splits) / "norm.json")
    if model == "auto":
        model = "unet" if _has_torch() else "pixel_lr"
    if model == "unet":
        from ch4hsi import evaluate, train
        for kv in ("tile=64", "epochs=8", "tiles_per_epoch=768", "val_tiles=128", "batch_size=16", "num_workers=0",
                   "base_channels=16", "depth=3", "stage_to_local=false", "warmup_epochs=1", "early_stop_patience=50"):
            k, v = kv.split("=")
            cfg.train[k] = type(cfg.train[k])(v) if not isinstance(cfg.train[k], bool) else v == "true"
        cfg.train.synth_aug.enabled = True
        train.run(cfg)
        evaluate.run(cfg)
        pred_dir = out / "preds"
        label = "U-Net (quick synthetic training)"

        def score(sc):
            p = pred_dir / f"{sc['meta']['scene_id']}.npy"
            return np.load(p).astype(np.float32) if p.exists() else None
        chain_score = _unet_chain_scorer(cfg, out)
    else:
        lr = PixelLogReg(norm).fit([Path(cfg.paths.scenes) / s for s in sp["train"]], seed=seed)
        label = "Pixel logistic regression"

        def score(sc):
            return lr.predict(sc["features"], sc["valid"])
        evaluate_scenes(cfg, score, out, label=label)
        chain_score = score
        (out / "preds").mkdir(exist_ok=True)       # cache like `evaluate`, so `--source real` works on this run
        for sid in sp["val"] + sp["test"]:
            np.save(out / "preds" / f"{sid}.npy", score(load_scene(Path(cfg.paths.scenes) / sid)).astype(np.float16))
    from ch4hsi.diagnostics import diagnose
    diagnose(cfg, out, score_fn=score, watermark="synthetic data")
    B = _common(cfg, out, score, label, source="synthetic")
    # sensor geometry straight from the in-memory store
    s = ss.store[B.scene["meta"]["scene_id"]]
    B.sensor = _sensor_from_cube(cfg, s["L"], ss.wl, ss.fwhm, ss.good, s["gx"], s["gy"],
                                 truth=np.sum(s["maps"], 0) if s["maps"] else None)
    negs = [sid for sid in sp["test"] + sp["val"] if ss.store[sid]["role"] == "neg"]
    B.inj_cube = _sensor_from_cube(cfg, ss.store[negs[0]]["L"], ss.wl, ss.fwhm, ss.good, ss.store[negs[0]]["gx"],
                                   ss.store[negs[0]]["gy"])
    B.pod = _mdl_lite(cfg, [(_sensor_from_cube(cfg, ss.store[n]["L"], ss.wl, ss.fwhm, ss.good, ss.store[n]["gx"],
                                               ss.store[n]["gy"]), n) for n in negs[:4]], chain_score, B.thr, seed)
    return B


def _unet_chain_scorer(cfg, out):
    import torch

    from ch4hsi.evaluate import load_model, predict_scene
    from ch4hsi.train import amp_dtype
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, norm, adt = load_model(out, cfg, dev), load_json(out / "norm.json"), amp_dtype(cfg.train.amp)
    return lambda sc: predict_scene(model, sc["features"], sc["valid"], norm, cfg.eval.infer_tile,
                                    cfg.eval.infer_stride, dev, adt)


def real_bundle(cfg, run: str) -> Bundle:
    cfg.run_name = run
    out = run_dir(cfg)
    if not (out / "thresholds.json").exists():
        raise SystemExit(f"{out} has no evaluation outputs; run `ch4hsi evaluate` first")
    mlabel = (load_json(out / "metrics_test.json") if (out / "metrics_test.json").exists() else {}).get("model_label", "U-Net")

    def score(sc):
        p = out / "preds" / f"{sc['meta']['scene_id']}.npy"
        return np.load(p).astype(np.float32) if p.exists() else None
    B = _common(cfg, out, score, mlabel, source="real")
    B.sensor = _sensor_from_l1b(cfg, B.scene["meta"])
    if B.neg_scene is not None:
        B.inj_cube = _sensor_from_l1b(cfg, B.neg_scene["meta"])
    files = sorted(out.glob("mdl_records*.csv"))
    if files:
        df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        B.pod = dict(df=df, fit=load_json(out / "mdl.json") if (out / "mdl.json").exists() else {})
    return B


def _common(cfg, out, score, label, source) -> Bundle:
    sp = load_json(Path(cfg.paths.splits) / "splits.json")["splits"]
    sdf = pd.read_csv(out / "per_scene_test.csv")
    pos = sdf[sdf.role == "pos"]
    # show a scene where plumes are clear but not trivial: the plume scene with the most labelled pixels
    sid = pos.sort_values("n_pos_px", ascending=False).scene_id.iloc[0]
    neg = sdf[sdf.role == "neg"].sort_values("model_pred_total", ascending=False)
    B = Bundle(cfg=cfg, out=out, source=source, label=label, thr=load_json(out / "thresholds.json"),
               scene=load_scene(Path(cfg.paths.scenes) / sid), splits=sp,
               scene_table=pd.read_csv(Path(cfg.paths.splits) / "scene_table.csv"))
    B.prob = score(B.scene)
    B.neg_scene = load_scene(Path(cfg.paths.scenes) / neg.scene_id.iloc[0]) if len(neg) else None
    B.neg_prob = score(B.neg_scene) if B.neg_scene is not None else None
    B.hist = dict(np.load(out / "pr_hist.npz")) if (out / "pr_hist.npz").exists() else None
    return B


def _sensor_from_cube(cfg, L, wl, fwhm, good, gx, gy, truth=None):
    from ch4hsi.physics.target import unit_absorption
    sel = band_selector(cfg.physics, cfg.preprocess)(wl, fwhm, good)
    k = unit_absorption(wl[sel["win"]], fwhm[sel["win"]], spec=cfg.physics.lut)
    return dict(win=np.ascontiguousarray(L[:, :, sel["win"]]), rgb=np.ascontiguousarray(L[:, :, sel["rgb"]]),
                wl=wl[sel["win"]], wl_all=wl, k=k, gx=gx, gy=gy, truth=truth, spectrum_full=L)


def _sensor_from_l1b(cfg, meta):
    try:
        from ch4hsi.io.emit import read_emit_l1b
        from ch4hsi.physics.target import load_lut, unit_absorption
        sel = band_selector(cfg.physics, cfg.preprocess)
        sc = read_emit_l1b(meta["l1b"], sel)
        idx = sel(sc.wavelengths, sc.fwhm, sc.good)
        k = unit_absorption(sc.wavelengths[idx["win"]], sc.fwhm[idx["win"]], lut=load_lut(cfg.physics.lut),
                            spec=cfg.physics.lut)
        return dict(win=sc.radiance["win"], rgb=sc.radiance["rgb"], wl=sc.wavelengths[idx["win"]],
                    wl_all=sc.wavelengths, k=k, gx=sc.glt_x, gy=sc.glt_y, truth=None, spectrum_full=None)
    except Exception as e:  # raw file deleted, netCDF4 missing, ...
        log.warning("sensor-geometry panels unavailable for %s (%s); using ortho products", meta["scene_id"], e)
        return None


def _chain(cfg, S, rad_win, extra=None):
    from ch4hsi.io.emit import valid_mask
    valid_s = valid_mask(S["win"]) & valid_mask(S["rgb"])
    return scene_products(rad_win, S["rgb"], S["k"], S["gx"], S["gy"], cfg.physics, cfg.preprocess, cfg.features,
                          valid=valid_s, extra_sensor=extra), valid_s


def _mdl_lite(cfg, cubes, chain_score, thr, seed):
    """Small version of `ch4hsi mdl` on in-memory synthetic cubes (same chain, same detection rule)."""
    M, E = cfg.mdl, cfg.eval
    recs = []
    for si, (S, sid) in enumerate(cubes):
        prod0, valid_s = _chain(cfg, S, S["win"])
        sc0 = dict(features=prod0["features"], valid=prod0["valid"], meta={"scene_id": f"{sid}_bg"})
        s0 = {"model": chain_score(sc0), "mf": mf_score(prod0["mf"], prod0["valid"], E.mf_smooth_sigma_px)}
        det0 = {m: remove_small((s0[m] >= thr[m]["threshold"]) & prod0["valid"], E.min_component_px) for m in s0}
        rng = np.random.default_rng([seed, si])
        srcs = random_sources(valid_s, M.plumes_per_scene, M.min_separation_px, margin=30, rng=rng)
        th = rng.uniform(0, 2 * np.pi, len(srcs))
        for q in M.flux_kgph:
            maps = [gaussian_plume_ppmm(valid_s.shape, s, q, M.wind_ms, t, max_age_s=M.max_plume_age_s,
                                        turbulence=M.turbulence, rng=rng) for s, t in zip(srcs, th)]
            tot = np.sum(maps, 0)
            ident = np.where(tot >= M.footprint_min_ppmm, np.argmax(maps, 0) + 1, 0).astype(np.float32)
            prod, _ = _chain(cfg, S, inject(S["win"], tot, S["k"]), {"inj_ppmm": tot, "inj_id": ident})
            sc = dict(features=prod["features"], valid=prod["valid"], meta={"scene_id": f"{sid}_q{q}"})
            dets = {"model": chain_score(sc), "mf": mf_score(prod["mf"], prod["valid"], E.mf_smooth_sigma_px)}
            inj_id = np.nan_to_num(prod["inj_id"]).astype(int)
            for j in range(1, len(srcs) + 1):
                fp = inj_id == j
                if not fp.any():
                    continue
                from scipy.ndimage import binary_dilation
                fpd = binary_dilation(fp)
                r = dict(scene_id=sid, q_kgph=q, plume=j, peak_ppmm=float(np.nanmax(prod["inj_ppmm"][fp])))
                for m in ("model", "mf"):
                    det = remove_small((dets[m] >= thr[m]["threshold"]) & prod["valid"], E.min_component_px)
                    r[f"det_{m}"] = bool(det[fpd].any())
                    r[f"pre_{m}"] = bool(det0[m][fpd].any())   # site already flagged before injection
                recs.append(r)
        log.info("MDL-lite scene %s: %d records", sid, len(recs))
    df = pd.DataFrame(recs)
    fit = {"wind_ms": M.wind_ms}
    for m in ("model", "mf"):
        d = df[~df[f"pre_{m}"]]            # as in `ch4hsi mdl-fit`: drop sites flagged before injection
        if d[f"det_{m}"].nunique() == 2:
            fit[m] = {"flux": pod_with_ci(d.q_kgph, d[f"det_{m}"], M.pod_levels, M.bootstrap, M.seed)}
    return dict(df=df, fit=fit)


# ====================================================================== drawing helpers
def _img(ax, a, cmap=None, vmin=None, vmax=None, ext=None):
    ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, extent=ext, interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)


def _rgb_sensor(S):
    rgb = S["rgb"].astype(np.float32)
    lo, hi = np.percentile(rgb, 2, axis=(0, 1)), np.percentile(rgb, 98, axis=(0, 1))
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1) ** 0.8


def _box(fig, x, y, w, h, title, stage, caption, color):
    """Rounded card with a colour rule, title, `ch4hsi <stage>` chip, caption; returns the content axes."""
    from matplotlib.patches import FancyBboxPatch
    fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                      transform=fig.transFigure, fc="white", ec=P.GRID, lw=1.2, zorder=-2))
    fig.patches.append(FancyBboxPatch((x, y + h - 0.006), w, 0.006, boxstyle="square,pad=0", transform=fig.transFigure,
                                      fc=color, ec="none", zorder=-1))
    fig.text(x + 0.008, y + h - 0.022, title, fontsize=10.5, weight="semibold", color=P.INK, va="top")
    if stage:
        fig.text(x + 0.008, y + h - 0.052, stage, fontsize=7.2, family="monospace", color=P.INK2, va="top",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#f3f2ef", ec="none"))
    fig.text(x + 0.008, y + 0.012, caption, fontsize=7.5, color=P.INK2, va="bottom")
    ax = fig.add_axes([x + 0.03, y + 0.1, w - 0.042, h - 0.19])
    ax.set_zorder(3)
    ax.set_facecolor("white")
    return ax


def _arrow(fig, p0, p1, text=None, dashed=False, rad=0.0):
    from matplotlib.patches import FancyArrowPatch
    a = FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=14, lw=1.6,
                        color=P.INK2, ls="--" if dashed else "-", connectionstyle=f"arc3,rad={rad}", zorder=5)
    a.set_zorder(6)
    fig.patches.append(a)
    if text:
        fig.text((p0[0] + p1[0]) / 2 + 0.004, (p0[1] + p1[1]) / 2, text, fontsize=7, color=P.INK2, ha="left",
                 va="center", style="italic")


def _plume_crop(sc, half=70):
    ids = sc["ids"] if sc["ids"].any() else sc["mask"]
    rr, cc = np.nonzero(ids > 0)
    H, W = sc["valid"].shape
    if not len(rr):
        return (slice(0, H), slice(0, W))
    r0, c0 = int(np.median(rr)), int(np.median(cc))
    return (slice(max(0, r0 - half), min(H, r0 + half)), slice(max(0, c0 - half), min(W, c0 + half)))


# ====================================================================== figures
def fig_workflow(B, path):
    plt = P.apply_style()
    cfg, sc = B.cfg, B.scene
    meta, valid = sc["meta"], sc["valid"]
    names = meta["channel_names"]
    fig = plt.figure(figsize=(17, 8.6))
    W, H, gx, gy = 0.178, 0.405, 0.022, 0.07
    xs = [0.02 + i * (W + gx) for i in range(5)]
    y1, y2 = 0.515, 0.515 - H - gy
    C = [P.C_MODEL, P.C_MODEL, P.C_MF, P.C_MF, P.C_MF_FIXED, P.C_MF_FIXED, P.C_MODEL, P.C_MF, P.C_MF_FIXED, P.C_EXTRA]
    S = B.sensor
    mf_vmax = max(800.0, float(np.nanpercentile(np.where(valid, sc["mf"], np.nan), 99.8)))

    # 1 radiance
    ax = _box(fig, xs[0], y1, W, H, "EMIT L1B radiance", "ch4hsi download",
              "285 bands, 381–2493 nm, 60 m; sensor geometry\n(downtrack × crosstrack), ~1.8 GB per scene", C[0])
    if S is not None:
        _img(ax, _rgb_sensor(S))
        ia = ax.inset_axes([0.52, 0.03, 0.46, 0.34])
        full = S.get("spectrum_full")
        if full is not None:
            spec = np.median(full.reshape(-1, full.shape[-1]), 0)
            ia.plot(S["wl_all"], spec, color=P.INK, lw=0.9)
            ia.axvspan(S["wl"].min(), S["wl"].max(), color=P.C_MF, alpha=0.25, lw=0)
        else:
            ia.plot(S["wl"], np.median(S["win"].reshape(-1, S["win"].shape[-1]), 0), color=P.INK, lw=0.9)
        ia.set_xticks([500, 1500, 2400]), ia.set_yticks([])
        ia.tick_params(labelsize=5.5, length=2, pad=1)
        ia.set_facecolor("#ffffffdd")
        ia.set_title("median spectrum", fontsize=6, pad=1)
    else:
        _img(ax, P.rgb_image(sc["features"], names, valid))
    # 2 target
    ax = _box(fig, xs[1], y1, W, H, "CH₄ target spectrum", "physics/target.py",
              "k(λ) = ∂ln L/∂(ppm·m) from the mag1c LUT,\nconvolved to band centres/FWHM (2122–2488 nm)", C[1])
    if S is not None:
        ax.plot(S["wl"], S["k"] * 1e5, color=P.C_MF, lw=1.6, marker="o", ms=2.5)
        ax.set_xlabel("wavelength (nm)", fontsize=7)
        ax.set_ylabel("k  (10⁻⁵ per ppm·m)", fontsize=7)
        ax.tick_params(labelsize=6.5)
    else:
        kb = np.asarray(meta.get("k_bins", []))
        ax.bar(np.arange(len(kb)), kb * 1e5, color=P.C_MF)
        ax.set(xlabel="SWIR bin", ylabel="mean k (10⁻⁵ / ppm·m)")
    # 3 MF sensor
    ax = _box(fig, xs[2], y1, W, H, "Column-wise matched filter", "ch4hsi preprocess",
              f"per-detector-column μ, Σ (shrinkage, 2 passes)\n→ enhancement in ppm·m, σ ≈ {meta.get('sigma_median_ppmm') or 0:.0f} ppm·m", C[2])
    if S is not None:
        from ch4hsi.physics.matched_filter import columnwise_mf
        mf_s, _, _ = columnwise_mf(S["win"], S["k"], None, cfg.physics.column_group, cfg.physics.shrinkage,
                                   cfg.physics.two_pass, cfg.physics.outlier_sigma, cfg.physics.albedo_correction)
        _img(ax, mf_s, P.ppmm_cmap(), 0, mf_vmax)
    else:
        _img(ax, np.where(valid, sc["mf"], np.nan), P.ppmm_cmap(), 0, mf_vmax)
    # 4 features
    ax = _box(fig, xs[3], y1, W, H, f"{len(names)} sensor-agnostic features", "ch4hsi preprocess",
              "MF, MF/σ, 12 SWIR residual bins L/μ−1,\nalbedo, visible RGB  → features.npy (C, H, W)", C[3])
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    _img(ax, np.zeros((2, 2)) * np.nan)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    show = [n for n in ("rgb_r", "albedo", "swir_resid_6", "swir_resid_9", "mf_snr", "mf") if n in names]
    crop = _plume_crop(sc, 60)
    for i, n in enumerate(show):
        a = np.asarray(sc["features"][names.index(n)], np.float32)[crop]
        v = valid[crop]
        a = np.where(v, a, np.nan)
        lo, hi = np.nanpercentile(a, [2, 98]) if np.isfinite(a).any() else (0, 1)
        if n.startswith("swir"):
            lo, hi = hi, lo                     # absorption = dips: show dips bright
        off = i * 0.085
        ex = [0.03 + off, 0.53 + off, 0.02 + off * 0.9, 0.52 + off * 0.9]
        cm = P.ppmm_cmap() if n.startswith("mf") else "Greys_r"
        ax.imshow(np.clip((a - lo) / (hi - lo + 1e-9), 0, 1), cmap=cm, extent=ex, zorder=i, vmin=0, vmax=1)
        ax.add_patch(plt.Rectangle((ex[0], ex[2]), 0.5, 0.5, fill=False, ec="white", lw=1.2, zorder=i + 0.5))
        if i == len(show) - 1:
            ax.text(ex[1] + 0.015, ex[3] - 0.02, n, fontsize=6.5, color=P.INK2, va="top", zorder=20)
        elif i == 0:
            ax.text(ex[0], ex[2] - 0.03, n, fontsize=6.5, color=P.INK2, va="top", zorder=20)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    # 5 ortho + labels
    ax = _box(fig, xs[4], y1, W, H, "GLT ortho + plume labels", "ch4hsi preprocess",
              "geometric lookup table → map grid (EPSG:4326);\nEMIT L2B CH₄ plume complexes rasterised as labels", C[4])
    ext = P.extent(meta)
    ax.imshow(np.where(valid, sc["mf"], np.nan), cmap=P.ppmm_cmap(), vmin=0, vmax=mf_vmax, extent=ext)
    if sc["mask"].any():
        ax.contour(np.flipud(sc["mask"].astype(float)), levels=[0.5], colors=P.C_MODEL, linewidths=1.2, extent=ext,
                   origin="lower")
    P.geo_axes(ax, ext)
    ax.tick_params(labelsize=6)
    # 6 split (row 2, right -> left)
    ax = _box(fig, xs[4], y2, W, H, "Geo-blocked split", "ch4hsi split",
              f"{cfg.split.block_deg:g}° lat/lon blocks keyed on plume location:\nno facility in both train and test", C[5])
    st, split_of = B.scene_table, {s: k for k, ids in B.splits.items() for s in ids}
    st = st.assign(split=st.scene_id.map(split_of))
    for k, c in (("train", P.C_MODEL), ("val", P.C_MF), ("test", P.C_MF_FIXED)):
        g = st[st.split == k]
        ax.scatter(g.lon, g.lat, s=18, color=c, label=f"{k} ({len(g)})", edgecolor="white", lw=0.5, zorder=3)
    ax.set_xlim(max(-180, st.lon.min() - 20), min(180, st.lon.max() + 20))
    ax.set_ylim(max(-80, st.lat.min() - 12), min(85, st.lat.max() + 18))
    ax.xaxis.set_major_formatter(P._deg_fmt("E", "W")), ax.yaxis.set_major_formatter(P._deg_fmt("N", "S"))
    ax.locator_params(nbins=4)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6.5, loc="lower left", ncol=3, handletextpad=0.1, columnspacing=0.6)
    # 7 model
    ax = _box(fig, xs[3], y2, W, H, "Segmentation model", "ch4hsi train",
              f"shown: {B.label}\nU-Net: 128² tiles, BCE+Dice, bf16 on GH200", C[6])
    if B.prob is not None:
        ax.imshow(np.where(valid, B.prob, np.nan), cmap=P.prob_cmap(), vmin=0, vmax=1, extent=ext)
        if sc["mask"].any():
            ax.contour(np.flipud(sc["mask"].astype(float)), levels=[0.5], colors=P.C_MF, linewidths=1.0, extent=ext,
                       origin="lower")
    P.geo_axes(ax, ext)
    ax.tick_params(labelsize=6)
    # 8 evaluation
    ax = _box(fig, xs[2], y2, W, H, "Evaluation (test)", "ch4hsi evaluate",
              "thresholds frozen on val; pixel P/R/F1/IoU/AP,\nplume recall, scene AUROC, 90% bootstrap CIs", C[7])
    if B.hist is not None:
        for m, lab in (("model", B.label.split(" (")[0]), ("mf", "Matched filter")):
            h = HistPR(0, 1, 1)
            h.edges, h.pos, h.neg = (B.hist[f"test_{m}_{k}"] for k in ("edges", "pos", "neg"))
            pr, rc, t = h.curve()
            ax.plot(rc, pr, color=P.METHOD_STYLE[m]["color"], lw=1.6, label=f"{lab}  AP {h.average_precision():.2f}")
            i = min(np.searchsorted(h.edges, B.thr[m]["threshold"], side="right") - 1, len(pr) - 1)
            ax.plot(rc[i], pr[i], "o", color=P.METHOD_STYLE[m]["color"], ms=6, mec="white", mew=1.5)
        ax.set(xlim=(0, 1), ylim=(0, 1.02))
        ax.set_xlabel("recall (pixel)", fontsize=7), ax.set_ylabel("precision", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.legend(fontsize=6.5, loc="lower left")
    # 9 MDL
    ax = _box(fig, xs[1], y2, W, H, "Minimum detection limit", "ch4hsi mdl",
              f"Gaussian plumes of known Q injected into L1B,\nfull chain re-run → POD(Q), MDL50/90 at {cfg.mdl.wind_ms:g} m/s", C[8])
    _pod_axes(ax, B, small=True)
    # 10 diagnostics + report
    ax = _box(fig, xs[0], y2, W, H, "Diagnostics + report", "ch4hsi diagnose · report",
              "error maps, calibration, threshold sweeps, ROC,\nfalse-alarm analysis → DIAGNOSTICS.md, REPORT.md", C[9])
    if B.prob is not None:
        pred = remove_small((B.prob >= B.thr["model"]["threshold"]) & valid, cfg.eval.min_component_px)
        ax.imshow(P.confusion_rgba(pred, sc["mask"] > 0, valid, P.rgb_image(sc["features"], names, valid)), extent=ext)
        P.confusion_legend(ax, "lower left")
    P.geo_axes(ax, ext)
    ax.tick_params(labelsize=6)

    # arrows
    ym1, ym2 = y1 + H * 0.55, y2 + H * 0.55
    for i in range(4):
        _arrow(fig, (xs[i] + W + 0.002, ym1), (xs[i + 1] - 0.002, ym1))
    _arrow(fig, (xs[4] + W / 2, y1 - 0.004), (xs[4] + W / 2, y2 + H + 0.004), "features + labels")
    for i in range(4, 0, -1):
        _arrow(fig, (xs[i] - 0.002, ym2), (xs[i - 1] + W + 0.002, ym2))
    _arrow(fig, (xs[1] + W / 2, y1 - 0.004), (xs[1] + W / 2, y2 + H + 0.004), "inject L·exp(k·ΔX)", dashed=True)
    src = "synthetic EMIT-format data" if B.source == "synthetic" else f"run {cfg.run_name}"
    fig.suptitle(f"ch4hsi — methane plume detection workflow  ·  every panel is real pipeline output ({src})",
                 fontsize=12.5, weight="semibold", color=P.INK, x=0.02, ha="left", y=0.985)
    P.watermark(fig, f"scene {meta['scene_id']}" + ("  ·  synthetic data" if B.source == "synthetic" else ""))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pod_axes(ax, B, small=False):
    pod = B.pod
    if not pod:
        ax.text(0.5, 0.5, "run `ch4hsi mdl`", ha="center", va="center", color=P.MUTED, transform=ax.transAxes)
        ax.set_axis_off()
        return
    df, fit = pod["df"], pod["fit"]
    qq = np.logspace(np.log10(df.q_kgph.min()), np.log10(df.q_kgph.max()), 200)
    for m in ("model", "mf"):
        d = df[~df[f"pre_{m}"].astype(bool)] if f"pre_{m}" in df else df
        g = d.groupby("q_kgph")[f"det_{m}"].agg(["sum", "count"])
        lo, hi = P.wilson(g["sum"], g["count"])
        r = g["sum"] / g["count"]
        col = P.METHOD_STYLE[m]["color"]
        ax.errorbar(g.index, r, yerr=[r - lo, hi - r], fmt="o", color=col, ms=4 if small else 5, capsize=0, lw=1)
        f = (fit.get(m) or {}).get("flux", {})
        name = B.label.split(" (")[0] if m == "model" else "Matched filter"
        if f.get("fit"):
            a, b = f["fit"]["a"], f["fit"]["b"]
            ax.plot(qq, 1 / (1 + np.exp(-(np.log10(qq) - a) / b)), color=col, lw=1.6,
                    label=f"{name}: MDL50 ≈ {f['mdl50']:.0f} kg/h")
        else:
            ax.plot([], [], color=col, label=name)
    ax.axhline(0.5, color=P.MUTED, lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.05)
    fs = 7 if small else 9
    ax.set_xlabel(f"source rate Q (kg/h), U = {fit.get('wind_ms', B.cfg.mdl.wind_ms)} m/s", fontsize=fs)
    ax.set_ylabel("probability of detection", fontsize=fs)
    ax.tick_params(labelsize=6.5 if small else 8)
    ax.legend(fontsize=6.5 if small else 8, loc="lower right")


def fig_scene_maps(B, path):
    plt = P.apply_style()
    rows = [(B.scene, B.prob, "plume scene")] + ([(B.neg_scene, B.neg_prob, "plume-free scene")] if B.neg_scene else [])
    fig, axs = plt.subplots(len(rows), 5, figsize=(18, 3.9 * len(rows)), squeeze=False)
    E = B.cfg.eval
    for r, (sc, prob, tag) in enumerate(rows):
        meta, valid, gt = sc["meta"], sc["valid"], sc["mask"] > 0
        ext = P.extent(meta)
        rgb = P.rgb_image(sc["features"], meta["channel_names"], valid)
        ax = axs[r]
        ax[0].imshow(rgb, extent=ext)
        ax[0].set_title(f"{tag}: true colour")
        vmax = max(1000.0, float(np.nanpercentile(np.where(valid, sc["mf"], np.nan), 99.8)))
        im = ax[1].imshow(np.where(valid, sc["mf"], np.nan), extent=ext, cmap=P.ppmm_cmap(), vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.02, label="CH₄ enhancement (ppm·m)")
        if gt.any():
            ax[1].contour(np.flipud(gt.astype(float)), levels=[0.5], colors=P.INK, linewidths=0.7, extent=ext,
                          origin="lower")
        ax[1].set_title("Matched filter (outline = label)")
        if prob is not None:
            im = ax[2].imshow(np.where(valid, prob, np.nan), extent=ext, cmap=P.prob_cmap(), vmin=0, vmax=1)
            fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.02, label="probability")
        ax[2].set_title(f"{B.label.split(' (')[0]} probability")
        for j, (m, s) in enumerate((("model", prob), ("mf", mf_score(sc["mf"], valid, E.mf_smooth_sigma_px)))):
            a = ax[3 + j]
            if s is None:
                continue
            pred = remove_small((s >= B.thr[m]["threshold"]) & valid, E.min_component_px)
            a.imshow(P.confusion_rgba(pred, gt, valid, rgb), extent=ext)
            tp, fp, fn = int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())
            nm = B.label.split(" (")[0] if m == "model" else "Matched filter"
            thr = B.thr[m]["threshold"]
            ts = f"p ≥ {thr:.2f}" if m == "model" else f"≥ {thr:.0f} ppm·m"
            a.set_title(f"{nm} ({ts}): " + (f"IoU {tp / max(tp + fp + fn, 1):.2f}" if gt.any() else f"{fp} FP px"))
            P.confusion_legend(a)
        for a in ax:
            P.geo_axes(a, ext)
    src = "synthetic EMIT-format scenes" if B.source == "synthetic" else f"test scenes, run {B.cfg.run_name}"
    fig.suptitle(f"Detection maps on the scene grid ({src}); thresholds frozen on validation", x=0.01, ha="left",
                 fontsize=11, weight="semibold")
    fig.tight_layout()
    P.watermark(fig, "synthetic data" if B.source == "synthetic" else "")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def fig_physics(B, path):
    S = B.sensor
    if S is None:
        log.warning("physics_check.png needs sensor-geometry radiance; skipped")
        return False
    plt = P.apply_style()
    cfg = B.cfg
    from ch4hsi.physics.matched_filter import columnwise_mf
    mf_s, sig, mu = columnwise_mf(S["win"], S["k"], None, cfg.physics.column_group, cfg.physics.shrinkage,
                                  cfg.physics.two_pass, cfg.physics.outlier_sigma, cfg.physics.albedo_correction)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].plot(S["wl"], S["k"] * 1e5, color=P.C_MF, marker="o", ms=3)
    axs[0].set(xlabel="wavelength (nm)", ylabel="k (10⁻⁵ per ppm·m)", title="CH₄ unit absorption in the MF window")
    truth = S.get("truth")
    plume = truth > 500 if truth is not None else mf_s > 3 * np.nanmedian(sig)
    if truth is not None:
        plume = truth > max(500.0, float(np.percentile(truth[truth > 100], 60))) if (truth > 100).any() else plume
    if plume.sum() > 5:
        rr, cc = np.nonzero(plume)
        ratio = S["win"][rr, cc] / mu[cc]                       # L / column background mean
        cont = np.abs(S["k"]) < 0.1 * np.abs(S["k"]).max()       # bands with ~no CH4 absorption
        A = np.stack([np.ones(cont.sum()), S["wl"][cont] - S["wl"].mean()], 1)
        coef, *_ = np.linalg.lstsq(A, ratio[:, cont].T, rcond=None)   # per-pixel linear continuum
        base = (np.stack([np.ones_like(S["wl"]), S["wl"] - S["wl"].mean()], 1) @ coef).T
        ratio = ratio / base                                     # remove the pixel's albedo / spectral slope
        dxp = truth[rr, cc] if truth is not None else mf_s[rr, cc]
        model_bl = np.exp(np.outer(dxp, S["k"]))
        q1, q3 = np.percentile(ratio, [25, 75], axis=0)
        axs[1].fill_between(S["wl"], q1, q3, color=P.C_MODEL, alpha=0.18, lw=0, label="plume pixels, IQR")
        axs[1].plot(S["wl"], np.median(ratio, 0), color=P.C_MODEL, label="plume pixels, median (continuum-normalised)")
        axs[1].plot(S["wl"], np.median(model_bl, 0), color=P.C_MF, ls="--",
                    label=f"Beer–Lambert exp(k·ΔX), median ΔX = {np.median(dxp):.0f} ppm·m"
                          + (" (true)" if truth is not None else " (MF)"))
        axs[1].axhline(1, color=P.MUTED, lw=0.6)
        axs[1].set(xlabel="wavelength (nm)", ylabel="radiance / column background mean",
                   title=f"In-plume absorption signature ({plume.sum()} px)")
        axs[1].legend(fontsize=7, loc="lower left")
    ok = np.isfinite(mf_s)
    if truth is not None:
        t, m = truth[ok].ravel(), mf_s[ok].ravel()
        sel = t > 50
        bgm = m[t < 1]
        axs[2].hexbin(t[sel], m[sel], gridsize=45, cmap=P.cmap(P.BLUE_RAMP[1:]), mincnt=1, bins="log")
        lim = [0, float(t[sel].max())]
        axs[2].plot(lim, lim, color=P.C_MF, lw=1.2, label="1:1")
        slope = float(np.polyfit(t[sel], m[sel], 1)[0])
        axs[2].set(xlabel="true injected enhancement (ppm·m)", ylabel="matched filter (ppm·m)",
                   title=f"MF recovery (slope {slope:.2f}; background σ {np.std(bgm):.0f} ppm·m)")
        axs[2].legend(loc="upper left")
    else:
        v = mf_s[ok]
        s0 = float(np.nanmedian(sig))
        bins = np.linspace(-4 * s0, 6 * s0, 80)
        axs[2].hist(v, bins=bins, density=True, color=P.C_MODEL, alpha=0.6, label="MF, all pixels")
        xx = np.linspace(bins[0], bins[-1], 300)
        axs[2].plot(xx, np.exp(-0.5 * (xx / s0) ** 2) / (s0 * np.sqrt(2 * np.pi)), color=P.C_MF,
                    label=f"N(0, σ = {s0:.0f})")
        axs[2].set_yscale("log")
        axs[2].set(xlabel="MF (ppm·m)", ylabel="density", title="MF background vs noise model")
        axs[2].legend()
    fig.tight_layout()
    P.watermark(fig, "synthetic data" if B.source == "synthetic" else "")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def fig_injection(B, path, qs=(150, 300, 600, 1200, 2400)):
    S = B.inj_cube
    plt = P.apply_style()
    cfg = B.cfg
    n = len(qs) if S is not None else 0
    fig = plt.figure(figsize=(2.7 * n + 5.8, 3.7))
    gs = fig.add_gridspec(1, n + 3, width_ratios=[1] * n + [0.06, 0.55, 1.9], wspace=0.08, left=0.02, right=0.98)
    if S is not None:
        from ch4hsi.io.emit import valid_mask
        valid_s = valid_mask(S["win"])
        H, Wd = valid_s.shape
        # put the source where the background MF is quiet (so the panels show the injected plume, not clutter)
        from scipy.ndimage import maximum_filter
        from ch4hsi.physics.matched_filter import columnwise_mf
        mf0, _, _ = columnwise_mf(S["win"], S["k"], valid_s, cfg.physics.column_group, cfg.physics.shrinkage,
                                  cfg.physics.two_pass, cfg.physics.outlier_sigma, cfg.physics.albedo_correction)
        clutter = maximum_filter(np.nan_to_num(mf0, nan=1e4), size=41)
        m = 45
        sub = clutter[m:H - m, m:Wd - m - 30]
        r, c = np.unravel_index(np.argmin(sub), sub.shape)
        src = (int(r + m), int(c + m))
        th = 0.35
        rng_seed = 5
        vmax = 1500
        im = None
        for i, q in enumerate(qs):
            E = gaussian_plume_ppmm(valid_s.shape, src, q, cfg.mdl.wind_ms, th, max_age_s=cfg.mdl.max_plume_age_s,
                                    turbulence=cfg.mdl.turbulence, rng=np.random.default_rng(rng_seed))
            prod, _ = _chain(cfg, S, inject(S["win"], E, S["k"]), {"inj": E})
            v = prod["valid"]
            det = remove_small((mf_score(prod["mf"], v, cfg.eval.mf_smooth_sigma_px) >= B.thr["mf"]["threshold"]) & v,
                               cfg.eval.min_component_px)
            inj = np.nan_to_num(prod["inj"])
            rr, cc = np.nonzero(inj > 100) if (inj > 100).any() else np.nonzero(inj >= inj.max())
            r0, c0 = int(np.median(rr)), int(np.median(cc))
            sl = (slice(max(0, r0 - 45), r0 + 45), slice(max(0, c0 - 45), c0 + 45))
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(np.where(v, prod["mf"], np.nan)[sl], cmap=P.ppmm_cmap(), vmin=0, vmax=vmax)
            ax.contour((inj >= cfg.mdl.footprint_min_ppmm)[sl].astype(float), levels=[0.5], colors=P.C_MODEL,
                       linewidths=0.9)
            fpm = inj >= cfg.mdl.footprint_min_ppmm
            if i == 0:
                prod0, _ = _chain(cfg, S, S["win"])
                det0 = remove_small((mf_score(prod0["mf"], prod0["valid"], cfg.eval.mf_smooth_sigma_px)
                                     >= B.thr["mf"]["threshold"]) & prod0["valid"], cfg.eval.min_component_px)
            from scipy.ndimage import binary_dilation
            pre = bool(det0[binary_dilation(fpm)].any())
            hit = bool(det[binary_dilation(fpm)].any())
            verdict = "site flagged before injection" if pre else ("MF detects" if hit else "MF misses")
            ax.set_title(f"Q = {q} kg/h  ·  {verdict}", fontsize=9, color=P.INK if hit and not pre else P.INK2)
            ax.set_xticks([]), ax.set_yticks([])
            ax.grid(False)
        cax = fig.add_subplot(gs[0, n])
        fig.colorbar(im, cax=cax, label="MF (ppm·m)")
    ax = fig.add_subplot(gs[0, n + 2])
    _pod_axes(ax, B)
    ax.set_title("Probability of detection (synthetic injection)")
    fig.suptitle(f"Beer–Lambert plume injection into radiance, U = {cfg.mdl.wind_ms:g} m/s "
                 "(blue outline = injected footprint ≥ 100 ppm·m)", x=0.01, ha="left", fontsize=11, weight="semibold")
    P.watermark(fig, "synthetic data" if B.source == "synthetic" else "")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ====================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", default=None)
    ap.add_argument("--set", "-s", action="append", default=[])
    ap.add_argument("--run", default=None, help="run name with evaluate outputs (real data)")
    ap.add_argument("--source", choices=["auto", "real", "synthetic"], default="auto")
    ap.add_argument("--model", choices=["auto", "unet", "pixel_lr"], default="auto",
                    help="synthetic mode: model to train (auto = U-Net if torch is installed)")
    ap.add_argument("--synthetic-root", default=str(REPO / "data" / "readme_synthetic"))
    ap.add_argument("--out", default=str(REPO / "docs" / "figures"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = a.source
    if src == "auto":
        cfg = load_config(a.config, a.set)
        ok = a.run and (Path(cfg.paths.runs) / a.run / "thresholds.json").exists()
        src = "real" if ok else "synthetic"
        log.info("data source: %s", src)
    if src == "real":
        if not a.run:
            raise SystemExit("--source real needs --run <run_name>")
        B = real_bundle(load_config(a.config, a.set), a.run)
    else:
        B = synthetic_bundle(Path(a.synthetic_root), a.model, a.seed)
    fig_workflow(B, out / "workflow.png")
    fig_scene_maps(B, out / "scene_maps.png")
    fig_physics(B, out / "physics_check.png")
    fig_injection(B, out / "mdl_injection.png")
    for name in ("02_threshold_sweep", "05_reliability", "08_false_alarms"):   # a taste of `ch4hsi diagnose`
        f = B.out / "diagnostics" / f"{name}.png"
        if f.exists():
            shutil.copy(f, out / f"diag_{name}.png")
    m = load_json(B.out / "metrics_test.json")
    log.info("figures written to %s  (model: %s | test F1 model %.2f vs MF %.2f)", out, B.label, m["model"]["f1"],
             m["mf"]["f1"])
    log.info("run directory with evaluate + diagnostics outputs: %s", B.out)


if __name__ == "__main__":
    main()
