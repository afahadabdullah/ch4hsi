"""Stage 7: minimum detection limit (MDL) by synthetic plume injection.

For each plume-free *test* scene we inject Gaussian plumes of known source rate Q (kg/h, at a fixed
wind speed) into the L1B radiance (Beer-Lambert, in sensor geometry), re-run the complete chain
(column-wise MF -> features -> GLT ortho -> model), and record whether each injected plume is detected
with the thresholds frozen from validation. A logistic probability-of-detection (POD) curve in
log10(Q) gives MDL50 / MDL90 with bootstrap CIs. Detection is also recorded against the peak
injected enhancement (ppm·m), and an analytic noise-based MDL is reported for comparison.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .config import Cfg, run_dir
from .evaluate import load_model, mf_score, predict_scene
from .features import band_selector, scene_products
from .io.emit import read_emit_l1b, valid_mask
from .metrics import pod_with_ci, remove_small
from .physics.plume_sim import gaussian_plume_ppmm, inject, random_sources
from .physics.target import load_lut, unit_absorption
from .physics.units import analytic_flux_mdl
from .utils import dump_json, get_logger, load_json, shard

log = get_logger(__name__)


def _detect(score, thr, valid, min_px):
    return remove_small((score >= thr) & valid, min_px)


def run(cfg: Cfg, task_id=None, num_tasks=None):
    import torch
    from .train import amp_dtype

    out = run_dir(cfg)
    M = cfg.mdl
    if task_id is None and "SLURM_ARRAY_TASK_ID" in os.environ:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    if task_id is not None and num_tasks is None:
        num_tasks = 1
    splits = load_json(Path(cfg.paths.splits) / "splits.json")["splits"]
    thr = load_json(out / "thresholds.json")
    norm = load_json(out / "norm.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adt = amp_dtype(cfg.train.amp)
    model = load_model(out, cfg, device)
    lut = load_lut(cfg.physics.lut)

    cand = []
    for sid in splits["test"]:
        meta = load_json(Path(cfg.paths.scenes) / sid / "meta.json")
        if meta["role"] == "neg" and Path(meta["l1b"]).exists():
            cand.append(meta)
    if not cand:  # fall back to plume scenes (injection sites avoid labelled plumes)
        cand = [load_json(Path(cfg.paths.scenes) / sid / "meta.json") for sid in splits["test"]]
        cand = [m for m in cand if Path(m["l1b"]).exists()]
    if not cand:
        raise RuntimeError("MDL needs raw L1B files of test scenes (preprocess.delete_raw must be false)")
    rng0 = np.random.default_rng(M.seed)
    rng0.shuffle(cand)
    cand = shard(cand[: M.n_scenes], task_id, num_tasks)
    methods = {"model": thr["model"]["threshold"], "mf": thr["mf"]["threshold"], "mf_fixed": thr["mf_fixed"]["threshold"]}
    sel = band_selector(cfg.physics, cfg.preprocess)
    recs, sig_rows = [], []
    for si, meta in enumerate(cand):
        sc = read_emit_l1b(meta["l1b"], sel)
        win = sel(sc.wavelengths, sc.fwhm, sc.good)["win"]
        k = unit_absorption(sc.wavelengths[win], sc.fwhm[win], lut=lut, spec=cfg.physics.lut)
        rad0, rgb = sc.radiance["win"], sc.radiance["rgb"]
        valid_s = valid_mask(rad0) & valid_mask(rgb)
        # existing labelled plumes (if any) are kept away from injection sites
        lab = np.load(Path(cfg.paths.scenes) / meta["scene_id"] / "mask.npy")

        def run_chain(rad, extra):
            prod = scene_products(rad, rgb, k, sc.glt_x, sc.glt_y, cfg.physics, cfg.preprocess, cfg.features,
                                  valid=valid_s, extra_sensor=extra)
            prob = predict_scene(model, prod["features"], prod["valid"], norm, cfg.eval.infer_tile,
                                 cfg.eval.infer_stride, device, adt)
            mfs = mf_score(prod["mf"], prod["valid"], cfg.eval.mf_smooth_sigma_px)
            dets = {m: _detect(prob if m == "model" else mfs, t, prod["valid"], cfg.eval.min_component_px)
                    for m, t in methods.items()}
            return prod, dets

        prod0, det0 = run_chain(rad0, None)
        sig_rows.append(dict(scene_id=meta["scene_id"], sigma_median=float(np.nanmedian(prod0["sigma"][prod0["valid"]])),
                             mf_bg_std=float(np.nanstd(prod0["mf"][prod0["valid"] & (lab == 0)]))))
        for rep in range(M.repeats):
            rng = np.random.default_rng([M.seed, si, rep, int(task_id or 0)])
            srcs = random_sources(valid_s, M.plumes_per_scene, M.min_separation_px, margin=30, rng=rng)
            thetas = rng.uniform(0, 2 * np.pi, len(srcs))
            for q in M.flux_kgph:
                maps = [gaussian_plume_ppmm(valid_s.shape, s, q, M.wind_ms, th, pixel_m=M.pixel_m,
                                            max_age_s=M.max_plume_age_s, turbulence=M.turbulence, rng=rng)
                        for s, th in zip(srcs, thetas)]
                total = np.sum(maps, axis=0)
                ident = np.where(total >= M.footprint_min_ppmm, np.argmax(maps, axis=0) + 1, 0).astype(np.float32)
                prod, dets = run_chain(inject(rad0, total, k), {"inj_ppmm": total, "inj_id": ident})
                inj_id = np.nan_to_num(prod["inj_id"], nan=0).astype(int)
                inj_ppmm = np.nan_to_num(prod["inj_ppmm"], nan=0)
                for j in range(1, len(srcs) + 1):
                    fp = inj_id == j
                    if fp.sum() == 0 or (lab[fp] > 0).any():
                        continue
                    fp_d = ndimage.binary_dilation(fp)   # 1-px tolerance for GLT resampling
                    rec = dict(scene_id=meta["scene_id"], rep=rep, q_kgph=q, wind_ms=M.wind_ms, plume=j,
                               peak_ppmm=float(inj_ppmm[fp].max()), footprint_px=int(fp.sum()))
                    for m in methods:
                        rec[f"det_{m}"] = bool(dets[m][fp_d].any())
                        rec[f"pre_{m}"] = bool(det0[m][fp_d].any())   # already "detected" before injection
                    recs.append(rec)
            log.info("scene %s rep %d done (%d records)", meta["scene_id"], rep, len(recs))
    tag = f"_{task_id:03d}" if task_id is not None else ""
    pd.DataFrame(recs).to_csv(out / f"mdl_records{tag}.csv", index=False)
    pd.DataFrame(sig_rows).to_csv(out / f"mdl_sigma{tag}.csv", index=False)
    if not num_tasks or num_tasks == 1:
        fit(cfg)


def fit(cfg: Cfg):
    out = run_dir(cfg)
    M = cfg.mdl
    df = pd.concat([pd.read_csv(p) for p in sorted(out.glob("mdl_records*.csv"))], ignore_index=True)
    sg = pd.concat([pd.read_csv(p) for p in sorted(out.glob("mdl_sigma*.csv"))], ignore_index=True)
    res = {"n_injections": int(len(df)), "n_scenes": int(df.scene_id.nunique()), "wind_ms": M.wind_ms}
    for m in ("model", "mf", "mf_fixed"):
        d = df[~df[f"pre_{m}"]]            # drop sites with pre-existing false alarms
        pod_q = d.groupby("q_kgph")[f"det_{m}"].mean().to_dict()
        r = dict(pod_by_flux=pod_q, pre_existing_rate=float(df[f"pre_{m}"].mean()))
        if d[f"det_{m}"].nunique() == 2:
            r["flux"] = pod_with_ci(d.q_kgph, d[f"det_{m}"], M.pod_levels, M.bootstrap, M.seed)
            r["peak_ppmm"] = pod_with_ci(d.peak_ppmm, d[f"det_{m}"], M.pod_levels, M.bootstrap, M.seed)
        res[m] = r
    sig = float(np.median(sg.sigma_median))
    res["noise"] = dict(sigma_median_ppmm=sig, mf_bg_std_median_ppmm=float(np.median(sg.mf_bg_std)),
                        analytic_mdl_kgph_3sigma_9px=analytic_flux_mdl(sig, 9, M.pixel_m, M.wind_ms))
    dump_json(res, out / "mdl.json")
    _plot(df, res, out / "figures")
    for m in ("model", "mf", "mf_fixed"):
        f = res[m].get("flux", {})
        log.info("%-8s MDL50 %s kg/h  MDL90 %s kg/h (U=%.1f m/s)", m, _fmt(f.get("mdl50")), _fmt(f.get("mdl90")), M.wind_ms)


def _fmt(v):
    return "n/a" if v is None else f"{v:.0f}"


def _plot(df, res, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    qq = np.logspace(np.log10(df.q_kgph.min()), np.log10(df.q_kgph.max()), 200)
    for m, lab, col in (("model", "U-Net", "#1f6feb"), ("mf", "Matched filter", "#d1242f")):
        d = df[~df[f"pre_{m}"]]
        g = d.groupby("q_kgph")[f"det_{m}"].mean()
        ax.plot(g.index, g.values, "o", color=col, ms=4)
        f = res[m].get("flux", {}).get("fit")
        if f:
            ax.plot(qq, 1 / (1 + np.exp(-(np.log10(qq) - f["a"]) / f["b"])), color=col,
                    label=f"{lab}: MDL50 {res[m]['flux']['mdl50']:.0f}, MDL90 {res[m]['flux']['mdl90']:.0f} kg/h")
    ax.set_xscale("log")
    ax.set(xlabel=f"Injected source rate (kg/h) at U = {res['wind_ms']} m/s", ylabel="Probability of detection",
           ylim=(-0.02, 1.02))
    ax.axhline(0.5, color="0.6", lw=0.6, ls="--")
    ax.axhline(0.9, color="0.6", lw=0.6, ls=":")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(fig_dir / "pod_curve.png", dpi=160)
    plt.close(fig)
