"""Stage 6: full-scene evaluation of the matched-filter baseline and the trained model.

Thresholds for both methods are chosen on the *validation* scenes (max pixel F1) and then frozen
for the test scenes. Reported on test:
  pixel     precision / recall / F1 / IoU (+ average precision from the PR curve)
  plume     recall (GT plume complexes hit) / precision (predicted components touching a plume)
  scene     AUROC of "scene contains a plume" using the max (smoothed) score
  per-plume detection vs plume size & MF strength (csv, for recall-vs-strength plots)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from .config import Cfg, run_dir
from .metrics import HistPR, auroc, object_counts, pixel_counts, prf, remove_small
from .utils import dump_json, get_logger, load_json

log = get_logger(__name__)


def load_model(out: Path, cfg: Cfg, device):
    import torch

    from .config import load_config
    from .models.unet import build_model
    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    # rebuild with the architecture the checkpoint was trained with
    tcfg = load_config(out / "config.yaml").train if (out / "config.yaml").exists() else cfg.train
    model = build_model(tcfg, ck["in_ch"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval()


def predict_scene(model, feats, valid, norm, tile, stride, device, adt=None, batch=16):
    """Sliding-window inference with a separable triangular blending window. Returns prob (H, W)."""
    import torch
    C, H, W = feats.shape
    mean = np.asarray(norm["mean"], np.float32)[:, None, None]
    std = np.asarray(norm["std"], np.float32)[:, None, None]
    ph, pw = max(0, tile - H), max(0, tile - W)
    nr = int(np.ceil((H + ph - tile) / stride)) + 1
    nc = int(np.ceil((W + pw - tile) / stride)) + 1
    Hp, Wp = (nr - 1) * stride + tile, (nc - 1) * stride + tile
    x = np.zeros((C, Hp, Wp), np.float32)
    x[:, :H, :W] = np.clip((np.asarray(feats, np.float32) - mean) / std, -10, 10)
    x[:, :H, :W][:, ~valid] = 0
    w1 = 1 - np.abs(np.linspace(-1, 1, tile)) * 0.9
    win = np.outer(w1, w1).astype(np.float32)
    acc = np.zeros((Hp, Wp), np.float32)
    wsum = np.zeros((Hp, Wp), np.float32)
    coords = [(r * stride, c * stride) for r in range(nr) for c in range(nc)]
    for i in range(0, len(coords), batch):
        cb = coords[i:i + batch]
        xb = torch.from_numpy(np.stack([x[:, r:r + tile, c:c + tile] for r, c in cb])).to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=adt, enabled=adt is not None):
            pb = torch.sigmoid(model(xb).float())[:, 0].cpu().numpy()
        for (r, c), p in zip(cb, pb):
            acc[r:r + tile, c:c + tile] += p * win
            wsum[r:r + tile, c:c + tile] += win
    prob = (acc / np.maximum(wsum, 1e-6))[:H, :W]
    prob[~valid] = 0
    return prob


def mf_score(mf, valid, sigma_px):
    """Normalised-convolution smoothing of the MF map (ppm·m); invalid pixels do not bleed in."""
    m = np.where(valid & np.isfinite(mf), mf, 0).astype(np.float32)
    if sigma_px <= 0:
        return m
    num = ndimage.gaussian_filter(m, sigma_px)
    den = ndimage.gaussian_filter(valid.astype(np.float32), sigma_px)
    s = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0)
    s[~valid] = 0
    return s


def load_scene(d):
    d = Path(d)
    return dict(meta=load_json(d / "meta.json"), features=np.load(d / "features.npy", mmap_mode="r"),
                valid=np.load(d / "valid.npy"), mask=np.load(d / "mask.npy"), ids=np.load(d / "plume_id.npy"),
                mf=np.load(d / "mf.npy"), sigma=np.load(d / "sigma.npy").astype(np.float32))


def scene_scores(cfg: Cfg, model, norm, sc, device, adt, cache_dir: Path | None):
    sid = sc["meta"]["scene_id"]
    cpath = cache_dir / f"{sid}.npy" if cache_dir else None
    if cpath is not None and cpath.exists():
        prob = np.load(cpath).astype(np.float32)
    else:
        prob = predict_scene(model, sc["features"], sc["valid"], norm, cfg.eval.infer_tile, cfg.eval.infer_stride,
                             device, adt)
        if cpath is not None:
            cpath.parent.mkdir(parents=True, exist_ok=True)
            np.save(cpath, prob.astype(np.float16))
    mfs = mf_score(sc["mf"], sc["valid"], cfg.eval.mf_smooth_sigma_px)
    return {"model": prob, "mf": mfs}


def run(cfg: Cfg):
    import torch

    from .train import amp_dtype
    out = run_dir(cfg)
    splits = load_json(Path(cfg.paths.splits) / "splits.json")["splits"]
    norm = load_json(out / "norm.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adt = amp_dtype(cfg.train.amp)
    model = load_model(out, cfg, device)
    E = cfg.eval
    cache = out / "preds" if E.save_predictions else None
    scene_dir = Path(cfg.paths.scenes)
    ranges = {"model": (0.0, 1.0), "mf": (0.0, float(E.mf_max_ppmm))}

    # ---- 1. thresholds on validation scenes
    hv = {m: HistPR(*ranges[m], E.n_bins) for m in ranges}
    for sid in splits["val"]:
        sc = load_scene(scene_dir / sid)
        s = scene_scores(cfg, model, norm, sc, device, adt, cache)
        for m in ranges:
            hv[m].update(s[m], sc["mask"] > 0, sc["valid"])
    thr = {m: hv[m].best_f1() for m in ranges}
    if hv["model"].pos.sum() == 0:
        log.warning("validation split has no plume pixels; falling back to default thresholds")
        thr = {"model": dict(threshold=0.5), "mf": dict(threshold=float(E.mf_fixed_threshold_ppmm))}
    thr["mf_fixed"] = dict(threshold=float(E.mf_fixed_threshold_ppmm))
    dump_json(thr, out / "thresholds.json")
    log.info("val-selected thresholds: model %.3f (F1 %.3f) | MF %.0f ppm·m (F1 %.3f)", thr["model"]["threshold"],
             thr["model"].get("f1", float("nan")), thr["mf"]["threshold"], thr["mf"].get("f1", float("nan")))

    # ---- 2. test
    methods = {"model": ("model", thr["model"]["threshold"]), "mf": ("mf", thr["mf"]["threshold"]),
               "mf_fixed": ("mf", thr["mf_fixed"]["threshold"])}
    ht = {m: HistPR(*ranges[m], E.n_bins) for m in ranges}
    agg = {m: dict(tp=0, fp=0, fn=0, gt_total=0, gt_detected=0, pred_total=0, pred_tp=0, pred_total_neg_scenes=0,
                   km2_neg=0.0) for m in methods}
    scene_rows, plume_rows = [], []
    examples = []
    for sid in splits["test"]:
        sc = load_scene(scene_dir / sid)
        s = scene_scores(cfg, model, norm, sc, device, adt, cache)
        valid, gt, ids = sc["valid"], sc["mask"] > 0, sc["ids"]
        for m in ranges:
            ht[m].update(s[m], gt, valid)
        row = dict(scene_id=sid, role=sc["meta"]["role"], n_pos_px=int(gt.sum()))
        gt_px = {}
        lab_ids = np.unique(ids[ids > 0])
        for pid in lab_ids:
            sel = ids == pid
            gt_px[int(pid)] = dict(n_px=int(sel.sum()), mf_max=float(np.nanmax(sc["mf"][sel])),
                                   mf_sum=float(np.nansum(np.clip(sc["mf"][sel], 0, None))))
        for m, (src, t) in methods.items():
            pred = remove_small((s[src] >= t) & valid, E.min_component_px)
            tp, fp, fn = pixel_counts(pred, gt, valid)
            oc = object_counts(pred, ids, valid)
            a = agg[m]
            a["tp"] += tp; a["fp"] += fp; a["fn"] += fn
            for k in ("gt_total", "gt_detected", "pred_total", "pred_tp"):
                a[k] += oc[k]
            if sc["meta"]["role"] == "neg":
                a["pred_total_neg_scenes"] += oc["pred_total"]
                a["km2_neg"] += valid.sum() * 0.0036       # 60 m pixels
            row[f"{m}_max"] = float(np.max(s[src][valid])) if valid.any() else 0.0
            row[f"{m}_iou"] = prf(tp, fp, fn)["iou"]
            for pid, det in oc["per_gt"].items():
                gt_px[int(pid)][f"det_{m}"] = bool(det)
        for pid, d in gt_px.items():
            plume_rows.append(dict(scene_id=sid, local_id=pid, **d))
        scene_rows.append(row)
        if len(examples) < E.n_example_figures and gt.sum() > 0:
            examples.append((sid, sc, s))
            if len(examples) == E.n_example_figures:
                _plot_examples(examples, thr, out / "figures")

    res = {"thresholds": thr, "n_test_scenes": len(scene_rows)}
    sdf = pd.DataFrame(scene_rows)
    for m, (src, _) in methods.items():
        a = agg[m]
        r = prf(a["tp"], a["fp"], a["fn"])
        r.update(plume_recall=a["gt_detected"] / max(a["gt_total"], 1),
                 plume_precision=a["pred_tp"] / max(a["pred_total"], 1),
                 n_gt_plumes=a["gt_total"], n_pred_components=a["pred_total"],
                 false_alarms_per_1000km2_neg=1000 * a["pred_total_neg_scenes"] / max(a["km2_neg"], 1e-9),
                 ap=ht[src].average_precision(),
                 scene_auroc=auroc(sdf[sdf.role == "pos"][f"{m}_max"], sdf[sdf.role == "neg"][f"{m}_max"]),
                 mean_scene_iou=float(np.nanmean(sdf[sdf.role == "pos"][f"{m}_iou"])) if (sdf.role == "pos").any() else None)
        res[m] = r
    dump_json(res, out / "metrics_test.json")
    sdf.to_csv(out / "per_scene_test.csv", index=False)
    pdf = pd.DataFrame(plume_rows)
    pdf.to_csv(out / "per_plume_test.csv", index=False)
    _plot_pr(ht, thr, out / "figures")
    if len(pdf):
        _plot_recall_vs_strength(pdf, out / "figures")
    if 0 < len(examples) < E.n_example_figures:
        _plot_examples(examples, thr, out / "figures")
    for m in methods:
        r = res[m]
        log.info("%-8s P %.3f R %.3f F1 %.3f IoU %.3f AP %.3f | plume R %.3f P %.3f | scene AUROC %.3f", m,
                 r["precision"], r["recall"], r["f1"], r["iou"], r["ap"], r["plume_recall"], r["plume_precision"],
                 r["scene_auroc"])


# ------------------------------------------------------------------ figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _plot_pr(ht, thr, fig_dir):
    plt = _plt()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for m, lab, col in (("model", "U-Net", "#1f6feb"), ("mf", "Matched filter", "#d1242f")):
        p, r, t = ht[m].curve()
        ax.plot(r, p, color=col, label=f"{lab} (AP {ht[m].average_precision():.2f})")
    ax.set(xlabel="Recall (pixel)", ylabel="Precision (pixel)", xlim=(0, 1), ylim=(0, 1.02), title="Test PR curve")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "pr_curve.png", dpi=160)
    plt.close(fig)


def _plot_recall_vs_strength(pdf, fig_dir):
    plt = _plt()
    fig_dir.mkdir(parents=True, exist_ok=True)
    bins = np.quantile(pdf.mf_sum, np.linspace(0, 1, 7))
    bins = np.unique(bins)
    if len(bins) < 3:
        return
    cat = pd.cut(pdf.mf_sum, bins, include_lowest=True)
    fig, ax = plt.subplots(figsize=(5, 3.6))
    x = np.arange(len(cat.cat.categories))
    for m, lab, col, off in (("det_model", "U-Net", "#1f6feb", -0.2), ("det_mf", "Matched filter", "#d1242f", 0.2)):
        if m in pdf:
            ax.bar(x + off, pdf.groupby(cat, observed=False)[m].mean().to_numpy(), width=0.4, color=col, label=lab)
    ax.set_xticks(x, [f"{int(b.left/1e3)}–{int(b.right/1e3)}k" for b in cat.cat.categories], rotation=30)
    ax.set(xlabel="Plume integrated MF (ppm·m·px)", ylabel="Plume recall", ylim=(0, 1.05))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "recall_vs_strength.png", dpi=160)
    plt.close(fig)


def _plot_examples(examples, thr, fig_dir):
    plt = _plt()
    fig_dir.mkdir(parents=True, exist_ok=True)
    for sid, sc, s in examples:
        ids = sc["ids"]
        rr, cc = np.nonzero(ids > 0)
        r0, c0 = int(np.median(rr)), int(np.median(cc))
        h = 96
        sl = (slice(max(0, r0 - h), r0 + h), slice(max(0, c0 - h), c0 + h))
        names = sc["meta"]["channel_names"]
        fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
        if "rgb_r" in names:
            i = [names.index(n) for n in ("rgb_r", "rgb_g", "rgb_b")]
            rgb = np.clip(np.moveaxis(np.asarray(sc["features"][i], np.float32), 0, -1)[sl] / 1.0, 0, 1)
            axs[0].imshow(rgb)
        axs[0].set_title("RGB")
        im = axs[1].imshow(np.where(sc["valid"], sc["mf"], np.nan)[sl], vmin=-500, vmax=2000, cmap="inferno")
        fig.colorbar(im, ax=axs[1], fraction=0.046, label="ppm·m")
        axs[1].set_title("Matched filter")
        axs[2].imshow(sc["mask"][sl], cmap="gray", vmin=0, vmax=1)
        axs[2].set_title("Label (plume complex)")
        axs[3].imshow(s["model"][sl], vmin=0, vmax=1, cmap="viridis")
        axs[3].contour(sc["mask"][sl], levels=[0.5], colors="w", linewidths=0.6)
        axs[3].set_title(f"U-Net prob (thr {thr['model']['threshold']:.2f})")
        for a in axs:
            a.set_xticks([]), a.set_yticks([])
        fig.suptitle(sid, fontsize=9)
        fig.tight_layout()
        fig.savefig(fig_dir / f"example_{sid}.png", dpi=130)
        plt.close(fig)
