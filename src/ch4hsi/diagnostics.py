"""Stage 6b: diagnostic plotting suite for a finished run (`ch4hsi diagnose`).

Reads what `train`, `evaluate` and `mdl` wrote into runs/<run_name>/ and produces
runs/<run_name>/diagnostics/ with numbered figures, error maps and DIAGNOSTICS.md:

  01 training curves          loss, val AP / F1 / P / R, learning rate (history.csv)
  02 threshold sweep          P, R, F1, IoU vs threshold on test; val-frozen operating point marked
  03 ROC                      pixel ROC (log FPR) + scene-level ROC ("does this scene contain a plume?")
  04 score distributions      plume vs background pixel scores for model and MF
  05 reliability              calibration of the model probability (ECE, Brier), probability histogram
  06 per-scene IoU            model vs MF IoU on each plume scene, IoU vs plume pixels
  07 plume detection          plume recall vs plume area and vs integrated MF (Wilson 90 % CIs)
  08 false alarms             FP rate vs albedo, per-scene FP rate vs noise, FP component sizes
  09 MF vs model              joint density of MF score and model probability (plume / background)
  10 noise                    column noise-equivalent sigma and MF background std per scene/split
  11 MDL                      POD vs source rate and vs peak ppm·m, per-scene spread (if `mdl` ran)
  12 split map                scene centres by split (geo-block leakage check)
  maps/                       georeferenced RGB | MF | model | TP/FP/FN for the best / worst plume
                              scenes and the plume-free scenes with the most false alarms

Everything is optional-input tolerant: a figure whose inputs are missing is skipped and noted.
Model probabilities come from runs/<run>/preds/ (written by `evaluate`), or are recomputed with
the checkpoint if torch is available; otherwise model panels are omitted (MF-only maps).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

from . import plotting as P
from .config import Cfg, run_dir
from .evaluate import load_scene, mf_score
from .metrics import EIGHT, HistPR, auroc, remove_small
from .utils import dump_json, get_logger, load_json

log = get_logger(__name__)


# ================================================================== entry points
def run(cfg: Cfg):
    out = run_dir(cfg)
    diagnose(cfg, out, score_fn=_default_score_fn(cfg, out))


def _default_score_fn(cfg: Cfg, out: Path):
    """Cached predictions from `evaluate`; fall back to the checkpoint when torch is available."""
    cache = out / "preds"
    state = {}

    def fn(sc):
        sid = sc["meta"]["scene_id"]
        p = cache / f"{sid}.npy"
        if p.exists():
            return np.load(p).astype(np.float32)
        if "model" not in state:
            state["model"] = None
            try:
                import torch

                from .evaluate import load_model
                from .train import amp_dtype
                if (out / "best.pt").exists():
                    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    state.update(model=load_model(out, cfg, dev), dev=dev, adt=amp_dtype(cfg.train.amp),
                                 norm=load_json(out / "norm.json"))
            except ImportError:
                log.warning("no cached predictions and no torch: model panels will be skipped")
        if state["model"] is None:
            return None
        from .evaluate import predict_scene
        return predict_scene(state["model"], sc["features"], sc["valid"], state["norm"], cfg.eval.infer_tile,
                             cfg.eval.infer_stride, state["dev"], state["adt"])

    return fn


def diagnose(cfg: Cfg, out: Path, score_fn=None, watermark: str = "") -> dict:
    """Build the full suite for run directory `out`. score_fn(scene) -> prob map or None."""
    out = Path(out)
    fig_dir = out / "diagnostics"
    (fig_dir / "maps").mkdir(parents=True, exist_ok=True)
    P.apply_style()
    D = cfg.get("diagnose", {}) or {}
    ctx = _Ctx(cfg, out, fig_dir, score_fn, watermark, D)
    steps = [("01_training_curves", ctx.training_curves), ("02_threshold_sweep", ctx.threshold_sweep),
             ("03_roc", ctx.roc), ("04_score_distributions", ctx.score_distributions),
             ("05_reliability", ctx.reliability), ("06_per_scene_iou", ctx.per_scene_iou),
             ("07_plume_detection", ctx.plume_detection), ("_scene_pass", ctx.scene_pass),
             ("08_false_alarms", ctx.false_alarms), ("09_mf_vs_model", ctx.mf_vs_model),
             ("10_noise", ctx.noise), ("11_mdl", ctx.mdl), ("12_split_map", ctx.split_map),
             ("maps", ctx.error_maps)]
    for name, fn in steps:
        try:
            msg = fn()
        except Exception as e:  # one broken figure must not kill the suite
            log.exception("diagnostic %s failed: %s", name, e)
            msg = f"failed: {e}"
        if not name.startswith("_"):
            ctx.status[name] = msg or "ok"
    dump_json(ctx.summary, fig_dir / "diagnostics_summary.json")
    ctx.write_index()
    log.info("diagnostics written to %s", fig_dir)
    return ctx.summary


# ================================================================== implementation
class _Ctx:
    def __init__(self, cfg, out, fig_dir, score_fn, wm, D):
        self.cfg, self.out, self.fig_dir, self.score_fn, self.wm = cfg, out, fig_dir, score_fn, wm
        self.n_maps = int(D.get("n_error_maps", 3))
        self.max_px = int(D.get("max_scatter_px", 200_000))
        self.status, self.summary, self.figs = {}, {}, []
        self.metrics = _load(out / "metrics_test.json")
        self.thr = _load(out / "thresholds.json")
        self.model_label = (self.metrics or {}).get("model_label", "U-Net")
        self.scene_df = _csv(out / "per_scene_test.csv")
        self.plume_df = _csv(out / "per_plume_test.csv")
        self.hist = dict(np.load(out / "pr_hist.npz")) if (out / "pr_hist.npz").exists() else None
        sp = Path(cfg.paths.splits) / "splits.json"
        self.splits = load_json(sp)["splits"] if sp.exists() else None
        self.pass_ = None

    # ---------------------------------------------------------- helpers
    def lab(self, m):
        return P.label(m, self.model_label)

    def save(self, fig, name, caption):
        P.watermark(fig, self.wm)
        path = self.fig_dir / f"{name}.png"
        fig.savefig(path)
        P.plt().close(fig)
        self.figs.append((name, caption))
        return "ok"

    def hpr(self, split, m):
        if self.hist is None or f"{split}_{m}_pos" not in self.hist:
            return None
        h = HistPR(0, 1, 1)
        h.edges, h.pos, h.neg = (self.hist[f"{split}_{m}_{k}"] for k in ("edges", "pos", "neg"))
        return h

    def thr_of(self, m):
        return float(self.thr[m]["threshold"]) if self.thr and m in self.thr else None

    # ---------------------------------------------------------- 01
    def training_curves(self):
        h = _csv(self.out / "history.csv")
        if h is None or h.empty:
            return "skipped: no history.csv"
        plt = P.plt()
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.3))
        ax[0].plot(h.epoch, h.train_loss, color=P.C_MODEL, label="train (BCE+Dice)")
        if "val_loss" in h:
            ax[0].plot(h.epoch, h.val_loss, color=P.C_MF, label="val (BCE)")
        ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
        ax[0].legend()
        for k, c, lab in (("val_ap", P.C_MODEL, "AP"), ("val_f1", P.C_MF, "best F1"),
                          ("val_precision", P.C_MF_FIXED, "P @ best F1"), ("val_recall", P.C_EXTRA, "R @ best F1")):
            if k in h:
                ax[1].plot(h.epoch, h[k], color=c, label=lab, lw=2 if k in ("val_ap", "val_f1") else 1.2)
        best = int(h.val_ap.idxmax()) if "val_ap" in h else None
        if best is not None:
            ax[1].axvline(h.epoch[best], color=P.MUTED, lw=0.8, ls="--")
            ax[1].annotate(f"best.pt (ep {int(h.epoch[best])})", (h.epoch[best], 0.02), fontsize=7, color=P.INK2,
                           xytext=(3, 0), textcoords="offset points")
        ax[1].set(xlabel="epoch", ylabel="validation tiles", ylim=(0, 1.02), title="Validation metrics")
        ax[1].legend(ncol=2)
        ax[2].plot(h.epoch, h.lr, color=P.C_MODEL)
        ax[2].set(xlabel="epoch", ylabel="learning rate (end of epoch)", title="Schedule")
        ax[2].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
        fig.tight_layout()
        self.summary["training"] = dict(epochs=int(len(h)), best_epoch=None if best is None else int(h.epoch[best]),
                                        best_val_ap=float(h.val_ap.max()) if "val_ap" in h else None)
        return self.save(fig, "01_training_curves", "Training loss, validation-tile metrics and LR schedule; dashed line = checkpoint kept as best.pt.")

    # ---------------------------------------------------------- 02
    def threshold_sweep(self):
        if self.hist is None:
            return "skipped: no pr_hist.npz (re-run evaluate)"
        plt = P.plt()
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
        for ax, m, xl in ((axs[0], "model", "probability threshold"), (axs[1], "mf", "MF threshold (ppm·m, smoothed)")):
            h = self.hpr("test", m)
            if h is None:
                continue
            prec, rec, t = h.curve()
            tp = np.cumsum(h.pos[::-1])[::-1].astype(float)
            fp = np.cumsum(h.neg[::-1])[::-1].astype(float)
            fn = h.pos.sum() - tp
            f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
            iou = tp / np.maximum(tp + fp + fn, 1)
            for y, c, lab in ((prec, P.C_MODEL, "precision"), (rec, P.C_MF, "recall"), (f1, P.C_MF_FIXED, "F1"),
                              (iou, P.C_EXTRA, "IoU")):
                ax.plot(t, y, color=c, label=lab, lw=2 if lab in ("F1", "IoU") else 1.3)
            tv = self.thr_of(m)
            if tv is not None:
                ax.axvline(tv, color=P.INK2, lw=0.9, ls="--")
                ax.annotate(f"val-frozen {tv:.2f}" if m == "model" else f"val-frozen {tv:.0f}", (tv, 1.0),
                            fontsize=7, color=P.INK2, xytext=(3, -10), textcoords="offset points")
                i_best = int(np.argmax(f1))
                self.summary.setdefault("threshold", {})[m] = dict(
                    val_frozen=tv, test_optimal=float(t[i_best]), test_f1_at_val=float(np.interp(tv, t, f1)),
                    test_f1_at_optimal=float(f1[i_best]))
            if m == "mf":
                tf = self.thr_of("mf_fixed")
                if tf is not None:
                    ax.axvline(tf, color=P.MUTED, lw=0.8, ls=":")
                    ax.annotate(f"fixed {tf:.0f}", (tf, 0.9), fontsize=7, color=P.MUTED, xytext=(3, 0),
                                textcoords="offset points")
                hi = t[np.searchsorted(np.cumsum(h.pos) / max(h.pos.sum(), 1), 0.999)] if h.pos.sum() else t[-1]
                ax.set_xlim(0, max(hi, (tf or 0) * 1.2))
            ax.set(xlabel=xl, ylabel="test pixels", ylim=(0, 1.02), title=f"{self.lab(m)}: threshold sweep")
            ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
        fig.tight_layout()
        return self.save(fig, "02_threshold_sweep", "Pixel precision/recall/F1/IoU as a function of threshold on the test split. The dashed line is the threshold frozen on validation; a large gap between it and the test F1 peak signals val/test shift.")

    # ---------------------------------------------------------- 03
    def roc(self):
        if self.hist is None:
            return "skipped: no pr_hist.npz"
        plt = P.plt()
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        for m in ("model", "mf"):
            h = self.hpr("test", m)
            if h is None or h.pos.sum() == 0 or h.neg.sum() == 0:
                continue
            tpr = np.cumsum(h.pos[::-1])[::-1] / h.pos.sum()
            fpr = np.cumsum(h.neg[::-1])[::-1] / h.neg.sum()
            o = np.argsort(fpr)
            a = float(np.trapezoid(tpr[o], fpr[o])) if hasattr(np, "trapezoid") else float(np.trapz(tpr[o], fpr[o]))
            axs[0].plot(np.maximum(fpr, 1e-7), tpr, color=P.METHOD_STYLE[m]["color"], label=f"{self.lab(m)} (AUC {a:.3f})")
            tv = self.thr_of(m)
            if tv is not None:
                i = min(np.searchsorted(h.edges, tv, side="right") - 1, len(tpr) - 1)
                axs[0].plot(max(fpr[i], 1e-7), tpr[i], "o", color=P.METHOD_STYLE[m]["color"], ms=8, mec=P.SURFACE, mew=2)
            self.summary.setdefault("pixel_auroc", {})[m] = a
        axs[0].set_xscale("log")
        nneg = max(int(h.neg.sum()), 10) if h is not None else 10
        axs[0].set_xlim(left=1.0 / nneg)
        axs[0].set(xlabel="false-positive rate (background pixels, log)", ylabel="true-positive rate",
                   title="Pixel ROC (dot = val-frozen threshold)", ylim=(0, 1.02))
        axs[0].legend(loc="lower right")
        sdf = self.scene_df
        if sdf is not None and (sdf.role == "pos").any() and (sdf.role == "neg").any():
            for m in ("model", "mf"):
                col = f"{m}_max"
                if col not in sdf:
                    continue
                s, y = sdf[col].to_numpy(float), (sdf.role == "pos").to_numpy()
                ts = np.unique(np.concatenate([s, [np.inf]]))[::-1]
                tpr = [(s[y] >= t).mean() for t in ts]
                fpr = [(s[~y] >= t).mean() for t in ts]
                axs[1].step(fpr, tpr, where="post", color=P.METHOD_STYLE[m]["color"],
                            label=f"{self.lab(m)} (AUC {auroc(s[y], s[~y]):.3f})")
            axs[1].plot([0, 1], [0, 1], color=P.GRID, lw=1)
            axs[1].set(xlabel="false-positive rate (plume-free scenes)", ylabel="true-positive rate (plume scenes)",
                       title="Scene ROC (score = max over scene)", xlim=(-0.02, 1.02), ylim=(0, 1.02))
            axs[1].legend(loc="lower right")
        else:
            axs[1].text(0.5, 0.5, "needs both plume and plume-free test scenes", ha="center", color=P.MUTED,
                        transform=axs[1].transAxes)
        fig.tight_layout()
        return self.save(fig, "03_roc", "Left: pixel-level ROC with log FPR (plume pixels are rare, so the low-FPR end is what matters). Right: scene-level ROC for flagging scenes that contain a plume.")

    # ---------------------------------------------------------- 04
    def score_distributions(self):
        if self.hist is None:
            return "skipped: no pr_hist.npz"
        plt = P.plt()
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
        for ax, m, xl in ((axs[0], "model", "model probability"), (axs[1], "mf", "MF score (ppm·m, smoothed)")):
            h = self.hpr("test", m)
            if h is None:
                continue
            e = h.edges
            nb = 50
            g = max(1, (len(e) - 1) // nb)
            edges = e[::g]
            pos = np.add.reduceat(h.pos[:-1], np.arange(0, len(e) - 1, g))[: len(edges) - 1]
            neg = np.add.reduceat(h.neg[:-1], np.arange(0, len(e) - 1, g))[: len(edges) - 1]
            for cnt, c, lab in ((neg, P.MUTED, "background"), (pos, P.METHOD_STYLE[m]["color"], "plume")):
                d = cnt / max(cnt.sum(), 1)
                ax.stairs(np.maximum(d, 1e-9), edges, color=c, lw=1.8, label=f"{lab} (n={int(cnt.sum()):,})",
                          fill=lab == "plume", alpha=0.35 if lab == "plume" else 1)
            tv = self.thr_of(m)
            if tv is not None:
                ax.axvline(tv, color=P.INK2, lw=0.9, ls="--")
            ax.set_yscale("log")
            ax.set(xlabel=xl, ylabel="fraction of pixels (log)", title=f"{self.lab(m)}: score distributions",
                   ylim=(1e-6, 1.5))
            ax.legend()
        fig.tight_layout()
        return self.save(fig, "04_score_distributions", "Test-pixel score histograms, plume vs background (dashed = val-frozen threshold). Overlap is the irreducible confusion at pixel level.")

    # ---------------------------------------------------------- 05
    def reliability(self):
        p = self.out / "reliability.npz"
        if not p.exists():
            return "skipped: no reliability.npz"
        r = dict(np.load(p))
        n, ps, pos = r["n"].astype(float), r["p_sum"], r["pos"].astype(float)
        if n.sum() == 0:
            return "skipped: empty"
        nb = len(n)
        conf = np.where(n > 0, ps / np.maximum(n, 1), np.nan)
        freq = np.where(n > 0, pos / np.maximum(n, 1), np.nan)
        ece = float(np.nansum(n / n.sum() * np.abs(conf - freq)))
        # Brier from binned data (exact for the bin means; within-bin spread ignored)
        brier = float(np.nansum(n * (conf ** 2 - 2 * conf * freq + freq)) / n.sum())
        lo, hi = P.wilson(pos, n)
        plt = P.plt()
        fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8), gridspec_kw=dict(width_ratios=[1.1, 1]))
        axs[0].plot([0, 1], [0, 1], color=P.GRID, lw=1.2)
        ok = n > 0
        axs[0].errorbar(conf[ok], freq[ok], yerr=[freq[ok] - lo[ok], hi[ok] - freq[ok]], fmt="o-", color=P.C_MODEL,
                        ms=5, lw=1.5, capsize=0, label=f"{self.model_label}  ECE {ece:.3f}, Brier {brier:.4f}")
        axs[0].set(xlabel="mean predicted probability", ylabel="observed plume fraction", xlim=(0, 1), ylim=(0, 1),
                   title="Reliability (test pixels, 90 % Wilson CI)")
        axs[0].legend(loc="upper left")
        centers = (np.arange(nb) + 0.5) / nb
        axs[1].bar(centers, n, width=0.9 / nb, color=P.C_MODEL)
        axs[1].set_yscale("log")
        axs[1].set(xlabel="predicted probability", ylabel="pixels (log)", title="Probability histogram")
        fig.tight_layout()
        self.summary["calibration"] = dict(ece=ece, brier_binned=brier, n_pixels=int(n.sum()))
        return self.save(fig, "05_reliability", "Calibration of the model probability. Pos-weighted BCE + Dice usually makes the network over-confident; this matters if probabilities are used for flux or alerting, not for the thresholded masks.")

    # ---------------------------------------------------------- 06
    def per_scene_iou(self):
        sdf = self.scene_df
        if sdf is None or "model_iou" not in sdf or not (sdf.role == "pos").any():
            return "skipped: no plume scenes in per_scene_test.csv"
        d = sdf[sdf.role == "pos"]
        plt = P.plt()
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        axs[0].plot([0, 1], [0, 1], color=P.GRID, lw=1.2)
        axs[0].scatter(d.mf_iou, d.model_iou, s=28, color=P.C_MODEL, edgecolor=P.SURFACE, lw=0.8, zorder=3)
        better = int((d.model_iou > d.mf_iou).sum())
        axs[0].set(xlabel=f"IoU — {self.lab('mf')}", ylabel=f"IoU — {self.model_label}", xlim=(-0.02, 1.02),
                   ylim=(-0.02, 1.02), title=f"Per plume scene ({better}/{len(d)} above diagonal)")
        for m in ("model", "mf"):
            axs[1].scatter(d.n_pos_px, d[f"{m}_iou"], s=24, color=P.METHOD_STYLE[m]["color"], label=self.lab(m),
                           edgecolor=P.SURFACE, lw=0.8)
        axs[1].set_xscale("log")
        axs[1].set(xlabel="labelled plume pixels in scene (log)", ylabel="scene IoU", ylim=(-0.02, 1.02),
                   title="IoU vs amount of plume")
        axs[1].legend()
        fig.tight_layout()
        self.summary["per_scene_iou"] = dict(n_pos_scenes=int(len(d)), model_better=better,
                                             median_model=float(d.model_iou.median()), median_mf=float(d.mf_iou.median()))
        return self.save(fig, "06_per_scene_iou", "Scene-by-scene IoU. Points below the diagonal are scenes where the matched filter beat the model — good candidates for the error maps.")

    # ---------------------------------------------------------- 07
    def plume_detection(self):
        pdf = self.plume_df
        if pdf is None or pdf.empty:
            return "skipped: no labelled plumes on test"
        plt = P.plt()
        fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.8))
        rows = {}
        for ax, col, xl in ((axs[0], "n_px", "plume area (pixels, log bins)"),
                            (axs[1], "mf_sum", "integrated MF over plume (ppm·m·px, log bins)")):
            x = pdf[col].clip(lower=1).to_numpy(float)
            nb = int(np.clip(len(pdf) // 15, 2, 8))
            edges = np.unique(np.quantile(np.log10(x), np.linspace(0, 1, nb + 1)))
            if len(edges) < 3:
                edges = np.linspace(np.log10(x.min()) - 0.01, np.log10(x.max()) + 0.01, 3)
            b = np.clip(np.digitize(np.log10(x), edges[1:-1]), 0, len(edges) - 2)
            xc = 10 ** (0.5 * (edges[:-1] + edges[1:]))
            for j, m in enumerate(("model", "mf", "mf_fixed")):
                c = f"det_{m}"
                if c not in pdf:
                    continue
                k = np.bincount(b, weights=pdf[c].astype(float), minlength=len(xc))
                n = np.bincount(b, minlength=len(xc)).astype(float)
                lo, hi = P.wilson(k, n)
                rate = np.where(n > 0, k / np.maximum(n, 1), np.nan)
                off = 10 ** ((j - 1) * 0.025)
                ax.errorbar(xc * off, rate, yerr=[rate - lo, hi - rate], fmt="o-", color=P.METHOD_STYLE[m]["color"],
                            label=self.lab(m), capsize=0, lw=1.5, ms=5)
                rows.setdefault(col, {})[m] = dict(bin_center=xc.tolist(), recall=rate.tolist(), n=n.tolist())
            ax.set_xscale("log")
            ax.set(xlabel=xl, ylabel="plume recall", ylim=(-0.02, 1.05))
            ax.legend(loc="lower right")
        axs[0].set_title(f"Plume-level recall vs size (n = {len(pdf)} plumes)")
        axs[1].set_title("Plume-level recall vs strength")
        fig.tight_layout()
        self.summary["plume_detection_bins"] = rows
        return self.save(fig, "07_plume_detection", "Plume-complex recall (≥1 detected pixel) in quantile bins of area and integrated MF, with 90 % Wilson intervals. The strength panel is the empirical analogue of the MDL POD curve.")

    # ---------------------------------------------------------- single pass over test scenes
    def scene_pass(self):
        if self.splits is None or self.thr is None:
            self.pass_ = None
            return "skipped"
        cfg, E = self.cfg, self.cfg.eval
        a_edges = np.linspace(0, 2.5, 11)
        mf_edges = np.linspace(-1000, float(E.mf_max_ppmm), 61)
        pr_edges = np.linspace(0, 1, 41)
        acc = dict(alb_n=np.zeros(len(a_edges) + 1), fa_sizes={m: [] for m in ("model", "mf", "mf_fixed")},
                   alb_fp={m: np.zeros(len(a_edges) + 1) for m in ("model", "mf", "mf_fixed")},
                   h2_pos=np.zeros((60, 40)), h2_neg=np.zeros((60, 40)), have_model=False)
        methods = {"model": ("model", self.thr_of("model")), "mf": ("mf", self.thr_of("mf")),
                   "mf_fixed": ("mf", self.thr_of("mf_fixed"))}
        scene_dir = Path(cfg.paths.scenes)
        for sid in self.splits["test"]:
            if not (scene_dir / sid / "meta.json").exists():
                continue
            sc = load_scene(scene_dir / sid)
            valid, gt = sc["valid"], sc["mask"] > 0
            s = {"mf": mf_score(sc["mf"], valid, E.mf_smooth_sigma_px)}
            pm = self.score_fn(sc) if self.score_fn else None
            if pm is not None:
                s["model"] = np.clip(np.asarray(pm, np.float32), 0, 1)
                acc["have_model"] = True
            names = sc["meta"]["channel_names"]
            alb = np.asarray(sc["features"][names.index("albedo")], np.float32) if "albedo" in names else None
            bg = valid & ~gt
            if alb is not None:
                ab = np.digitize(alb[bg], a_edges)
                acc["alb_n"] += np.bincount(ab, minlength=len(a_edges) + 1)
            for m, (src, t) in methods.items():
                if src not in s or t is None:
                    continue
                pred = remove_small((s[src] >= t) & valid, E.min_component_px)
                if alb is not None:
                    acc["alb_fp"][m] += np.bincount(np.digitize(alb[pred & bg], a_edges), minlength=len(a_edges) + 1)
                lab, n = ndimage.label(pred, structure=EIGHT)
                if n:
                    sizes = np.bincount(lab.ravel())[1:]
                    touch = ndimage.maximum(gt, lab, index=np.arange(1, n + 1)).astype(bool)
                    acc["fa_sizes"][m] += sizes[~touch].tolist()
            if "model" in s:
                for g, key in ((gt & valid, "h2_pos"), (bg, "h2_neg")):
                    h, _, _ = np.histogram2d(np.clip(s["mf"][g], mf_edges[0], mf_edges[-1] - 1e-3),
                                             s["model"][g], bins=[mf_edges, pr_edges])
                    acc[key] += h
        acc.update(a_edges=a_edges, mf_edges=mf_edges, pr_edges=pr_edges)
        self.pass_ = acc
        return "ok"

    # ---------------------------------------------------------- 08
    def false_alarms(self):
        acc, sdf = self.pass_, self.scene_df
        if acc is None and sdf is None:
            return "skipped"
        plt = P.plt()
        fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.6))
        ms = [m for m in ("model", "mf", "mf_fixed") if not (m == "model" and acc is not None and not acc["have_model"])]
        if acc is not None and acc["alb_n"].sum():
            e = acc["a_edges"]
            xc = np.concatenate([[e[0] - 0.125], 0.5 * (e[:-1] + e[1:]), [e[-1] + 0.125]])
            n = acc["alb_n"]
            keep = n > 200
            for m in ms:
                rate = acc["alb_fp"][m] / np.maximum(n, 1) * 1e3
                axs[0].plot(xc[keep], rate[keep], "o-", color=P.METHOD_STYLE[m]["color"], label=self.lab(m), ms=4)
            axs[0].set_yscale("symlog", linthresh=0.1)
            axs[0].set(xlabel="albedo (window radiance / scene median)", ylabel="FP pixels per 1000 background px",
                       title="False positives vs surface brightness")
            axs[0].legend()
        if sdf is not None and "sigma_median_ppmm" in sdf and "model_fp" in sdf:
            for m in ms:
                r = sdf[f"{m}_fp"] / sdf.n_valid_px.clip(lower=1) * 1e3
                axs[1].scatter(sdf.sigma_median_ppmm, r, s=22, color=P.METHOD_STYLE[m]["color"], label=self.lab(m),
                               edgecolor=P.SURFACE, lw=0.7)
            axs[1].set_yscale("symlog", linthresh=0.1)
            axs[1].set(xlabel="scene median column σ (ppm·m)", ylabel="FP pixels per 1000 valid px",
                       title="Per-scene false positives vs noise")
            axs[1].legend()
        if acc is not None:
            allsz = [s for m in ms for s in acc["fa_sizes"][m]]
            if allsz:
                bins = np.unique(np.logspace(0, np.log10(max(allsz) + 1), 16).astype(int))
                for m in ms:
                    if acc["fa_sizes"][m]:
                        axs[2].hist(acc["fa_sizes"][m], bins=bins, histtype="step", lw=1.8,
                                    color=P.METHOD_STYLE[m]["color"], label=f"{self.lab(m)} (n={len(acc['fa_sizes'][m])})")
                axs[2].set_xscale("log")
                axs[2].legend()
            axs[2].set(xlabel="false-alarm component size (pixels)", ylabel="components",
                       title="False-alarm components (touch no label)")
            self.summary["false_alarm_components"] = {m: len(acc["fa_sizes"][m]) for m in ms}
        fig.tight_layout()
        return self.save(fig, "08_false_alarms", "Where false alarms come from. MF false positives classically rise over bright or spectrally unusual surfaces; a learned model should flatten the albedo curve. Remember uncatalogued real plumes also count as FPs here.")

    # ---------------------------------------------------------- 09
    def mf_vs_model(self):
        acc = self.pass_
        if acc is None or not acc["have_model"]:
            return "skipped: no model probabilities"
        plt = P.plt()
        from matplotlib.colors import LogNorm
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        ext = [acc["mf_edges"][0], acc["mf_edges"][-1], 0, 1]
        for ax, key, title, ramp in ((axs[0], "h2_neg", "background pixels", P.cmap(P.BLUE_RAMP[1:])),
                                     (axs[1], "h2_pos", "plume pixels", P.cmap(P.ORANGE_RAMP[1:]))):
            h = acc[key].T
            if h.sum() == 0:
                continue
            im = ax.imshow(np.where(h > 0, h, np.nan), origin="lower", extent=ext, aspect="auto", cmap=ramp,
                           norm=LogNorm(vmin=1, vmax=max(h.max(), 2)))
            fig.colorbar(im, ax=ax, label="pixels", fraction=0.046)
            for m, ls in (("mf", "--"), ("mf_fixed", ":")):
                t = self.thr_of(m)
                if t is not None:
                    ax.axvline(t, color=P.INK2, lw=0.8, ls=ls)
            t = self.thr_of("model")
            if t is not None:
                ax.axhline(t, color=P.INK2, lw=0.8, ls="--")
            ax.set(xlabel="MF score (ppm·m)", title=title)
            ax.grid(False)
        axs[0].set_ylabel(f"{self.model_label} probability")
        fig.tight_layout()
        return self.save(fig, "09_mf_vs_model", "Joint density of MF score and model probability on test pixels (dashed: val-frozen thresholds; dotted: fixed MF threshold). Background mass above the horizontal line and left of the vertical line = model-only false alarms; plume mass below/right = what the model adds or loses versus MF.")

    # ---------------------------------------------------------- 10
    def noise(self):
        rows = []
        for m in sorted(Path(self.cfg.paths.scenes).glob("*/meta.json")):
            meta = load_json(m)
            rows.append(dict(scene_id=meta["scene_id"], role=meta["role"], sigma=meta.get("sigma_median_ppmm"),
                             bgstd=meta.get("mf_bg_std_ppmm")))
        if not rows:
            return "skipped: no scenes"
        df = pd.DataFrame(rows).dropna()
        split_of = {s: k for k, ids in (self.splits or {}).items() for s in ids}
        df["split"] = df.scene_id.map(split_of).fillna("unused")
        plt = P.plt()
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
        cols = {"train": P.C_MODEL, "val": P.C_MF, "test": P.C_MF_FIXED, "unused": P.MUTED}
        for k, g in df.groupby("split"):
            axs[0].scatter(g.sigma, g.bgstd, s=24, color=cols.get(k, P.MUTED), label=f"{k} ({len(g)})",
                           edgecolor=P.SURFACE, lw=0.7)
        lim = [0, float(max(df.sigma.max(), df.bgstd.max()) * 1.05)]
        axs[0].plot(lim, lim, color=P.GRID, lw=1.2)
        axs[0].set(xlabel="median column noise-equivalent σ (ppm·m)", ylabel="MF std over background (ppm·m)",
                   title="Noise model vs observed MF spread")
        axs[0].legend()
        bins = np.linspace(0, lim[1], 25)
        for k in ("train", "val", "test"):
            g = df[df.split == k]
            if len(g):
                axs[1].hist(g.sigma, bins=bins, histtype="step", lw=1.8, color=cols[k], label=k, density=True)
        axs[1].set(xlabel="median column σ (ppm·m)", ylabel="density", title="σ distribution by split")
        axs[1].legend()
        fig.tight_layout()
        self.summary["noise"] = dict(sigma_median=float(df.sigma.median()), bgstd_median=float(df.bgstd.median()),
                                     ratio_median=float((df.bgstd / df.sigma).median()))
        return self.save(fig, "10_noise", "Left: observed MF background spread vs the filter's own noise estimate σ = (tᵀC⁻¹t)^-½ (points far above the diagonal = surface clutter / non-Gaussian background). Right: σ by split — train and test should overlap.")

    # ---------------------------------------------------------- 11
    def mdl(self):
        files = sorted(self.out.glob("mdl_records*.csv"))
        if not files:
            return "skipped: mdl stage not run"
        df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        mdl = _load(self.out / "mdl.json") or {}
        plt = P.plt()
        fig, axs = plt.subplots(1, 3, figsize=(14, 3.7))
        for m in ("model", "mf", "mf_fixed"):
            c = f"det_{m}"
            if c not in df:
                continue
            d = df[~df[f"pre_{m}"]]
            st = P.METHOD_STYLE[m]
            g = d.groupby("q_kgph")[c].agg(["sum", "count"])
            lo, hi = P.wilson(g["sum"], g["count"])
            r = g["sum"] / g["count"]
            axs[0].errorbar(g.index, r, yerr=[r - lo, hi - r], fmt="o", color=st["color"], ms=5, capsize=0)
            f = mdl.get(m, {}).get("flux", {}).get("fit")
            if f:
                qq = np.logspace(np.log10(df.q_kgph.min()), np.log10(df.q_kgph.max()), 200)
                axs[0].plot(qq, 1 / (1 + np.exp(-(np.log10(qq) - f["a"]) / f["b"])), color=st["color"],
                            label=f"{self.lab(m)}: MDL50 {mdl[m]['flux']['mdl50']:.0f} kg/h")
            else:
                axs[0].plot([], [], color=st["color"], label=self.lab(m))
            pk = d.peak_ppmm.clip(lower=1)
            edges = np.unique(np.quantile(np.log10(pk), np.linspace(0, 1, 9)))
            if len(edges) >= 3:
                b = np.clip(np.digitize(np.log10(pk), edges[1:-1]), 0, len(edges) - 2)
                k = np.bincount(b, weights=d[c].astype(float), minlength=len(edges) - 1)
                n = np.bincount(b, minlength=len(edges) - 1).astype(float)
                lo, hi = P.wilson(k, n)
                rr = k / np.maximum(n, 1)
                axs[1].errorbar(10 ** (0.5 * (edges[:-1] + edges[1:])), rr, yerr=[rr - lo, hi - rr], fmt="o-",
                                color=st["color"], label=self.lab(m), ms=5, capsize=0, lw=1.5)
            ps = d.groupby(["scene_id", "q_kgph"])[c].mean().unstack()
            for sid, row in ps.iterrows():
                axs[2].plot(row.index, row.values, color=st["color"], lw=0.8, alpha=0.45)
        for ax in axs:
            ax.set_xscale("log")
            ax.axhline(0.5, color=P.MUTED, lw=0.6, ls="--")
            ax.axhline(0.9, color=P.MUTED, lw=0.6, ls=":")
            ax.set_ylim(-0.02, 1.05)
        axs[0].set(xlabel=f"source rate (kg/h) at U = {mdl.get('wind_ms', '?')} m/s", ylabel="probability of detection",
                   title="POD vs source rate")
        axs[0].legend(loc="lower right")
        axs[1].set(xlabel="peak injected enhancement (ppm·m)", title="POD vs peak enhancement")
        axs[1].legend(loc="lower right")
        axs[2].set(xlabel="source rate (kg/h)", title="Per-scene POD (spread = background dependence)")
        fig.tight_layout()
        return self.save(fig, "11_mdl", "Synthetic-injection detection. The per-scene panel shows how much the MDL depends on the background scene; the peak-ppm·m panel removes the wind/plume-model assumption.")

    # ---------------------------------------------------------- 12
    def split_map(self):
        st = _csv(Path(self.cfg.paths.splits) / "scene_table.csv")
        if st is None or st.empty or self.splits is None:
            return "skipped: no splits/scene_table.csv"
        split_of = {s: k for k, ids in self.splits.items() for s in ids}
        st["split"] = st.scene_id.map(split_of)
        plt = P.plt()
        fig, ax = plt.subplots(figsize=(9, 4.6))
        regs = (self.cfg.get("scenes", {}) or {}).get("negative_regions", {}) or {}
        from matplotlib.patches import Rectangle
        for name, (w, s, e, n) in regs.items():
            ax.add_patch(Rectangle((w, s), e - w, n - s, fill=False, ec=P.MUTED, lw=0.8, ls="--"))
            ax.text(w, n + 0.4, name, fontsize=6, color=P.MUTED)
        for k, mk in (("train", "o"), ("val", "s"), ("test", "D")):
            for role, face in (("pos", True), ("neg", False)):
                g = st[(st.split == k) & (st.role == role)]
                if len(g):
                    c = {"train": P.C_MODEL, "val": P.C_MF, "test": P.C_MF_FIXED}[k]
                    ax.scatter(g.lon, g.lat, s=30, marker=mk, facecolor=c if face else "none", edgecolor=c, lw=1.2,
                               label=f"{k} · {'plume' if role == 'pos' else 'plume-free'} ({len(g)})")
        bs = float(self.cfg.split.get("block_deg", 1.0))
        ax.set(xlabel="longitude (°)", ylabel="latitude (°)",
               title=f"Scene centres by split ({self.cfg.split.method}, {bs:g}° blocks)")
        ys = np.concatenate([st.lat.to_numpy(float)] + [np.array([s_, n_]) for (_, s_, _, n_) in regs.values()])
        xs = np.concatenate([st.lon.to_numpy(float)] + [np.array([w_, e_]) for (w_, _, e_, _) in regs.values()])
        ax.set_xlim(xs.min() - 8, xs.max() + 8)
        ax.set_ylim(ys.min() - 5, ys.max() + 8)
        ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
        fig.tight_layout()
        return self.save(fig, "12_split_map", "Scene locations by split (filled = plume scenes, open = plume-free; dashed boxes = hard-negative regions). With geo_block splitting no block should host two splits.")

    # ---------------------------------------------------------- maps
    def error_maps(self):
        sdf = self.scene_df
        if sdf is None or sdf.empty or self.thr is None:
            return "skipped: no per_scene_test.csv"
        pos = sdf[sdf.role == "pos"].sort_values("model_iou", ascending=False)
        neg = sdf[sdf.role == "neg"].sort_values("model_pred_total", ascending=False)
        pick = []
        k = self.n_maps
        pick += [("best", s) for s in pos.scene_id.head(k)]
        pick += [("worst", s) for s in pos.scene_id.tail(k) if s not in pos.scene_id.head(k).tolist()]
        pick += [("false-alarm", s) for s in neg.scene_id.head(k) if neg.set_index("scene_id").loc[s, "model_pred_total"]
                 + neg.set_index("scene_id").loc[s, "mf_pred_total"] > 0]
        done = 0
        for tag, sid in pick:
            if self._map(tag, sid):
                done += 1
        return f"ok ({done} maps)"

    def _map(self, tag, sid):
        cfg, E = self.cfg, self.cfg.eval
        d = Path(cfg.paths.scenes) / sid
        if not (d / "meta.json").exists():
            return False
        sc = load_scene(d)
        meta, valid, gt = sc["meta"], sc["valid"], sc["mask"] > 0
        ext = P.extent(meta)
        mfs = mf_score(sc["mf"], valid, E.mf_smooth_sigma_px)
        pm = self.score_fn(sc) if self.score_fn else None
        rgb = P.rgb_image(sc["features"], meta["channel_names"], valid)
        plt = P.plt()
        show_model = pm is not None
        ncol = 5 if show_model else 4
        fig, axs = plt.subplots(1, ncol, figsize=(3.3 * ncol, 3.6))
        axs[0].imshow(rgb, extent=ext)
        axs[0].set_title("True colour")
        vmax = max(1000.0, float(np.nanpercentile(np.where(valid, sc["mf"], np.nan), 99.9)) if valid.any() else 1000)
        im = axs[1].imshow(np.where(valid, sc["mf"], np.nan), extent=ext, cmap=P.ppmm_cmap(), vmin=0, vmax=vmax)
        fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.02, label="ppm·m")
        if gt.any():
            axs[1].contour(np.flipud(gt.astype(float)), levels=[0.5], colors=P.INK, linewidths=0.6,
                           extent=ext, origin="lower")
        axs[1].set_title("Matched filter (label outline)")
        i = 2
        if show_model:
            im = axs[2].imshow(np.where(valid, pm, np.nan), extent=ext, cmap=P.prob_cmap(), vmin=0, vmax=1)
            fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.02, label="probability")
            axs[2].set_title(f"{self.model_label} probability")
            i = 3
        for j, (m, score) in enumerate((("model", pm), ("mf", mfs))):
            if score is None:
                continue
            pred = remove_small((score >= self.thr_of(m)) & valid, E.min_component_px)
            axs[i].imshow(P.confusion_rgba(pred, gt, valid, rgb), extent=ext)
            tp, fp, fn = int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())
            iou = tp / max(tp + fp + fn, 1)
            axs[i].set_title(f"{self.lab(m).split(' (')[0]}: IoU {iou:.2f}" if gt.any() else
                             f"{self.lab(m).split(' (')[0]}: {fp} FP px")
            P.confusion_legend(axs[i])
            i += 1
        for a in axs:
            P.geo_axes(a, ext)
        fig.suptitle(f"{tag}: {sid}  ({meta['role']}, σ≈{meta.get('sigma_median_ppmm') or 0:.0f} ppm·m)", fontsize=9,
                     color=P.INK2)
        fig.tight_layout()
        P.watermark(fig, self.wm)
        fig.savefig(self.fig_dir / "maps" / f"{tag}_{sid}.png")
        plt.close(fig)
        return True

    # ---------------------------------------------------------- index
    def write_index(self):
        L = [f"# Diagnostics — run `{self.cfg.run_name}`\n"]
        if self.metrics:
            L.append("| method | F1 | 90% CI | IoU | 90% CI | plume recall | 90% CI |")
            L.append("|---|---|---|---|---|---|---|")
            for m in ("model", "mf", "mf_fixed"):
                r = self.metrics.get(m)
                if not r:
                    continue
                ci = r.get("ci90", {})
                L.append(f"| {self.lab(m)} | {_f(r.get('f1'))} | {_ci(ci.get('f1'))} | {_f(r.get('iou'))} | "
                         f"{_ci(ci.get('iou'))} | {_f(r.get('plume_recall'))} | {_ci(ci.get('plume_recall'))} |")
            L.append("")
        cal = self.summary.get("calibration")
        if cal:
            L.append(f"Calibration: ECE {cal['ece']:.3f}, Brier {cal['brier_binned']:.4f}.\n")
        for name, cap in self.figs:
            L.append(f"### {name.replace('_', ' ', 1)}\n\n![{name}]({name}.png)\n\n{cap}\n")
        maps = sorted((self.fig_dir / "maps").glob("*.png"))
        if maps:
            L.append("### Error maps\n")
            L += [f"![{p.stem}](maps/{p.name})" for p in maps]
        skipped = {k: v for k, v in self.status.items() if v not in ("ok",) and not str(v).startswith("ok")}
        if skipped:
            L.append("\n### Skipped\n")
            L += [f"- `{k}`: {v}" for k, v in skipped.items()]
        (self.fig_dir / "DIAGNOSTICS.md").write_text("\n".join(L) + "\n")


# ================================================================== small utils
def _load(p):
    return load_json(p) if Path(p).exists() else None


def _csv(p):
    try:
        return pd.read_csv(p) if Path(p).exists() else None
    except pd.errors.EmptyDataError:
        return None


def _f(v, p=3):
    return "–" if v is None or (isinstance(v, float) and v != v) else f"{v:.{p}f}"


def _ci(c):
    return f"{c[0]:.3f}–{c[1]:.3f}" if c else "–"

