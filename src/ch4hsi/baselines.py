"""Light-weight learned baseline: per-pixel logistic regression on the same normalised feature stack.

It sees exactly the U-Net's inputs but no spatial context, so "U-Net minus pixel-LR" isolates what the
convolutional context adds on top of the spectral features. Pure numpy/scipy (no torch), so it also
serves as the model stand-in for README figures and tests where torch is unavailable.

    ch4hsi baseline-lr [--set run_name=...]     -> runs/<run_name>_pixel_lr/ (same files as `evaluate`)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize

from .config import Cfg
from .utils import dump_json, get_logger, load_json

log = get_logger(__name__)


class PixelLogReg:
    def __init__(self, norm: dict, l2: float = 1e-2, smooth_sigma_px: float = 1.0):
        self.mean = np.asarray(norm["mean"], np.float32)
        self.std = np.asarray(norm["std"], np.float32)
        self.names = list(norm.get("channel_names", []))
        self.l2, self.smooth = float(l2), float(smooth_sigma_px)
        self.w = None

    def _x(self, feats_chw, rr=None, cc=None):
        f = np.asarray(feats_chw[:, rr, cc] if rr is not None else feats_chw, np.float32)
        z = (f - self.mean.reshape((-1,) + (1,) * (f.ndim - 1))) / self.std.reshape((-1,) + (1,) * (f.ndim - 1))
        return np.clip(np.nan_to_num(z), -10, 10)

    def fit(self, scene_dirs, n_per_scene: int = 20000, pos_frac: float = 0.5, seed: int = 0):
        """Class-balanced pixel sample from the training scenes, L2-regularised logistic loss (L-BFGS)."""
        rng = np.random.default_rng(seed)
        X, Y = [], []
        for d in scene_dirs:
            d = Path(d)
            f = np.load(d / "features.npy", mmap_mode="r")
            v, m = np.load(d / "valid.npy"), np.load(d / "mask.npy") > 0
            pos, neg = np.flatnonzero((v & m).ravel()), np.flatnonzero((v & ~m).ravel())
            n_pos = min(len(pos), int(n_per_scene * pos_frac))
            pick = np.concatenate([rng.choice(pos, n_pos, replace=False) if n_pos else pos[:0],
                                   rng.choice(neg, min(len(neg), n_per_scene - n_pos), replace=False)])
            rr, cc = np.unravel_index(np.sort(pick), v.shape)
            X.append(self._x(f, rr, cc).T)
            Y.append(m[rr, cc])
        X = np.concatenate(X).astype(np.float64)
        y = np.concatenate(Y).astype(np.float64)
        wpos = 0.5 / max(y.mean(), 1e-6)
        wneg = 0.5 / max(1 - y.mean(), 1e-6)
        sw = np.where(y > 0, wpos, wneg)
        Xb = np.hstack([X, np.ones((len(X), 1))])

        def f_and_g(w):
            z = Xb @ w
            p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
            loss = -np.sum(sw * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))) / len(y)
            g = Xb.T @ (sw * (p - y)) / len(y)
            loss += 0.5 * self.l2 * np.sum(w[:-1] ** 2)
            g[:-1] += self.l2 * w[:-1]
            return loss, g

        res = minimize(f_and_g, np.zeros(Xb.shape[1]), jac=True, method="L-BFGS-B", options=dict(maxiter=500))
        self.w = res.x
        log.info("pixel-LR fitted on %d px (%.1f%% plume), loss %.4f, converged=%s", len(y), 100 * y.mean(),
                 res.fun, res.success)
        return self

    def predict(self, feats_chw, valid):
        z = np.tensordot(self.w[:-1], self._x(feats_chw), axes=(0, 0)) + self.w[-1]
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        p = np.where(valid, p, 0).astype(np.float32)
        if self.smooth > 0:           # same normalised-convolution smoothing as the MF score
            num = ndimage.gaussian_filter(p, self.smooth)
            den = ndimage.gaussian_filter(valid.astype(np.float32), self.smooth)
            p = np.where(den > 1e-3, num / np.maximum(den, 1e-3), 0).astype(np.float32)
            p[~valid] = 0
        return p

    def coefficients(self):
        return dict(zip(self.names or [f"c{i}" for i in range(len(self.w) - 1)], self.w[:-1].tolist()),
                    bias=float(self.w[-1]))


def run(cfg: Cfg):
    """`ch4hsi baseline-lr`: fit on train scenes, evaluate like the U-Net into runs/<run_name>_pixel_lr/."""
    from .config import copy_cfg, run_dir, save_config
    from .diagnostics import diagnose
    from .evaluate import evaluate_scenes

    splits = load_json(Path(cfg.paths.splits) / "splits.json")["splits"]
    norm = load_json(Path(cfg.paths.splits) / "norm.json")
    c2 = copy_cfg(cfg)
    c2.run_name = f"{cfg.run_name}_pixel_lr"
    out = run_dir(c2)
    save_config(c2, out / "config.yaml")
    dump_json(norm, out / "norm.json")
    lr = PixelLogReg(norm).fit([Path(cfg.paths.scenes) / s for s in splits["train"]], seed=int(cfg.train.seed))
    dump_json(lr.coefficients(), out / "pixel_lr_coefficients.json")
    evaluate_scenes(c2, lambda sc: lr.predict(sc["features"], sc["valid"]), out, label="Pixel logistic regression")
    diagnose(c2, out, score_fn=lambda sc: lr.predict(sc["features"], sc["valid"]))
