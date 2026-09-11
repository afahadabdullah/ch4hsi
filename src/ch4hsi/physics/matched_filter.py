"""Column-wise (per detector column) matched filter for pushbroom imaging spectrometers.

For each detector column j (or group of columns) with background mean mu_j and covariance C_j,
the target signature is t_j = mu_j * k (first-order Beer-Lambert), and

    alpha = (x - mu)^T C^-1 t / (t^T C^-1 t)          [ppm·m]
    sigma_alpha = (t^T C^-1 t)^-1/2                   [ppm·m, noise-equivalent enhancement]

With `albedo_correction` the estimate is divided by r = x.mu / mu.mu (mag1c-style), which removes
the bias over bright/dark surfaces. `two_pass` re-estimates (mu, C) after masking positive
outliers (likely plume pixels) so plumes do not contaminate their own background.
"""
from __future__ import annotations

import numpy as np


def _stats(X, W, shrinkage):
    """X: (g, n, b) float64, W: (g, n) weights {0,1}. Returns mu (g,b), C (g,b,b)."""
    n = W.sum(axis=1).clip(min=2)[:, None]
    mu = (X * W[..., None]).sum(axis=1) / n
    Xc = (X - mu[:, None, :]) * W[..., None]
    C = np.einsum("gnb,gnd->gbd", Xc, Xc) / (n[..., None] - 1)
    d = np.einsum("gbb->gb", C)
    C = (1 - shrinkage) * C
    idx = np.arange(C.shape[-1])
    C[:, idx, idx] += shrinkage * d
    # tiny ridge for numerical safety
    C[:, idx, idx] += 1e-9 * d.mean(axis=1, keepdims=True) + 1e-12
    return mu, C


def _apply(X, mu, C, k, albedo_correction):
    t = mu * k[None, :]                                       # (g, b)
    Cit = np.linalg.solve(C, t[..., None])[..., 0]            # (g, b)
    denom = np.einsum("gb,gb->g", t, Cit)                     # (g,)
    num = np.einsum("gnb,gb->gn", X - mu[:, None, :], Cit)
    if albedo_correction:
        r = np.einsum("gnb,gb->gn", X, mu) / np.einsum("gb,gb->g", mu, mu)[:, None]
        r = np.where(np.abs(r) > 1e-3, r, 1.0)
    else:
        r = 1.0
    mf = num / (denom[:, None] * r)
    sigma = 1.0 / np.sqrt(np.clip(denom, 1e-30, None))
    return mf, sigma


def columnwise_mf(rad: np.ndarray, k: np.ndarray, valid: np.ndarray | None = None, column_group: int = 1,
                  shrinkage: float = 1e-3, two_pass: bool = True, outlier_sigma: float = 3.0,
                  albedo_correction: bool = True, chunk_cols: int = 64):
    """rad: (rows, cols, b) radiance in the retrieval window; k: (b,) per ppm·m.

    Returns mf (rows, cols) ppm·m [NaN where invalid], sigma (cols,) ppm·m, mu (cols, b).
    """
    rows, cols, b = rad.shape
    if valid is None:
        valid = np.all(np.isfinite(rad), axis=-1)
    G = max(1, int(column_group))
    ngroups = int(np.ceil(cols / G))
    mf_out = np.full((rows, cols), np.nan, dtype=np.float32)
    sig_out = np.full(cols, np.nan, dtype=np.float32)
    mu_out = np.full((cols, b), np.nan, dtype=np.float32)
    k = np.asarray(k, dtype=np.float64)

    groups_per_chunk = max(1, chunk_cols // G)
    with np.errstate(divide="ignore", invalid="ignore"):
        _loop(rad, valid, k, G, ngroups, groups_per_chunk, rows, cols, b, shrinkage, two_pass,
              outlier_sigma, albedo_correction, mf_out, sig_out, mu_out)
    return mf_out, sig_out, mu_out


def _loop(rad, valid, k, G, ngroups, groups_per_chunk, rows, cols, b, shrinkage, two_pass,
          outlier_sigma, albedo_correction, mf_out, sig_out, mu_out):
    for g0 in range(0, ngroups, groups_per_chunk):
        g1 = min(ngroups, g0 + groups_per_chunk)
        c0, c1 = g0 * G, min(cols, g1 * G)
        ng = g1 - g0
        pad = ng * G - (c1 - c0)
        Xs = rad[:, c0:c1, :].astype(np.float64)
        Ws = valid[:, c0:c1].astype(np.float64)
        Xs = np.where(Ws[..., None] > 0, Xs, 0.0)
        if pad:
            Xs = np.concatenate([Xs, np.zeros((rows, pad, b))], axis=1)
            Ws = np.concatenate([Ws, np.zeros((rows, pad))], axis=1)
        # (rows, ng*G, b) -> (ng, G*rows, b): group = consecutive columns
        X = Xs.reshape(rows, ng, G, b).transpose(1, 2, 0, 3).reshape(ng, G * rows, b)
        W = Ws.reshape(rows, ng, G).transpose(1, 2, 0).reshape(ng, G * rows)

        mu, C = _stats(X, W, shrinkage)
        mf, sigma = _apply(X, mu, C, k, albedo_correction)
        if two_pass:
            W2 = W * (mf < outlier_sigma * sigma[:, None])
            enough = W2.sum(axis=1) > max(10, b + 2)
            W2[~enough] = W[~enough]
            mu, C = _stats(X, W2, shrinkage)
            mf, sigma = _apply(X, mu, C, k, albedo_correction)

        mf = mf.reshape(ng, G, rows).transpose(2, 0, 1).reshape(rows, ng * G)[:, : c1 - c0]
        W = W.reshape(ng, G, rows).transpose(2, 0, 1).reshape(rows, ng * G)[:, : c1 - c0]
        mf_out[:, c0:c1] = np.where(W > 0, mf, np.nan)
        sig_out[c0:c1] = np.repeat(sigma, G)[: c1 - c0]
        mu_out[c0:c1] = np.repeat(mu, G, axis=0)[: c1 - c0]
