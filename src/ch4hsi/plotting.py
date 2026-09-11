"""Shared matplotlib style, colours and map helpers for evaluation / diagnostics / README figures.

Colours follow one fixed categorical order (never cycled): model = blue, tuned MF = orange,
fixed-threshold MF = aqua. Magnitudes use single-hue ramps; TP/FP/FN overlays reuse the same three
validated slots and always ship with a legend.
"""
from __future__ import annotations

import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"

C_MODEL = "#2a78d6"     # slot 1 blue
C_MF = "#eb6834"        # slot 2 orange
C_MF_FIXED = "#1baf7a"  # slot 3 aqua
C_EXTRA = "#4a3aa7"     # violet (only when a 4th, non-adjacent series is unavoidable)

METHOD_STYLE = {
    "model": dict(color=C_MODEL, label="U-Net"),
    "mf": dict(color=C_MF, label="Matched filter (val-tuned)"),
    "mf_fixed": dict(color=C_MF_FIXED, label="Matched filter (1000 ppm·m)"),
}

# TP / FP / FN overlay colours (same three slots, legend always drawn)
C_TP, C_FP, C_FN = C_MF_FIXED, C_MF, C_MODEL

BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
ORANGE_RAMP = ["#fcfcfb", "#fbe0cf", "#f4b08a", "#eb6834", "#b5461b", "#5e210a"]


def plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    return _plt


def apply_style():
    p = plt()
    p.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
        "axes.titleweight": "semibold", "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2.0, "lines.markersize": 5,
        "font.size": 9, "text.color": INK, "figure.dpi": 110, "savefig.bbox": "tight", "savefig.dpi": 160,
        "image.interpolation": "nearest",
    })
    return p


def cmap(ramp, name="ramp", bad=None):
    from matplotlib.colors import LinearSegmentedColormap
    cm = LinearSegmentedColormap.from_list(name, ramp)
    cm.set_bad(bad or "#d9d8d3")
    return cm


def prob_cmap():
    return cmap(BLUE_RAMP, "prob")


def ppmm_cmap():
    return cmap(ORANGE_RAMP, "ppmm")


def label(method: str, model_label: str | None = None) -> str:
    if method == "model" and model_label:
        return model_label
    return METHOD_STYLE[method]["label"]


# ------------------------------------------------------------------ maps
def extent(meta_or_gt, shape=None):
    """imshow extent [W, E, S, N] from a scene meta dict or a GDAL geotransform + shape."""
    if isinstance(meta_or_gt, dict):
        gt, (H, W) = meta_or_gt["geotransform"], meta_or_gt["shape"]
    else:
        gt, (H, W) = meta_or_gt, shape
    return [gt[0], gt[0] + W * gt[1], gt[3] + H * gt[5], gt[3]]


def geo_axes(ax, ext, crop=None):
    """Lon/lat ticks with a 1/cos(lat) aspect so pixels look square on the ground."""
    lat0 = 0.5 * (ext[2] + ext[3])
    ax.set_aspect(1.0 / max(np.cos(np.deg2rad(lat0)), 0.2))
    if crop is not None:
        ax.set_xlim(crop[0], crop[1])
        ax.set_ylim(crop[2], crop[3])
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_formatter(_deg_fmt("E", "W"))
    ax.yaxis.set_major_formatter(_deg_fmt("N", "S"))
    ax.locator_params(nbins=4)
    ax.grid(False)


def _deg_fmt(pos, neg):
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, _: f"{abs(v):.2f}°{pos if v >= 0 else neg}")


def crop_extent(meta, rc_center, half_px):
    """Geographic window of ±half_px pixels around (row, col) on the scene grid."""
    gt = meta["geotransform"]
    H, W = meta["shape"]
    r0, r1 = max(0, rc_center[0] - half_px), min(H, rc_center[0] + half_px)
    c0, c1 = max(0, rc_center[1] - half_px), min(W, rc_center[1] + half_px)
    return [gt[0] + c0 * gt[1], gt[0] + c1 * gt[1], gt[3] + r1 * gt[5], gt[3] + r0 * gt[5]]


def rgb_image(features, names, valid, gamma=0.8):
    """Stretched true-colour composite from the rgb_* feature channels (grey if unavailable)."""
    H, W = valid.shape
    if all(n in names for n in ("rgb_r", "rgb_g", "rgb_b")):
        rgb = np.stack([np.asarray(features[names.index(n)], np.float32) for n in ("rgb_r", "rgb_g", "rgb_b")], -1)
        v = rgb[valid]
        lo, hi = (np.percentile(v, 2, axis=0), np.percentile(v, 98, axis=0)) if len(v) else (0, 1)
        rgb = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1) ** gamma
    else:
        rgb = np.full((H, W, 3), 0.75, np.float32)
    rgba = np.concatenate([rgb, valid[..., None].astype(np.float32)], -1)
    return rgba


def confusion_rgba(pred, gt, valid, base=None):
    """RGBA overlay: TP / FP / FN coloured over a faded base image."""
    from matplotlib.colors import to_rgb
    H, W = valid.shape
    img = np.ones((H, W, 4), np.float32)
    if base is not None:
        g = base[..., :3].mean(-1, keepdims=True)
        img[..., :3] = 0.55 + 0.4 * g
    else:
        img[..., :3] = 0.92
    for m, c in ((pred & gt, C_TP), (pred & ~gt, C_FP), (~pred & gt, C_FN)):
        img[m, :3] = to_rgb(c)
    img[..., 3] = valid.astype(np.float32)
    return img


def confusion_legend(ax, loc="lower left"):
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C_TP, label="TP"), Patch(color=C_FP, label="FP"), Patch(color=C_FN, label="FN")],
              loc=loc, fontsize=7, ncol=3, handlelength=1, columnspacing=0.8, frameon=True, facecolor=SURFACE,
              edgecolor=GRID, framealpha=0.9)


def wilson(k, n, z=1.645):
    """Wilson score interval (90 % by default) for k successes out of n."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = k / n
        d = 1 + z ** 2 / n
        c = (p + z ** 2 / (2 * n)) / d
        h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    # clamp so that lo <= p <= hi exactly (float round-off at p = 0 or 1 breaks errorbar)
    return np.fmin(c - h, p), np.fmax(c + h, p)


def watermark(fig, text):
    if text:
        fig.text(0.995, 0.005, text, ha="right", va="bottom", fontsize=7, color=MUTED, style="italic")
