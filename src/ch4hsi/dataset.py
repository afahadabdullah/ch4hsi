"""Tile sampling from per-scene .npy stacks (memory-mapped; cheap random crops)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .physics.plume_sim import gaussian_plume_ppmm
from .utils import load_json


class Scene:
    def __init__(self, d: str | Path, max_pos: int = 20000, seed: int = 0):
        self.dir = Path(d)
        self.meta = load_json(self.dir / "meta.json")
        self.id = self.meta["scene_id"]
        self.H, self.W = self.meta["shape"]
        mask = np.load(self.dir / "mask.npy")
        pos = np.argwhere(mask > 0)
        if len(pos) > max_pos:
            pos = pos[np.random.default_rng(seed).choice(len(pos), max_pos, replace=False)]
        self.pos = pos
        v = np.load(self.dir / "valid.npy")
        vi = np.argwhere(v[::8, ::8]) * 8          # coarse list of valid anchors for random crops
        self.valid_anchor = vi
        self._f = self._m = self._v = self._s = None

    def arrays(self):
        if self._f is None:                        # opened lazily inside each worker
            self._f = np.load(self.dir / "features.npy", mmap_mode="r")
            self._m = np.load(self.dir / "mask.npy", mmap_mode="r")
            self._v = np.load(self.dir / "valid.npy", mmap_mode="r")
            self._s = np.load(self.dir / "sigma.npy", mmap_mode="r")
        return self._f, self._m, self._v, self._s


def crop(a, r0, c0, T, fill=0):
    """Crop a[..., r0:r0+T, c0:c0+T] with padding outside the array."""
    H, W = a.shape[-2:]
    out = np.full(a.shape[:-2] + (T, T), fill, dtype=a.dtype)
    rs, re_ = max(r0, 0), min(r0 + T, H)
    cs, ce = max(c0, 0), min(c0 + T, W)
    if re_ > rs and ce > cs:
        out[..., rs - r0:re_ - r0, cs - c0:ce - c0] = a[..., rs:re_, cs:ce]
    return out


class TileDataset(Dataset):
    def __init__(self, scene_dirs, norm: dict, tcfg, mode: str = "train", seed: int = 0):
        self.scenes = [Scene(d, seed=seed) for d in scene_dirs]
        self.pos_scenes = [i for i, s in enumerate(self.scenes) if len(s.pos)]
        self.mean = np.asarray(norm["mean"], np.float32)[:, None, None]
        self.std = np.asarray(norm["std"], np.float32)[:, None, None]
        self.names = norm["channel_names"]
        self.T = int(tcfg.tile)
        self.mode = mode
        self.tcfg = tcfg
        self.seed = seed
        self.epoch = 0
        self.n = int(tcfg.tiles_per_epoch if mode == "train" else tcfg.val_tiles)
        self.fixed = None
        if mode != "train":           # deterministic validation tiles
            rng = np.random.default_rng(seed + 12345)
            self.fixed = [self._pick(rng) for _ in range(self.n)]

    def set_epoch(self, e):
        self.epoch = e

    def __len__(self):
        return self.n

    def _pick(self, rng):
        T = self.T
        if self.pos_scenes and rng.random() < self.tcfg.pos_tile_frac:
            si = self.pos_scenes[rng.integers(len(self.pos_scenes))]
            r, c = self.scenes[si].pos[rng.integers(len(self.scenes[si].pos))]
            jitter = rng.integers(-T // 3, T // 3 + 1, size=2)
            return si, int(r - T // 2 + jitter[0]), int(c - T // 2 + jitter[1])
        si = int(rng.integers(len(self.scenes)))
        s = self.scenes[si]
        if len(s.valid_anchor):
            r, c = s.valid_anchor[rng.integers(len(s.valid_anchor))]
            return si, int(r - T // 2), int(c - T // 2)
        return si, int(rng.integers(0, max(1, s.H - T))), int(rng.integers(0, max(1, s.W - T)))

    def _synth(self, x_raw, y, sigma, meta, rng):
        """Paste a Gaussian plume into raw (un-normalised) features, consistently across channels."""
        sa = self.tcfg.synth_aug
        T = self.T
        q = float(np.exp(rng.uniform(np.log(sa.flux_kgph[0]), np.log(sa.flux_kgph[1]))))
        u = float(rng.uniform(*sa.wind_ms))
        E = gaussian_plume_ppmm((T, T), (int(rng.integers(T // 4, 3 * T // 4)), int(rng.integers(T // 4, 3 * T // 4))),
                                q, u, float(rng.uniform(0, 2 * np.pi)), pixel_m=60.0,
                                max_age_s=float(rng.uniform(300, 900)), turbulence=0.3, rng=rng, supersample=3)
        names = self.names
        if "mf" in names:
            x_raw[names.index("mf")] += E
        if "mf_snr" in names:
            x_raw[names.index("mf_snr")] += E / np.maximum(sigma, 1.0)
        kb = meta.get("k_bins")
        if kb:
            for i, kk in enumerate(kb):
                n = f"swir_resid_{i}"
                if n in names:
                    j = names.index(n)
                    x_raw[j] = (1 + x_raw[j]) * np.exp(kk * E) - 1
        y = np.maximum(y, (E > sa.label_min_ppmm).astype(y.dtype))
        return x_raw, y

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i]) if self.mode == "train" else np.random.default_rng([self.seed, i])
        si, r0, c0 = self.fixed[i] if self.fixed is not None else self._pick(rng)
        s = self.scenes[si]
        f, m, v, sg = s.arrays()
        T = self.T
        x = crop(f, r0, c0, T).astype(np.float32)
        y = crop(m, r0, c0, T).astype(np.float32)
        val = crop(v, r0, c0, T, fill=False)
        sig = crop(sg, r0, c0, T).astype(np.float32)
        sa = self.tcfg.synth_aug
        if self.mode == "train" and sa.enabled and y.sum() == 0 and rng.random() < sa.prob:
            x, y = self._synth(x, y, sig, s.meta, rng)
        x = (x - self.mean) / self.std
        x = np.clip(x, -10, 10)
        x[:, ~val] = 0.0
        y[~val] = 0.0
        if self.mode == "train":
            k = int(rng.integers(4))
            x, y, val = np.rot90(x, k, (1, 2)), np.rot90(y, k), np.rot90(val, k)
            if rng.random() < 0.5:
                x, y, val = x[:, :, ::-1], y[:, ::-1], val[:, ::-1]
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))[None],
                torch.from_numpy(np.ascontiguousarray(val))[None])
