"""Stage 5: train the segmentation model (single GPU, AMP, resumable for Slurm requeue)."""
from __future__ import annotations

import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Cfg, run_dir, save_config
from .dataset import TileDataset
from .metrics import HistPR
from .models.unet import build_model
from .utils import dump_json, get_logger, load_json

log = get_logger(__name__)


def stage_scenes(scene_ids, cfg: Cfg) -> dict:
    """Copy scene stacks to node-local NVMe (Prism: /lscratch) for fast random reads."""
    src_root = Path(cfg.paths.scenes)
    if not cfg.train.stage_to_local:
        return {s: str(src_root / s) for s in scene_ids}
    dst_root = Path(cfg.paths.local_scratch) / "ch4hsi_scenes"
    try:
        dst_root.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for s in scene_ids:
            d = dst_root / s
            if not (d / "meta.json").exists():
                shutil.copytree(src_root / s, d, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("mf.npy", "plume_id.npy"))
        log.info("staged %d scenes to %s in %.0fs", len(scene_ids), dst_root, time.time() - t0)
        return {s: str(dst_root / s) for s in scene_ids}
    except OSError as e:
        log.warning("staging to %s failed (%s); reading from %s", dst_root, e, src_root)
        return {s: str(src_root / s) for s in scene_ids}


def amp_dtype(mode: str):
    if mode == "off" or not torch.cuda.is_available():
        return None
    if mode == "bf16" or (mode == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


def masked_loss(logits, y, valid, pos_weight, dice_w):
    v = valid.float()
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none",
                                             pos_weight=torch.tensor(pos_weight, device=logits.device))
    bce = (bce * v).sum() / v.sum().clamp(min=1)
    p = torch.sigmoid(logits) * v
    inter = (p * y).sum()
    dice = 1 - (2 * inter + 1) / (p.sum() + (y * v).sum() + 1)
    return bce + dice_w * dice, bce.detach(), dice.detach()


def lr_at(epoch_f, tcfg):
    if epoch_f < tcfg.warmup_epochs:
        return tcfg.lr * max(epoch_f / max(tcfg.warmup_epochs, 1e-8), 0.02)
    t = (epoch_f - tcfg.warmup_epochs) / max(1, tcfg.epochs - tcfg.warmup_epochs)
    return tcfg.lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))


@torch.no_grad()
def validate(model, loader, device, adt):
    model.eval()
    h = HistPR(0, 1, 1000)
    tot, n = 0.0, 0
    for x, y, v in loader:
        x, y, v = x.to(device, non_blocking=True), y.to(device), v.to(device)
        with torch.autocast("cuda", dtype=adt, enabled=adt is not None):
            logits = model(x)
        logits = logits.float()
        loss, _, _ = masked_loss(logits, y, v, 1.0, 0.0)
        tot += float(loss) * len(x)
        n += len(x)
        h.update(torch.sigmoid(logits).cpu().numpy(), y.cpu().numpy() > 0.5, v.cpu().numpy())
    best = h.best_f1()
    return dict(val_loss=tot / max(n, 1), val_ap=h.average_precision(), val_f1=best["f1"],
                val_thr=best["threshold"], val_precision=best["precision"], val_recall=best["recall"])


def run(cfg: Cfg):
    tc = cfg.train
    out = run_dir(cfg)
    save_config(cfg, out / "config.yaml")
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    splits = load_json(Path(cfg.paths.splits) / "splits.json")["splits"]
    norm = load_json(Path(cfg.paths.splits) / "norm.json")
    dump_json(norm, out / "norm.json")
    dirs = stage_scenes(splits["train"] + splits["val"], cfg)
    ds_tr = TileDataset([dirs[s] for s in splits["train"]], norm, tc, "train", tc.seed)
    ds_va = TileDataset([dirs[s] for s in splits["val"]], norm, tc, "val", tc.seed)
    log.info("train scenes %d (%d with plumes) | val scenes %d", len(ds_tr.scenes), len(ds_tr.pos_scenes), len(ds_va.scenes))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adt = amp_dtype(tc.amp)
    nw = int(tc.num_workers)
    if "SLURM_CPUS_PER_TASK" in os.environ:
        nw = min(nw, int(os.environ["SLURM_CPUS_PER_TASK"]) - 1)
    nw = max(0, nw)
    dl_kw = dict(batch_size=tc.batch_size, num_workers=nw, pin_memory=device.type == "cuda")
    dl_va = DataLoader(ds_va, shuffle=False, persistent_workers=nw > 0, **dl_kw)

    raw_model = build_model(tc, len(norm["mean"])).to(device).to(memory_format=torch.channels_last)
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if n_gpus > 1:
        gpu_names = ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpus))
        log.info("multi-GPU enabled: using DataParallel across %d GPUs (%s)", n_gpus, gpu_names)
        model = torch.nn.DataParallel(raw_model)
    else:
        model = raw_model
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=adt == torch.float16)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=adt == torch.float16)
    start, best, bad = 0, -1.0, 0
    hist = []
    last = out / "last.pt"
    if last.exists():  # resume after preemption / time limit
        ck = torch.load(last, map_location=device, weights_only=False)
        raw_model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start, best, bad, hist = ck["epoch"] + 1, ck["best"], ck["bad"], ck["hist"]
        log.info("resumed from epoch %d (best val AP %.4f)", start, best)
    log.info("device=%s (gpus=%d) amp=%s workers=%d params=%.2fM", device, n_gpus, adt, nw, sum(p.numel() for p in raw_model.parameters()) / 1e6)

    steps = len(ds_tr) // tc.batch_size
    for epoch in range(start, tc.epochs):
        ds_tr.set_epoch(epoch)   # epoch is part of the per-tile RNG seed; workers are re-forked each epoch
        dl_tr = DataLoader(ds_tr, shuffle=False, drop_last=True, **dl_kw)
        model.train()
        t0, run_loss = time.time(), 0.0
        for it, (x, y, v) in enumerate(dl_tr):
            for g in opt.param_groups:
                g["lr"] = lr_at(epoch + it / steps, tc)
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y, v = y.to(device, non_blocking=True), v.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=adt, enabled=adt is not None):
                logits = model(x)
            loss, bce, dice = masked_loss(logits.float(), y, v, tc.bce_pos_weight, tc.dice_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            run_loss += float(loss)
        vm = validate(model, dl_va, device, adt)
        rec = dict(epoch=epoch, train_loss=run_loss / max(steps, 1), lr=opt.param_groups[0]["lr"],
                   time_s=time.time() - t0, **vm)
        hist.append(rec)
        improved = vm["val_ap"] > best
        if improved:
            best, bad = vm["val_ap"], 0
            torch.save(dict(model=raw_model.state_dict(), epoch=epoch, val=vm, in_ch=len(norm["mean"])), out / "best.pt")
        else:
            bad += 1
        torch.save(dict(model=raw_model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
                        best=best, bad=bad, hist=hist), last)
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        log.info("ep %3d | loss %.4f | val AP %.4f F1 %.3f (P %.3f R %.3f) | %.0fs %s", epoch, rec["train_loss"],
                 vm["val_ap"], vm["val_f1"], vm["val_precision"], vm["val_recall"], rec["time_s"], "*" if improved else "")
        if bad >= tc.early_stop_patience:
            log.info("early stopping (no val AP improvement for %d epochs)", bad)
            break
    dump_json(dict(best_val_ap=best, epochs_run=len(hist)), out / "train_summary.json")
