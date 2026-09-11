"""Command-line entry point:  ch4hsi <stage> [--config cfg.yaml] [--set key=value ...]"""
from __future__ import annotations

import argparse
import sys

from .config import load_config
from .utils import get_logger

STAGES = {
    "fetch-labels":   ("1  plume-complex catalogue (+ COG/GeoJSON assets)", "ch4hsi.catalog.plumes:run"),
    "resolve-scenes": ("2a match plumes to EMIT L1B scenes; sample plume-free negatives", "ch4hsi.catalog.scenes:run_resolve"),
    "download":       ("2b download L1B radiance (array-shardable)", "ch4hsi.catalog.scenes:run_download"),
    "fetch-enh":      ("   optional: official L2B CH4 enhancement for MF cross-check", "ch4hsi.catalog.scenes:run_fetch_enh"),
    "preprocess":     ("3  MF + features + ortho + labels per scene (array-shardable)", "ch4hsi.preprocess:run"),
    "mf-check":       ("   optional: compare our MF with EMIT L2B CH4ENH", "ch4hsi.crosscheck:run"),
    "split":          ("4  geo-blocked train/val/test split + normalisation stats", "ch4hsi.splits:run"),
    "train":          ("5  train U-Net", "ch4hsi.train:run"),
    "evaluate":       ("6  MF baseline vs model on test scenes", "ch4hsi.evaluate:run"),
    "diagnose":       ("6b diagnostic figure suite (curves, calibration, error maps, ...)", "ch4hsi.diagnostics:run"),
    "baseline-lr":    ("   pixel logistic-regression baseline, evaluated + diagnosed (CPU)", "ch4hsi.baselines:run"),
    "mdl":            ("7  synthetic-injection minimum detection limit (array-shardable)", "ch4hsi.mdl:run"),
    "mdl-fit":        ("7b merge MDL shards and fit POD curves", "ch4hsi.mdl:fit"),
    "report":         ("8  REPORT.md with tables, figures and resume bullets", "ch4hsi.report:run"),
    "aviris-fetch":   ("   experimental: AVIRIS-NG radiance for labelled flight lines", "ch4hsi.aviris:fetch"),
    "aviris-eval":    ("   experimental: cross-sensor evaluation on AVIRIS-NG", "ch4hsi.aviris:evaluate"),
    "make-fake-data": ("   write a tiny synthetic EMIT-format dataset for smoke tests", "ch4hsi.fake:run"),
}
SHARDABLE = {"download", "preprocess", "mdl"}


def _resolve(target):
    mod, fn = target.split(":")
    return getattr(__import__(mod, fromlist=[fn]), fn)


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", "-c", default=None, help="experiment YAML layered over configs/default.yaml")
    common.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE", help="override a config value")
    common.add_argument("--task-id", type=int, default=None, help="shard index (default $SLURM_ARRAY_TASK_ID)")
    common.add_argument("--num-tasks", type=int, default=None, help="number of shards (default $SLURM_ARRAY_TASK_COUNT)")
    ap = argparse.ArgumentParser(prog="ch4hsi", description="Hyperspectral methane plume detection (EMIT / AVIRIS-NG)")
    sub = ap.add_subparsers(dest="stage", required=True)
    for name, (help_, _) in STAGES.items():
        sub.add_parser(name, help=help_, parents=[common])
    sub.add_parser("show-config", help="print the resolved configuration", parents=[common])
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    log = get_logger()
    if a.stage == "show-config":
        import yaml
        print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        return 0
    log.info("stage %s | data_root=%s | run=%s", a.stage, cfg.paths.data_root, cfg.run_name)
    fn = _resolve(STAGES[a.stage][1])
    if a.stage in SHARDABLE:
        fn(cfg, a.task_id, a.num_tasks)
    else:
        fn(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
