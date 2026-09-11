"""Small shared helpers: logging, sharding, JSON I/O."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np


def get_logger(name: str = "ch4hsi") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.propagate = False           # every module logger has its own handler; avoid duplicate lines
        log.setLevel(os.environ.get("CH4HSI_LOGLEVEL", "INFO"))
    return log


def shard(items: list, task_id: int | None, num_tasks: int | None) -> list:
    """Round-robin shard for Slurm array jobs. task_id defaults to $SLURM_ARRAY_TASK_ID."""
    if task_id is None:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if num_tasks is None:
        num_tasks = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    return items[task_id::num_tasks]


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, cls=_NpEncoder)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


class Timer:
    def __init__(self, label: str, log: logging.Logger | None = None):
        self.label, self.log = label, log or get_logger()

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.log.info("%s: %.1fs", self.label, time.time() - self.t0)
