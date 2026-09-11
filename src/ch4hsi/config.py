"""Configuration loading: YAML + ${ENV:-default} expansion + dotted CLI overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def _expand(s: str) -> str:
    """Expand ${VAR} and ${VAR:-default}; defaults may themselves contain ${...} (nested braces)."""
    out, i = [], 0
    while i < len(s):
        j = s.find("${", i)
        if j < 0:
            out.append(s[i:])
            break
        out.append(s[i:j])
        depth, k = 1, j + 2
        while k < len(s) and depth:
            if s.startswith("${", k):
                depth, k = depth + 1, k + 2
                continue
            if s[k] == "}":
                depth -= 1
            k += 1
        if depth:                       # unbalanced: leave the rest untouched
            out.append(s[j:])
            break
        inner = s[j + 2:k - 1]
        name, sep, default = inner.partition(":-")
        val = os.environ.get(name)
        out.append(val if val not in (None, "") else (_expand(default) if sep else ""))
        i = k
    return "".join(out)


def _walk(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _walk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk(v) for v in obj]
    if isinstance(obj, str):
        return _expand(obj)
    return obj


class Cfg(dict):
    """dict with attribute access (recursive)."""

    def __getattr__(self, k):
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return v

    def __setattr__(self, k, v):
        self[k] = v

    @staticmethod
    def wrap(obj):
        if isinstance(obj, dict):
            return Cfg({k: Cfg.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Cfg.wrap(v) for v in obj]
        return obj

    def to_dict(self):
        def un(o):
            if isinstance(o, dict):
                return {k: un(v) for k, v in o.items()}
            if isinstance(o, list):
                return [un(v) for v in o]
            return o
        return un(self)


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _set_dotted(d: dict, key: str, value: str):
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = yaml.safe_load(value)


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def load_config(path: str | os.PathLike | None = None, overrides: list[str] | None = None) -> Cfg:
    """Load default.yaml, then an optional experiment YAML on top, then `a.b=c` overrides."""
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    if path and Path(path).resolve() != DEFAULT_CONFIG:
        with open(path) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like key.sub=value, got {ov!r}")
        k, v = ov.split("=", 1)
        _set_dotted(cfg, k, v)
    cfg = _walk(cfg)
    root = Path(cfg["paths"]["data_root"])
    for sub in ("catalog", "raw", "scenes", "splits", "runs"):
        if not cfg["paths"].get(sub):
            cfg["paths"][sub] = str(root / sub)
    return Cfg.wrap(cfg)


def save_config(cfg: Cfg, path: str | os.PathLike):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)


def run_dir(cfg: Cfg) -> Path:
    d = Path(cfg.paths.runs) / cfg.run_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def copy_cfg(cfg: Cfg) -> Cfg:
    return Cfg.wrap(copy.deepcopy(cfg.to_dict()))
