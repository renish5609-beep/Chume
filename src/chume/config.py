from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import yaml


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "inherit" in cfg:
        parent = load_config(path.parent / cfg.pop("inherit"))
        cfg = deep_update(parent, cfg)
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
