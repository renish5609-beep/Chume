from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ..chips import Chip, generic_stats
from ..config import load_config
from ..io import find_chip, load_chip, read_table
from .massbalance import add_massbalance, add_wind
from .plume import plume_features, wind_uv
from .site_context import add_site_context


def _one(row: pd.Series, cfg: dict) -> dict:
    cols, cc, fc = cfg["columns"], cfg["chips"], cfg["features"]
    sid = str(row[cols["id"]])
    out: dict = {cols["id"]: sid}
    path = find_chip(cfg["paths"]["chip_dir"], sid, cc["pattern"])
    if path is None:
        out["chip_missing"] = 1.0
        return out
    arr = load_chip(path)
    chip = Chip(arr, cc["band_order"], cc["scale"], cc["scl_bad"], cc.get("cldprb_bad", 40.0))
    out["chip_missing"] = 0.0
    if fc.get("generic_chip", True):
        out.update(generic_stats(chip))
    if fc.get("plume", True):
        u, v = wind_uv(row, cols)
        rng = np.random.RandomState(zlib.crc32(sid.encode()) % (2**31))
        out.update(plume_features(chip, u, v, cfg, rng))
    return out


def build(cfg: dict, n_jobs: int = -1) -> pd.DataFrame:
    cols = cfg["columns"]
    tr = read_table(cfg["paths"]["train_csv"], cols["id"])
    te = read_table(cfg["paths"]["test_csv"], cols["id"])
    tr["_split"], te["_split"] = "train", "test"
    df = pd.concat([tr, te], ignore_index=True)
    rows = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_one)(r, cfg) for _, r in df.iterrows())
    chipf = pd.DataFrame(rows)
    df = df.merge(chipf, on=cols["id"], how="left")
    uv = np.array([wind_uv(r, cols) for _, r in df.iterrows()], dtype=float)
    df = add_wind(df, uv[:, 0], uv[:, 1])
    if cfg["features"].get("massbalance", True):
        df = add_massbalance(df, cfg["features"]["plume_radii"])
    if cfg["features"].get("site_context", False):
        base = [c for c in df.columns if c.startswith(("p_", "hot_", "tai_", "g_"))]
        df = add_site_context(df, cols["site"], base)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--random_angle", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config, {"features": {"plume_random_angle": True}} if a.random_angle else None)
    df = build(cfg, a.n_jobs)
    tag = "rand" if cfg["features"].get("plume_random_angle") else "true"
    out = Path(a.out or Path(cfg["paths"]["processed"]) / f"features_{tag}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out} shape={df.shape}")


if __name__ == "__main__":
    main()
