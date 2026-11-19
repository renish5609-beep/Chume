"""Run one config override over the 3 CV seeds and append the result to docs/EXPERIMENTS.md."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .train import run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--over", default="{}", help="JSON config override (applied after --over_file)")
    ap.add_argument("--over_file", default=None, help="JSON file with a base override")
    ap.add_argument("--features", default="data/processed/features_true.parquet")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seeds", default=None, help="comma list; default cv.seeds")
    ap.add_argument("--out", default="docs/EXPERIMENTS.md")
    a = ap.parse_args()
    from .config import deep_update

    over = json.loads(open(a.over_file).read()) if a.over_file else {}
    over = deep_update(over, json.loads(a.over))
    base = load_config(a.config, over)
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else base["cv"]["seeds"]
    targets = base["columns"]["targets"]
    scores, per = [], []
    for s in seeds:
        rep = run(load_config(a.config, {**over, "seed": s}), a.features, f"exp_{a.name}_s{s}")
        scores.append(rep["final"][0])
        per.append(rep["final"][1])
        print(a.name, s, round(rep["final"][0], 5), flush=True)
    m, sd = float(np.mean(scores)), float(np.std(scores))
    pt = {t: float(np.mean([p[t] for p in per])) for t in targets}
    out = Path(a.out)
    if not out.exists():
        out.write_text(
            "# Experiments (site-grouped CV, mean over seeds)\n\n| name | MCRMSE | std | "
            + " | ".join(targets)
            + " | seeds | override |\n|"
            + "---|" * (len(targets) + 5)
            + "\n"
        )
    with out.open("a") as f:
        f.write(
            f"| {a.name} | {m:.5f} | {sd:.5f} | "
            + " | ".join(f"{pt[t]:.4f}" for t in targets)
            + f" | {len(seeds)} | `{json.dumps(over)}` |\n"
        )
    print(f"{a.name}: {m:.5f} +- {sd:.5f}", json.dumps({t: round(v, 4) for t, v in pt.items()}))


if __name__ == "__main__":
    main()
