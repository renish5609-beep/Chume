from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .train import run

ABLATIONS = [
    (
        "01_tabular",
        {
            "features": {"generic_chip": False, "plume": False, "massbalance": False},
            "models": {"chain": False, "gate": False, "blend": False, "catboost": False},
            "targets": {"transform": "none"},
        },
        "true",
    ),
    (
        "02_generic_chip",
        {
            "features": {"plume": False, "massbalance": False},
            "models": {"chain": False, "gate": False, "blend": False, "catboost": False},
            "targets": {"transform": "none"},
        },
        "true",
    ),
    (
        "03_plume",
        {
            "features": {"massbalance": False},
            "models": {"chain": False, "gate": False, "blend": False, "catboost": False},
            "targets": {"transform": "none"},
        },
        "true",
    ),
    (
        "04_plume_random",
        {
            "features": {"massbalance": False},
            "models": {"chain": False, "gate": False, "blend": False, "catboost": False},
            "targets": {"transform": "none"},
        },
        "rand",
    ),
    (
        "05_massbalance",
        {"models": {"chain": False, "gate": False, "blend": False, "catboost": False}},
        "true",
    ),
    ("06_chain", {"models": {"gate": False, "blend": False, "catboost": False}}, "true"),
    ("07_gate", {"models": {"blend": False, "catboost": False}}, "true"),
    ("08_full_chume", {}, "true"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default="docs/ABLATIONS.md")
    a = ap.parse_args()
    base = load_config(a.config)
    rows = []
    for name, over, feat in ABLATIONS:
        if a.only and name not in a.only:
            continue
        scores, per = [], []
        for s in base["cv"]["seeds"]:
            cfg = load_config(a.config, {**over, "seed": s})
            rep = run(cfg, str(Path(a.processed) / f"features_{feat}.parquet"), f"abl_{name}_s{s}")
            scores.append(rep["final"][0])
            per.append(rep["final"][1])
        targets = base["columns"]["targets"]
        rows.append(
            (
                name,
                float(np.mean(scores)),
                float(np.std(scores)),
                {t: float(np.mean([p[t] for p in per])) for t in targets},
            )
        )
        print(name, rows[-1][1], rows[-1][2])
    targets = base["columns"]["targets"]
    lines = [
        "| Config | MCRMSE | std | " + " | ".join(targets) + " |",
        "|" + "---|" * (3 + len(targets)),
    ]
    for name, m, sd, pt in rows:
        lines.append(
            f"| {name} | {m:.5f} | {sd:.5f} | " + " | ".join(f"{pt[t]:.5f}" for t in targets) + " |"
        )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("# Ablations (repeated site-grouped CV)\n\n" + "\n".join(lines) + "\n")
    Path(a.out).with_suffix(".json").write_text(json.dumps(rows, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
