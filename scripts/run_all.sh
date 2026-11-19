#!/usr/bin/env bash
# Full Chume pipeline on the real data (after bash scripts/download.sh).
set -euo pipefail
CFG=${1:-configs/base.yaml}
python -m pytest -q tests
python -m chume.recon --config "$CFG"
python -m chume.features.build --config "$CFG"
python -m chume.features.build --config "$CFG" --random_angle
python -m chume.adversarial --config "$CFG"
# Phase-2/3 ablations on the full feature set (rows 01-05 of docs/ABLATIONS.md)
python -m chume.ablate --config "$CFG" --only 01_tabular 02_generic_chip --out docs/BASELINES.md
python -m chume.ablate --config "$CFG" --only 03_plume 04_plume_random 05_massbalance --out docs/ABLATIONS_phase3.md
# Final picks A and B (rows 15 and 19) plus row 14, 3 CV seeds each
python -m chume.experiment compact_chain --over_file configs/final.json
python -m chume.experiment compact_cat_blend --over_file configs/cat_blend.json
python -m chume.experiment pickB_conservative --over_file configs/final_B.json
python - <<'PY'
import numpy as np, pandas as pd, pathlib
for name, src in [("final_A_chain", "exp_compact_chain"), ("final_B_conservative", "exp_pickB_conservative")]:
    subs = [pd.read_csv(f"outputs/runs/{src}_s{s}/submission.csv", dtype={"sample_id": str}) for s in (42, 7, 2026)]
    out = subs[0].copy(); T = out.columns[1:]
    out[T] = np.mean([s[T].values for s in subs], axis=0)
    d = pathlib.Path(f"outputs/runs/{name}"); d.mkdir(parents=True, exist_ok=True)
    out.to_csv(d / "submission.csv", index=False)
PY
python -m chume.validate outputs/runs/final_A_chain/submission.csv
python -m chume.validate outputs/runs/final_B_conservative/submission.csv
python -m chume.figures --run outputs/runs/exp_compact_chain_s42 --ablations docs/ABLATIONS.json --out docs/figures
