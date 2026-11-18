#!/usr/bin/env bash
set -euo pipefail
CFG=${1:-configs/base.yaml}
python -m pytest -q tests
python -m chume.recon --config "$CFG"
python -m chume.features.build --config "$CFG"
python -m chume.features.build --config "$CFG" --random_angle
python -m chume.adversarial --config "$CFG"
python -m chume.train --config "$CFG" --run_id chume_full
python -m chume.validate outputs/runs/chume_full/submission.csv
python -m chume.ablate --config "$CFG"
python -m chume.figures --run outputs/runs/chume_full
