#!/usr/bin/env bash
set -euo pipefail
RUN=${1:?usage: submit.sh <run_dir> "<message>"}
MSG=${2:-chume}
python -m chume.validate "$RUN/submission.csv"
kaggle competitions submit -c obscura -f "$RUN/submission.csv" -m "$MSG"
