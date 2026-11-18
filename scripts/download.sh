#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw
kaggle competitions download -c obscura -p data/raw
cd data/raw && for z in *.zip; do unzip -oq "$z" && rm "$z"; done
ls -la
