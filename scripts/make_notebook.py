"""Build notebooks/chume_kaggle.ipynb: a self-contained Kaggle notebook.

The notebook re-creates the chume package from embedded source (code only, no data),
points the config at /kaggle/input/obscura, builds features, trains the final config,
and writes /kaggle/working/submission.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(p for p in (ROOT / "src" / "chume").rglob("*.py")) + [
    ROOT / "pyproject.toml",
    ROOT / "configs" / "base.yaml",
    ROOT / "configs" / "final.json",
]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


def main() -> None:
    files = {str(p.relative_to(ROOT)): p.read_text() for p in FILES}
    cells = [
        md(
            "# Chume: OBSCURA final submission\n\n"
            "Self-contained reproduction of the final Chume submission. The cell below writes the `chume` "
            "package (code only) to `/kaggle/working/chume`, installs it, builds chip features from "
            "`/kaggle/input/obscura`, trains the final configuration with site-grouped CV, and writes "
            "`submission.csv`.\n\n"
            "Hardware used for the reference run: 4 CPU cores, 15 GB RAM, no GPU. Feature build about 2 "
            "minutes per 4,439 chips with 4 workers; training about 3 minutes per seed (3 seeds). No internet access and no "
            "external data or pretrained weights are needed for the final configuration."
        ),
        code(
            "import json, os, pathlib\n"
            f"FILES = json.loads({json.dumps(json.dumps(files))})\n"
            "root = pathlib.Path('/kaggle/working/chume')\n"
            "for rel, txt in FILES.items():\n"
            "    p = root / rel\n"
            "    p.parent.mkdir(parents=True, exist_ok=True)\n"
            "    p.write_text(txt)\n"
            "print(len(FILES), 'files written')"
        ),
        code(
            "%cd /kaggle/working/chume\n"
            "!pip install -q --no-deps --no-build-isolation -e .\n"
            "import glob\n"
            "IN = sorted(glob.glob('/kaggle/input/*/train.csv'))[0].rsplit('/', 1)[0]\n"
            "print('input dir', IN)"
        ),
        code(
            "import yaml\n"
            "cfg = yaml.safe_load(open('configs/base.yaml'))\n"
            "cfg['paths'].update({'raw': IN, 'train_csv': f'{IN}/train.csv', 'test_csv': f'{IN}/test.csv',\n"
            "    'chip_dir': f'{IN}/images', 'sample_submission': f'{IN}/sample_submission.csv',\n"
            "    'processed': '/kaggle/working/processed', 'runs': '/kaggle/working/runs'})\n"
            "yaml.safe_dump(cfg, open('configs/kaggle.yaml', 'w'))\n"
            "!python -m chume.features.build --config configs/kaggle.yaml --n_jobs 4"
        ),
        code(
            "import json\n"
            "import numpy as np, pandas as pd\n"
            "from chume.config import load_config\n"
            "from chume.train import run\n"
            "over = json.load(open('configs/final.json'))\n"
            "subs, cvs = [], []\n"
            "for seed in (42, 7, 2026):  # the 3 CV seeds; test predictions are averaged\n"
            "    rep = run(load_config('configs/kaggle.yaml', {**over, 'seed': seed}),\n"
            "              '/kaggle/working/processed/features_true.parquet', f'final_s{seed}')\n"
            "    cvs.append(rep['final'][0])\n"
            "    subs.append(pd.read_csv(f'/kaggle/working/runs/final_s{seed}/submission.csv', dtype={'sample_id': str}))\n"
            "    print(seed, 'CV MCRMSE', round(rep['final'][0], 5), {k: round(v, 4) for k, v in rep['final'][1].items()})\n"
            "print('mean CV', np.mean(cvs), '+-', np.std(cvs))\n"
            "T = ['t_ch4', 't_no2', 't_co', 't_aai', 't_firms']\n"
            "sub = subs[0].copy()\n"
            "sub[T] = np.mean([s[T].values for s in subs], axis=0)"
        ),
        code(
            "import pandas as pd, shutil\n"
            "from chume.validate import validate_submission\n"
            "test = pd.read_csv(f'{IN}/test.csv', dtype={'sample_id': str})\n"
            "validate_submission(sub, test['sample_id'], 'sample_id',\n"
            "                    ['t_ch4', 't_no2', 't_co', 't_aai', 't_firms'])\n"
            "sub.to_csv('/kaggle/working/submission.csv', index=False)\n"
            "sub.head()"
        ),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / "notebooks" / "chume_kaggle.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({len(files)} embedded files)")


if __name__ == "__main__":
    main()
