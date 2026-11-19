# Chume

Compact physics features (thermal-anomaly and SWIR indices, wind and mass-balance terms) and a strict
cross-target chain for the OBSCURA Satellite AI Research Challenge (Kaggle `obscura`): predicting five
satellite columns (CH4, NO2, CO, aerosol index, FIRMS fire/flare) around unseen industrial sites from
Sentinel-2 chips and weather/terrain features.

No competition data, chips or targets are stored in this repo.

## Results (site-grouped 5-fold CV, mean ± std over seeds 42, 7, 2026)

| Model | CV MCRMSE | Public LB |
|---|---|---|
| Mean predictor (same grouped CV) | 1.0092 | |
| Baseline: tabular + generic chip stats | 0.93833 ± 0.00167 | 1.15504 |
| **Pick A**: compact physics + signed-log targets + strict chain (`configs/final.json`) | **0.91252 ± 0.00069** | 1.14134 |
| **Pick B** (conservative): compact physics + signed-log targets, LightGBM only, skill gate capped at w ≤ 1.0 (`configs/final_B.json`) | 0.91979 ± 0.00008 | 1.13004 |
| (earlier Pick B, superseded) LGBM/CatBoost NNLS blend (`configs/cat_blend.json`) | 0.91254 ± 0.00123 | 1.13569 |

Full ablation table: `docs/ABLATIONS.md`. Research log: `docs/RESEARCH_LOG.md`. Data notes: `docs/DATA_NOTES.md`.
Write-up: `docs/WRITEUP.md` / `docs/WRITEUP.pdf`.

## Install

Python 3.10+ (developed on 3.13).

```bash
pip install -e ".[dev,geo]"
```

Exact versions used: `requirements.txt`. The embeddings experiment (row 13, not used in the final
submissions) also needs `torch` and `timm`, plus the SSL4EO-S12 checkpoint in `data/weights/`
(see `src/chume/features/embeddings.py`).

## Reproduce

```bash
pytest -q
```

```bash
bash scripts/download.sh          # needs Kaggle credentials and accepted competition rules
```

```bash
bash scripts/run_all.sh           # recon, features, adversarial, ablations, final picks A and B, figures
```

```bash
bash scripts/submit.sh outputs/runs/final_A_chain "Pick A"
```

Each final pick trains with the three CV seeds and averages the three test predictions.
Runtime on 4 CPU cores / 15 GB RAM, no GPU: feature build ~2 min per pass (4 workers), Pick A ~1 min per
seed, Pick B ~1 min per seed, the full `run_all.sh` about 1 hour (mostly the full-feature ablations 03-05).

Kaggle: `notebooks/chume_kaggle.ipynb` is self-contained (the package source is embedded; generate it with
`python scripts/make_notebook.py`). It reads `/kaggle/input/obscura`, rebuilds features, trains Pick A over
the three seeds and writes `submission.csv`. Checked locally against the submitted file: identical
predictions, 3 min 52 s end to end on 4 CPUs.

## Layout

- `src/chume/`: `chips` (bands, SCL + CLDPRB cloud mask, derived layers), `features/` (plume, massbalance,
  embeddings, site_context, build), `models/` (gbdt, chain, blend, gate), `train`, `experiment`, `ablate`,
  `adversarial`, `figures`, `recon`, `validate`, `metric`, `cv`.
- `configs/`: `base.yaml` (real data), `compact.json` (66-feature physics set), `final.json` (Pick A),
  `final_B.json` (Pick B, conservative), `cat_blend.json` (ablation row 14), `synthetic.yaml`.
- `docs/`: data notes, research log, ablations, experiments, figures, write-up.
- `scripts/`: download, run_all, submit, make_notebook, synthetic data generator.

## Disclosures

Libraries: numpy, pandas, pyarrow, scikit-learn, SciPy, LightGBM, CatBoost, rasterio/tifffile, matplotlib,
PyYAML, joblib. Tested but not in the final submissions: PyTorch, timm, and the SSL4EO-S12 ResNet50 MoCo
checkpoint (published by TorchGeo on Hugging Face). No external data.
