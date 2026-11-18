# Chume

Wind-aligned plume physics plus a cross-target chain for the OBSCURA Satellite AI Research Challenge (Kaggle `obscura`).

## Quick start

```
pip install -e ".[dev]"
pytest -q
bash scripts/download.sh          # needs ~/.kaggle/kaggle.json
bash scripts/run_all.sh           # recon, features, adversarial, train, validate, ablations, figures
bash scripts/submit.sh outputs/runs/chume_full "chume full"
```

Before the first real run, set the column names, chip pattern, band order, and wind convention in `configs/base.yaml` (see `docs/DATA_NOTES.md`).

## Layout

- `src/chume/` holds the package: `chips`, `features/` (plume, massbalance, embeddings, site_context, build), `models/` (gbdt, chain, blend, gate), `train`, `ablate`, `adversarial`, `figures`, `recon`, `validate`.
- `configs/` holds `base.yaml` and `synthetic.yaml`.
- `docs/` holds the research background, runbook, write-up template, and the ablation and data notes.
- `scripts/` holds download, run_all, submit, and the synthetic data generator.

No competition data is stored in this repo.
