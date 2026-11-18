# Chume research log

Rule (Section 5): a change is a win only if mean MCRMSE over CV seeds [42, 7, 2026] improves by more than the
seed-to-seed std. All CV is StratifiedGroupKFold by `site_key`, 5 folds. Numbers trace to
`outputs/experiments.jsonl` and `docs/BASELINES.md` / `docs/ABLATIONS.md`.

## 2026-11-18 — Bootstrap and recon

- Extracted starter repo from CHUME_MASTER.pdf; 7/7 tests pass; synthetic smoke run final MCRMSE 1.793
  (mean 3.657), matching the PDF's pre-flight table.
- Real data downloaded. Config fixed for the real layout (see DATA_NOTES): `images/*.tif`, 14 bands with an
  extra CLDPRB band (now a cloud mask, not a reflectance band), meteorological wind columns.
- Bug fix: the tabular-only ablation kept `clear_frac` (a chip statistic without the `g_` prefix). Fixed
  before any baseline was recorded.
- Leak checks: none found (DATA_NOTES, "Leak checks").
- Pretrained encoders confirmed allowed by the team (Kaggle Rules tab). SSL4EO-S12 ResNet50 MoCo
  checkpoint (`resnet50_sentinel2_all_moco-df8b932e.pth`, hf.co/torchgeo) loaded into a timm ResNet50
  with 13 input channels, all keys matched (torchgeo itself not installed; same weights).

## 2026-11-18 — Baselines (Phase 2)

| Config | MCRMSE (mean of 3 seeds) | std | t_ch4 | t_no2 | t_co | t_aai | t_firms |
|---|---|---|---|---|---|---|---|
| train mean (in-sample) | 1.0000 | - | | | | | |
| 01_tabular (LGBM, 29 raw features + wind u/v/ws) | 0.94137 | 0.00131 | 1.01277 | 0.98363 | 0.98775 | 0.65169 | 1.07102 |
| 02_generic_chip (+ per-band p10/p50/p90, SCL fractions) | 0.93833 | 0.00167 | 1.01039 | 1.01035 | 0.98984 | 0.65121 | 1.02987 |

- 02 beats 01 by 0.0030, more than the std (~0.0015): a small win, almost entirely FIRMS (1.071 -> 1.030).
  NO2 gets worse with chip stats (0.984 -> 1.010).
- Only AAI has real skill (0.65). CH4 and FIRMS are *worse than the mean* (> 1.0); NO2 and CO barely beat it.
  The gate (Phase 4) should shrink CH4/FIRMS toward the mean.
- Note: the "mean" row uses the full-train mean, so it is slightly optimistic; under site-grouped CV the
  mean predictor would score a little above 1.0.

### Kaggle submissions

| Date | Run | CV | Public LB | Note |
|---|---|---|---|---|
| 2026-11-18 | abl_02_generic_chip_s42 | 0.93760 (this seed) | 1.15504 | pipeline/format check (Section 4) |

- Public LB is far above CV (1.155 vs 0.938). Consistent with the README: test facilities are shifted
  relative to the train mean and come from a later time period. CV is still the selection criterion
  (Section 0), but CV-vs-LB gap must be watched per milestone.
