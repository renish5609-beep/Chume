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

## 2026-11-18/19 — Phase 3: plume, random-angle control, mass balance (full feature set)

| Config | MCRMSE | std | t_ch4 | t_no2 | t_co | t_aai | t_firms |
|---|---|---|---|---|---|---|---|
| 03_plume | 0.94874 | 0.00119 | 1.01578 | 1.04383 | 1.00508 | 0.67009 | 1.00892 |
| 04_plume_random | 0.94789 | 0.00107 | 1.01278 | 1.05116 | 1.00423 | 0.66489 | 1.00639 |
| 05_massbalance | 0.94104 | 0.00061 | 1.00693 | 1.01991 | 1.00298 | 0.66769 | 1.00767 |

- Adding all ~400 plume columns made things **worse** than 02 (0.9487 vs 0.9383): FIRMS improved
  (1.030 -> 1.009) but NO2, CO and AAI got worse. With ~65 training sites per fold, hundreds of correlated
  chip features let the trees memorize site appearance.
- **Key control:** random wind angle (0.9479) is *not worse* than the true angle (0.9487). Wind alignment
  adds nothing measurable beyond radial core/ring structure here.
- Honest reference: the mean predictor under the same site-grouped CV scores **1.0092**, not 1.000 (the
  `stages.mean` number in metrics.json uses the full-train mean). Between-site share of target variance:
  ch4 0.32, no2 0.46, co 0.34, aai 0.57, firms 0.64.
- Adversarial validation (full feature set): train-vs-test AUC **0.997**. Top drift: elevation, slope,
  scl_6 (water), nmdi, then many plume/SWIR features. Elevation/slope are near-constant per site, i.e.
  site fingerprints.
- Univariate (Spearman, train): NO2 vs wind speed |rho| 0.32 (dilution); FIRMS vs `tai_n_pos_all` 0.25,
  `tai_max` 0.22; AAI vs median reflectance ~0.6 (bright, dusty scenes).

## 2026-11-19 — Phase 4: compact model, transforms, gate, chain, embeddings

All rows: 3 seeds, `python -m chume.experiment` (docs/EXPERIMENTS.md).

- Gate on 02 features: 0.94053 vs 0.93833 without -> **gate hurts**. Reverted.
- Drop elevation/slope (adversarial pruning, step 8.4.1): 0.93695 (gain 0.0036 >> std 0.0005). **Win**, kept.
- Stronger regularization (num_leaves 7, min_data_in_leaf 80, feature_fraction 0.5, lr 0.02): 0.93755
  vs 0.94053 with gate. Kept.
- Clip to train range: identical (no prediction left the range). No effect.
- **Compact physics set** (66 features: 27 raw minus elevation/slope, band medians, 4 SCL fractions,
  hot-pixel/TAI counts and maxima, a few core/ring and down-up SWIR contrasts, wind and mass-balance
  terms; `configs/compact.json`): **0.92174** (-0.0152 vs 0.93695). **Big win**, FIRMS 1.004 -> 0.948.
- Random angle on the compact set: 0.92248 vs 0.92174 (difference 0.0007 < std 0.001). Control
  confirms: no measurable wind-alignment effect.
- Per-target transform search (`auto`): **0.91403** (-0.0077). Signed log chosen for all five targets
  (heavy tails). **Win**, kept.
- Gate on compact: 0.92439 vs 0.92174. Hurts again. The gate shrinks toward the training mean, but the
  error is dominated by site-level shifts of the mean itself, so shrinkage removes useful site signal.
- Huber loss: 0.91452 vs 0.91403. No gain, reverted.
- SSL4EO-S12 embeddings (+32 PCA dims): 0.91817, worse overall (NO2 improves 0.992 -> 0.972, CO/FIRMS worsen).
  Not kept.
- + CatBoost (depth 4, l2 10) and NNLS blend: **0.91254 ± 0.00123** (gain 0.0015 > std). Win -> first Pick B (later replaced, see below).
- + strict chain (LGBM base), blend with LGBM: **0.91252 ± 0.00069**. Win, AAI 0.650 -> 0.634 -> Pick A.
- LGBM + CatBoost + chain + blend: 0.91354 (3-member NNLS blend overfits slightly). Not better.
- chain_include_self: 0.91490 ± 0.0025. Worse.
- Full Chume with gate: 0.91871. Worse.

## 2026-11-19 — Submissions and final picks

| Date | Submission | CV (3 seeds) | Public LB |
|---|---|---|---|
| 2026-11-18 | baseline 02_generic_chip (seed 42) | 0.93833 | 1.15504 |
| 2026-11-19 | Pick A: compact + slog + chain, 3-seed avg | 0.91252 | 1.14134 |
| 2026-11-19 | Pick B: compact + slog + LGBM/CatBoost blend, 3-seed avg | 0.91254 | 1.13569 |

- CV and public LB move in the same direction (CV -0.026, LB -0.014 to -0.019). The absolute gap (~0.23)
  matches the README's warning that new facilities and a later time period shift every target.
- A and B are tied on CV; A has the lower std, B uses a different learner (CatBoost) and is better on the
  public split. Keeping both as finals hedges the CV-vs-LB uncertainty. Not choosing by public LB.
- Site context (7.6): **not applicable**. Test rows have no `site_key`, so site-level features cannot be
  computed for test without inferring site membership, which rule 2 forbids. Not built, not used.
- Kaggle notebook (`notebooks/chume_kaggle.ipynb`) reproduces Pick A's submitted file exactly
  (max abs diff 0.0), 3 min 52 s on 4 CPUs.

## 2026-11-19 — Pick B replaced by a conservative variant

- Old Pick A (chain, 0.91252 ± 0.00069) and old Pick B (CatBoost blend, 0.91254 ± 0.00123) share features,
  transforms and LightGBM settings; their CV differs by 0.00002 and test predictions correlate 0.92-0.99
  per target. They are effectively the same model, so B gave little hedge.
- New Pick B (`configs/final_B.json`): LightGBM only (1 blend member instead of 2) and the skill gate with
  its weight capped at 1.0 (new config key `models.gate_w_max`). CV **0.91979 ± 0.00008** (CH4 1.0145,
  NO2 0.9964, CO 0.9919, AAI 0.6515, FIRMS 0.9446). Worse than A by 0.0073 but the most stable model
  tested. Gate weights (seeds 42/7/2026): CH4 0.03/0.14/0.09 (near-mean prediction), NO2 0.65/0.57/0.67,
  CO 1.00/0.90/0.70, AAI ~1.0, FIRMS 1.0.
- Uploaded: public LB **1.13004**, the best public score so far despite worse CV. Consistent with the site
  shift: shrinking the weakest target (CH4) toward the mean costs little on CV but is safer at new sites.
  Selection still follows the plan: A = best CV, B = conservative.
- Adversarial validation re-run on the final 66-feature set: AUC **0.979** (full 447-feature set 0.997).
  Top drift on the final set: nmdi, scl_6 (water), p_r_ch4_norm_cr0.25, p_hot_ratio_core_max, g_B01_p50.

| Date | Submission | CV (3 seeds) | Public LB |
|---|---|---|---|
| 2026-11-19 | Pick B (conservative): LGBM only, gate w <= 1.0, 3-seed avg | 0.91979 | 1.13004 |
