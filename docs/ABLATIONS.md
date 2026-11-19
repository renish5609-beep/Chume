# Ablations (site-grouped 5-fold CV, mean ± std over seeds 42, 7, 2026)

Reference: mean predictor under the same grouped CV = **1.0092** (t_ch4 1.015, t_no2 1.010, t_co 1.007, t_aai 1.009, t_firms 1.005).

| Config | MCRMSE | std | t_ch4 | t_no2 | t_co | t_aai | t_firms |
|---|---|---|---|---|---|---|---|
| 01 tabular LGBM (29 raw + wind u/v/ws) | 0.94137 | 0.00131 | 1.0128 | 0.9836 | 0.9878 | 0.6517 | 1.0710 |
| 02 + generic chip stats | 0.93833 | 0.00167 | 1.0104 | 1.0104 | 0.9898 | 0.6512 | 1.0299 |
| 03 + wind-aligned plume (all ~400 feats) | 0.94874 | 0.00119 | 1.0158 | 1.0438 | 1.0051 | 0.6701 | 1.0089 |
| 04 plume, RANDOM wind angle (control) | 0.94789 | 0.00107 | 1.0128 | 1.0512 | 1.0042 | 0.6649 | 1.0064 |
| 05 + mass balance + transforms (all feats) | 0.94104 | 0.00061 | 1.0069 | 1.0199 | 1.0030 | 0.6677 | 1.0077 |
| 06 02-features + skill gate | 0.94053 | 0.00044 | 1.0168 | 1.0140 | 0.9960 | 0.6568 | 1.0191 |
| 07 ... + drop elevation/slope (adversarial pruning) | 0.93695 | 0.00045 | 1.0165 | 1.0068 | 1.0013 | 0.6564 | 1.0037 |
| 08 compact physics set (66 feats), regularized | 0.92174 | 0.00106 | 1.0077 | 1.0149 | 0.9876 | 0.6507 | 0.9478 |
| 09 compact, RANDOM wind angle (control) | 0.92248 | 0.00125 | 1.0088 | 1.0145 | 0.9883 | 0.6508 | 0.9500 |
| 10 compact + per-target transforms | 0.91403 | 0.00060 | 1.0033 | 0.9921 | 0.9879 | 0.6503 | 0.9365 |
| 11 compact + skill gate | 0.92439 | 0.00036 | 1.0155 | 1.0034 | 0.9913 | 0.6584 | 0.9533 |
| 12 compact+transforms, Huber loss | 0.91452 | 0.00098 | 1.0032 | 0.9925 | 0.9890 | 0.6506 | 0.9373 |
| 13 compact+transforms + SSL4EO-S12 embeddings | 0.91817 | 0.00148 | 1.0090 | 0.9720 | 1.0014 | 0.6525 | 0.9561 |
| 14 + CatBoost, NNLS blend | 0.91254 | 0.00123 | 0.9994 | 0.9911 | 0.9884 | 0.6464 | 0.9374 |
| 15 + strict cross-target chain  [Pick A] | 0.91252 | 0.00069 | 0.9980 | 0.9984 | 0.9895 | 0.6338 | 0.9429 |
| 16 + CatBoost + chain + blend | 0.91354 | 0.00134 | 0.9991 | 1.0031 | 0.9893 | 0.6325 | 0.9437 |
| 17 ... chain_include_self | 0.91490 | 0.00250 | 1.0035 | 1.0031 | 0.9903 | 0.6368 | 0.9408 |
| 18 full Chume (LGBM+Cat+chain+blend+gate) | 0.91871 | 0.00028 | 1.0114 | 0.9997 | 0.9925 | 0.6367 | 0.9532 |
| 19 10 + gate capped at w ≤ 1.0, LGBM only  [Pick B, conservative] | 0.91979 | 0.00008 | 1.0145 | 0.9964 | 0.9919 | 0.6515 | 0.9446 |

Sources: rows 01-05 from `python -m chume.ablate` (docs/BASELINES.md, docs/ABLATIONS_phase3.md); rows 06-19 from `python -m chume.experiment` (docs/EXPERIMENTS.md). Run-level numbers are in outputs/experiments.jsonl.
