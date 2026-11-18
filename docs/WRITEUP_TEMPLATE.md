# Write-up template (fill with real numbers only)

Title: **Chume: Wind-Aligned Plume Physics and Cross-Target Chaining for Observing Blocked Satellite Columns**

## 1. Project and team information
- Team name on Kaggle: [ ]
- Members (name, university, Kaggle username): [ ]
- Kaggle team / profile URL: [ ]
- Code: [GitHub URL, code only] / [Kaggle notebook URL]
- Final private leaderboard MCRMSE / rank: [fill after Nov 28]

## 2. Problem understanding
In 3 to 5 sentences: five atmospheric enhancement targets around industrial facilities, Sentinel-2 chips plus 29 auxiliary features, targets missing where the satellite column was blocked, MCRMSE metric, and test facilities unseen in training. State the central difficulty: **generalizing to new sites**. State the central hypothesis: **physically motivated, wind-aligned, within-scene contrasts transfer across sites better than generic image statistics.**

## 3. Data exploration, preprocessing, feature engineering, missing targets
- Dataset facts from DATA_NOTES: chip size, bands, scaling, cloud mask, wind variables and convention, number of sites.
- Target figure (`targets.png`): correlations and missingness. Explain the missingness (blocked or invalid retrievals), why blanks aren't zeros, and why each target is trained only on observed rows.
- Preprocessing: reflectance conversion, SCL cloud/shadow masking.
- Feature families (with the `regions.png` diagram): generic chip stats; derived SWIR indices (B12/B11 methane ratio, TAI, B12/B8A hot ratio, NDBI, haze); wind-aligned core/downwind/upwind/crosswind regions at 3 radii; mass-balance features; [embeddings, if used]. Cite Varon et al. 2021, Liu et al. 2021, and the IME literature (RESEARCH_BACKGROUND).

## 4. Baseline models and lessons learned
Table: mean predictor, tabular LGBM, + generic chip stats. Lesson: what generic chip stats buy you, and why they plateau under site shift (adversarial validation AUC, top drifting features).

## 5. Experiments and methodology
- Stage 1: per-target LightGBM and CatBoost, observed rows only, inner site-grouped early stopping, 3 seeds; final hyperparameters as a table.
- Target transforms chosen per target (from `metrics.json`).
- Stage 2: strict nested cross-target chain. Explain why nesting prevents leakage.
- Blend (NNLS) and skill gate, both nested. Report the gate weight per target and interpret it (for example "AAI w = 0.31: weak but real skill").
- Ablation table (`docs/ABLATIONS.md`) with mean ± std over 3 CV seeds, and `ablation.png`.
- **The key control:** true wind angle vs random angle. State the result and what it means.

## 6. Validation and results
- CV protocol: StratifiedGroupKFold by site (5 folds, stratified on target availability), repeated over 3 seeds; nested blending and gating.
- Final CV MCRMSE overall and per target; OOF scatter (`oof_scatter.png`).
- Table: CV vs public leaderboard for each submitted milestone; comment on agreement.
- Feature importance (`importance.png`): are the physics features near the top for CH4 and FIRMS?

## 7. Findings and limitations (include what failed)
- What worked, ranked by CV gain.
- What didn't work, with numbers.
- Limitations: Sentinel-2 methane sensitivity is limited to strong sources over homogeneous surfaces; single-scene (no multi-pass reference allowed); uncertain wind convention and chip geometry assumptions; AAI has no UV analogue in Sentinel-2; small data, so CV variance is high.
- Future work: multi-temporal references if allowed; physically simulated plume augmentation; uncertainty-aware gating.

## 8. Code and reproducibility
- Repo layout, `pip install -e .`, `bash scripts/run_all.sh`, seeds, hardware, runtime, and the Kaggle notebook link.
- **Disclosures:** libraries (numpy, pandas, scikit-learn, LightGBM, CatBoost, SciPy, matplotlib, [torch, torchgeo]) and pretrained models ([SSL4EO-S12 ResNet50 MoCo via TorchGeo, or "none"]). No external data used.
- Rules compliance statement: no external data, no site identification, no leaderboard probing, no metadata exploitation.

---

# Devpost description template

**Chume** predicts five blocked atmospheric columns (CH4, NO2, CO, aerosol index, fire/flare activity) around unseen industrial facilities from Sentinel-2 chips and auxiliary weather and terrain features.

**Key idea:** plumes are carried by wind. Instead of describing the whole scene, Chume measures *downwind minus upwind* contrasts of SWIR indices known to respond to methane (B12/B11) and to high-temperature sources (Thermal Anomaly Index). It adds mass-balance features (enhancement scales roughly with emission divided by wind speed), chains the five targets with a leak-free nested stack, and shrinks each target toward its mean by an out-of-fold fitted gate, so a target without real skill can't hurt the score.

**Validation:** site-grouped cross-validation repeated over 3 seeds, with nested blending and gating. A random-wind-angle control tests whether the wind alignment itself carries the signal.

**Results:** CV MCRMSE [ ] (baseline [ ]), private LB [ ]. See the research PDF for ablations, figures, and limitations.

**Built with:** Python, LightGBM, CatBoost, scikit-learn, SciPy, matplotlib[, PyTorch, TorchGeo].
**Links:** Kaggle profile/team [ ], code [ ], notebook [ ].
