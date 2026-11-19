# Devpost description

**Chume** predicts five blocked atmospheric columns (CH4, NO2, CO, aerosol index, fire/flare activity)
around unseen industrial facilities from Sentinel-2 chips and auxiliary weather and terrain features.

**Key idea:** plumes come from point sources and are carried by wind. Instead of describing the whole
scene, Chume measures a compact set of physics features:

- thermal-anomaly and hot-pixel signals (TAI, B12/B8A) for flares;
- the B12/B11 methane ratio's core-vs-ring contrast;
- wind and mass-balance terms, since enhancement is diluted by wind speed.

Targets are modelled in signed-log space. A leak-free nested cross-target chain lets each gas borrow
strength from the others.

We also tested the hypothesis that *downwind-minus-upwind* contrasts carry the signal. A random-wind-angle
control showed they do not on this data (0.9225 vs 0.9217 CV, within noise). The gain comes from compact,
transferable physics features: using 66 of them instead of ~470 cut CV error by 0.015. Dropping
site-fingerprint terrain features (adversarial AUC 0.997) helped too.

**Validation:** site-grouped cross-validation repeated over 3 seeds, with nested blending, a
random-wind-angle control, and adversarial validation.

**Results:** CV MCRMSE **0.9125** (tabular + chip-stats baseline 0.9383; mean predictor 1.0092).
Public LB 1.1413 (Pick A, best CV) / 1.1300 (Pick B, conservative: gated, CV 0.9198) for the two final
picks (baseline 1.1550). The ~0.22 CV-to-LB gap comes from test sites that differ as a population
(adversarial AUC 0.98 even on the final features), not only as individual new facilities. Private LB [fill after Nov 28].
See the research PDF for ablations, figures and limitations.

**Built with:** Python, LightGBM, CatBoost, scikit-learn, SciPy, pandas, rasterio, matplotlib. PyTorch,
timm and the SSL4EO-S12 pretrained encoder were tested but are not used in the final submissions.

**Links:** Kaggle profile/team [ ], code [ ], notebook [ ].
