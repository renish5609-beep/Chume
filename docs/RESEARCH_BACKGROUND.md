# Research Background: why Chume's features should work

This section gives the physical reasoning behind each feature family, so the write-up can explain why each one exists rather than just reporting importances. References are numbered and listed at the end.

## 1. What the five targets physically are

| Target | What the measurement is | What drives it near an industrial site | Chume's best proxies |
|---|---|---|---|
| `t_ch4` | Methane column enhancement over background | Leaks, venting, and incomplete flaring at oil/gas and industrial facilities. Diluted by wind. | B12/B11 SWIR ratio contrast downwind vs upwind; mass balance (× wind speed) |
| `t_no2` | NO2 column enhancement | High-temperature combustion (flares, furnaces, power generation). Short lifetime, so it's local. | Heat/flare proxies (TAI, hot pixels), built-up fraction, chained FIRMS/CO predictions |
| `t_co` | CO column enhancement | Incomplete combustion (flaring, fires, smelting). Longer lifetime, so it also carries a regional background. | Flare proxies, chained NO2/FIRMS, weather |
| `t_aai` | UV absorbing aerosol index | Smoke and dust; positive values mean UV-absorbing aerosol. Very sensitive to clouds. | Haze/coastal-band proxies (B01/B04), flare activity, clear-sky fraction |
| `t_firms` | Thermal fire/flare activity (FIRMS-style active-fire signal) | Active flares and fires at the site | TAI and hot-pixel counts in the core, max hot ratio |

**Why some targets are blank ("blocked columns").** The atmospheric products are retrieved only for scenes that pass quality and cloud screening. AAI in particular is known to be strongly affected by clouds [5]. A blank therefore means "not retrievable," not "zero." Chume trains each target only on observed rows, and the metric ignores test rows with no truth, so models only need to be right under observable conditions.

## 2. Methane in Sentinel-2 SWIR

Sentinel-2 MSI has two SWIR bands: B11 (about 1560 to 1660 nm) and B12 (about 2090 to 2290 nm). Methane absorbs much more in B12 than in B11. Varon et al. (2021) [1] showed that combining the two bands, especially against a non-plume reference, can detect and quantify large methane point sources from Sentinel-2. Their best method, multi-band multi-pass (MBMP), worked down to about 3 t/h over favorable, homogeneous surfaces. Zhang et al. (2022) [2] extended this with multi-date comparisons and calibrated it against a controlled-release test.

**What Chume borrows:** we can't use other dates or any external imagery, so we use the single-scene spatial version of the same idea. The B12/B11 ratio, normalized by the chip's clear-sky median, is compared downwind versus upwind and crosswind of the site. A methane plume should lower the ratio downwind relative to upwind. Surface albedo differences affect both directions, so the contrast cancels much of the land-cover signal that confuses generic statistics.

**Honest limitation:** Sentinel-2 is only sensitive to strong sources over bright, homogeneous surfaces. Many samples will have no visible plume. The skill gate and the target's mean fallback handle this, and the write-up should state it plainly.

## 3. Flares and high-temperature sources

Gas flares burn at very high temperatures (above roughly 1300 K), so their emission peaks in the SWIR and shows up strongly in B12, and to a lesser degree B11 [3]. Liu et al. (2021) [4] proposed a tri-spectral Thermal Anomaly Index for Sentinel-2 that combines the two SWIR bands with the NIR band B8A, which barely responds to high temperatures. Simpler ratios like B12/B8A and B12/B11 above 1 are classic active-fire indicators.

**What Chume borrows:** `tai = (B12 - B11)/B8A`, `hot_ratio = B12/B8A`, counts of pixels above several thresholds in the core and across the whole chip, and the max value. These should drive `t_firms` directly and `t_no2`, `t_co`, and `t_aai` indirectly through combustion. Bright rooftops are a known source of false positives [3], which is why the clear-sky mask and the built-up index (NDBI) are included as context.

## 4. Absorbing aerosol index

TROPOMI's AAI compares the observed UV spectral contrast at two wavelengths (340/380 nm or 354/388 nm) against a purely Rayleigh-scattering atmosphere. Positive residues indicate absorbing aerosol such as dust or smoke [6, 7]. Sentinel-2 has no UV bands, so Chume can only proxy aerosol through the coastal/aerosol band B01, haze ratios (B01/B04), and the combustion signals. Expect AAI to be the hardest target. If its gate weight comes out near 0, that's a correct finding, and it should be reported.

## 5. Mass balance: why wind speed matters

For a steady point source, the integrated mass enhancement (IME) method gives the emission rate as Q = U_eff · IME / L, where IME is the total excess mass in the plume, L is the plume length scale, and U_eff is an effective wind speed [8, 9]. Rearranged: **the enhancement you observe scales roughly with emission / wind speed.** Higher wind dilutes the plume and lowers the observed column. Large-eddy-simulation work found that the IME depends nonlinearly and strongly on background wind speed [10], which is why both wind-scaled features and wind-scaled targets are tested rather than assumed.

**What Chume borrows:**
1. Features like `heat_proxy / ws` (expected enhancement given a source of that strength) and `methane_contrast × ws` (an emission-like quantity).
2. A target transform: train on `y × ws` (closer to emission, which may vary more smoothly across sites), then divide by ws at prediction time. CV decides per target.
3. Calm-wind handling: below about 1 m/s, the plume direction is undefined, so directional contrasts are masked out and radial (core vs ring) features take over.

## 6. Why wind alignment helps generalization to unseen sites

Gradient-boosted trees on whole-chip percentiles learn what kind of place a site is: desert, farmland, urban. With test facilities never seen in training, that knowledge transfers poorly. A downwind-minus-upwind contrast is a within-scene difference, so it largely removes the site's own background. That's exactly the shift problem this competition sets. The random-angle control (computing the same features with a random direction) is what separates a real plume effect from plain center-vs-surroundings structure.

## 7. Cross-target structure

Combustion produces NO2, CO, aerosol, and thermal signal together, and methane often co-occurs with flaring at oil and gas sites. When targets are correlated and differently missing, the predictions for one carry information about the others. A strict stacked chain (stage-1 predictions of the other targets as features, built with nested folds so nothing leaks) exploits this. It also helps a target whose labels are sparse.

## 8. Pretrained Sentinel-2 encoders

SSL4EO-S12 [11] is a large self-supervised Sentinel-1/2 pretraining dataset. Its ResNet and ViT weights, including MoCo, DINO, and MAE variants for all 13 Sentinel-2 bands, are distributed through TorchGeo [12] (for example `ResNet50_Weights.SENTINEL2_ALL_MOCO`). Frozen embeddings compressed with PCA give the GBDT a compact description of the scene without the overfitting risk of fine-tuning on 3.2k samples. Use them only if the Kaggle rules allow pretrained models, and disclose them.

## 9. Validation theory for this setting

- **Group shift:** random K-fold would leak site identity and badly overstate skill. GroupKFold on site mimics the test condition.
- **Small data:** with about 3.2k rows and 5 targets, fold assignment noise is large. Use repeated CV (3 fold seeds) and treat improvements smaller than the seed std as noise.
- **RMSE and heavy tails:** RMSE punishes large misses on extreme rows, and confident wrong predictions on unseen sites are the main way to lose. Shrinking toward the mean (the gate) is the RMSE-optimal response to low skill.
- **Leaderboard shake-up:** with about 1.2k test rows split into public and private, the public leaderboard is a noisy estimate. Select by CV.

## References

1. Varon, D. J., Jervis, D., McKeever, J., Spence, I., Gains, D., and Jacob, D. J. (2021). High-frequency monitoring of anomalous methane point sources with multispectral Sentinel-2 satellite observations. *Atmos. Meas. Tech.*, 14, 2771 to 2785. doi:10.5194/amt-14-2771-2021
2. Zhang, Z., Sherwin, E. D., Varon, D. J., and Brandt, A. R. (2022). Detecting and quantifying methane emissions from oil and gas production: algorithm development with ground-truth calibration based on Sentinel-2 satellite imagery. *Atmos. Meas. Tech.*, 15, 7155. doi:10.5194/amt-15-7155-2022
3. Faruolo, M., et al. (2022, 2023). Daytime Approach for gas Flaring Investigation (DAFI) using Landsat 8/9 OLI and Sentinel-2 MSI. Project site: sites.google.com/view/flaringsitesinventory
4. Liu, Y., Zhi, W., Xu, B., Xu, W., and Wu, W. (2021). Detecting high-temperature anomalies from Sentinel-2 MSI images. *ISPRS J. Photogramm. Remote Sens.*, 177, 174 to 193.
5. Kooreman, M. L., et al. (2020). Effects of clouds on the UV Absorbing Aerosol Index from TROPOMI. *Atmos. Meas. Tech.*, 13, 6407 to 6426. doi:10.5194/amt-13-6407-2020
6. ESA Sentinel-5P Level-2 Aerosol Index technical guide. sentinel.esa.int, Sentinel-5P Level-2 Aerosol Index.
7. Google Earth Engine catalog, COPERNICUS/S5P/OFFL/L3_AER_AI (UV Aerosol Index, 354/388 nm).
8. Varon, D. J., et al. (2018). Quantifying methane point sources from fine-scale satellite observations of atmospheric methane plumes. *Atmos. Meas. Tech.*, 11, 5673 to 5686.
9. Frankenberg, C., et al. (2016). Airborne methane remote measurements reveal heavy-tail flux distribution in Four Corners region. *PNAS*, 113, 9734 to 9739.
10. Jongaramrungruang, S., Frankenberg, C., et al. (2019). Towards accurate methane point-source quantification from high-resolution 2-D plume imagery. *Atmos. Meas. Tech.*, 12, 6667 to 6681.
11. Wang, Y., et al. (2022). SSL4EO-S12: A large-scale multi-modal, multi-temporal dataset for self-supervised learning in Earth observation. arXiv:2211.07044
12. Stewart, A. J., et al. (2022). TorchGeo: Deep learning with geospatial data. arXiv:2111.08872
