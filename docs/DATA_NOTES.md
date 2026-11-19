# DATA_NOTES (Phase 1 recon, 2026-11-18)

Sources: `data/raw/README.md` (organizer dictionary), `python -m chume.recon` (`docs/DATA_NOTES_auto.md`),
hand inspection of `outputs/recon/*.png`, and one-off checks recorded below. Only aggregate statistics
appear here; no rows, chips or targets are committed.

## Files

| File | Notes |
|---|---|
| `data/raw/train.csv` | 3,218 rows, 36 columns: `sample_id`, `site_key`, 5 targets, 29 features |
| `data/raw/test.csv` | 1,221 rows, 30 columns: `sample_id` + 29 features (no `site_key`) |
| `data/raw/sample_submission.csv` | `sample_id,t_ch4,t_no2,t_co,t_aai,t_firms` |
| `data/raw/images/<sample_id>.tif` | 4,439 GeoTIFFs, one per sample |
| `data/raw/README.md` | organizer data dictionary |

Download: 7.78 GB zip, ~9.2 GB unpacked. `configs/base.yaml > paths.chip_dir` changed `chips` -> `images`.

## Chips

- 384 x 384 px, uint16, **14 bands**: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12, **SCL, CLDPRB**.
  No B10. No georeferencing and no metadata (rasterio warns `NotGeoreferencedWarning`; now silenced).
- B1-B12 are L2A surface reflectance x1e-4 -> `chips.scale: 10000` is correct. Observed maxima up to ~18,000 DN
  on bright industrial roofs.
- SCL present (values 0-11). CLDPRB = s2cloudless probability 0-100. Chips are **not** cloud-masked.
- Config: `band_order` keeps the code's internal names (`B01 ... B12`, `SCL`) and appends `CLDPRB`; the
  mapping is positional and identical to the organizer order (B1=B01 etc.). Names were not changed to
  `B1` because `chips.py` keys its derived layers on `B01/B04/B08/B8A/B11/B12`.
- `chips.py` change: CLDPRB is popped out of the reflectance bands (it was being treated as a band) and
  pixels with CLDPRB >= `chips.cldprb_bad` (40, the same threshold the organizers use for
  `s2_cloud_frac`) are masked in addition to SCL classes 0, 1, 3, 8, 9, 10.
- `chips.pattern: "{id}.tif"`.
- **Footprint varies by site**: "Each site's frame was curated to its facility, so physical extent differs
  between sites and is not disclosed." Some chips are visibly upsampled (blurry). Pixel size is therefore
  not constant; plume radii are fractions of the chip half-width, which is the right unit here.
- **Facility at chip center: yes.** The contact sheet (`outputs/recon/contact_sheet_swir.png`) shows
  refineries / tank farms / plants centred in every frame. Core/ring geometry kept at the chip center.
- **North-up: unknown** (no georeferencing). Kept `north_up: true`; the random-angle control (04) decides.

## Numeric features (29)

`month_sin, month_cos, temperature, humidity, wind_speed, wind_direction, precipitation, solar_rad,
elevation, slope, s2_cloud_frac, ndvi, evi, savi, ndre, ndwi, mndwi, awei, nmdi, nbri, nbr2, bsi, bi,
swir_ratio, ndii, hsfi, ohi, b11_mean, b12_mean`.

- Weather (ERA5-Land, quantized), terrain (GLO-30 elevation/slope, quantized 50 m / 1 deg), and 18 chip-mean
  spectral indices computed by the organizers on the cloud-masked chip.
- Missing: spectral indices 0.1% in both splits; all others 0%.
- `month_sin/cos` are provided (month only, no day). They are used as given; no date-like features are derived.
- **Train vs test drift (KS statistic)**: elevation 0.34 (train mean 173 m vs test 461 m), bsi 0.19,
  ndii 0.19, nbri 0.17, slope 0.16, hsfi 0.14, humidity 0.13, s2_cloud_frac 0.12. Test sites are
  higher and more barren on average. Adversarial validation (Phase 5) will quantify this.

### Wind

- `columns.wind_speed: wind_speed`, `columns.wind_dir: wind_direction`,
  `wind_dir_convention: meteorological_from`. The dictionary says "Meteorological wind direction near
  overpass", i.e. the direction the wind blows **from**; the code converts to the wind-to vector.
- Speed 0-14 m/s (quantized 0.5), direction quantized to 10 deg. Calm (< 1 m/s): train 1.2%, test 2.8%.
- **Convention check (visual)**: overlaid the downwind/upwind cones on 30 true-colour chips (highest
  `t_aai`, wind >= 3 m/s, one per site; `outputs/recon/wind_overlay.png`). High-AAI scenes are dominated by
  regional dust haze, not discrete plumes, so no chip shows a plume clear enough to confirm or refute the
  direction. **Result: inconclusive; convention left as documented.** The random-angle control (04 vs 03)
  is the real test.

## Targets

All five are **z-scores** with train statistics (mean 0, std 1). Units hidden.

| target | observed | skew | notes |
|---|---|---|---|
| t_ch4 | 1,113 (34.6%) | -0.17 | missing 65.4%; 76/81 sites have at least one value |
| t_no2 | 3,130 (97.3%) | 3.04 | heavy right tail (max 7.6) |
| t_co | 3,188 (99.1%) | 1.70 | tails both sides (-6.7 .. 9.1) |
| t_aai | 3,218 (100%) | 0.72 | |
| t_firms | 3,218 (100%) | 1.98 | **82.6% sit at the floor -0.439 = "no detection"** (a real low value, not missing) |

- Pairwise correlations are weak: no2-co 0.35, ch4-firms -0.23, ch4-co 0.13, aai-firms 0.11, others ~0.
  The PDF's premise that combustion couples NO2/CO/AAI/FIRMS strongly is only weakly supported; the chain
  is a hypothesis to test.
- Availability patterns (ch4,no2,co,aai,firms): 01111 = 2,014, 11111 = 1,094, 00111 = 65, 01011 = 18,
  10111 = 15, 00011 = 8, 11011 = 4.
- Organizer note: the mean predictor scores ~1.0 in-distribution and **higher on held-out facilities**,
  because test sites are shifted. Targets are noisy relative to retrieval error.

## Sites

- Train: 81 sites, 4-72 rows per site (median 39). Test: no `site_key` (by design); README says test
  facilities never appear in train and come from **later, non-overlapping time stretches** (private test is
  the latest). So there are two shifts: site and time.
- `columns.site: site_key` (unchanged). CV is StratifiedGroupKFold by site (sklearn warns that the rarest
  availability pattern has 4 members; harmless).

## Leak checks (rule 6)

- Exact duplicate pixel arrays across all 4,439 chips: **0**.
- Near-duplicate scenes: standardized 24x24 RGB thumbnails, correlation of each test chip to its closest
  train chip. Within train, a chip's best match from a *different* site is > 0.9 for 0% of rows (same-site
  median 0.97). 32 test chips (2.6%) exceed 0.9, all matched to a single train site. Visual check
  (`outputs/recon/leak_pairs.png`): **different facilities** on similar flat desert backgrounds; the
  correlation is driven by the uniform sand gradient. **Not a leak.**
- `sample_id` is a 16-hex anonymized hash; not used as a feature (rule 3). No file metadata exists to exploit.
- Nothing found that looks like a leak.

## Contradictions with CHUME_MASTER.pdf (the data wins)

1. Chip folder is `images/`, not `chips/`; files are `.tif`.
2. 14 bands, not 13: an extra **CLDPRB** band after SCL. Handled in `chips.py`.
3. Targets are z-scores, so "enhancement scales with emission / wind speed" applies only up to an unknown
   affine map; the `ws_mult` transform multiplies a z-score (with an arbitrary zero) by wind speed, which
   is physically less meaningful than assumed. Kept as a candidate; CV decides.
4. `t_firms` is a zero-inflated variable with a floor at -0.439, not a smooth intensity.
5. Weak inter-target correlation (see above) weakens the chain's motivation.
6. Test differs from train in **time** as well as site; private test is the latest.
7. Chip physical extent differs per site (frames curated per facility), so pixel size is not constant.
8. Wind convention could not be confirmed visually (no visible discrete plumes).
