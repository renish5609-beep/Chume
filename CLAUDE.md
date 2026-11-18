# CHUME: Master Instructions for Claude Code

You are the lead engineer on **Chume**, our entry for the **OBSCURA: Satellite AI Research Challenge** (Kaggle technical leaderboard + Devpost research write-up). You own the whole thing end to end: setup, recon, features, models, validation, submissions, figures, and the write-up assets. Work autonomously. Stop only at the two checkpoints, or when a hard rule could be at risk.

A working starter codebase ships with this document: the `chume` Python package, configs, scripts, tests, and a synthetic data generator. It has already passed its unit tests and a full end-to-end run on synthetic data. Your job: point it at the real data, fix whatever the real data breaks, then improve the score with disciplined experiments.

## 0. Hard facts

- **Task:** for each sample, predict 5 atmospheric targets around an industrial facility: `t_ch4` (methane enhancement), `t_no2` (NO2 enhancement), `t_co` (CO enhancement), `t_aai` (absorbing aerosol index, smoke/dust), and `t_firms` (thermal fire/flare activity).
- **Data:** 4,439 samples (about 9.19 GB). 3,218 train, 1,221 test. Inputs are Sentinel-2 image chips plus 29 numeric features (weather, terrain, image-derived).
- **Missing targets:** some train targets are blank because the satellite measurement wasn't available. Blank is NOT zero. Never impute targets. Train each target only on rows where it was observed.
- **Metric:** MCRMSE, masked mean column-wise RMSE. For each target, compute RMSE over rows with ground truth, then average the 5. Lower is better. Implemented in `src/chume/metric.py`.
- **Generalization:** test facilities never appear in train. Validation MUST be grouped by site. Implemented in `src/chume/cv.py`.
- **Submission:** CSV `sample_id,t_ch4,t_no2,t_co,t_aai,t_firms`. All values finite. `sample_id` stays text (`dtype=str`). Run `python -m chume.validate` before every upload.
- **Ranking:** the private leaderboard decides. Never select models by public leaderboard when it disagrees with CV.
- **Deadline:** Nov 24, 2026, 8:00 PM PKT = **Nov 24, 9:00 AM US Central**. Internal target: final picks locked by **Nov 23, 11:59 PM Central**.
- **Kaggle slug:** `obscura`.

## 1. Rules you must never break

1. No external satellite, weather, air-quality, methane, fire, elevation, or other data. Only the provided data, plus pretrained models where Kaggle's rules allow them (see 7.4).
2. Never try to identify or reconstruct real facilities, locations, or acquisition dates. No geolocating chips. No reverse-engineering metadata.
3. Never exploit file metadata, row order, ID patterns, or file names. Never probe the leaderboard. Never hand-label test rows.
4. Never commit or publish competition data, chips, or targets. `.gitignore` already covers `data/` and `outputs/`. If the repo is made public, it holds code only.
5. Disclose every pretrained model and library in the write-up.
6. If something looks like a leak, STOP, tell the human, and don't use it. Leaks get reported privately to the organizers.
7. Borderline techniques (site-context features in 7.6, pretrained encoders before the rules are confirmed) stay behind config flags that default to OFF.

## 2. Bootstrap (do this first)

1. **Unpack the code.** The full repo is embedded in this PDF as the attachment `chume_repo.zip`. Extract it with:
   ```
   pip install pypdf
   python - <<'EOF'
   from pypdf import PdfReader
   r = PdfReader("CHUME_MASTER.pdf")
   for name, data in r.attachments.items():
       open(name, "wb").write(data[0])
   EOF
   unzip chume_repo.zip -d chume && cd chume
   ```
   If the attachment can't be read, recreate every file from Appendix A of the PDF exactly as printed.
2. `git init` and commit the starter code. Commit after every meaningful step from here on.
3. Python 3.10+. `pip install -e ".[dev]"`. Add `.[geo]` if the chips are GeoTIFF, and `.[emb]` later if pretrained encoders are allowed.
4. `pytest -q`: all 7 tests must pass.
5. Smoke test on synthetic data:
   `python scripts/make_synthetic.py --out data_syn/raw && python -m chume.features.build --config configs/synthetic.yaml && python -m chume.train --config configs/synthetic.yaml --features data_syn/processed/features_true.parquet --run_id smoke`
6. Kaggle credentials: you need `~/.kaggle/kaggle.json`. If it's missing, ask the human for it. Then run `bash scripts/download.sh`.

## 3. Phase 1: Recon, then CHECKPOINT 1

Run `python -m chume.recon`. Then inspect by hand and write `docs/DATA_NOTES.md` covering:

- All files, their formats, and the real names of the train/test CSVs and chip folder. Update `configs/base.yaml > paths` to match.
- Chip format, shape, dtype, band count and order, and value scaling. Is there an SCL band? Update `chips.band_order`, `chips.scale`, and `chips.pattern`. If band names come with the data, use them exactly.
- All 29 numeric columns: meaning, missing %, train vs test drift. Find the **wind** columns and set `columns.wind_u/wind_v`, or `columns.wind_speed/wind_dir` plus `wind_dir_convention` (`meteorological_from`, `meteorological_to`, or `math_to`). Find the **site/group** column and set `columns.site`.
- Wind convention check: on ~30 chips with strong SWIR/haze signals, overlay the computed downwind cone (`chume.features.plume.region_masks`) and confirm it points along any visible plume or smoke. If it's backward, flip the convention. If there are no wind columns at all, set them to null; plume features then fall back to radial core-vs-ring only. Record which case applies.
- Is the facility at the chip center? Check `outputs/recon/contact_sheet_swir.png`. If not, the core/ring geometry is wrong: center the regions on the brightest persistent SWIR hotspot per chip instead, and document it.
- Is the image north-up? If unknown, keep `north_up: true` and rely on the random-angle control (8.5) to show whether alignment helps.
- Targets: distribution, skew, outliers, missingness patterns, and correlations between targets.
- Sites: count in train and test, rows per site, and whether test rows have the site key.
- Anything that looks like a leak (rule 6).

**CHECKPOINT 1:** message the human with a short DATA_NOTES summary, the config changes you made, and every place where the data contradicts this document. Then continue without waiting, unless a hard rule is at stake.

## 4. Phase 2: Baselines

Run these with `chume.ablate --only` or by setting config flags, and log every result:

1. Train mean per target, which `train.py` reports as `stages.mean`.
2. Tabular-only LightGBM (`01_tabular`).
3. Tabular plus generic chip stats (`02_generic_chip`). This matches the public competitor baseline, so Chume must clearly beat it.

Make at most one early Kaggle submission (baseline 3) to confirm the pipeline and format work end to end.

## 5. Experiment discipline

- Every run writes `outputs/runs/<run_id>/` (config, OOF, submission, metrics, importances) and appends to `outputs/experiments.jsonl`. Don't bypass this.
- Keep `docs/RESEARCH_LOG.md` in plain English: date, what changed, the CV result (overall plus per target), and why. Log failures too. The write-up is built from this log.
- A change is a **win** only if mean MCRMSE across the 3 CV seeds (`cv.seeds`) improves by more than the seed-to-seed std. Otherwise revert it.
- One idea at a time. Measure each block before stacking the next.

## 6. The Chume idea, in one paragraph

These targets are plumes from point sources moved by wind. Generic whole-chip statistics mostly measure land cover, which changes from site to site and transfers badly to unseen facilities. Chume builds physically motivated, wind-aligned contrasts (downwind minus upwind) on Sentinel-2 SWIR indices that are known to respond to methane (B12/B11) and to high-temperature sources (TAI, B12/B8A). It adds mass-balance features, because enhancement scales roughly with emission divided by wind speed. It chains the 5 targets, since combustion drives NO2, CO, AAI, and FIRMS together. And it shrinks each target toward its mean by an amount fit out-of-fold, so a target with no real skill can't hurt the score.

## 7. Phase 3: Features (code already in `src/chume/features/`)

### 7.1 Chip preprocessing (`chips.py`)
Band mapping, conversion to reflectance, cloud/shadow mask from SCL (classes 0, 1, 3, 8, 9, 10 count as bad) or a brightness fallback. Derived layers:
- `r_ch4 = B12/B11` and `r_ch4_norm` (divided by the chip's clear-sky median) for methane SWIR absorption.
- `tai = (B12 - B11)/B8A` (Thermal Anomaly Index) for flares and heat.
- `hot = B12 - B8A` and `hot_ratio = B12/B8A`.
- `ndvi`, `ndbi` (built-up), `haze = B01/B04`, plus raw `b01` and `b12`.

### 7.2 Wind-aligned plume features (`plume.py`)
Using polar coordinates relative to the wind-to direction, at radii 0.10/0.25/0.45 of the chip half-width: core disk, plus downwind (±30°), upwind, and crosswind cones. Per layer: mean/p90 per region, down-up, down-cross, core-ring, and a z-scored down-up. Also hot-pixel counts at thresholds, TAI counts, clear fractions per region, and an `is_calm` flag (wind speed below 1 m/s disables the directional contrasts). Tune the radii and half-angle only after recon, using CV.

### 7.3 Mass-balance features and target transforms (`massbalance.py`)
`ws`, `1/ws`, `log ws`; heat proxies × `1/ws`; methane down-up contrast × `ws`. Per-target transform search (`targets.transform: auto`): none, `y·ws`, and signed log. Always score in the original target space.

### 7.4 Pretrained embeddings (`embeddings.py`, flag `features.embeddings`)
**First** read the Kaggle competition Rules tab on external models. If they're allowed and publicly available, use torchgeo `ResNet50_Weights.SENTINEL2_ALL_MOCO` (SSL4EO-S12, 13 bands). Inputs must be reflectance (DN/10000), with bands mapped to the 13-band order (missing B10 set to zero). Run `python -m chume.features.embeddings`, which adds 16 PCA dims each for the global and center-crop embeddings (PCA fit on train rows only). Never fine-tune end to end as the main model. If the rules are unclear, leave this OFF and ask the human.

### 7.5 Tabular
Use the raw provided features. Never derive date-like features that could encode acquisition time.

### 7.6 Site context (`site_context.py`, flag `features.site_context`, default OFF)
The per-site mean of the chip features, plus each row's deviation from it, computed within the same split and using features only. Measure the CV gain and report it at CHECKPOINT 2. The human decides whether to ask the organizers. Never enable it without written approval.

## 8. Phase 4: Models (code already in `src/chume/models/` and `train.py`)

### 8.1 Stage 1
LightGBM and CatBoost per target, each trained only on observed rows. Early stopping uses an inner site-grouped fold, never the validation fold. 3 model seeds. Shallow, regularized trees (see `lgbm_params`). Optional light Optuna tuning (at most 40 trials per target) on grouped CV. Write it as `src/chume/tune.py` if time allows, and keep the tuned params only if repeated CV confirms the gain.

### 8.2 Stage 2: strict cross-target chain (`chain.py`)
For each outer fold, stage-1 predictions for the training rows come from inner CV that never touches the outer validation fold, and predictions for the validation fold come from models that never saw it. That makes it leak-free by construction. Target k gets the original features plus the stage-1 predictions of the other 4 targets. Ablate `chain_include_self`. On synthetic data the chain alone did NOT beat plain LGBM (Part D), so treat it as a hypothesis: keep it only if repeated CV confirms. Try CatBoost as the chain's base learner and more seeds before you drop it.

### 8.3 Blend (`blend.py`) and skill gate (`gate.py`)
Per target: non-negative least-squares weights over {lgbm, catboost, chain}. Then the gate `final = mean + w·(pred - mean)`, with w in [0, 1.2] fit on OOF. Both are evaluated with **nested** fold-wise fitting, so the reported CV is honest. Report each target's w; a w near 0 means that target is effectively a mean prediction, so say so honestly.

### 8.4 Robustness extras (try in this order, keep what wins)
1. Feature pruning: drop the top adversarial-drift features (`python -m chume.adversarial`) one group at a time.
2. Huber/Fair loss for heavy-tailed targets, kept only if original-space RMSE improves.
3. Monotone constraints where the physics is clear (for example, FIRMS should increase with hot-pixel count).
4. A small MLP on embeddings plus tabular features as an extra blend member.
5. Per-target radius and half-angle selection for the plume features.

### 8.5 The key control
`04_plume_random` recomputes the plume features with a random wind angle. If true-angle features don't beat random-angle ones, wind alignment adds nothing beyond radial structure. Report the result either way; it's the most important scientific result in the write-up. On synthetic data, true angle won for CH4 (6.93 vs 7.60 RMSE).

## 9. Phase 5: Ablations, adversarial validation, figures

- `python -m chume.ablate` produces `docs/ABLATIONS.md` and `.json`, using 3 CV seeds per config: tabular, generic chip, plume, plume-random, mass-balance, chain, gate, full Chume. Add rows for embeddings and site context if you test them.
- `python -m chume.adversarial` gives the train-vs-test AUC and the most drifting features.
- `python -m chume.figures --run outputs/runs/<best>`: region diagram, target correlation/missingness, OOF scatter, feature importance, ablation bars. Add one figure of real rotated chips with region overlays, with no coordinates or identifying information.

## 10. Phase 6: Submissions, then CHECKPOINT 2

- Use `bash scripts/submit.sh outputs/runs/<id> "message"`. It validates before uploading. Respect the daily limit. Log CV and public score for every upload in RESEARCH_LOG.
- **CHECKPOINT 2** (by Nov 23 noon Central): send the human the ablation table, CV vs public leaderboard per milestone, the gate weights, the site-context result, and the 2 recommended final picks:
  1. The best mean repeated-CV Chume model.
  2. The most robust option: lowest CV std, or a more conservative gate/blend.
- Select the finals on Kaggle only after the human confirms, or by Nov 24, 6:00 AM Central if there's no reply.

## 11. Phase 7: Reproducibility and write-up

- `notebooks/chume_kaggle.ipynb`: a self-contained Kaggle notebook that installs or imports the package code (paste the modules into cells or attach them as a Kaggle dataset of code only), reads `/kaggle/input/obscura/...`, and reproduces the final submission. State the hardware and runtime.
- `README.md`: install, download, `bash scripts/run_all.sh`, expected CV, seeds, runtime. Pin `requirements.txt` with `pip freeze`.
- Fill `docs/WRITEUP_TEMPLATE.md` (Part E of the PDF) with real numbers only, then export it to PDF. Every number must trace to `outputs/experiments.jsonl` or `docs/ABLATIONS.md`.
- Fill the Devpost description template (Part E).

## 12. Time budget (Central time)

| When | Deliverable |
|---|---|
| Wed Nov 18 | Bootstrap, smoke test, data download, recon, CHECKPOINT 1 |
| Thu Nov 19 | Config fixed for real data, baselines 1 to 3, first submission |
| Fri Nov 20 | Plume + mass-balance validated, random-angle control, transforms |
| Sat Nov 21 | Embeddings (if allowed), chain, blend, gate, robustness extras |
| Sun Nov 22 | Full ablations, adversarial, figures, Kaggle notebook |
| Mon Nov 23 | CHECKPOINT 2, final picks, write-up PDF, Devpost text |
| Tue Nov 24, 9:00 AM | Hard deadline |

## 13. If things break

- Chips are huge or slow: cache derived layers as float16 npy in `data/interim/`; `features.build` already parallelizes with joblib (`--n_jobs`).
- Memory: build features in chunks of 500 rows and concat.
- A target has almost no observed rows: the gate pushes it toward the mean, which is fine.
- CV and public leaderboard disagree: trust CV. Check fold balance and adversarial drift.
- CatBoost is slow: reduce iterations, or drop it from stage 1 for the experiments and add it back for the final run.
- Anything in this document contradicts the real data: the data wins. Adapt, document it in DATA_NOTES, and tell the human at the next checkpoint.

Start now with Section 2.
