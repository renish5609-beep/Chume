# Runbook: commands, checkpoints, strategy

## Before anything (human)

- [ ] **Eligibility:** OBSCURA requires currently enrolled college/university students above the age of majority in their country of residence (18 in the US). Confirm with the organizers before investing the week if anyone on the team is under 18.
- [ ] Every team member is registered on Kaggle (one account each) and has accepted the competition rules.
- [ ] The Kaggle team is formed, and the Devpost project lists the exact same people.
- [ ] Get `kaggle.json` (Kaggle > Settings > API > Create New Token) and put it in `~/.kaggle/kaggle.json` with `chmod 600`.
- [ ] Read the Kaggle Rules tab for the external/pretrained model policy, and tell Claude Code what it says.
- [ ] Join the BTAJI WhatsApp community / Kaggle discussion for announcements.

## Handing this to Claude Code (human)

1. Make an empty folder and put `CHUME_MASTER.pdf` in it.
2. Open Claude Code in that folder and paste:

> Read CHUME_MASTER.pdf completely. It is your full instruction set. Extract the embedded chume_repo.zip attachment as described in Part A, Section 2, copy CLAUDE.md to the repo root, and execute the plan from Section 2 onward. Stop only at the two checkpoints or if a hard rule is at risk.

3. Give it `kaggle.json` when it asks.
4. Answer CHECKPOINT 1 (around Wed/Thu) and CHECKPOINT 2 (Mon Nov 23, by noon).

## Command reference

| Purpose | Command |
|---|---|
| Install | `pip install -e ".[dev]"` (plus `.[geo]` for GeoTIFF, `.[emb]` for encoders) |
| Tests | `pytest -q` |
| Synthetic smoke test | `python scripts/make_synthetic.py --out data_syn/raw` then build and train with `configs/synthetic.yaml` |
| Download data | `bash scripts/download.sh` |
| Recon | `python -m chume.recon` |
| Build features | `python -m chume.features.build` and `python -m chume.features.build --random_angle` |
| Add embeddings | `python -m chume.features.embeddings` (only if allowed) |
| Train full Chume | `python -m chume.train --run_id <name>` |
| Different CV seed | `python -m chume.train --seed 7 --run_id <name>_s7` |
| Ablations | `python -m chume.ablate` (or `--only 03_plume 04_plume_random`) |
| Adversarial validation | `python -m chume.adversarial` |
| Figures | `python -m chume.figures --run outputs/runs/<name>` |
| Validate submission | `python -m chume.validate outputs/runs/<name>/submission.csv` |
| Submit | `bash scripts/submit.sh outputs/runs/<name> "msg"` |
| Everything | `bash scripts/run_all.sh` |

## Outputs produced per run

`outputs/runs/<id>/`: `config.json`, `folds.csv`, `oof.csv`, `submission.csv`, `metrics.json` (per-stage scores, blend weights, gate weights per fold, chosen transforms), and `importance_lgbm.csv`. `outputs/experiments.jsonl` gets one line per run.

## Pre-flight validation already done

The starter code was run end to end on a synthetic dataset (520 samples, 48×48 chips, 13 bands with SCL, wind given as speed plus meteorological direction, a site-grouped split, and random target masking). Results:

- 7/7 unit tests pass: masked metric, grouped folds, downwind geometry, meteorological wind conversion, synthetic plume detection, gate shrinkage, and submission validator.
- Feature build: 464 columns.
- Adversarial AUC about 0.53 (no train/test drift, as expected for synthetic data).
- Submission validated (columns, row count, string IDs, finite values).
- Ablations on synthetic data: see the table in Part D. The key control behaved as designed: wind-aligned plume features beat the random-angle control on CH4.

These numbers only prove the code works. They say nothing about real-data performance.

## Final submission strategy

1. **Pick A:** the best mean MCRMSE across 3 CV seeds.
2. **Pick B:** a conservative variant, such as gate weights capped at 1.0, fewer blend members, or the most stable fold std.
3. Never choose by public leaderboard. With about 1.2k test rows, the public/private split is noisy.
4. Select both on Kaggle before Nov 24, 6:00 AM Central (deadline is 9:00 AM Central).

## Risk register

| Risk | Sign | Response |
|---|---|---|
| Wind convention backward | Random-angle control beats true angle | Flip `wind_dir_convention`, re-check overlays |
| Chips not centered on the site | Core features have no importance; hot pixels off-center | Re-center on the persistent SWIR hotspot |
| Pretrained models disallowed | Rules tab | Keep embeddings OFF; nothing else depends on them |
| Site context disallowed | Organizer reply | Keep OFF (default) |
| Heavy-tailed target dominates the metric | One target's RMSE far above the others | Try a transform, Huber, and gate; check outliers in DATA_NOTES |
| Overfitting via tuning | CV gain smaller than seed std | Revert; keep the default params |
| Running out of time | Behind the time budget | Skip embeddings and extras; ship plume + mass balance + chain + gate |
| Data leak noticed | Anything too good to be true | Stop, report privately, don't use it |
