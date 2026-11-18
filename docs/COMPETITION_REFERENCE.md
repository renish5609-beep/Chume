# Competition reference (from the official Devpost rules and overview)

## Format
- Organizer: BTAJI Crew (a student-led AI and data science community). Kaggle hosts the technical competition; Devpost hosts registration and the research documentation.
- Both are required for prizes and certificates: a valid Kaggle submission AND a complete Devpost project (title, description, team info, Kaggle profile URL, code references, reproducibility details, research PDF). No video required.

## Eligibility and teams
- Currently enrolled students at recognized colleges or universities who are above the age of majority in their country of residence, and who meet Kaggle's age rules. Organizers may ask for proof of enrollment and identity.
- Teams of up to 20 eligible students; cross-university teams are allowed. One Kaggle account per person, one team per person. Kaggle and Devpost team membership must match exactly.

## Data rules (summary)
- Don't identify or reconstruct real facilities, locations, acquisition dates, or hidden targets.
- No outside satellite, weather, air-quality, methane, fire, elevation, or other prohibited data used to identify samples or recover answers.
- No manual labeling of test rows, leaderboard probing, or exploiting metadata, identifiers, or row order.
- Pretrained models must satisfy the official Kaggle rules and be disclosed in the write-up.
- Don't redistribute the dataset, chips, tables, or targets outside Kaggle. No private sharing of code or predictions between teams. Public code must not expose data.
- Report potential leaks privately to the Kaggle organizers; don't exploit them.
- Winners must provide reproducible code and documentation on request.

## Evaluation and judging
- Metric: Masked Mean Column-wise RMSE (lower is better). The private leaderboard decides the technical ranking.
- Judging also reviews research quality and reproducibility (clarity of model, preprocessing, training, validation, and results) and documentation and rules compliance.
- Judges listed: Muhammad Ibrahim Qasmi (Kaggle 3x Grandmaster), Zulqarnain Ali, Tayyab Sajjad.

## Prizes
- 1st $25, 2nd $15, 3rd $10 (per entry, not per member). The pool may grow if sponsors join. Digital participation certificates go to every eligible student who completes both submissions.

## Timeline (PKT, GMT+5) and US Central equivalents
| Event | PKT | US Central |
|---|---|---|
| Launch | Oct 24, 8:00 PM | Oct 24, 10:00 AM |
| Devpost submissions open | Nov 18 | Nov 17/18 |
| **Final deadline** | **Nov 24, 8:00 PM** | **Nov 24, 9:00 AM** |
| Winners announced | Nov 28, 8:00 PM | Nov 28, 9:00 AM |

## Suggested write-up sections (organizers)
Project and team info; problem understanding; data exploration, preprocessing, feature engineering, missing targets; baselines and lessons; experiments and methodology; validation and results (site-grouped validation, MCRMSE, charts, leaderboard); findings and limitations including unsuccessful approaches; code and reproducibility.
