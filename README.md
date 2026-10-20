# CardioTwin | TechBuilders

**Happiest Health Digital Twin Challenge 2026 — Public repository submission**

> **Research prototype only.** All benchmark patients, longitudinal measurements and event outcomes are synthetic. The simulated five-year cardiovascular event probability is **not** a clinically validated estimate, diagnosis, or near-term emergency prediction.

## Team and institution

| Field | Details |
|---|---|
| Team | **TechBuilders** |
| College | **Manipal Institute of Technology, Manipal** |
| Incubator | Not specified / not claimed |
| Team members and individual roles | **To be filled by team before submission** |
| Team leader | **To be filled by team before submission** |
| GitHub owner | `aniketh-234` (intended; repository not yet published) |
| Demo video (minimum 20 minutes) | **ADD VIDEO LINK AFTER RECORDING** |

## Project title
**CardioTwin — Explainable Cardiovascular Digital Twin with Hybrid CNN and Trend Review**

## Problem statement and healthcare use case
Static EHR-style records do not describe recent day-to-day changes. CardioTwin demonstrates a research workflow that links historical risk factors with 28 days of synthetic daily physiological measurements, updates a virtual patient's rolling state as new days are ingested, and shows an inspectable simulated five-year cardiovascular event estimate alongside separate trend-review reasons. The intended conceptual use case is clinician review of longitudinal changes, **not clinical deployment or real patient triage**.

## Technical stack and ML framework
- **Python / PyTorch**: neural model training and inference (MLP, 1D CNN, LSTM, GRU).
- **scikit-learn**: logistic static/fusion baselines, preprocessing and evaluation.
- **NumPy / pandas**: synthetic data generation and tabular processing.
- **Python standard-library HTTP backend**: endpoints, input validation, inference orchestration and data-quality gating.
- **SQLite**: persistent synthetic patient state and event journal.
- **HTML / CSS / JavaScript**: interactive browser dashboard, ingestion, replay, trend inspection, explanations and what-if comparison.

The **original validation-selected model is logistic fusion** (lowest validation Brier score). **Hybrid CNN + rules** is the interactive demonstration mode, not the proven best model. Rules never alter the CNN probability. Explanations are reference-replacement sensitivities, **not SHAP or causal effects**.

## Required submission artifacts

| Deliverable | Repository path |
|---|---|
| Full implementation and checkpoints | `backend/`, `frontend/`, `ml/`, `artifacts/`, `data/` |
| Detailed architecture diagram (PPT) | [`docs/architecture_detailed.pptx`](docs/architecture_detailed.pptx) |
| Detailed architecture diagram (PDF) | [`docs/architecture_detailed.pdf`](docs/architecture_detailed.pdf) |
| Project presentation (PPT) | [`docs/CardioTwin_Hackathon_Presentation.pptx`](docs/CardioTwin_Hackathon_Presentation.pptx) |
| Original project presentation (PDF; older baseline) | `docs/presentation.pdf` — historical reference only |
| Model and data cards | `docs/MODEL_CARD.md`, `docs/DATA_CARD.md` |
| Reproduction / API documentation | `docs/VERIFICATION.md`, `docs/API.md` |
| Open-source license | [`LICENSE`](LICENSE) — MIT, copyright CardioTwin contributors 2026 |
| Video | **PENDING: add unlisted video URL here after recording** |

## Key measured outcomes (synthetic held-out test, 800 patients / 177 simulated events)

| Review policy | Detected events | False alerts | Missed events |
|---|---:|---:|---:|
| CNN, 50% threshold | 80 | 39 | 97 |
| CNN 50% OR rules | 89 | 78 | 88 |
| CNN, validation-matched lower threshold | 101 | 80 | 76 |

The rules increase event detection relative to CNN at 50%, but also increase false alerts. The exploratory matched-alert comparison **does not establish incremental predictive value from rules**. See `artifacts/hybrid_evaluation.json`.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m backend.server
```

Open `http://127.0.0.1:8000/`. Use synthetic demo patients only. See the original technical documentation below for training, evaluation, testing and limitations.

## License and third-party notices
Original project source and generated sandbox data are distributed under the included **MIT License**. Third-party dependencies retain their own licenses; no external Synthea archive is redistributed here.

---

# Detailed technical documentation (from the supplied project)

# CardioTwin

## Hybrid prototype update — 16 October 2026

### Exploratory comparison at similar validation alert rates

Run `python -m ml.evaluate_hybrid` followed by `python -m ml.compare_review_policies` to reproduce the evidence shown in the dashboard. The second command adds a comparison to `artifacts/hybrid_evaluation.json` and saves patient-level outputs in `artifacts/review_comparison_patients.json`.

The CNN threshold of **0.3793189525604248** was chosen solely to match the hybrid's **153/800 validation alerts**, without using outcome labels in threshold selection. Tied candidate alert counts favor fewer alerts, then a higher threshold. Models, data and live inference settings are unchanged.

| Existing test set | Alerts | Events detected | False alerts | Missed events | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN at 50% | 119 | 80 | 39 | 97 | 45.2% | 67.2% | 0.541 |
| CNN at 50% OR rules | 167 | 89 | 78 | 88 | 50.3% | 53.3% | 0.517 |
| CNN at validation-matched threshold | 181 | 101 | 80 | 76 | 57.1% | 55.8% | 0.564 |

These results do not demonstrate added predictive value from the trend rules for the five-year synthetic target. The rules remain a separate transparent trend-review feature. Validation rates match exactly; test alert counts differ. This is exploratory: the test set was already inspected before this comparison was proposed, and validation was previously used for CNN early stopping. The threshold is not a clinical operating point. No new independent validation is claimed.

The browser now starts in **Hybrid · CNN + rules** mode. It uses the existing CNN checkpoint and unchanged dataset/labels, alongside an independent rule trace. The original validation-selected winner remains logistic fusion; the hybrid demo is not a new validation winner.

- Trend checks compare the latest seven days with the preceding 21, requiring at least 5 and 14 observed values respectively. At least two directional shifts of 0.5 training standard deviations flag review. This is a fixed engineering heuristic, not a clinical threshold.
- Missing-data rules take precedence and withhold predictions. Rule flags never modify the CNN probability.
- Explainability includes 20 sensor/week reference-replacement differences, historical feature sensitivities, and before/after probability changes on ingestion. Missing values are preserved in temporal explanations. These are non-additive sensitivities, not SHAP or causal treatment effects.
- The **Replay trend example (7 days)** button appends deliberately extreme synthetic demo readings. Reset restores the original patient. This replay is a UI demonstration, not evaluation data.
- Model evidence now shows the held-out review-policy tradeoff: CNN detects 80/177 events with 39 false alerts; CNN OR rule flag detects 89/177 with 78 false alerts. No overall superiority or clinical benefit is claimed.

On this computer, double-click `start.bat` (uses the installed `.venv`) and open http://127.0.0.1:8000. On another computer follow the setup below. Verify with `python -m unittest discover -s tests -p "test_*.py" -v`; reproduce the hybrid comparison with `python -m ml.evaluate_hybrid`.

Hybrid implementation: `backend/rules.py`, `backend/engine.py`, `frontend/app.js`. Report: `artifacts/hybrid_evaluation.json`. API adds `/api/hybrid-metrics` and `model=hybrid_cnn`. Patient predictions include `rules`, `review_status`, and `explanation`; ingestion also returns `update_comparison` and persists it in the journal.

The original PDF presentation, architecture PDF, verification record and `checksums.json` describe the initial package; they have not been regenerated for this update. The original checksum manifest remains useful to confirm that data and trained artifacts were preserved, but is not a manifest of the modified source.

A local, working digital-twin research prototype for the Happiest Health Digital Twin Challenge 2026. It combines a historical patient profile with 28 days of dynamic wearable and home-cuff summaries to estimate a **hypothetical five-year cardiovascular event probability under a custom simulator**.

**All benchmark patients, sensor readings and outcomes are synthetic. The measured results below are not clinical accuracy.** This package replaces the earlier label-free model scaffolding with a runnable, trained sandbox.

## Team details - complete before submission

- Team name: Tech Builders
- College/incubator: Manipal Institute of Technology, Manipal
- Team members and roles: TO BE PROVIDED
- Team leader contact: provide on the submission platform as required
- Public repository URL: TO BE PROVIDED
- Minimum 20-minute demonstration video URL: TO BE PROVIDED

## Run on Windows

Install Python 3.12, extract this folder, open a terminal inside it, then:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python -m backend.server
```

Open **http://127.0.0.1:8000**. The supplied checkpoints mean you do not need to retrain first. For a CPU-only PyTorch installation, install it from https://download.pytorch.org/whl/cpu before the remaining requirements. `start.bat` starts the server after your environment is active.

On macOS/Linux, activate with `source .venv/bin/activate`, then use the same commands. Stop the server with Ctrl+C. All browser assets are local and require no CDN. The server binds to localhost only. This is not an authenticated production service and must not receive real patient data.

## What works

- Patient dashboard with historical profile, rolling signal chart, data coverage and model output.
- Real inference from saved MLP, 1D CNN, LSTM, GRU and logistic models.
- Daily reading ingestion, accelerated seven-day replay and persisted SQLite state.
- Explicit insufficient-data behavior after simulated sensor dropout.
- Non-mutating what-if comparisons and feature sensitivity against training medians.
- Shared test-set evidence, confidence intervals and event-level audit journal.
- Snapshot JSON export and one-click patient reset.

The model predicts only within the simulator's assumptions. The dashboard is a conceptual doctor workflow, not a medical device. Dynamic inputs are daily summaries, including home-cuff BP; they are not raw ECG, continuous blood pressure or an implemented commercial-device integration.

## Actual training results

One deterministic split: 2,400 training, 800 validation and 800 test patients. There are 177 test events. All models use the same split. Accuracy uses a fixed threshold of 0.5.

| Model | Test accuracy | ROC-AUC | Event recall |
|---|---:|---:|---:|
| MLP | 82.38% | 0.826 | 45.20% |
| 1D CNN | 83.00% | 0.839 | 45.20% |
| LSTM | 82.25% | 0.835 | 41.81% |
| GRU | 82.50% | 0.838 | 42.94% |
| Logistic fusion | 82.12% | 0.840 | 35.59% |
| Logistic static only | 81.38% | 0.814 | 26.55% |
| Prevalence baseline | 77.88% | 0.500 | 0.00% |

**Default: logistic fusion**, selected by lowest validation Brier score (0.1288). **Selected neural model: 1D CNN**, validation Brier 0.1304. CNN accuracy has a Wilson 95% interval of approximately 80.24%-85.44%, conditional on this synthetic test set. These intervals do not quantify real-world uncertainty or simulator misspecification. Differences between close models are not proof of superiority.

The modest event recall at threshold 0.5 is a material limitation. We do not choose an operating threshold from the test set or claim that a high accuracy makes this suitable for screening. See `artifacts/metrics.json` for complete metrics and confusion counts.

## Data and target

The original supplied Synthea archive contains 108 patients. An exploratory historical audit at 2020-01-01 found 5 qualifying subsequent events, 56 candidate negatives and 7 with insufficient follow-up. Last encounters are only a follow-up proxy. These counts are insufficient for credible neural evaluation and were not used to train the delivered models. See `docs/synthea_audit.json` and `ml/audit_synthea.py`.

The delivered benchmark is **a separate custom simulation**, not an expanded Synthea cohort. It contains 4,000 adults, 28 daily records each, and 888 simulated outcomes. Historical features and dynamic trajectories share latent synthetic factors. A documented stochastic equation generates a hypothetical event within five years. Latent factors, generating probabilities, event dates and target labels are excluded from input features. All simulated patients start eligible and have complete five-year endpoint follow-up. The clinical definition is hypothetical, not adjudicated MI/stroke or a validated composite.

This does not establish that a month of wearable readings predicts real five-year events. The synthetic design deliberately creates a relationship for testing the integrated pipeline. See `docs/MODEL_CARD.md` and `docs/DATA_CARD.md`.

## Reproduce training

```bash
python -m ml.generate --patients 4000 --seed 2026
python -m ml.train --seed 42 --epochs 100
python -m unittest discover -s tests -p "test_*.py" -v
```

Training saves checkpoints, coefficients, train-fitted preprocessing, split identifiers, per-patient test predictions, metrics and validation histories in `artifacts/`. CPU runs are deterministic in the recorded environment; floating-point/library changes can alter results. Stop the server before retraining, then restart it to load new artifacts. Regeneration replaces the bundled sandbox data. Delete `runtime.sqlite3` if you intentionally want to discard all locally persisted demo updates after regenerating.

For a fresh Synthea audit, supply your original archive:

```bash
python -m ml.audit_synthea /path/to/synthea_sample_data_csv_latest.zip
```

## Technical stack and architecture

Python, PyTorch, NumPy, pandas and scikit-learn handle simulation, preparation, training and inference. A standard-library HTTP backend serves a responsive HTML/CSS/JavaScript frontend. SQLite stores demo updates and an event journal. No cloud account, API key or paid service is required.

Architecture: `docs/architecture.pdf`. Presentation: `docs/presentation.pdf`. API: `docs/API.md`. Timed walkthrough: `docs/DEMO_SCRIPT.md`.

## Submission checklist

The official rules require a public GitHub repository, team/college information, problem statement, technical details, open-source license, architecture PDF/PPT, presentation PDF/PPT and a **minimum 20-minute demonstration video**. Use folder name `Team Name_College Name`. Submit the leader's details and repository link through the platform.

- [x] Working frontend and backend
- [x] Trained models and measured synthetic evaluation
- [x] Source, reproducible data generator and included synthetic dataset
- [x] Architecture and presentation PDFs
- [x] MIT license for the original project code and generated sandbox data
- [x] Timed 22-minute recording plan
- [ ] Replace team/college placeholders
- [ ] Record and add the required video
- [ ] Publish repository, verify public links and submit

The repository has not been published and no submission has been sent. The original Synthea archive is not redistributed in this package. Dependencies retain their respective licenses. Rules checked on 13 October 2026: https://unstop.com/hackathons/crp-digital-twin-challenge-2026-happiest-health-1757873/amp . The page lists prototype submission ending 20 October 2026, 19:00 IST, and registration ending 18 October, 19:00 IST; confirm these in your account before submission. Its shortlist counts conflict in different sections.
