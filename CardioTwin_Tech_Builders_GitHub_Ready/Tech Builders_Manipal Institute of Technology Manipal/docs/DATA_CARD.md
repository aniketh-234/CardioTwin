# Data card and provenance

## Included benchmark

Source: original custom synthetic simulator `ml/generate.py`, seed 2026. There is no download, patient linkage, real patient information or duplication from the original Synthea archive. The supplied data and generator are included under the project MIT license. No claim of population representativeness is made.

| File | Contents |
|---|---|
| data/static.csv | 4,000 synthetic profiles, 8 numeric features plus identifier |
| data/daily_readings.csv | 112,000 dated daily summaries, six dynamic features |
| data/outcomes.csv | One dated synthetic target per patient, complete follow-up assumption |
| data/benchmark.npz | Matching arrays used for training |
| data/metadata.json | Generation metadata, definitions and input bounds |
| data/demo_patients.json | 24 subjects from the test partition, without outcome labels |

Simulation observations span 2021-01-01 through 2021-01-28. The prediction index is 2021-01-29 UTC. Follow-up ends 2026-01-29. These dates are fictional timelines. They do not imply that actual data collection or clinical follow-up happened.

## Feature dictionary

| Feature | Unit / coding | Source assumption |
|---|---|---|
| age | years, 30-80 | Historical demographics |
| sex_male | binary 0/1 | Simplified synthetic demographic field |
| bmi | kg/m2 | Historical clinical measurement |
| systolic_bp | mmHg | Historical office BP |
| total_cholesterol, hdl | mg/dL | Historical labs |
| smoker, diabetes | binary 0/1 | Historical status indicators |
| resting_hr | bpm | Daily wearable summary |
| hrv_rmssd | ms | Daily wearable HRV summary |
| steps | count/day | Daily wearable activity summary |
| sleep_hours | hours/day | Daily wearable sleep summary |
| home_systolic_bp | mmHg | Daily connected home-cuff summary |
| coverage | proportion 0-1 | Synthetic data availability |

Four percent of each dynamic measurement, excluding coverage, is independently set missing in generation. Coverage is a separate simulated summary; its relationship to missingness is simplified. CSV blanks and JSON nulls mean missing, not zero. Patient identifiers, dates, outcome labels and simulator latent variables are never model inputs. The simulator creates daily records directly and does not derive them from real raw sensor waveforms.

## Original supplied Synthea archive

`docs/synthea_audit.json` records its hash and the source audit. It contains 108 patients and 68,648 observations. At a candidate 2020 index, only five subsequent five-year events appear under the exploratory code list. This sample and the previously generated August 2026 wearables are not the training data for this package. Their timing does not support the new five-year benchmark. The archive is not redistributed; rerun `ml/audit_synthea.py` against your original file if needed.

## Rules alignment

The official competition page permits synthetic EHR and sensor data. That permission supports a sandbox proof of concept; it does not turn synthetic accuracy into clinical validation or guarantee acceptance by the jury. The integrated static/dynamic prediction path is implemented here. Biological fidelity and clinical generalization remain future research work.

Rules source: https://unstop.com/hackathons/crp-digital-twin-challenge-2026-happiest-health-1757873/amp

Technical references: https://docs.pytorch.org/docs/stable/nn.html and https://scikit-learn.org/stable/modules/model_evaluation.html
