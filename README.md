# CardioTwin — Explainable Cardiovascular Digital Twin

**Tech Builders | Manipal Institute of Technology, Manipal**  
**Happiest Health Digital Twin Challenge 2026**  
**Public repository:** https://github.com/aniketh-234/CardioTwin

> **Research prototype — synthetic data only.** CardioTwin estimates a *hypothetical five-year cardiovascular event* under a custom simulator. The output is not a clinically validated risk estimate, diagnosis, imminent heart-attack warning, or medical recommendation. Do not enter real patient data.

## Team and submission details

| Field | Information |
|---|---|
| Team | **Tech Builders** |
| Institution | **Manipal Institute of Technology, Manipal** |
| Incubator | Not specified |
| Team members and roles | **[Add member names and responsibilities]** |
| Team leader | **[Add leader name]** |
| Repository | [aniketh-234/CardioTwin](https://github.com/aniketh-234/CardioTwin) |
| Required demonstration video (minimum 20 minutes) | **[Add accessible video URL after recording]** |
| License | [MIT](CardioTwin_Tech_Builders_GitHub_Ready/Tech%20Builders_Manipal%20Institute%20of%20Technology%20Manipal/LICENSE) — project source; dependencies retain their own licenses |

## Problem statement and healthcare use case

Static electronic health records capture background factors but do not show how day-to-day physiological measurements evolve. CardioTwin connects a synthetic historical health profile with a rolling **28-day** sequence of daily sensor-style summaries. After each new day's readings are ingested, the virtual patient state updates and a saved trained model runs inference again. The dashboard presents the model estimate, explicit trend-review reasons, data-quality status, and sensitivity-based explanations.

**Intended research use case:** inspect longitudinal changes in a simulated cardiovascular-risk review workflow. **Not intended:** clinical triage, treatment recommendations, real patient monitoring, or acute-event detection.

## Submission deliverables

All links below point to files verified in this public repository.

| Requirement | Verified repository link |
|---|---|
| Source code | [Backend](https://github.com/aniketh-234/CardioTwin/tree/main/CardioTwin_Tech_Builders_GitHub_Ready/Tech%20Builders_Manipal%20Institute%20of%20Technology%20Manipal/backend), [Frontend](https://github.com/aniketh-234/CardioTwin/tree/main/CardioTwin_Tech_Builders_GitHub_Ready/Tech%20Builders_Manipal%20Institute%20of%20Technology%20Manipal/frontend), [ML](https://github.com/aniketh-234/CardioTwin/tree/main/CardioTwin_Tech_Builders_GitHub_Ready/Tech%20Builders_Manipal%20Institute%20of%20Technology%20Manipal/ml) |
| Saved model and evaluation artifacts | [Artifacts](https://github.com/aniketh-234/CardioTwin/tree/main/CardioTwin_Tech_Builders_GitHub_Ready/Tech%20Builders_Manipal%20Institute%20of%20Technology%20Manipal/artifacts) |
| Detailed architecture diagram (PDF) | [Open architecture PDF](https://github.com/aniketh-234/CardioTwin/blob/main/CardioTwin_Tech_Builders_GitHub_Ready/Tech%20Builders_Manipal%20Institute%20of%20Technology%20Manipal/docs/architecture_detailed.pdf) |
| Detailed architecture diagram (PPT) | [Download architecture PPT](https://github.com/aniketh-234/CardioTwin/blob/main/CardioTwin_Tech_Builders_GitHub_Ready/Tech%20Builders_Manipal%20Institute%20of%20Technology%20Manipal/docs/architecture_detailed.pptx) |
| Project presentation (PPT) | [Download presentation PPT](https://github.com/aniketh-234/CardioTwin/blob/main/CardioTwin_Tech_Builders_GitHub_Ready/Tech%20Builders_Manipal%20Institute%20of%20Technology%20Manipal/docs/CardioTwin_Hackathon_Presentation.pptx) |
| Project presentation (PDF) | [Open presentation PDF](https://github.com/aniketh-234/CardioTwin/blob/main/CardioTwin_Tech_Builders_GitHub_Ready/Tech%20Builders_Manipal%20Institute%20of%20Technology%20Manipal/docs/presentation.pdf) |
| Model and data cards | [Model card](https://github.com/aniketh-234/CardioTwin/blob/main/CardioTwin_Tech_Builders_GitHub_Ready/Tech%20Builders_Manipal%20Institute%20of%20Technology%20Manipal/docs/MODEL_CARD.md), [Data card](https://github.com/aniketh-234/CardioTwin/blob/main/CardioTwin_Tech_Builders_GitHub_Ready/Tech%20Builders_Manipal%20Institute%20of%20Technology%20Manipal/docs/DATA_CARD.md) |
| API and verification | [API documentation](https://github.com/aniketh-234/CardioTwin/blob/main/CardioTwin_Tech_Builders_GitHub_Ready/Tech%20Builders_Manipal%20Institute%20of%20Technology%20Manipal/docs/API.md), [Verification notes](https://github.com/aniketh-234/CardioTwin/blob/main/CardioTwin_Tech_Builders_GitHub_Ready/Tech%20Builders_Manipal%20Institute%20of%20Technology%20Manipal/docs/VERIFICATION.md) |
| Dependencies | [requirements.txt](https://github.com/aniketh-234/CardioTwin/blob/main/CardioTwin_Tech_Builders_GitHub_Ready/Tech%20Builders_Manipal%20Institute%20of%20Technology%20Manipal/requirements.txt) |
| Open-source license | [MIT LICENSE](https://github.com/aniketh-234/CardioTwin/blob/main/CardioTwin_Tech_Builders_GitHub_Ready/Tech%20Builders_Manipal%20Institute%20of%20Technology%20Manipal/LICENSE) |
| Minimum 20-minute video | **Pending: add the video URL after recording** |

**Version note:** The verified architecture and presentation files above are the currently committed versions. The separately prepared final slide files must be uploaded before they can be linked here as final versions.

**Project location:** The runnable application is in `CardioTwin_Tech_Builders_GitHub_Ready/Tech Builders_Manipal Institute of Technology Manipal/` within this repository. Open that folder before running the setup commands below.

## Dataset and prediction target

The benchmark consists of **4,000 synthetic adults**, each with **28 daily records** (**112,000 daily rows**), and **888 simulated events**. Patient-level splits: **2,400 training**, **800 validation**, and **800 test** patients. The test split contains **177 simulated events**. No patient's days are shared between training and test.

- **Historical EHR-style fields (8):** age, sex, BMI, systolic blood pressure, total cholesterol, HDL cholesterol, smoking status, diabetes status.
- **Daily fields:** resting heart rate, heart-rate variability, steps, sleep duration, home-cuff systolic blood pressure, and measurement coverage.
- **Target:** a simulator-generated first cardiovascular event within five years after the prediction index, *not* an observed or clinically adjudicated endpoint.

The custom simulator links static factors, longitudinal trajectories and outcomes for the **same synthetic identity**. Hidden generating variables, event dates, outcomes and labels are excluded from model inputs. The supplied Synthea sample was separately audited but **was not used to train these models** because the available follow-up events were insufficient for credible neural evaluation. See `docs/DATA_CARD.md` and `docs/MODEL_CARD.md`.

## Technical stack and models

| Layer | Implementation |
|---|---|
| Frontend | HTML, CSS, JavaScript; interactive patient dashboard |
| Backend | Python standard-library HTTP server; input validation and inference orchestration |
| ML | PyTorch MLP, temporal 1D CNN, LSTM, GRU; scikit-learn logistic static/fusion baselines |
| Data and preprocessing | NumPy, pandas, scikit-learn; train-fitted preprocessing and missingness indicators |
| Persistence | SQLite for synthetic patient state and event journal |
| Reproducibility | Saved checkpoints, split identifiers, evaluation artifacts, unit tests |

**Model-selection distinction:** logistic fusion achieved the lowest validation Brier score and was the original validation-selected model. **Hybrid CNN + rules** is the interactive dashboard demonstration mode, **not** a proven superior model.

## Detailed system architecture

```text
 Synthetic EHR profile (8 static features)      Synthetic daily readings (5 signals + coverage)
                  |                                           |
                  +-------------------+-----------------------+
                                      |
                         Python HTTP API / validation
                         - schema and range checks
                         - synthetic patient identity
                                      |
                        28-day rolling patient state
                         - SQLite persistence + journal
                                      |
                      Preprocessing and readiness gates
                      - train-fitted transforms
                      - missingness masks / coverage
                      - insufficient data => WITHHOLD
                                      |
                   +------------------+------------------+
                   |                                     |
         Learned prediction                         Trend review rules
         static + temporal branch                   latest 7d vs previous 21d
         1D CNN / fusion head                       per-signal reasons; >=2 flags
         synthetic 5-year probability               (does NOT change probability)
                   |                                     |
                   +------------------+------------------+
                                      |
                       Explanation and presentation
                       - sensor/week sensitivities
                       - historical sensitivities
                       - non-mutating scenario comparison
                       - prediction / unavailable status
                                      |
                       Browser dashboard + snapshot
                       - inspect, ingest, reset, replay
                       - trajectory, reasons, journal
```

### What happens after a new day is entered

1. Browser submits fictional HR, HRV, steps, sleep, home BP and coverage.
2. Backend validates patient identity, fields and acceptable ranges.
3. A dated record is appended, the simulated day advances, and only the newest 28 days remain.
4. Data-quality gates check recent coverage, missingness and historical feature availability.
5. If ready, saved preprocessing and a trained model checkpoint produce a new inference **without retraining**.
6. Trend rules compare the latest 7 days with the preceding 21 days and return inspectable reasons.
7. Reference-replacement sensitivities and optional non-mutating scenario comparisons support model inspection.
8. Dashboard shows the updated trajectory, prediction or withheld state, review reasons, before/after change and event journal.

Trend checks require at least **5 observed values in the recent 7 days** and **14 in the preceding 21 days** for each signal. A directional change of **0.5 training-population standard deviations** can trigger a signal rule; at least **two adverse signals** produce a multiple-trend review flag. These are engineering heuristics, **not clinical thresholds**. Rule flags never modify the CNN probability.

## Evaluation: held-out synthetic test set

**800 test patients; 177 simulated events.** The alert policies below are threshold-specific and should not be interpreted as clinical performance.

| Review policy | Total alerts | Events detected | False alerts | Missed events | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| CNN at 50% | 119 | 80 | 39 | 97 | 45.2% | 67.2% |
| CNN at 50% **OR** trend-review flag | 167 | 89 | 78 | 88 | 50.3% | 53.3% |
| CNN at validation-matched 37.93% | 181 | 101 | 80 | 76 | 57.1% | 55.8% |

CNN test accuracy was **83.00%** and ROC-AUC approximately **0.839**. The prevalence baseline achieved **77.88%** accuracy while detecting **zero** events at the displayed threshold; accuracy alone is therefore insufficient. See `artifacts/metrics.json` and `artifacts/hybrid_evaluation.json`.

The hybrid policy detected nine more events than CNN at 50%, but generated 39 additional false alerts. The exploratory lower-threshold CNN comparison detected more events at a similar alert workload. **The experiment does not establish added predictive value from the rules for the synthetic five-year target.** The validation-matched threshold was 0.3793189525604248; the original test set had already been inspected before this follow-up analysis. This is **not independent clinical validation**.

## Run locally

Use **Python 3.12** where possible. From the project root:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m backend.server
```

Open **http://127.0.0.1:8000/**. Saved model artifacts allow inference without retraining. The application is local-only and must not receive real patient data.

### Reproduce and test

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python -m ml.evaluate_hybrid
python -m ml.compare_review_policies
```

To regenerate the synthetic benchmark and retrain models (this **replaces** the bundled simulation and trained outputs):

```bash
python -m ml.generate --patients 4000 --seed 2026
python -m ml.train --seed 42 --epochs 100
```

See `docs/VERIFICATION.md`, `docs/API.md` and the data/model cards for additional setup and evaluation detail.

## Demonstration workflow

1. Select and reset a prepared synthetic patient; inspect the historical profile and trajectory.
2. Ingest one fictional daily record and show the advanced date, changed rolling window and new inference.
3. Reset and replay the **trend example (7 days)** to inspect triggered review reasons.
4. Inspect historical and sensor/week sensitivities; run a what-if scenario without overwriting history.
5. Reset and demonstrate sensor dropout, where the application withholds predictions for insufficient data.
6. Open model evidence and compare alert policies, false alerts and missed events.

**Video (required; at least 20 minutes):** [ADD LINK AFTER RECORDING]. Ensure the video is accessible to judges and contains substantive explanation **and** live prototype demonstration.

## Limitations and responsible use

- All training, testing, benchmark patients and demonstrated readings are **synthetic**.
- No commercial wearable or production EHR integration is implemented; daily readings are simulated summaries, not raw ECG or continuous BP.
- No independent real-patient validation, clinically justified thresholds, subgroup evaluation or production deployment is claimed.
- Sensitivities are reference-replacement differences, **not SHAP values**, additive feature attribution, or causal treatment effects.
- Scenario sliders show model response to alternative inputs; they do **not** establish intervention benefit.
- The prototype lacks production authentication, governance and clinical security controls.

## Open-source license

Project-authored code and synthetic benchmark assets are provided under the repository's **[MIT License](CardioTwin_Tech_Builders_GitHub_Ready/Tech%20Builders_Manipal%20Institute%20of%20Technology%20Manipal/LICENSE)**. Third-party packages retain their own licenses. The original Synthea archive is **not redistributed**.

## Before submitting

- [ ] Add all team members, individual roles and team leader's name.
- [ ] Confirm the final presentation and detailed architecture files are committed at the paths above.
- [ ] Add the accessible minimum-20-minute video link.
- [ ] Verify repository links and demo instructions from a signed-out browser.
- [ ] Submit the public repository link and required team-leader details through the official challenge portal.

**Repository:** https://github.com/aniketh-234/CardioTwin
