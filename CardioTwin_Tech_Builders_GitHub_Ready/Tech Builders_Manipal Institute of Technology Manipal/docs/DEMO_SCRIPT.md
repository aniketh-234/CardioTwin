# 22-minute recording plan

This is a presenter runbook, not a completed video. Record at least 20 minutes as required by the competition. Use the supplied presentation PDF and live app. Speak in your own voice, explain the code and leave time for the interactions below. Add your real team/college details first. Do not imply any clinical validation or invent team experience.

## 00:00-02:00 - Problem and honest scope

Show slides 1-2. Introduce the team and your chosen cardiovascular use case. Explain that the prototype combines an EHR-style history with a changing patient state. State explicitly: all benchmark patients and outcomes are synthetic, and this demonstration does not estimate a real person's cardiovascular risk. Explain that the goal is to prove ingestion, fusion, inference, review and traceability in one working application.

Suggested opening: “CardioTwin is our working research prototype for a cardiovascular digital twin. We combine a historical profile with a rolling month of sensor summaries, update the virtual patient when a new record arrives, and show the model's evidence and limitations. We trained and evaluated the models on a transparent synthetic benchmark. The results measure this simulator, not clinical performance.”

Explain why this is a focused prototype rather than a full physiology model. Identify a clinician as the conceptual reviewer, without claiming you have already conducted clinician interviews.

## 02:00-04:30 - Dataset and target

Show slide 3 and `docs/synthea_audit.json`. Explain the original sample's label limitation: an exploratory historical cohort contained only five future events. Do not describe all other patients as confirmed disease-free. Explain why joining unrelated EHR and wearable subjects would be invalid.

Open `ml/generate.py`. Walk through the 4,000 synthetic profiles, 28 days and stochastic future outcome. Explain that latent burden drives both trajectories and the outcome by assumption. The generating equation is visible and unvalidated. There are 888 simulated events, while the held-out test partition contains 177. Show that event dates occur after the prediction index and that no outcome date enters the model.

Pause on the distinction between simulated outcomes and observed clinical outcomes. Explain what matched longitudinal data would be needed to replace this simulation.

## 04:30-07:00 - Architecture and models

Show slides 4-5. Trace historical profile and daily summaries into validation and preprocessing. Explain missingness masks and why normalization statistics use training subjects only. Walk through MLP, CNN, LSTM and GRU branches in `ml/models.py`. Show the static branch and the fusion head. Explain how CNN/recurrent models use temporal organization while the logistic baseline uses mean daily features.

Show `artifacts/splits.json`: patients are disjoint across training, validation and test. Explain early stopping and validation Brier selection. Show why selecting a model based on test accuracy would undermine an independent evaluation. Explain that an architecture is useful only if it improves relevant evidence, and that this benchmark does not establish such clinical evidence.

## 07:00-10:00 - Patient workspace

Open http://127.0.0.1:8000. Select the first synthetic patient and reset it. Explain the historical profile, simulation date, 28-day trajectory, coverage and five-year synthetic score. Use the signal selector to inspect heart rate, HRV, steps, sleep and connected-cuff BP. Point out missing gaps on the chart.

Switch from the validation-selected logistic model to CNN, then LSTM. Explain that different learned approximations produce different outputs; this is not a confidence interval. Return to the selected model. Show the feature sensitivities and read their reference definition. Explain that they are median-replacement differences, not SHAP values or causes.

Export a snapshot. Open it briefly to show the current profile, prediction and explicit synthetic disclosure. Avoid presenting the number as a diagnosis or as a probability valid for a real patient.

## 10:00-13:00 - Dynamic updates and insufficient data

Change the daily input values within the supported ranges and click Ingest next day. Show the updated date, update counter, rolling chart and recomputed score. Explain that the model receives a changed window and performs inference; it does not retrain every time.

Click Replay 7 days. Narrate the chronological progression and distinguish the accelerated demo from real time. Open the event journal and show dates, recorded inputs and the model used. Return to the patient workspace.

Click Demo sensor dropout. Show that the score becomes withheld after insufficient recent data. Explain the explicit engineering thresholds and why they need clinical validation. Reset the patient and show recovery. This is a useful live failure-mode demonstration: the interface does not equate missing signals with a healthy patient.

## 13:00-15:00 - Scenario comparison

Adjust the BP, steps and sleep sliders. Click Compare scenario. Explain that this is a non-mutating alternate input profile. The result may increase or decrease depending on the model and current patient; do not promise a particular reduction.

State: “This is model sensitivity, not an estimate of treatment benefit. Setting a field to a different value does not model what happens biologically if a patient follows an intervention.” Show that the saved history does not change. Explain that a validated causal or physiological model would be a separate research task.

## 15:00-18:00 - Actual results and limitations

Open Model evidence and slide 6. Show the shared 800-patient test set, majority/prevalence baseline and all learned models. The CNN achieved 83.00% test accuracy and approximately 0.839 ROC-AUC. Logistic fusion achieved approximately 0.840 ROC-AUC and had the best validation Brier score, so it is the default.

Emphasize the event recall column. At threshold 0.5, the default model detects only about 35.6% of simulated events. The prevalence baseline achieves about 77.9% accuracy while detecting none, showing why accuracy alone is misleading. The prototype does not propose 0.5 as a medical threshold.

Explain the confidence interval's scope. It covers sampling variability conditional on this synthetic test set; it does not include real-world distribution shift or simulator errors. Do not claim that 83% accuracy proves clinical success or that a small difference between architectures is statistically decisive.

## 18:00-20:30 - Reproducibility and implementation

Show the repository structure. Open `README.md`, `artifacts/metrics.json`, checkpoints and the model card. Demonstrate the unit-test command and explain prediction parity between saved evaluation outputs and the serving API. Show the API specification and the SQLite journal implementation.

Explain that it runs locally with no paid API dependency. State that the server binds to localhost, has no user authentication and is not a production clinical service. Distinguish implemented capabilities from roadmap items: daily ingestion and inference are implemented; commercial-device integration and real-patient validation are not.

## 20:30-22:00 - Next evidence and closing

Show slides 8-10. Explain the next steps: obtain matched longitudinal records, define and adjudicate the endpoint, handle censoring, validate externally, evaluate calibration and subgroup behavior, and work with clinicians on acceptable workflow and thresholds.

Close with a concrete description: “We deliver a reproducible sandbox with four trained neural models, a baseline-selected inference service, a usable patient dashboard and explicit handling of missing data. Our next milestone is a clinically grounded dataset and independent evaluation.”

Show the accessible repository and video links after publishing. Ensure the final recording lasts at least 20 minutes and has readable screen text and audible narration. Do not pad a short recording with silence to satisfy duration.

## Judge questions to rehearse

- Why is the benchmark synthetic? Access constraints plus insufficient usable labels in the supplied sample; the rules permit simulation.
- Does the reported accuracy apply to patients? No. It evaluates recovery of a hypothetical synthetic outcome process.
- Why default to logistic regression? It has the best validation Brier score here; complexity is not the selection criterion.
- Why is recall low? The fixed 0.5 operating threshold and event prevalence yield many false negatives; clinical thresholding would need separate development and validation.
- What makes the patient twin dynamic? New dated summaries update persisted state, rolling features and inference through an actual API.
- Are scenario comparisons treatment recommendations? No. They are input sensitivities with no causal interpretation.
- Can you deploy to a hospital today? No. Authentication, governance, real-data evaluation and clinical validation remain necessary.
