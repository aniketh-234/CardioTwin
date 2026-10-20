# Model card

## Intended use

Demonstrate a locally runnable cardiovascular digital-twin pipeline under transparent synthetic assumptions. Users can inspect a patient profile, update daily sensor summaries, recompute a hypothetical five-year event probability and inspect model evidence. Outputs are unsuitable for diagnosis, triage, treatment, insurance or real-person risk assessment.

## Training population and target

4,000 independently generated adult profiles aged 30-80. The profiles do not represent an established Indian or other clinical population. There are 28 observations per patient and one binary label per patient. The target is an abstract, hypothetical first cardiovascular event within five years, not a coded/adjudicated MI, stroke or cardiovascular-death composite. All subjects are assumed event-free at index and fully observed throughout the horizon. This simplifying assumption avoids censoring in the sandbox and would need replacement in a clinical cohort.

## Outcome generation

`ml/generate.py` makes its synthetic assumptions explicit. With independent noise `epsilon ~ Normal(0,0.45)`, a latent burden and a latent trajectory trend:

```text
logit = -2.2 + 0.045*(age-50) + 0.025*(systolic_bp-125)
        + 0.014*(total_cholesterol-200) - 0.025*(hdl-50)
        + 0.6*smoker + 0.7*diabetes + 0.10*(bmi-27)
        + 0.65*burden + 0.5*trend + epsilon
probability = sigmoid(logit)
event ~ Bernoulli(probability)
```

These coefficients are unvalidated engineering assumptions. Dynamic variables correlate with burden and trend by construction. Training tests whether a model recovers the designed association, not whether wearables truly predict disease. The outcome equation is not Framingham, ASCVD, PREVENT or any other established medical score. Event dates are assigned after the prediction index, conditional on the simulated binary event. They do not constitute a survival model.

## Models

- MLP: static dense branch plus flattened 28-day dynamic dense branch.
- CNN: static dense branch plus two temporal Conv1d layers and average pooling.
- LSTM/GRU: static dense branch plus one recurrent layer, using the final output.
- Neural hidden width: 16. Fusion head: 32 -> 16 -> 1, ReLU, dropout 0.2.
- Adam, learning rate 0.002, weight decay 0.001, batches of 128, BCEWithLogitsLoss.
- Maximum 100 epochs, patience 12, stopping on validation Brier score.
- Logistic baselines: static-only and static plus mean normalized daily inputs.

Median imputation and scaling use training patients only. Missingness indicators are appended. All neural models use the same stratified 60/20/20 patient split, seed 42. The synthetic generator uses seed 2026. No model sees simulator probabilities, latent states or post-index outcomes as inputs.

## Selection and interpretation

Logistic fusion has the lowest validation Brier score and is the default. CNN has the lowest validation Brier among neural models. The browser lets a reviewer inspect all neural models without changing the recorded selection rule. Brier is a probability error score, not a standalone proof of clinical calibration. The fixed 0.5 threshold produces limited recall and is not an endorsed medical operating point.

See `artifacts/metrics.json` for all numbers, `test_predictions.csv` for per-patient outputs, `splits.json` for patient partitions and `training_history.json` for validation curves. The confidence interval is the Wilson interval for accuracy conditional on this test set. It excludes data-generation uncertainty, model-selection uncertainty and distribution shift.

## Dynamic updates and scenarios

Ingestion drops the oldest daily summary and appends the next day's summary. The fixed trained model recomputes a score. It does not retrain on each patient reading. These shifted windows have no new observed outcome labels and are exploratory predictions, not separately validated evaluations.

Recent seven-day coverage below 0.6, recent sensor missingness above 0.4, or more than half of static values missing cause abstention. These are engineering demonstration thresholds, not clinical standards. There is no age-specific calibration or out-of-distribution detector beyond input bounds.

Feature sensitivities replace one static feature or the dynamic history with training medians and measure the score difference. Correlated features and interactions mean these values are neither additive SHAP explanations nor causal effects. Scenario sliders replace input values only; they do not simulate the biological effects of an intervention.

## Limitations and next work

Synthetic distribution and hand-designed labels dominate performance. There is no evidence for transfer to real patients or the Indian population, no external or prospective validation, no subgroup fairness claim, no survival/competing-risk analysis and no causal treatment model. A real deployment needs a matched, appropriately governed longitudinal cohort, a precise adjudicated endpoint, ascertainment and censoring rules, independent validation, calibration and clinician-led workflow testing.

The local web server has no user authentication and is intentionally bound to loopback. Do not expose it publicly or use actual clinical records. The code and supplied benchmark are an engineering submission starter, not a deployable clinical product.
