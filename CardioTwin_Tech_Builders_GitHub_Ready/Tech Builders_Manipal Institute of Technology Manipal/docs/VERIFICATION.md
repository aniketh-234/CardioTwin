# Verification record

Executed on 13 October 2026 using the versions in `artifacts/environment.json`.

## Training

- Generated 4,000 distinct synthetic patients with 28 daily summaries each.
- Trained MLP, CNN, LSTM, GRU and two logistic baselines on the same patient partitions.
- Saved four PyTorch checkpoints and both logistic coefficient sets.
- Evaluated 800 held-out patients and retained per-patient predictions.
- Selected the default model using validation Brier score.

## Automated checks

Seven Python tests passed:

1. Patient partition isolation and post-index event dates.
2. Serving predictions match recorded held-out predictions for all six learned models.
3. Insufficient dynamic data withholds the probability.
4. Invalid shapes and out-of-range inputs are rejected.
5. Stored preprocessing matches training-only fitting.
6. API scenarios do not mutate patient state and duplicate daily writes are rejected.
7. Unknown inference models return a validation error.

Python syntax compilation also passed.

## Browser and document checks

Chromium checks passed for initial loading, scenario comparison, daily ingestion, dropout abstention, reset, evidence-table loading and journal display. There were no JavaScript page errors. The desktop and 390-pixel mobile layouts were visually inspected; the mobile page had no horizontal overflow.

The ten-page presentation and one-page architecture PDF were rendered and visually reviewed. A diagram label overlap was corrected before delivery.

## Not verified or completed

- No clinical validation, external cohort evaluation or real sensor-device integration.
- No production security, multi-user deployment or hospital-system integration.
- No team identity verification, public GitHub publication, video recording or competition submission.
- The timed recording guide is not a completed 20-minute video.

Re-run Python checks with `python -m unittest discover -s tests -p "test_*.py" -v`.
For browser checks install `requirements-dev.txt`, run `python -m playwright install chromium`, start the server, then run `python tests/browser_smoke.py`. The browser check writes disposable screenshots outside the project folder and resets its selected demo patient.
