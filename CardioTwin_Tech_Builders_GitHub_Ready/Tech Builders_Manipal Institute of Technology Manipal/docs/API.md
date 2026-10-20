# Local API

Start with `python -m backend.server`. Base URL: `http://127.0.0.1:8000`. All payloads are JSON. No authentication is implemented; the server binds to loopback and rejects cross-origin writes. Patient IDs must be from the bundled demo list. Writes persist to local SQLite. This API is for synthetic demo data only.

| Method | Route | Purpose |
|---|---|---|
| GET | /api/health | Runtime status |
| GET | /api/meta | Data definition and bounds |
| GET | /api/patients | Demo patient list |
| GET | /api/patients/{id}?model=cnn | Current profile, sequence and prediction |
| GET | /api/metrics | Recorded training evaluation |
| GET | /api/audit | Latest 30 persisted write events |
| POST | /api/ingest | Append a daily record |
| POST | /api/scenario | Non-mutating input comparison |
| POST | /api/reset | Restore a fixture and journal the reset |

Omit `model` to use the validation-selected model. Valid selectable models: mlp, cnn, lstm, gru, logistic_static and logistic_fusion.

Example ingest body, using an actual ID from `/api/patients`:

```json
{
  "patient_id": "SIM-01396",
  "date": "2021-01-29",
  "model": "cnn",
  "readings": [72, 45, 6500, 7.0, 130, 0.95]
}
```

Readings order: resting heart rate, HRV, steps, sleep hours, cuff systolic BP, coverage. A new date must be exactly one day after the current last date. Duplicates are rejected. Missing values use JSON null. The accepted feature ranges are in `/api/meta`. Invalid shapes, ranges, unknown patients/models and malformed JSON return HTTP 400. Unknown routes return 404. Prediction abstention is a successful response with `status: insufficient_data` and `probability: null`.

Example scenario:

```json
{
  "patient_id": "SIM-01396",
  "adjustments": {"systolic_bp": 125, "steps": 8000, "sleep_hours": 7.5}
}
```

Scenario BP changes both historical BP and every daily cuff value. Steps and sleep changes replace the corresponding daily values across the window. The result describes a hypothetical changed input profile, not a medical intervention. No patient record is modified.

Daily replay is accelerated simulation. There is no WebSocket, live commercial-device connection or automatic model retraining. Snapshot export is browser-generated JSON from the current patient response.
