# Live positioning API contract

Contract version: `0.1`

The service uses HTTP JSON on port `8765`. Responses allow cross-origin requests and disable caching. WebSocket streaming is not implemented in this version.

## Endpoints

| Method | Path | Response |
|---|---|---|
| `GET` | `/api/meta` | HTTP 200 with experiment geometry and stream identity |
| `GET` | `/api/status` | HTTP 200 with backend and link state |
| `GET` | `/api/predict/latest` | HTTP 200 with the latest prediction, or HTTP 204 when none is usable |
| `GET` | `/ws/predictions` | HTTP 501; poll the HTTP endpoints instead |

Poll `/api/status` and `/api/predict/latest` approximately every 0.5 seconds. A repeated response with the same `session_id` and `seq` is the same prediction, not another sample.

## Coordinate system

The standard grid has five columns, three rows and 0.8 m spacing. Label numbering is row-major:

```text
y=2   10  11  12  13  14
y=1    5   6   7   8   9
y=0    0   1   2   3   4
       x increases to the right
```

For label `l`, `x = l % cols`, `y = l // cols`, `x_m = x × spacing_m` and `y_m = y × spacing_m`. Clients should use the values returned by `/api/meta` instead of hard-coding this geometry.

## `GET /api/meta`

The response has this shape:

```json
{
  "contract_version": "0.1",
  "source": "live",
  "session_id": "d14-1a2b3c4d",
  "grid": {"cols": 5, "rows": 3, "spacing_m": 0.8},
  "cells": [{"label": 0, "x": 0, "y": 0, "x_m": 0.0, "y_m": 0.0}],
  "devices": [{"id": "piA", "kind": "pi", "x_m": -0.9, "y_m": 2.0, "h_m": 1.1}],
  "links": [{"id": "piA_apX", "pi": "piA", "ap": "apX"}],
  "window_frames": 16,
  "frame_rate_hz": 10
}
```

`cells`, `devices` and `links` contain all configured entries. `source` is `live` for Pi input and `replay` when the server is started with `--replay`. `session_id` is unique to one server process.

## `GET /api/status`

Example:

```json
{
  "type": "status",
  "contract_version": "0.1",
  "source": "live",
  "session_id": "d14-1a2b3c4d",
  "timestamp": "2026-10-20T14:03:10.000+11:00",
  "state": "ready",
  "progress": null,
  "calibrated_at": "2026-10-20T13:48:30.000+11:00",
  "links": {
    "piA_apX": {"online": true, "fps": 10.1, "last_frame_age_s": 0.08}
  },
  "message": null
}
```

The `links` object contains all four required links. State values are:

| State | Meaning |
|---|---|
| `offline` | No live frames are available, or model training failed |
| `calibrating` | The labelled calibration round is incomplete |
| `training` | The service is parsing the calibration round and training the live model |
| `ready` | The model and all four links are available |
| `degraded` | One or more required links are offline; new predictions are paused |

`progress` is a number from 0 to 1 during calibration and otherwise `null`. `message` is explanatory text for display, not a machine-readable state. A link is considered offline after its input is older than the live buffer limit.

## `GET /api/predict/latest`

Before the first usable prediction, or while the current status is neither `ready` nor `degraded`, the endpoint returns HTTP 204 with no body. Otherwise it returns the most recently produced prediction:

```json
{
  "type": "prediction",
  "contract_version": "0.1",
  "source": "live",
  "session_id": "d14-1a2b3c4d",
  "model_id": "F-MLP",
  "seq": 1532,
  "timestamp": "2026-10-20T14:03:12.481+11:00",
  "window": {
    "start": "2026-10-20T14:03:10.881+11:00",
    "end": "2026-10-20T14:03:12.481+11:00",
    "frames": 16
  },
  "label": 7,
  "x": 2,
  "y": 1,
  "x_m": 1.6,
  "y_m": 0.8,
  "confidence": 0.62,
  "probs": [0.00, 0.01, 0.08, 0.01, 0.00, 0.00, 0.07, 0.62, 0.09, 0.01, 0.00, 0.01, 0.08, 0.02, 0.00],
  "expected_m": {"x": 1.64, "y": 0.81},
  "links_used": ["piA_apX", "piA_apY", "piB_apX", "piB_apY"],
  "state": "ready"
}
```

Field rules:

- `seq` increases within one process and restarts for a new `session_id`.
- `timestamp` equals `window.end`; timestamps are ISO 8601 values with a time-zone offset.
- `probs` contains one probability per cell in label order.
- `label` is the maximum-probability class and `confidence` is its probability.
- `expected_m` is a probability-weighted display coordinate. The formal classifier output remains `label`.
- `links_used` lists the links used for this prediction.

When a link fails after a prediction has been produced, status becomes `degraded` and inference pauses. The endpoint may still return its last prediction; clients must use `seq`, timestamp and current status to avoid treating it as fresh.

## Error responses

Unknown paths return HTTP 404 with a JSON `error` field. `/ws/predictions` returns HTTP 501 with an explanation. HTTP 204 has no JSON body and means “no current prediction”, not a parsing failure.
