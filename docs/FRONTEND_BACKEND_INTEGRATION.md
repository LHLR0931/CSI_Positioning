# Frontend–backend integration

The Unity client has two data paths:

- **Live** polls the Python backend running on the experiment computer.
- **Replay** reads the fixed D1 result file bundled under `unity-app/Assets/Resources/Replay/`.

Replay is for repeatable inspection of saved results. It is not a substitute for current CSI capture, calibration or live inference.

## Live HTTP interface

The default base address is:

```text
http://<backend-computer-ip>:8765
```

The implemented endpoints are:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/meta` | Grid, cells, devices, links and session identity |
| `GET` | `/api/status` | Backend state and per-link health |
| `GET` | `/api/predict/latest` | Most recent prediction, or HTTP 204 when none is available |

The client should poll status and the latest prediction about every 0.5 seconds. `/ws/predictions` deliberately returns HTTP 501 in this backend version.

The complete field definitions are in [`backend/API_CONTRACT.md`](../backend/API_CONTRACT.md). Important identity fields are:

- `contract_version`: interface version, currently `0.1`
- `source`: `live` or `replay`
- `session_id`: changes each time the backend starts
- `model_id`: model name in prediction responses
- `seq`: prediction sequence within one backend process
- `window.start` and `window.end`: CSI window timestamps

The client must treat a changed `session_id` as a new stream. It must not merge old and new sequence numbers or retain an old prediction as current.

## Backend states

| State | Meaning | Client behaviour |
|---|---|---|
| `offline` | No usable live input or no model | Show the message and do not advance positioning |
| `calibrating` | A labelled calibration round is being collected | Show `progress`; no prediction is expected |
| `training` | The current-session model is being trained | Wait for completion |
| `ready` | Four links and the model are available | Accept fresh predictions |
| `degraded` | At least one CSI link is unavailable | Show the failed links and pause positioning |

This implementation pauses prediction when any required link is unavailable. A previously returned prediction does not become a new prediction simply because it is returned again; use `session_id` and `seq` for de-duplication.

## Start a live service

Set the two Pi addresses in the current PowerShell session, then start the service from the repository root:

```powershell
$env:CSI_PIA_HOST = "<piA-address>"
$env:CSI_PIB_HOST = "<piB-address>"
python -X utf8 -m backend.live.server --session <session> --round 1
```

If the session has no trained live model, the server watches the named calibration round, trains after all 15 points are present, and then enters `ready`. For an offline transport test, use a captured round:

```powershell
python -X utf8 -m backend.live.server --session <session> --replay data\<session>\phase2_r1
```

The `--replay` option replays CSI packets through the live pipeline; it is different from the Unity result replay.

## Integration checks

Before a real run, verify all of the following:

1. `/api/meta` reports the expected grid, devices and a new `session_id`.
2. `/api/status` reaches `ready` with all four links online.
3. `/api/predict/latest` changes `seq` and contains 15 probabilities.
4. The Unity client displays `source`, session and model information without relabelling replay or mock data as live.
5. Disconnecting a required link produces `degraded` and stops route progress.
6. Restarting the backend starts a new session in the client.

These checks confirm transport and state handling. Accuracy and navigation performance require a separately recorded ground truth experiment.
