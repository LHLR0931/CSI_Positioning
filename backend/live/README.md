# Live F-MLP backend

The live pipeline receives CSI from two Raspberry Pi devices, separates and aligns four links, applies a 16-frame F-MLP model, and serves the result over HTTP. The interface is defined in [`../API_CONTRACT.md`](../API_CONTRACT.md).

## Requirements

Complete the receiver setup, clock synchronisation, warm-up and smoke test in [`../../capture/采集手册.md`](../../capture/采集手册.md). Set the current device addresses in the PowerShell process that will run the server:

```powershell
$env:CSI_PIA_HOST = "<piA-address>"
$env:CSI_PIB_HOST = "<piB-address>"
$env:CSI_APX_BSSID = "<apX-bssid>"
$env:CSI_APY_BSSID = "<apY-bssid>"
```

## Same-day calibration and service

Start the backend from the repository root:

```powershell
conda activate csi
python -X utf8 -m backend.live.server --session <session> --round 1
```

If a live model for the session is already available under `backend/artifacts/live/<session>/`, it is loaded immediately. Otherwise the service enters `calibrating` and watches the requested round. In another terminal, collect all 15 labelled positions:

```powershell
conda activate csi
.\phase2_capture.ps1 -Session <session> -Round 1 -RoomNote "<environment-note>"
```

After the round is complete, the service parses the data, trains the model and enters `ready`. Training products and `live_*.jsonl` prediction logs are written to `backend/artifacts/live/<session>/` and are excluded from Git.

The server listens on `0.0.0.0:8765`. A phone on the same network should use `http://<computer-ip>:8765`. Allow inbound access to this port only on the intended private network.

## Runtime behaviour

- The input is a 16-frame window across all four links.
- The prediction loop produces at most one new result every 0.4 seconds.
- A required link older than the buffer limit changes status to `degraded` and pauses new predictions.
- Restoring all four links allows the next complete window to resume inference.
- Each server start creates a new `session_id`; `seq` restarts from zero.

Do not reuse a model after the devices or CSI driver have been reset without running the required same-day calibration procedure. Live demonstration output is not an accuracy result unless it is paired with an independently controlled ground truth protocol.

## Replay transport test

An existing pcap round can exercise the same buffering, inference and HTTP path without connecting to the Pi devices:

```powershell
python -X utf8 -m backend.live.server --session <session> --replay data\<session>\phase2_r1
```

This mode reports `source = "replay"` and must not be presented as a live capture.

## Checks

```powershell
python -m pytest tests/test_live.py -q
python tests/check_client_rules.py http://127.0.0.1:8765 20
```

The second command requires a running service and validates the client-facing response rules for 20 seconds.
