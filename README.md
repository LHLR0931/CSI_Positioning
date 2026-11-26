# Wi-Fi CSI Device-Free Indoor Positioning

This repository contains the reproducible code and derived data interface for a four-link Wi-Fi channel-state-information (CSI) indoor-localisation experiment. Two access points and two Nexmon-enabled Raspberry Pi receivers form four links around a 5 × 3 grid with 0.8 m spacing.

The experimental code covers CSI parsing, leakage-safe preprocessing, D1 training and evaluation, D+3 recalibration experiments, diagnostic analyses, and an HTTP back end for live demonstrations. The Unity application is an optional Android demonstration client.

## Repository layout

| Path | Purpose |
|---|---|
| `backend/data/` | PCAP parsing, session definitions, loading, splitting, and preprocessing |
| `backend/models/` | Single-link and multi-link models |
| `backend/training/` | D1, cross-day, fusion, ablation, and diagnostic experiments |
| `backend/live/` | Same-day calibration, F-MLP inference, and HTTP service |
| `capture/` | Raspberry Pi setup scripts and the data-collection manual |
| `tests/` | Automated tests for data, metrics, live parsing, and interfaces |
| `docs/DATASET.md` | Reproduction-data download and integrity instructions |
| `report/健康分预登记.md` | Frozen link-health preregistration used by `crossday.py` |
| `unity-app/` | Optional Unity/Android demonstration client |

## Environment

The tested environment is Python 3.11 with PyTorch 2.11.0 and CUDA 12.8. A compatible NVIDIA driver is required for GPU training.

```powershell
conda create -n csi python=3.11 -y
conda activate csi
cd <repository-directory>
python -m pip install -r requirements-tested.txt
python -X utf8 -m pytest -q
```

`requirements-tested.txt` records the final experiment environment. `requirements.txt` provides compatible version ranges for later development.

## Reproduction data

Download `csi-reproduction-d1.zip` and `csi-reproduction-d3.zip` from the [repository Releases](https://github.com/LHLR0931/CSI_Positioning/releases). Verify their SHA-256 values and extract both archives into the repository root as described in [`docs/DATASET.md`](docs/DATASET.md).

The extracted layout must contain:

```text
datasets/raw/d1/static_raw64.npz
datasets/raw/d3/static_raw64.npz
data/d1/session_meta.json
data/d3/session_meta.json
```

Raw PCAP captures are not required to reproduce the final training and evaluation pipeline.

## Reproduce the experiments

Run commands from the repository root in the `csi` environment. Training outputs are written below `backend/artifacts/`.

### 1. Random-split diagnostic

```powershell
python -X utf8 -m backend.training.random_split_check --seed 42 1 2
python -X utf8 -m backend.training.random_split_check --summary
```

### 2. D1 model selection and final round-wise evaluation

```powershell
python -X utf8 -m backend.training.train_d1 --phase select --task cls --seed 42 1 2
python -X utf8 -m backend.training.train_d1 --phase select --summary
python -X utf8 -m backend.training.train_d1 --phase final --task cls --seed 42 1 2
python -X utf8 -m backend.training.train_d1 --phase final --summary
python -X utf8 -m backend.training.best_single_link_check
```

### 3. Frozen-feature fusion comparison

```powershell
python -X utf8 -m backend.training.frozen_fusion --phase select
python -X utf8 -m backend.training.frozen_fusion --phase final
python -X utf8 -m backend.training.frozen_fusion --summary
```

### 4. D+3 recalibration and link-health evaluation

```powershell
python -X utf8 -m backend.training.crossday train --day d3
python -X utf8 -m backend.training.crossday evaluate --day d3 --unblind
python -X utf8 -m backend.training.crossday report
```

`crossday.py` verifies the frozen preregistration file before running the D+3 analysis.

### 5. D+3 F-MLP and diagnostic analyses

```powershell
python -X utf8 -m backend.training.frozen_fusion_d3
python -X utf8 -m backend.training.frozen_fusion_nocal
python -X utf8 -m backend.training.e5_d3
python -X utf8 -m backend.training.fmlp_factor_ablation_d3
python -X utf8 -m backend.training.temporal_point_analysis
```

The file headers in `backend/training/` describe the required preceding artifacts, fixed decision rules, and output paths for each analysis.

## Collect a new session

Real device addresses are supplied only in the current terminal session:

```powershell
$env:CSI_PIA_HOST = "<piA-address>"
$env:CSI_PIB_HOST = "<piB-address>"
$env:CSI_APX_BSSID = "<apX-bssid>"
$env:CSI_APY_BSSID = "<apY-bssid>"
```

Copy the scripts in `capture/pi/` to both receivers, follow [`capture/采集手册.md`](capture/采集手册.md), and run:

```powershell
.\phase2_capture.ps1 -Session smoke -Round 1 -Only 0
.\phase2_capture.ps1 -Session <new-session> -Round 1 -RoomNote "<environment-note>"
python -X utf8 -m backend.data.load_real_stage2 <new-session> --sc-from d1
```

Register the new session and round roles in `backend/data/sessions.py` before training or evaluation.

## Live back end

After collecting a same-day calibration round, start the HTTP service with:

```powershell
python -X utf8 -m backend.live.server --session <session> --round 1
```

The service polls four live links, trains or loads an F-MLP model, and exposes `/api/meta`, `/api/status`, and `/api/predict/latest` on port 8765. See [`backend/live/README.md`](backend/live/README.md) and [`backend/API_CONTRACT.md`](backend/API_CONTRACT.md).

## Optional Unity client

The Unity project uses Unity 6000.6.3f1 with Android Build Support. Build and interface instructions are in [`unity-app/README.md`](unity-app/README.md). The bundled replay is a historical held-out demonstration and is not a substitute for a live same-day calibration experiment.

## Verification

```powershell
python -X utf8 -m pytest -q
python -m pip check
```

The data archives have independent SHA-256 values in [`docs/dataset_checksums.sha256`](docs/dataset_checksums.sha256).
