# Reproduction data

The training and analysis scripts require two data archives that are distributed separately from the source repository:

- `csi-reproduction-d1.zip`: D1 rounds r1–r10.
- `csi-reproduction-d3.zip`: D+3 rounds r1–r6.

Each archive contains the derived 64-subcarrier CSI array, the truncation report, the session metadata, and the per-round timing metadata. Hardware BSSIDs in the array metadata are replaced by the logical identifiers `apX` and `apY`; this does not alter the CSI samples or experiment labels. The archives do not contain raw PCAP captures, anomaly captures, photographs, or historical pilot sessions.

## Installation

Download both archives and extract them into the repository root. The resulting layout must include:

```text
data/d1/session_meta.json
data/d1/phase2_r1/points_meta.csv
data/d3/session_meta.json
data/d3/phase2_r1/points_meta.csv
datasets/raw/d1/static_raw64.npz
datasets/raw/d3/static_raw64.npz
```

Existing empty directories may be merged when extracting. Do not place the archives themselves inside `data/` or `datasets/`.

## Integrity check

Compare the SHA-256 values of the downloaded archives with `docs/dataset_checksums.sha256`:

```powershell
Get-FileHash .\csi-reproduction-d1.zip -Algorithm SHA256
Get-FileHash .\csi-reproduction-d3.zip -Algorithm SHA256
```

Each archive also contains `dataset_manifest.json`, which records the path, size, and SHA-256 value of every included file.

## Rebuilding the archives

If the verified local data are already present, rebuild both archives with:

```powershell
python -X utf8 tools/package_reproduction_data.py
```

The generated archives are written to `_github_release/`. They should be distributed as release assets rather than committed to the source repository.

Raw PCAP data are not required to rerun the final training and evaluation pipeline. To process a newly collected session, follow `capture/采集手册.md` and run `backend.data.load_real_stage2` as described in the main README.
