"""Build deterministic D1 and D+3 data archives for experiment reproduction."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "_github_release"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
SESSIONS = {
    "d1": tuple(range(1, 11)),
    "d3": tuple(range(1, 7)),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_files(session: str, rounds: tuple[int, ...]) -> list[Path]:
    paths = [
        ROOT / "datasets" / "raw" / session / "static_raw64.npz",
        ROOT / "datasets" / "raw" / session / "truncation_report.json",
        ROOT / "data" / session / "session_meta.json",
    ]
    paths.extend(
        ROOT / "data" / session / f"phase2_r{round_id}" / "points_meta.csv"
        for round_id in rounds
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        formatted = "\n".join(str(path.relative_to(ROOT)) for path in missing)
        raise FileNotFoundError(f"Missing required data files:\n{formatted}")
    return paths


def zip_info(name: str, compression: int = zipfile.ZIP_STORED) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
    info.compress_type = compression
    info.external_attr = 0o100644 << 16
    return info


def add_file(archive: zipfile.ZipFile, source: Path, archive_name: str) -> None:
    with source.open("rb") as src, archive.open(zip_info(archive_name), "w") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)


def write_sanitized_npz(source: Path, destination: Path) -> None:
    """Copy an NPZ while replacing hardware BSSIDs with logical AP identifiers."""
    with np.load(source, allow_pickle=False) as arrays:
        links = np.asarray(arrays["links"]).astype(str)
        logical_bssids = np.asarray(
            [link.rsplit("_", 1)[-1] for link in links], dtype="<U3"
        )
        if set(logical_bssids.tolist()) != {"apX", "apY"}:
            raise ValueError(f"Unexpected link identifiers in {source}")

        with zipfile.ZipFile(destination, "w", allowZip64=True) as archive:
            for name in arrays.files:
                array = logical_bssids if name == "bssid" else arrays[name]
                buffer = io.BytesIO()
                np.lib.format.write_array(buffer, np.asarray(array), allow_pickle=False)
                archive.writestr(
                    zip_info(f"{name}.npy", zipfile.ZIP_DEFLATED),
                    buffer.getvalue(),
                )


def build_session(session: str, rounds: tuple[int, ...]) -> Path:
    files = required_files(session, rounds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"csi-reproduction-{session}.zip"
    temporary = output.with_suffix(".zip.tmp")
    sanitized_npz = OUT_DIR / f".{session}.static_raw64.sanitized.npz"
    if temporary.exists():
        temporary.unlink()
    if sanitized_npz.exists():
        sanitized_npz.unlink()

    source_npz = ROOT / "datasets" / "raw" / session / "static_raw64.npz"
    try:
        write_sanitized_npz(source_npz, sanitized_npz)
        package_files = [
            (
                sanitized_npz if path == source_npz else path,
                path.relative_to(ROOT).as_posix(),
            )
            for path in files
        ]
        entries = [
            {
                "path": archive_name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path, archive_name in package_files
        ]

        manifest = {
            "format_version": 1,
            "session": session,
            "extract_to": "repository root",
            "metadata_policy": "Hardware BSSIDs are replaced by apX/apY logical identifiers.",
            "files": entries,
        }
        manifest_bytes = (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for path, archive_name in package_files:
                add_file(archive, path, archive_name)
            archive.writestr(zip_info("dataset_manifest.json"), manifest_bytes)

        with zipfile.ZipFile(temporary, "r") as archive:
            corrupt = archive.testzip()
            if corrupt is not None:
                raise RuntimeError(f"Archive integrity check failed: {corrupt}")
            expected = {entry["path"] for entry in entries} | {"dataset_manifest.json"}
            if set(archive.namelist()) != expected:
                raise RuntimeError(f"Unexpected archive contents in {temporary}")

        temporary.replace(output)
    finally:
        sanitized_npz.unlink(missing_ok=True)
    return output


def main() -> None:
    for session, rounds in SESSIONS.items():
        output = build_session(session, rounds)
        print(f"{sha256(output)}  {output.name}")


if __name__ == "__main__":
    main()
