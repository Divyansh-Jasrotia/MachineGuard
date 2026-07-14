"""Download and curate the 6 dB MIMII fan, pump, and valve recordings.

The MIMII archive layout can differ by release.  This script discovers the
available ``id_XX`` folders rather than assuming a fixed range of model IDs.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import requests


RECORD_URL = "https://zenodo.org/records/3384388/files/{archive}?download=1"
ARCHIVES = {
    "fan": "6_dB_fan.zip",
    "pump": "6_dB_pump.zip",
    "valve": "6_dB_valve.zip",
}
TARGET_COUNTS = {"normal": 200, "abnormal": 100}


def _format_bytes(value: float) -> str:
    """Format a byte count for compact terminal progress output."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _format_duration(seconds: float) -> str:
    """Format an ETA without displaying a misleading value for unknown time."""
    if seconds == float("inf"):
        return "unknown"
    seconds = max(0, round(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:d}:{seconds:02d}"


def _download_archive(url: str, archive_path: Path) -> None:
    """Stream an archive to disk while reporting percent, speed, and ETA."""
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        started = time.monotonic()
        last_report = started
        with archive_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                output.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 0.25 or (total and downloaded >= total):
                    elapsed = max(now - started, 0.001)
                    speed = downloaded / elapsed
                    percent = f"{downloaded / total * 100:6.2f}%" if total else "   n/a"
                    remaining = (total - downloaded) / speed if total and speed else float("inf")
                    total_text = _format_bytes(total) if total else "unknown total"
                    message = (
                        f"\r  {percent}  {_format_bytes(downloaded)} / {total_text}  "
                        f"{_format_bytes(speed)}/s  ETA {_format_duration(remaining)}"
                    )
                    print(message, end="", file=sys.stdout, flush=True)
                    last_report = now
    print(file=sys.stdout, flush=True)


def _label_and_model(path: Path) -> tuple[str, str] | None:
    """Infer a MIMII label and model ID from a WAV path in an archive."""
    parts = [part.lower() for part in path.parts]
    filename = path.name.lower()
    label = next((name for name in TARGET_COUNTS if name in parts), None)
    if label is None:
        label = next((name for name in TARGET_COUNTS if filename.startswith(f"{name}_")), None)
    model = next((part for part in parts if part.startswith("id_") and part[3:].isdigit()), None)
    if label is None or model is None:
        return None
    return label, model


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract a ZIP archive while rejecting paths that escape its directory."""
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        zipped.extractall(destination)


def _sample_across_models(grouped: dict[str, list[Path]], count: int, rng: random.Random) -> list[Path]:
    """Select up to ``count`` clips in a round-robin across discovered IDs."""
    available = {model: sorted(files) for model, files in grouped.items()}
    for files in available.values():
        rng.shuffle(files)

    selected: list[Path] = []
    while len(selected) < count:
        progress = False
        for model in sorted(available):
            if available[model] and len(selected) < count:
                selected.append(available[model].pop())
                progress = True
        if not progress:
            break
    if len(selected) != count:
        raise RuntimeError(f"Needed {count} clips but found only {len(selected)} across {sorted(grouped)}")
    return selected


def _copy_curated_files(extracted: Path, machine_type: str, output_root: Path) -> dict[str, int]:
    """Copy deterministic, ID-balanced selections to ``data/<type>/<label>``."""
    grouped: dict[str, dict[str, list[Path]]] = {
        label: defaultdict(list) for label in TARGET_COUNTS
    }
    for wav_path in extracted.rglob("*.wav"):
        classified = _label_and_model(wav_path.relative_to(extracted))
        if classified is not None:
            label, model = classified
            grouped[label][model].append(wav_path)

    rng = random.Random(f"mimii-3384388-{machine_type}")
    selections = {
        label: _sample_across_models(grouped[label], count, rng)
        for label, count in TARGET_COUNTS.items()
    }

    for label, files in selections.items():
        label_directory = output_root / machine_type / label
        if label_directory.exists():
            shutil.rmtree(label_directory)
        label_directory.mkdir(parents=True, exist_ok=True)
        for index, source in enumerate(files, start=1):
            # Prefixing retains the source ID and prevents duplicate basenames.
            model = _label_and_model(source.relative_to(extracted))[1]
            target = label_directory / f"{model}_{index:03d}_{source.name}"
            shutil.copy2(source, target)
    return {label: len(files) for label, files in selections.items()}


def download_mimii_subsets(destination: str | Path = "data", dry_run: bool = False) -> dict[str, dict[str, int] | str]:
    """Download, extract, curate, and remove the three requested MIMII archives."""
    output_root = Path(destination)
    if dry_run:
        return {
            machine: f"would download {archive} and retain {TARGET_COUNTS['normal']} normal / {TARGET_COUNTS['abnormal']} abnormal WAVs"
            for machine, archive in ARCHIVES.items()
        }

    summary: dict[str, dict[str, int]] = {}
    with tempfile.TemporaryDirectory(prefix="machineguard-mimii-") as temporary:
        temporary_path = Path(temporary)
        for machine_type, archive_name in ARCHIVES.items():
            archive_path = temporary_path / archive_name
            extract_path = temporary_path / machine_type
            url = RECORD_URL.format(archive=archive_name)
            print(f"Downloading {url}")
            try:
                _download_archive(url, archive_path)
                print(f"Extracting {archive_name} …")
                _safe_extract(archive_path, extract_path)
                summary[machine_type] = _copy_curated_files(extract_path, machine_type, output_root)
                print(f"Curated {machine_type}: {summary[machine_type]}")
            finally:
                # TemporaryDirectory also cleans up, but this makes archive removal explicit.
                archive_path.unlink(missing_ok=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("data"), help="Curated data directory")
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without changing files")
    args = parser.parse_args()
    summary = download_mimii_subsets(args.destination, args.dry_run)
    for machine_type, result in summary.items():
        print(f"{machine_type}: {result}")


if __name__ == "__main__":
    main()
