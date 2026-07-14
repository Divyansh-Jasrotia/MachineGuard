"""Copy representative normal and abnormal MIMII clips into the demo assets."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


MACHINE_TYPES = ("fan", "pump", "valve")
SAMPLES = {"normal": "healthy", "abnormal": "faulty"}


def _first_wav(directory: Path) -> Path:
    files = sorted(path for path in directory.glob("*.wav") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No WAV files found in {directory}")
    return files[0]


def make_samples(data_root: str | Path = "data", destination: str | Path = "assets/samples") -> dict[str, str]:
    """Copy one labelled clip for each machine/condition pair.

    The files are taken from the curated data produced by ``download_data.py``;
    therefore a normal source is named ``healthy`` and an abnormal source is
    named ``faulty`` for the Gradio demo.
    """
    source_root = Path(data_root)
    sample_root = Path(destination)
    sample_root.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for machine_type in MACHINE_TYPES:
        for label, display_label in SAMPLES.items():
            source = _first_wav(source_root / machine_type / label)
            target = sample_root / f"{machine_type}_{display_label}.wav"
            shutil.copy2(source, target)
            copied[target.name] = str(source)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Curated MIMII data directory")
    parser.add_argument("--destination", type=Path, default=Path("assets/samples"), help="Demo sample directory")
    args = parser.parse_args()
    for target, source in make_samples(args.data_root, args.destination).items():
        print(f"{target} <- {source}")


if __name__ == "__main__":
    main()
