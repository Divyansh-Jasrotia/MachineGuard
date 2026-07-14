"""Train calibrated MIMII anomaly detectors and report held-out ROC-AUC."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# Allow both ``python scripts/train.py`` and ``python -m scripts.train``.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.anomaly_model import calibrate, fit_detector, save_detector, score_detector
from lib.audio_processing import extract_features, load_audio


MACHINE_TYPES = ("fan", "pump", "valve")


def _feature_matrix(folder: Path) -> np.ndarray:
    """Load all WAVs below a folder as one 14-column feature matrix."""
    files = sorted(path for path in folder.rglob("*.wav") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No WAV clips found in {folder}")
    features = []
    for index, path in enumerate(files, start=1):
        try:
            features.append(extract_features(load_audio(path)))
        except Exception as error:
            raise RuntimeError(f"Could not process {path}: {error}") from error
        print(f"\r  features: {index}/{len(files)}", end="", flush=True)
    print()
    return np.vstack(features)


def _best_threshold(normal_scores: np.ndarray, abnormal_scores: np.ndarray) -> float:
    """Choose the validation threshold that maximizes Youden's J statistic."""
    labels = np.concatenate((np.zeros(len(normal_scores)), np.ones(len(abnormal_scores))))
    scores = np.concatenate((normal_scores, abnormal_scores))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    valid = np.isfinite(thresholds)
    if not np.any(valid):
        return 60.0
    return float(np.clip(thresholds[valid][np.argmax((true_positive_rate - false_positive_rate)[valid])], 0, 100))


def _evaluate_detector(
    train_normals: np.ndarray,
    validation_normals: np.ndarray,
    abnormal_features: np.ndarray | None,
    detector: str,
) -> tuple[dict, float | None, float]:
    """Fit, calibrate, calculate real AUC, and tune a validation threshold."""
    bundle = fit_detector(train_normals, detector=detector)
    calibrate(bundle, validation_normals)
    normal_scores = score_detector(bundle, validation_normals)
    if abnormal_features is None:
        bundle["calibration"]["threshold"] = 60.0
        return bundle, None, 60.0

    abnormal_scores = score_detector(bundle, abnormal_features)
    labels = np.concatenate((np.zeros(len(normal_scores)), np.ones(len(abnormal_scores))))
    scores = np.concatenate((normal_scores, abnormal_scores))
    auc = float(roc_auc_score(labels, scores))
    threshold = _best_threshold(normal_scores, abnormal_scores)
    bundle["calibration"]["threshold"] = threshold
    return bundle, auc, threshold


def train_machine(
    machine_type: str,
    normal_features: np.ndarray,
    abnormal_features: np.ndarray | None,
    requested_detector: str = "gmm",
) -> tuple[dict, dict]:
    """Train one type, automatically falling back to IsolationForest below AUC .75."""
    if len(normal_features) < 5:
        raise ValueError("At least five normal clips are required for an 80/20 split and 4-component GMM.")
    train_normals, validation_normals = train_test_split(normal_features, test_size=0.20, random_state=42)
    bundle, auc, threshold = _evaluate_detector(
        train_normals, validation_normals, abnormal_features, requested_detector
    )
    note = ""
    if requested_detector == "gmm" and auc is not None and auc < 0.75:
        gmm_auc = auc
        bundle, auc, threshold = _evaluate_detector(
            train_normals, validation_normals, abnormal_features, "isolation_forest"
        )
        note = f"GMM AUC {gmm_auc:.3f} was below 0.750; switched to IsolationForest."
    metrics = {
        "machine_type": machine_type,
        "detector": bundle["detector"],
        "auc": auc,
        "threshold": threshold,
        "note": note,
    }
    return bundle, metrics


def _write_metrics(metrics: list[dict], output: Path) -> None:
    """Write only measured values; types without abnormal validation show N/A."""
    rows = ["# MachineGuard training metrics", "", "| Machine | Detector | ROC-AUC | Threshold | Notes |", "|---|---|---:|---:|---|"]
    for result in metrics:
        auc = f"{result['auc']:.3f}" if result["auc"] is not None else "N/A (normal-only custom fit)"
        note = result["note"] or ""
        rows.append(f"| {result['machine_type']} | {result['detector']} | {auc} | {result['threshold']:.2f} | {note} |")
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def train_all(
    data_root: str | Path = "data",
    models_dir: str | Path = "models",
    detector: str = "gmm",
    custom_folder: str | Path | None = None,
    machine_type: str | None = None,
) -> list[dict]:
    """Train all curated types, or one normal-only custom folder when supplied."""
    if detector not in ("gmm", "isolation_forest"):
        raise ValueError("detector must be 'gmm' or 'isolation_forest'")
    data_root, models_dir = Path(data_root), Path(models_dir)
    if custom_folder is not None:
        if machine_type not in MACHINE_TYPES:
            raise ValueError("--machine-type fan, pump, or valve is required with --custom-folder")
        jobs = [(machine_type, Path(custom_folder), None)]
    else:
        jobs = [(kind, data_root / kind / "normal", data_root / kind / "abnormal") for kind in MACHINE_TYPES]

    metrics: list[dict] = []
    for kind, normal_folder, abnormal_folder in jobs:
        print(f"Training {kind} from {normal_folder}")
        normals = _feature_matrix(normal_folder)
        abnormals = _feature_matrix(abnormal_folder) if abnormal_folder is not None else None
        bundle, result = train_machine(kind, normals, abnormals, detector)
        save_detector(bundle, models_dir / f"{kind}.joblib")
        metrics.append(result)
        print(f"  saved {models_dir / f'{kind}.joblib'}; AUC={result['auc']}; threshold={result['threshold']:.2f}")
    _write_metrics(metrics, models_dir.parent / "metrics.md")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--detector", choices=("gmm", "isolation_forest"), default="gmm")
    parser.add_argument("--custom-folder", type=Path, help="Normal recordings for one machine-type refit")
    parser.add_argument("--machine-type", choices=MACHINE_TYPES, help="Type to use with --custom-folder")
    args = parser.parse_args()
    train_all(args.data_root, args.models_dir, args.detector, args.custom_folder, args.machine_type)


if __name__ == "__main__":
    main()
