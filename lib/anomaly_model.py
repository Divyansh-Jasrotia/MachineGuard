"""Calibrated one-class anomaly detectors for machine-sound features."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


DETECTORS = ("gmm", "isolation_forest")


def fit_detector(normal_features: np.ndarray, detector: str = "gmm", random_state: int = 42) -> dict:
    """Fit a scaler and one-class detector using normal features only."""
    features = np.asarray(normal_features, dtype=np.float64)
    if features.ndim != 2 or len(features) < 4:
        raise ValueError("At least four normal feature vectors are required to train a detector.")
    if detector not in DETECTORS:
        raise ValueError(f"detector must be one of {DETECTORS}")

    scaler = StandardScaler().fit(features)
    scaled = scaler.transform(features)
    if detector == "gmm":
        model = GaussianMixture(n_components=4, covariance_type="full", random_state=random_state).fit(scaled)
    else:
        model = IsolationForest(n_estimators=300, random_state=random_state, n_jobs=-1).fit(scaled)
    return {"detector": detector, "scaler": scaler, "model": model, "calibration": None}


def raw_anomaly_scores(bundle: dict, features: np.ndarray) -> np.ndarray:
    """Return uncalibrated anomaly scores where larger values mean less normal."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("features must be one feature vector or a two-dimensional feature matrix.")
    scaled = bundle["scaler"].transform(values)
    if bundle["detector"] == "gmm":
        return -bundle["model"].score_samples(scaled)
    return -bundle["model"].score_samples(scaled)


def calibrate(bundle: dict, normal_validation_features: np.ndarray, threshold: float = 60.0) -> dict:
    """Fit 0–100 min-max calibration from the held-out normal distribution."""
    raw = raw_anomaly_scores(bundle, normal_validation_features)
    low, high = float(np.min(raw)), float(np.max(raw))
    # A constant normal-validation distribution remains a valid, stable bundle.
    if np.isclose(low, high):
        high = low + 1e-8
    bundle["calibration"] = {"raw_min": low, "raw_max": high, "threshold": float(threshold)}
    return bundle


def score_detector(bundle: dict, features: np.ndarray) -> float | np.ndarray:
    """Return min-max calibrated anomaly scores clipped to the 0–100 range."""
    calibration = bundle.get("calibration")
    if calibration is None:
        raise ValueError("Detector has not been calibrated.")
    values = np.asarray(features)
    raw = raw_anomaly_scores(bundle, values)
    scores = 100.0 * (raw - calibration["raw_min"]) / (calibration["raw_max"] - calibration["raw_min"])
    scores = np.clip(scores, 0.0, 100.0)
    return float(scores[0]) if values.ndim == 1 else scores


def train_detector(normal_features: np.ndarray, detector: str = "gmm", random_state: int = 42) -> dict:
    """Compatibility wrapper for fitting a detector from normal features."""
    return fit_detector(normal_features, detector, random_state)


def save_detector(bundle: dict, path: str | Path) -> Path:
    """Save scaler, model, detector choice, and calibration in one joblib file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    return output


def load_detector(path: str | Path) -> dict:
    """Load a detector bundle created by :func:`save_detector`."""
    return joblib.load(path)
