"""Audio loading and fixed-length feature extraction for MachineGuard."""

from __future__ import annotations

from os import PathLike

import librosa
import numpy as np
import soundfile as sf


TARGET_RMS = 0.1
SAMPLE_RATE = 16_000


def load_audio(
    path_or_array: str | PathLike[str] | np.ndarray | tuple[int, np.ndarray] | tuple[np.ndarray, int] | list | dict,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """Load mono ``float32`` audio at ``sr`` and RMS-normalize it afterwards.

    File paths are decoded and resampled by librosa. Arrays are assumed already
    to use ``sr``; alternatively pass ``(array, source_sample_rate)`` or
    ``(source_sample_rate, array)`` when an array has a different rate.
    Stereo arrays are averaged to mono.
    """
    if isinstance(path_or_array, (str, PathLike)):
        y, original_sr = sf.read(path_or_array, dtype="float32")
        if y.size == 0:
            raise ValueError(f"Audio file {path_or_array} contains no samples.")
        if y.ndim == 2:
            axis = 0 if y.shape[0] <= y.shape[1] else 1
            y = np.mean(y, axis=axis, dtype=np.float32)
        if original_sr != sr:
            y = librosa.resample(y, orig_sr=original_sr, target_sr=sr)
    else:
        source_sr = sr
        audio = path_or_array
        if isinstance(path_or_array, dict):
            if "sample_rate" in path_or_array and "data" in path_or_array:
                source_sr = int(path_or_array["sample_rate"])
                audio = path_or_array["data"]
            elif "sr" in path_or_array and "array" in path_or_array:
                source_sr = int(path_or_array["sr"])
                audio = path_or_array["array"]
            elif "path" in path_or_array:
                return load_audio(path_or_array["path"], sr=sr)
            elif "name" in path_or_array and "mime_type" in path_or_array and "data" in path_or_array:
                # Gradio may pass recorded audio as a dict with metadata keys.
                source_sr = int(path_or_array.get("sample_rate", sr))
                audio = path_or_array["data"]
            else:
                raise ValueError(
                    "Unsupported audio dict format. Expected keys like 'sample_rate' and 'data'."
                )
        if isinstance(path_or_array, (tuple, list)):
            if len(path_or_array) != 2:
                raise ValueError(
                    "Audio tuple/list must be either (audio, sample_rate) or (sample_rate, audio)."
                )
            first, second = path_or_array
            if isinstance(first, (int, np.integer)) and not isinstance(second, (int, np.integer)):
                source_sr, audio = int(first), second
            elif isinstance(second, (int, np.integer)) and not isinstance(first, (int, np.integer)):
                source_sr, audio = int(second), first
            elif isinstance(first, (str, PathLike)):
                return load_audio(first, sr=sr)
            elif isinstance(second, (str, PathLike)):
                return load_audio(second, sr=sr)
            elif isinstance(first, np.ndarray):
                audio = first
                source_sr = sr
            elif isinstance(second, np.ndarray):
                audio = second
                source_sr = sr
            else:
                audio = first
                source_sr = sr
        y = np.asarray(audio, dtype=np.float32)
        if y.ndim == 2:
            # Handles both (channels, samples) and (samples, channels).
            axis = 0 if y.shape[0] <= y.shape[1] else 1
            y = np.mean(y, axis=axis, dtype=np.float32)
        if y.ndim != 1:
            raise ValueError("Audio must be a one-dimensional mono array or a two-dimensional channel array.")
        if source_sr != sr:
            y = librosa.resample(y, orig_sr=source_sr, target_sr=sr)

    y = np.ascontiguousarray(y, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float32)))) if y.size else 0.0
    if rms > 1e-8:
        y = y * (TARGET_RMS / rms)
    return y.astype(np.float32, copy=False)


def compute_melspec(y: np.ndarray) -> np.ndarray:
    """Return a 64-band log-mel spectrogram for plotting a 16 kHz signal."""
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("compute_melspec expects a non-empty mono signal.")
    mel_power = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64, power=2.0
    )
    return librosa.power_to_db(mel_power, ref=np.max).astype(np.float32)


def extract_features(y: np.ndarray) -> np.ndarray:
    """Return a 14-dimensional acoustic feature vector for a 16 kHz signal.

    Vector order: RMS mean; centroid, bandwidth, rolloff, and flatness each as
    mean/std; ZCR mean; then energy ratios for 0–500 Hz, 0.5–2 kHz, 2–5 kHz,
    and 5–8 kHz.  The latter four bands act as fault-fingerprint features.
    """
    y = np.asarray(y, dtype=np.float32)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("extract_features expects a non-empty mono signal.")

    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE, n_fft=1024, hop_length=512)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE, n_fft=1024, hop_length=512)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE, n_fft=1024, hop_length=512)[0]
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=1024, hop_length=512)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=512)[0]

    spectrum = np.abs(librosa.stft(y, n_fft=1024, hop_length=512)) ** 2
    frequencies = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=1024)
    total_energy = float(np.sum(spectrum))
    bands = ((0, 500), (500, 2_000), (2_000, 5_000), (5_000, 8_000))
    band_ratios = [
        float(np.sum(spectrum[(frequencies >= low) & (frequencies <= high)])) / max(total_energy, 1e-12)
        for low, high in bands
    ]

    features = [
        float(np.mean(rms)),
        float(np.mean(centroid)), float(np.std(centroid)),
        float(np.mean(bandwidth)), float(np.std(bandwidth)),
        float(np.mean(rolloff)), float(np.std(rolloff)),
        float(np.mean(flatness)), float(np.std(flatness)),
        float(np.mean(zcr)),
        *band_ratios,
    ]
    return np.asarray(features, dtype=np.float32)


if __name__ == "__main__":
    from pathlib import Path

    sample = Path(__file__).resolve().parents[1] / "assets" / "samples" / "fan_healthy.wav"
    audio = load_audio(sample)
    mel = compute_melspec(audio)
    features = extract_features(audio)
    print(f"audio shape: {audio.shape}")
    print(f"mel shape: {mel.shape}")
    print(f"features shape: {features.shape}")
    print(f"features: {features}")
