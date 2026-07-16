"""MachineGuard's Gradio product interface."""
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from os import PathLike
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from lib.anomaly_model import load_detector, score_detector
from lib.audio_processing import SAMPLE_RATE, compute_melspec, extract_features, load_audio
from lib.diagnosis import build_evidence, generate_narrative, synthesize_verdict


ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "assets" / "samples"
MODEL_DIR = ROOT / "models"
HISTORY_HEADERS = ["Time", "Machine", "Score", "Verdict"]
MIN_RECORDING_SECONDS = 5
MAX_RECORDING_SECONDS = 12


@lru_cache(maxsize=3)
def _model_for(machine_type: str):
    """Load a trained bundle once per Gradio session process, if one exists."""
    try:
        return load_detector(MODEL_DIR / f"{machine_type}.joblib")
    except Exception:
        return None


def _mel_plot(mel):
    """Render lib-produced mel data as a compact dark-friendly plot."""
    figure, axis = plt.subplots(figsize=(8, 3.4))
    image = axis.imshow(mel, aspect="auto", origin="lower", cmap="magma")
    axis.set(title="Log-mel spectrogram", xlabel="Frames", ylabel="Mel band")
    figure.colorbar(image, ax=axis, label="dB")
    figure.tight_layout()
    return figure


def _warm_analysis_pipeline():
    """Compile/load local analysis dependencies before a user records audio."""
    # Librosa uses Numba for parts of feature extraction.  Its first invocation
    # can compile for a noticeable time; doing it during app startup prevents a
    # recording from appearing to freeze after the user presses Stop.
    warm_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    extract_features(warm_audio)
    compute_melspec(warm_audio)
    for kind in ("fan", "pump", "valve"):
        _model_for(kind)


def _health_label(score: float) -> dict[str, float]:
    """Map the calibrated score to the product's color-coded health state."""
    if score > 60:
        state, dot = "FAULT", "🔴"
    elif score > 35:
        state, dot = "WARNING", "🟡"
    else:
        state, dot = "HEALTHY", "🟢"
    return {f"{dot} {state} — {score:.1f} / 100": 1.0}


def analyze(audio_input, machine_type, language, history):
    """Thin UI adapter around the audio, model, and diagnosis libraries."""
    history = list(history or [])
    if not audio_input:
        return {"Awaiting audio": 1.0}, None, "Upload or record a machine clip to begin.", "", None, history, history
    try:
        payload_type = type(audio_input).__name__
        if isinstance(audio_input, (tuple, list)) and len(audio_input) == 2:
            payload_desc = f"tuple(sample_rate={audio_input[0]}, shape={getattr(audio_input[1], 'shape', type(audio_input[1]).__name__)})"
        elif isinstance(audio_input, dict):
            payload_desc = f"dict(keys={list(audio_input.keys())})"
        elif isinstance(audio_input, (str, PathLike)):
            audio_path = Path(audio_input)
            if audio_path.exists():
                payload_desc = f"path={audio_input} size={audio_path.stat().st_size} bytes"
            else:
                payload_desc = f"path={audio_input} missing"
        else:
            payload_desc = f"{payload_type}"

        audio = load_audio(audio_input)
        if audio.size == 0:
            raise ValueError("Loaded audio contains no samples.")
        silence_ratio = float(np.mean(np.abs(audio) < 1e-3))
        if silence_ratio > 0.8:
            return {
                "Awaiting audio": 1.0
            }, None, (
                "### Recording appears mostly silent\n"
                "Please record again with a stronger machine sound."
            ), "", None, history, history
        features = extract_features(audio)
        model = _model_for(machine_type)
        score = float(score_detector(model, features)) if model is not None else 0.0
        model_name = f"{model['detector']}_v1" if model is not None else "model_not_trained"
        evidence = build_evidence(
            machine_type,
            score,
            {"clip_seconds": len(audio) / SAMPLE_RATE, "model": model_name},
            features[-4:],
        )
        narrative = generate_narrative(evidence, language)
        verdict = evidence["verdict"]
        history.insert(0, [datetime.now().strftime("%H:%M:%S"), machine_type, round(score, 1), verdict])
        history = history[:50]

        return _health_label(score), _mel_plot(compute_melspec(audio)), narrative, narrative, None, history, history
    except Exception as error:
        return {"Analysis unavailable": 1.0}, None, f"### Unable to analyse this clip\n`{error}`", "", None, history, history


def speak_verdict(narrative: str):
    """Synthesize the displayed diagnosis when the user asks to hear it."""
    audio_path, error_message = synthesize_verdict(narrative or "")
    if audio_path:
        return audio_path, "Speech is ready. Press Play in the audio player."
    return None, f"Speech could not be generated. {error_message}"


def choose_sample(filename: str, machine_type: str):
    """Supply a sample path and its machine type to a chained Gradio event."""
    return str(SAMPLE_DIR / filename), machine_type


dark_theme = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate",
).set(body_background_fill="#07111f", block_background_fill="#0e1b2d")

css = """
footer {display: none !important}
.gradio-container {background: #07111f !important; color: #e6f0ff !important; max-width: 1240px !important}
#title {text-align:center; margin: 20px 0 2px; font-size: 2.25rem}
#tagline {text-align:center; color:#91a4bd; margin:0 0 20px}
#footer {text-align:center; color:#91a4bd; margin:24px 0 8px}
#recording-tip {background:#0d2236; border:1px solid #1c4969; border-radius:10px; color:#b9dff6; margin:8px 0 14px; padding:10px 13px}
#result-heading {margin:2px 0 8px; color:#d9edff}
.primary-action button {min-height:48px; font-size:1.05rem; font-weight:700}
.sample-row button {border-color:#274764 !important}
"""


with gr.Blocks(theme=dark_theme, css=css, title="MachineGuard") as demo:
    gr.Markdown("# MachineGuard", elem_id="title")
    gr.Markdown("**Listen early. Maintain smarter.** · AI-assisted machine-sound screening", elem_id="tagline")

    session_history = gr.State([])
    narrative_state = gr.State("")
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(
                sources=["microphone", "upload"],
                # A finalized WAV path is more reliable than the in-memory
                # microphone payload on Gradio 4.x, which can be incomplete
                # when the stop event fires.
                type="filepath",
                format="wav",
                label="Machine audio (record 5–12 seconds)",
                min_length=MIN_RECORDING_SECONDS,
                max_length=MAX_RECORDING_SECONDS,
                # Brave can lock up while Gradio continuously draws the live
                # recording waveform. Use the browser's native audio player
                # during capture instead.
                waveform_options=gr.WaveformOptions(show_recording_waveform=False, show_controls=False),
            )
            gr.Markdown(
                "**How to record:** place the microphone near the machine, capture a steady 5–12 second sound, then press Stop. Analysis starts automatically.",
                elem_id="recording-tip",
            )
            machine_type = gr.Dropdown(["fan", "pump", "valve"], value="fan", label="Machine type", info="Choose the machine you recorded.")
            language = gr.Radio([("English", "en"), ("हिन्दी", "hi")], value="en", label="Diagnosis language")
            analyze_button = gr.Button("Analyze recording", variant="primary", elem_classes=["primary-action"])
        with gr.Column(scale=1):
            gr.Markdown("#### Analysis result", elem_id="result-heading")
            gauge = gr.Label(label="Health gauge")
            mel_plot = gr.Plot(label="Mel-spectrogram")

    with gr.Accordion("Try a reference sample", open=False):
        gr.Markdown("Select a clip, then choose **Analyze recording** to check the analysis flow before recording from a real machine.")
        with gr.Row(elem_classes=["sample-row"]):
            healthy_fan = gr.Button("Healthy Fan")
            faulty_fan = gr.Button("Faulty Fan")
            healthy_pump = gr.Button("Healthy Pump")
            faulty_pump = gr.Button("Faulty Pump")
            healthy_valve = gr.Button("Healthy Valve")
            faulty_valve = gr.Button("Faulty Valve")

    diagnosis = gr.Markdown("### Diagnosis\nYour maintenance guidance will appear here.")
    speak_button = gr.Button("Speak verdict with Gnani", variant="secondary", visible=True)
    verdict_audio = gr.Audio(label="Spoken verdict", type="filepath", interactive=False)
    speech_status = gr.Markdown("")
    history_table = gr.Dataframe(headers=HISTORY_HEADERS, value=[], interactive=False, label="Session history")

    outputs = [gauge, mel_plot, diagnosis, narrative_state, verdict_audio, history_table, session_history]
    analyze_button.click(analyze, [audio, machine_type, language, session_history], outputs, queue=False)
    # ``change`` also fires while the microphone input is being updated.  That
    # can start analysis on its first tiny buffer and freeze the recorder.
    # Listen only for the explicit stop action instead.
    audio.stop_recording(analyze, [audio, machine_type, language, session_history], outputs, queue=False)
    speak_button.click(speak_verdict, narrative_state, [verdict_audio, speech_status], queue=False)
    for button, filename, kind in (
        (healthy_fan, "fan_healthy.wav", "fan"),
        (faulty_fan, "fan_faulty.wav", "fan"),
        (healthy_pump, "pump_healthy.wav", "pump"),
        (faulty_pump, "pump_faulty.wav", "pump"),
        (healthy_valve, "valve_healthy.wav", "valve"),
        (faulty_valve, "valve_faulty.wav", "valve"),
    ):
        button.click(
            lambda file=filename, type_=kind: choose_sample(file, type_),
            outputs=[audio, machine_type],
        )

    gr.Markdown("Built for practical preventive maintenance · [View on GitHub](https://github.com/)", elem_id="footer")


if __name__ == "__main__":
    _warm_analysis_pipeline()
    demo.launch(share=False)
