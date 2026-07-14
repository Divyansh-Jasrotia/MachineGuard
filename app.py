"""MachineGuard's Gradio product interface."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt

from lib.anomaly_model import load_detector, score_detector
from lib.audio_processing import compute_melspec, extract_features, load_audio
from lib.diagnosis import build_evidence, generate_narrative


ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = ROOT / "assets" / "samples"
MODEL_DIR = ROOT / "models"
HISTORY_HEADERS = ["Time", "Machine", "Score", "Verdict"]


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


def _health_label(score: float) -> dict[str, float]:
    """Map the calibrated score to the product's color-coded health state."""
    if score > 60:
        state, dot = "FAULT", "🔴"
    elif score > 35:
        state, dot = "WARNING", "🟡"
    else:
        state, dot = "HEALTHY", "🟢"
    return {f"{dot} {state} — {score:.1f} / 100": 1.0}


def analyze(audio_path, machine_type, language, history):
    """Thin UI adapter around the audio, model, and diagnosis libraries."""
    history = list(history or [])
    if not audio_path:
        return {"Awaiting audio": 1.0}, None, "Upload or record a machine clip to begin.", history, history
    try:
        audio = load_audio(audio_path)
        features = extract_features(audio)
        model = _model_for(machine_type)
        score = float(score_detector(model, features)) if model is not None else 0.0
        model_name = f"{model['detector']}_v1" if model is not None else "model_not_trained"
        evidence = build_evidence(
            machine_type,
            score,
            {"clip_seconds": len(audio) / 16_000, "model": model_name},
            features[-4:],
        )
        narrative = generate_narrative(evidence, language)
        verdict = evidence["verdict"]
        history.insert(0, [datetime.now().strftime("%H:%M:%S"), machine_type, round(score, 1), verdict])
        history = history[:50]
        return _health_label(score), _mel_plot(compute_melspec(audio)), narrative, history, history
    except Exception as error:
        return {"Analysis unavailable": 1.0}, None, f"### Unable to analyse this clip\n`{error}`", history, history


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
.gradio-container {background: #07111f !important; color: #e6f0ff !important}
#title {text-align:center; margin-bottom: 0}
#tagline {text-align:center; color:#91a4bd; margin-top:0}
#footer {text-align:center; color:#91a4bd; margin-top:20px}
"""


with gr.Blocks(theme=dark_theme, css=css, title="MachineGuard") as demo:
    gr.Markdown("# MachineGuard", elem_id="title")
    gr.Markdown("**Listen early. Maintain smarter.**", elem_id="tagline")

    session_history = gr.State([])
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Machine audio")
            machine_type = gr.Dropdown(["fan", "pump", "valve"], value="fan", label="Machine type")
            language = gr.Radio([("English", "en"), ("हिन्दी", "hi")], value="en", label="Diagnosis language")
            analyze_button = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=1):
            gauge = gr.Label(label="Health gauge")
            mel_plot = gr.Plot(label="Mel-spectrogram")

    gr.Markdown("### Try a sample")
    with gr.Row():
        healthy_fan = gr.Button("Healthy Fan")
        faulty_fan = gr.Button("Faulty Fan")
        healthy_pump = gr.Button("Healthy Pump")
        faulty_pump = gr.Button("Faulty Pump")
        healthy_valve = gr.Button("Healthy Valve")
        faulty_valve = gr.Button("Faulty Valve")

    diagnosis = gr.Markdown("### Diagnosis\nYour maintenance guidance will appear here.")
    history_table = gr.Dataframe(headers=HISTORY_HEADERS, value=[], interactive=False, label="Session history")

    outputs = [gauge, mel_plot, diagnosis, history_table, session_history]
    analyze_button.click(analyze, [audio, machine_type, language, session_history], outputs)
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
        ).then(analyze, [audio, machine_type, language, session_history], outputs)

    gr.Markdown("Built for practical preventive maintenance · [View on GitHub](https://github.com/)", elem_id="footer")


if __name__ == "__main__":
    demo.launch(share=True)
