"""MachineGuard's Gradio product interface."""
from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

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


PANEL_BG = "#0d2236"
PANEL_BORDER = "#1c4969"
TEXT_MUTED = "#91a4bd"
TEXT_LIGHT = "#e6f0ff"


def _mel_plot(mel):
    """Render lib-produced mel data styled to match the app's dark cards."""
    figure, axis = plt.subplots(figsize=(8, 3.4))
    figure.patch.set_facecolor(PANEL_BG)
    axis.set_facecolor(PANEL_BG)
    image = axis.imshow(mel, aspect="auto", origin="lower", cmap="magma")
    axis.set_title("Log-mel spectrogram", color=TEXT_LIGHT, fontsize=11, pad=10)
    axis.set_xlabel("Frames", color=TEXT_MUTED, fontsize=9)
    axis.set_ylabel("Mel band", color=TEXT_MUTED, fontsize=9)
    axis.tick_params(colors=TEXT_MUTED, labelsize=8)
    for spine in axis.spines.values():
        spine.set_color(PANEL_BORDER)
    colorbar = figure.colorbar(image, ax=axis, label="dB")
    colorbar.ax.yaxis.set_tick_params(color=TEXT_MUTED, labelcolor=TEXT_MUTED, labelsize=8)
    colorbar.outline.set_edgecolor(PANEL_BORDER)
    colorbar.set_label("dB", color=TEXT_MUTED, fontsize=9)
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


def _severity(score: float) -> tuple[str, str, str]:
    """Return (state label, dot emoji, color) so every gauge/table use agrees."""
    if score > 60:
        return "FAULT", "🔴", "#ef4444"
    if score > 35:
        return "WARNING", "🟡", "#f5b400"
    return "HEALTHY", "🟢", "#22c55e"


def _health_html(score: float) -> str:
    """Render a gauge whose fill color always matches the verdict's severity."""
    state, dot, color = _severity(score)
    pct = max(0.0, min(100.0, score))
    return f"""
    <div class="health-gauge mg-card">
      <div class="health-gauge-label" style="color:{color}">{dot} {state} — {score:.1f} / 100</div>
      <div class="health-gauge-track">
        <div class="health-gauge-fill" style="width:{pct}%; background:{color}"></div>
      </div>
      <div class="health-gauge-scale"><span>0 · Healthy</span><span>100 · Severe fault</span></div>
    </div>
    """


def _status_html(text: str, color: str = "#91a4bd") -> str:
    """Neutral gauge state for the awaiting/unavailable placeholders."""
    return f"""
    <div class="health-gauge mg-card">
      <div class="health-gauge-label" style="color:{color}">{text}</div>
      <div class="health-gauge-track"><div class="health-gauge-fill" style="width:0%"></div></div>
      <div class="health-gauge-scale"><span>0 · Healthy</span><span>100 · Severe fault</span></div>
    </div>
    """


def analyze(audio_input, machine_type, language, history):
    """Thin UI adapter around the audio, model, and diagnosis libraries."""
    history = list(history or [])
    if not audio_input:
        return _status_html("⚪ Awaiting audio"), None, "Upload or record a machine clip to begin.", "", None, history, history
    try:
        audio = load_audio(audio_input)
        if audio.size == 0:
            raise ValueError("Loaded audio contains no samples.")
        silence_ratio = float(np.mean(np.abs(audio) < 1e-3))
        if silence_ratio > 0.8:
            return _status_html("⚪ Awaiting audio"), None, (
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
        _, dot, _ = _severity(score)
        history.insert(0, [datetime.now().strftime("%H:%M:%S"), machine_type, round(score, 1), f"{dot} {verdict}"])
        history = history[:50]

        return _health_html(score), _mel_plot(compute_melspec(audio)), narrative, narrative, None, history, history
    except Exception as error:
        return _status_html("⚠️ Analysis unavailable", "#f5b400"), None, f"### Unable to analyse this clip\n`{error}`", "", None, history, history


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
#tagline {text-align:center; color:#91a4bd; margin:0 0 10px}
#footer {text-align:center; color:#91a4bd; margin:24px 0 8px}
#result-heading {margin:2px 0 12px; color:#d9edff; font-size:1.05rem}
.primary-action button {min-height:48px; font-size:1.05rem; font-weight:700}
.sample-row button {border-color:#274764 !important}

/* Shared card system: every panel (tip, selectors, gauge, plot, diagnosis)
   uses the exact same background/border/radius/padding/spacing so nothing
   looks mismatched next to anything else. */
:root {
  --mg-bg: #0d2236;
  --mg-border: #1c4969;
  --mg-radius: 12px;
  --mg-pad: 16px;
  --mg-gap: 16px;
}
.mg-card {
  background: var(--mg-bg) !important;
  border: 1px solid var(--mg-border) !important;
  border-radius: var(--mg-radius) !important;
  padding: var(--mg-pad) !important;
  margin: 0 0 var(--mg-gap) 0 !important;
}
#left-col > *, #right-col > * { margin-bottom: var(--mg-gap) !important; }
#left-col > *:last-child, #right-col > *:last-child { margin-bottom: 0 !important; }

#badge-row { display:flex; justify-content:center; margin:0 0 20px; }
#badge-row .badge-pill {
  background: var(--mg-bg); border:1px solid var(--mg-border); border-radius:999px;
  padding:6px 16px; font-size:.82rem; color:#b9dff6; text-align:center;
}

#recording-tip { color:#b9dff6; }
#selector-row { gap: var(--mg-gap) !important; }
#selector-row .gr-form, #selector-row > div { background: transparent !important; border: none !important; box-shadow:none !important; }

.health-gauge-label {font-size:1.2rem; font-weight:700; margin-bottom:8px}
.health-gauge-track {height:10px; border-radius:6px; background:#182a40; overflow:hidden}
.health-gauge-fill {height:100%; transition:width .3s ease}
.health-gauge-scale {display:flex; justify-content:space-between; color:#7f93ad; font-size:.78rem; margin-top:6px}

#mel-plot-card { padding: 4px !important; }
#mel-plot-card .plot-container { background: var(--mg-bg) !important; }

#score-note {color:#7f93ad; font-size:.85rem; margin:6px 0 10px}

#sample-accordion { border-color: var(--mg-border) !important; background: var(--mg-bg) !important; margin: 0 0 var(--mg-gap) 0 !important; }

#diagnosis-card { padding-top: 4px !important; }

#speak-card .primary-action { margin-bottom: 4px !important; }
#speak-status { color:#7f93ad; font-size:.85rem; margin:4px 0 0 }

#history-table table thead th {
  background: var(--mg-bg) !important;
  color: #d9edff !important;
  border-color: var(--mg-border) !important;
}
#history-table table { border-color: var(--mg-border) !important; }
"""


with gr.Blocks(theme=dark_theme, css=css, title="MachineGuard") as demo:
    gr.Markdown("# MachineGuard", elem_id="title")
    gr.Markdown("**Listen early. Maintain smarter.** · AI-assisted machine-sound screening", elem_id="tagline")
    gr.HTML(
        '<div id="badge-row"><span class="badge-pill">MIMII dataset · GMM anomaly detection · '
        "fan 0.87 / pump 0.95 / valve 0.94 ROC-AUC</span></div>"
    )

    session_history = gr.State([])
    narrative_state = gr.State("")

    with gr.Accordion("🎧 Try a reference sample", open=True, elem_id="sample-accordion"):
        gr.Markdown("Select a clip, then choose **Analyze recording** to check the analysis flow before recording from a real machine.")
        with gr.Row(elem_classes=["sample-row"]):
            healthy_fan = gr.Button("🟢 Healthy Fan")
            faulty_fan = gr.Button("🔴 Faulty Fan")
            healthy_pump = gr.Button("🟢 Healthy Pump")
            faulty_pump = gr.Button("🔴 Faulty Pump")
            healthy_valve = gr.Button("🟢 Healthy Valve")
            faulty_valve = gr.Button("🔴 Faulty Valve")

    with gr.Row():
        with gr.Column(scale=1, elem_id="left-col"):
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
            with gr.Row(elem_id="selector-row"):
                machine_type = gr.Dropdown(["fan", "pump", "valve"], value="fan", label="Machine type", info="Choose the machine you recorded.")
                language = gr.Radio([("English", "en"), ("हिन्दी", "hi")], value="en", label="Diagnosis language")
            analyze_button = gr.Button("Analyze recording", variant="primary", elem_classes=["primary-action"])
        with gr.Column(scale=1, elem_id="right-col"):
            gr.Markdown("#### Analysis result", elem_id="result-heading")
            gauge = gr.HTML(_status_html("⚪ Awaiting audio"), label="Health gauge")
            mel_plot = gr.Plot(label="Mel-spectrogram", elem_id="mel-plot-card")

    diagnosis = gr.Markdown("### Diagnosis\nYour maintenance guidance will appear here.", elem_id="diagnosis-card", elem_classes=["mg-card"])
    gr.Markdown(
        "Score is a calibrated anomaly rating: **0 = healthy**, **100 = severe fault** "
        "(🟢 ≤35 healthy · 🟡 36–60 warning · 🔴 >60 fault).",
        elem_id="score-note",
    )

    with gr.Group(elem_id="speak-card", elem_classes=["mg-card"]):
        gr.Markdown("#### 🔊 Hear the diagnosis")
        speak_button = gr.Button("🔊 Speak verdict (Gnani Timbre TTS)", variant="secondary", elem_classes=["primary-action"])
        verdict_audio = gr.Audio(label="Spoken verdict", type="filepath", interactive=False)
        speech_status = gr.Markdown("", elem_id="speak-status")

    history_table = gr.Dataframe(headers=HISTORY_HEADERS, value=[], interactive=False, label="Session history", elem_id="history-table")

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

    gr.Markdown(
        "Built for practical preventive maintenance · [View on GitHub](https://github.com/Divyansh-Jasrotia/MachineGuard)",
        elem_id="footer",
    )


if __name__ == "__main__":
    _warm_analysis_pipeline()
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), share=False)
