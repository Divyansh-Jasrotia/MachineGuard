---
title: MachineGuard
emoji: 🔊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# MachineGuard

MachineGuard is a lightweight preventive-maintenance demo that listens to fan, pump, and valve recordings, turns each clip into acoustic features, scores how unusual it is compared with normal machine sound, and explains the result in plain language. It is built for a small-factory owner: upload or record audio, review a 0–100 health signal and mel-spectrogram, then act on clearly labelled *heuristic* maintenance hypotheses rather than treating the result as a certain diagnosis.

<!-- TODO: fill in the Hugging Face Space URL after deploying -->
**Live demo:** TODO — add the Hugging Face Spaces URL here after deploying.

## Run locally in 60 seconds

```powershell
cd machineguard
pip install -r requirements.txt
python app.py
```

Open the local URL shown by Gradio. The interface supports microphone or file upload, six demo-sample buttons, English/Hindi diagnosis, and session-only history.

To prepare the MIMII subset and train real models:

```powershell
python scripts/download_data.py
python scripts/make_samples.py
python scripts/train.py
```

## Measured metrics

`metrics.md` is written by `scripts/train.py` after training on the held-out MIMII split.

| Machine | Detector | ROC-AUC | Threshold |
|---|---|---:|---:|
| Fan | GMM | 0.870 | 4.39 |
| Pump | GMM | 0.954 | 33.50 |
| Valve | GMM | 0.937 | 58.91 |

The training script reports the measured held-out ROC-AUC only. If a GMM is below 0.75 for a machine type, it switches that type to IsolationForest and records the reason in `metrics.md`.

## Optional LLM narrative

MachineGuard always works without an LLM: it falls back to a deterministic English or Hindi maintenance template. Select a provider with environment variables before starting `app.py`.

| Provider | PowerShell setup |
|---|---|
| No remote LLM (default) | `$env:LLM_PROVIDER="none"` |
| Groq | `$env:LLM_PROVIDER="groq"`<br>`$env:GROQ_API_KEY="..."`<br>Optional: `$env:GROQ_MODEL="llama-3.1-8b-instant"` |
| Gemini | `$env:LLM_PROVIDER="gemini"`<br>`$env:GEMINI_API_KEY="..."`<br>Optional: `$env:GEMINI_MODEL="gemini-1.5-flash"` |

Network calls use a 3-second timeout with no retry. Missing keys or provider failures automatically use the local template, so a demo remains responsive.

## Optional spoken verdict (Gnani TTS)

To add a playable spoken version of the diagnosis, create a Gnani Timbre API key and set it before starting the app:

```powershell
$env:GNANI_API_KEY="..."
# Optional defaults:
$env:GNANI_TTS_VOICE="Karan"
$env:GNANI_TTS_MODEL="vachana-voice-v3"
$env:GNANI_TTS_TIMEOUT_SECONDS="30"
```

After analysis, click **Speak verdict** to send the finished diagnosis text to Gnani's REST TTS endpoint. The returned MP3 appears in the **Spoken verdict** player. Without the key, the app shows a configuration message and text diagnosis continues normally.

## Architecture

```text
Audio in (microphone / upload / sample WAV)
                  |
                  v
Feature extraction
(16 kHz normalization, log-mel, 14 acoustic features)
                  |
                  v
Anomaly model
(StandardScaler + GMM, or IsolationForest fallback)
                  |
                  v
Diagnosis engine
(band-ratio evidence + optional Groq/Gemini narrative)
                  |
                  v
Gradio UI
(health gauge, mel plot, EN/HI diagnosis, session history)
```

## Project notes

- Dataset source: MIMII 6 dB fan, pump, and valve archives from Zenodo record 3384388.
- Scores are calibrated from held-out normal clips; `> 60` is anomalous by default, with the final threshold tuned during training.
- This is a decision-support demo, not a substitute for a physical inspection or safety process.
