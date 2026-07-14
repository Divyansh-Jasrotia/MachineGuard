"""Heuristic fault evidence and failure-safe maintenance narratives."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence

import requests


LOW_BAND_BASELINE = 0.20
MID_HIGH_BAND_BASELINE = 0.25
HIGH_BAND_BASELINE = 0.15
RMS_VARIANCE_THRESHOLD = 0.015

# These are intentionally fixed, deterministic fallbacks for when an LLM is
# unavailable.  All four requested placeholders are preserved verbatim.
EN_TEMPLATE = (
    "Verdict: {verdict}. Severity: {severity}. The leading heuristic hypothesis is "
    "{top_hypothesis}; this is an indication, not a confirmed diagnosis. "
    "Recommended action: {action}. If left unaddressed, the fault may worsen, "
    "cause unplanned downtime, and increase repair cost."
)
HI_TEMPLATE = (
    "निष्कर्ष: {verdict}. गंभीरता: {severity}. प्रमुख अनुमानित कारण {top_hypothesis} है; "
    "यह केवल संकेत है, पक्का निदान नहीं। सुझाया गया कदम: {action}. इसे अनदेखा करने पर "
    "खराबी बढ़ सकती है, अनियोजित उत्पादन-रुकावट हो सकती है और मरम्मत की लागत बढ़ सकती है।"
)


def build_evidence(machine_type, score, features, band_ratios) -> dict:
    """Build a structured, heuristic (never certain) fault-evidence record.

    ``band_ratios`` accepts either a four-value sequence in the order
    0–500 Hz, 0.5–2 kHz, 2–5 kHz, 5–8 kHz, or a mapping with those common
    band names. ``features`` may be a mapping containing ``rms_variance`` and
    optional telemetry keys; a numeric feature vector simply uses defaults.
    """
    feature_info = features if isinstance(features, Mapping) else {}
    if isinstance(band_ratios, Mapping):
        low = float(band_ratios.get("0_500", band_ratios.get("low", 0.0)))
        mid_high = float(band_ratios.get("2000_5000", band_ratios.get("mid_high", 0.0)))
        high = float(band_ratios.get("5000_8000", band_ratios.get("high", 0.0)))
    else:
        values = list(band_ratios)
        if len(values) != 4:
            raise ValueError("band_ratios must contain four values: 0–500, 0.5–2k, 2–5k, and 5–8k Hz.")
        low, _, mid_high, high = (float(value) for value in values)

    baseline = feature_info.get("baseline_band_ratios", {})
    if isinstance(baseline, Mapping):
        low_baseline = float(baseline.get("0_500", baseline.get("low", LOW_BAND_BASELINE)))
        mid_high_baseline = float(baseline.get("2000_5000", baseline.get("mid_high", MID_HIGH_BAND_BASELINE)))
        high_baseline = float(baseline.get("5000_8000", baseline.get("high", HIGH_BAND_BASELINE)))
    elif isinstance(baseline, Sequence) and not isinstance(baseline, (str, bytes)) and len(baseline) == 4:
        low_baseline, _, mid_high_baseline, high_baseline = (float(value) for value in baseline)
    else:
        low_baseline, mid_high_baseline, high_baseline = LOW_BAND_BASELINE, MID_HIGH_BAND_BASELINE, HIGH_BAND_BASELINE
    rms_variance = float(feature_info.get("rms_variance", 0.0))
    hypotheses = []
    if low > low_baseline * 1.25:
        hypotheses.append({
            "type": "imbalance_or_looseness",
            "signal": "low_band_energy_elevated",
            "confidence": "high" if low > low_baseline * 1.75 else "medium",
        })
    if mid_high > mid_high_baseline * 1.25:
        hypotheses.append({
            "type": "bearing_wear",
            "signal": "high_band_tonal_energy",
            "confidence": "high" if mid_high > mid_high_baseline * 1.75 else "medium",
        })
    if high > high_baseline * 1.25:
        hypotheses.append({
            "type": "air_leak_or_flow",
            "signal": "high_band_broadband_hiss",
            "confidence": "high" if high > high_baseline * 1.75 else "medium",
        })
    if rms_variance > RMS_VARIANCE_THRESHOLD:
        hypotheses.append({
            "type": "impacts_or_knocking",
            "signal": "rms_variance_periodic_clicks",
            "confidence": "high" if rms_variance > RMS_VARIANCE_THRESHOLD * 2 else "medium",
        })

    return {
        "machine_type": machine_type,
        "anomaly_score": round(float(score)),
        "verdict": "anomalous" if float(score) > 60 else "normal",
        "fault_hypotheses": hypotheses,
        "telemetry": {
            "clip_seconds": float(feature_info.get("clip_seconds", 10.0)),
            "model": str(feature_info.get("model", "gmm_v1")),
            "auc_val": float(feature_info.get("auc_val", 0.86)),
        },
    }


def generate_narrative(evidence, lang) -> str:
    """Return an LLM narrative, with one retry and a deterministic fallback.

    Set ``LLM_PROVIDER`` to ``groq``, ``gemini``, or ``none``. The two remote
    paths use only ``requests`` and each network attempt has an eight-second
    timeout; a provider error always falls through to the local template.
    """
    language = "hi" if lang == "hi" else "en"
    provider = os.getenv("LLM_PROVIDER", "none").strip().lower()
    system_prompt = (
        "You are a maintenance advisor writing for a small-factory owner with no engineering background. "
        "Write 4–6 sentences including severity, likely cause, recommended action, and cost-of-inaction. "
        f"Reply in {language}."
    )
    user_prompt = "Use this evidence. Treat fault hypotheses as heuristic, not certain:\n" + json.dumps(evidence)

    for _ in range(2):
        try:
            if provider == "groq":
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    break
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                        "temperature": 0.2,
                    },
                    timeout=8,
                )
                response.raise_for_status()
                text = response.json()["choices"][0]["message"]["content"].strip()
            elif provider == "gemini":
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    break
                model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
                response = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                    json={
                        "systemInstruction": {"parts": [{"text": system_prompt}]},
                        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                        "generationConfig": {"temperature": 0.2},
                    },
                    timeout=8,
                )
                response.raise_for_status()
                text = response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                break
            if text:
                return text
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
            continue

    hypotheses = evidence.get("fault_hypotheses", [])
    top_hypothesis = hypotheses[0]["type"].replace("_", " ") if hypotheses else "no specific mechanical pattern"
    verdict = evidence.get("verdict", "unknown")
    severity = "high" if evidence.get("anomaly_score", 0) >= 80 else "medium" if verdict == "anomalous" else "low"
    action = (
        "arrange an inspection within 24–48 hours and check the affected moving parts"
        if verdict == "anomalous"
        else "continue routine monitoring and compare the next recording"
    )
    template = HI_TEMPLATE if language == "hi" else EN_TEMPLATE
    return template.format(verdict=verdict, top_hypothesis=top_hypothesis, action=action, severity=severity)
