# MachineGuard

## Problem

Machine faults are often first detectable in sound, but reviewing recordings manually is slow.

## Demo

MachineGuard accepts a microphone recording or uploaded audio, lets a user choose fan, pump, or valve, and presents an anomaly label, plot, diagnosis, and history row.

## Architecture

The Gradio app is deliberately thin. Audio preprocessing, anomaly modeling, and diagnosis live in separate `lib/` modules so the demo can later be connected to MIMII data and trained detectors.

## Current scope

This repository is a runnable stub scaffold: it makes no network calls and requires no downloaded dataset or trained model.
