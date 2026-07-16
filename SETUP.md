# MachineGuard setup

Use these steps to create a working local environment and run MachineGuard on Windows.

## Requirements

- Python 3.9 or newer (Python 3.9 is known to work with this project).
- PowerShell.
- A modern browser. Chrome or Edge is recommended for microphone recording.

Check Python is available:

```powershell
py -3.9 --version
```

## First-time setup

Open PowerShell in the project folder and run:

```powershell
cd D:\Coding\MachineGuard\MachineGuard
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The pinned packages in `requirements.txt` include compatible Gradio,
Hugging Face Hub, and Pydantic versions. Do not install newer versions of
those packages separately unless you update and test the application.

## Run the app

```powershell
cd D:\Coding\MachineGuard\MachineGuard
.\.venv\Scripts\python.exe app.py
```

Wait for the message containing the local URL, then open:

```text
http://127.0.0.1:7860
```

Keep the PowerShell window open while using the app. Press `Ctrl+C` in that
window to stop the server.

## Microphone recording

1. Open the local URL in Chrome or Edge.
2. Allow microphone access when the browser asks.
3. Select the machine type.
4. Record a 5–12 second clip and press Stop.
5. The app analyzes the clip after recording stops.

Use `127.0.0.1:7860` consistently. Browsers treat `localhost:7860` and
`127.0.0.1:7860` as different sites with separate microphone permissions.

## Troubleshooting

### The virtual environment does not run

Recreate it:

```powershell
Remove-Item -LiteralPath .venv -Recurse -Force
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Browser cannot record or records silence

- Check Windows **Settings > System > Sound > Input** and select the right microphone.
- Check **Settings > Privacy & security > Microphone** and enable microphone access.
- In browser site settings for `127.0.0.1:7860`, set Microphone to **Allow**.
- Restart the app and hard-refresh the page with `Ctrl+Shift+R`.
- If Brave freezes or gives incomplete recordings, use Chrome or Edge.

### Dependency installation fails

Ensure that PowerShell has internet access, then run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --prefer-binary
```

### Optional LLM narrative

The app works without an API key and uses a local maintenance template by default.
To enable a provider for the current PowerShell session, set `LLM_PROVIDER` and
the corresponding API key before launching the app. See `README.md` for the
available provider variables.
