# Copilot Kinect

Kinect classroom analytics prototype with a Flask dashboard, RGB/depth alignment,
YOLO pose detection, face recognition, attendance tracking, and metric export
utilities.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Optional metric upload is configured through an environment variable:

```powershell
$env:POWER_AUTOMATE_UPLOAD_URL="https://..."
```

Create `data/administrators.json` from the example file before running locally:

```powershell
Copy-Item data\administrators.example.json data\administrators.json
```

## Run

```powershell
.\.venv\Scripts\python.exe app.py
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

## Notes

Large generated artifacts are intentionally ignored by git, including recordings,
history exports, embedding databases, and local YOLO model weights. Keep those in
local storage or a release/artifact bucket instead of committing them to GitHub.
