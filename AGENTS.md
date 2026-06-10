# Repository Guidelines

## Project Shape

Copilot Kinect is a Windows-oriented Flask application for classroom analytics
with Kinect RGB/depth input, YOLO pose detection, InsightFace recognition,
student attendance workflows, pose/depth metrics, CSV export, optional Power
Automate upload, and K-means analysis outputs.

The main runtime entry point is `app.py`. Core vision logic lives in
`src/vision/`, UI templates live in `templates/`, static assets live in
`static/`, operational scripts live in `scripts/`, tests live in `tests/`, and
generated clustering artifacts live in `clustering/`.

For a fuller file-by-file categorization, see `docs/project_file_map.md`.

## Environment

This project targets:

- Windows
- Python 3.10
- NVIDIA GPU / CUDA 12.6 for the configured PyTorch wheels
- Kinect Runtime / SDK as appropriate for Kinect v1 or Kinect v2

Create and install the environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` includes CUDA 12.6 PyTorch wheels, `ultralytics`,
`onnxruntime-gpu`, `insightface`, `comtypes`, and `pykinect2` from GitHub.
Dependency installs may need network access and can take time.

## Common Commands

Run the Flask app:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open:

```text
http://127.0.0.1:5000/
http://127.0.0.1:5000/dashboard
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

Rebuild face embeddings:

```powershell
.\.venv\Scripts\python.exe scripts\rebuild_face_db.py
```

Check GPU / ONNX runtime:

```powershell
.\.venv\Scripts\python.exe scripts\check_gpu_runtime.py
```

Run K-means analysis:

```powershell
.\.venv\Scripts\python.exe clustering\k_means.py
```

## Runtime Data And Secrets

Useful data files and folders:

- `data/administrators.example.json`: template account/course config.
- `data/administrators.json`: local account/course config.
- `data/user.json`: student/user data.
- `data/student_faces/`: training images.
- `data/embeddings/face_embeddings.json`: generated face embedding database.
- `data/kinect_alignment_profiles.json`: RGB/depth alignment profiles.
- `data/runtime_tuning_profile.json`: runtime tuning thresholds.
- `history/`: per-minute classroom metric CSV exports.

Do not commit secrets, webhook URLs, or private face embeddings unless the user
explicitly asks. The optional upload webhook is configured with:

```powershell
$env:POWER_AUTOMATE_UPLOAD_URL="https://..."
```

## Models

YOLO pose models are expected under `models/yolo/`. The app can use
`YOLO_POSE_MODEL` to override the default model path:

```powershell
$env:YOLO_POSE_MODEL="models/yolo/your-model.pt"
```

Large `.pt`, `.onnx`, video, embedding, and local capture artifacts should be
treated as generated or local runtime assets unless the task is explicitly about
versioning them.

## Main Modules

- `app.py`: Flask routes, dashboard APIs, Kinect stream endpoints, training
  upload/capture endpoints, attendance controls, minute CSV export, and optional
  Power Automate upload.
- `src/vision/kinect_service.py`: Kinect v1/v2/video-mode capture, RGB/depth
  frames, status, and MJPEG stream helpers.
- `src/vision/rgb_depth_alignment.py`: RGB/depth alignment profiles and mapping
  helpers.
- `src/vision/attendance_pipeline.py`: YOLO pose flow, temporary/confirmed
  person tracking, face recognition integration, attendance state, metrics, and
  annotated frames.
- `src/vision/face_recognition_db.py`: student face image management, embedding
  database rebuild/load, and frame recognition.
- `src/vision/pose_depth_metrics.py`: presence, focus, head stability, fatigue,
  posture angle, desk distance, stillness, hand raise, and shared-attention
  metrics.
- `clustering/k_means.py`: converts `history/classroom-metrics-*.csv` into
  synthetic and aggregated student features, then writes clustering
  CSV/plots/markdown into `clustering/`.

## Important Routes

Common app routes include:

- `GET /`
- `POST /login`
- `POST /logout`
- `GET /kinect/color_feed`
- `GET /kinect/depth_feed`
- `POST /api/kinect/connect`
- `POST /api/kinect/disconnect`
- `GET /api/kinect/status`
- `GET /api/face-recognition/status`
- `GET /api/attendance/status`
- `POST /api/attendance/course`
- `POST /api/attendance/start`
- `POST /api/attendance/stop`
- `POST /api/attendance/confirm`
- `POST /api/attendance/toggle`
- `GET /api/training/students`
- `POST /api/training/upload`
- `POST /api/training/capture`
- `POST /api/training/capture-frame`
- `POST /api/training/capture/reset`
- `POST /api/face-recognition/rebuild`

## Testing Notes

The current automated test coverage is focused on `PoseDepthMetricEngine` hand
raise behavior in `tests/test_pose_depth_metrics.py`. Prefer adding focused
unit tests around metric math and pure helper logic. Hardware-dependent Kinect,
camera, GPU, and model tests should be guarded or run through script-level
manual checks.

## Editing Guidance

- Preserve the existing Flask route/API shapes unless the user asks for a
  breaking change.
- Be careful with long-running background threads in `app.py`; the
  `MinuteStudentCsvExporter` starts at import/runtime and is stopped through
  `atexit`.
- Avoid importing heavy GPU/model dependencies in tests unless needed.
- Prefer deterministic outputs for analysis scripts. `clustering/k_means.py`
  currently uses `RANDOM_SEED = 42`.
- Use `rg` / `rg --files` for searching.
- Use `apply_patch` for manual code edits.

## Working Tree Caution

This repository may already have user-generated changes. At init time, there
there were dirty K-means artifacts under the clustering output directory, a
modified K-means script, and an untracked detailed project markdown file. Do not
revert or delete user changes unless the user explicitly requests it. Generated
images, CSVs, history exports, videos, and embedding files should be handled
deliberately.
