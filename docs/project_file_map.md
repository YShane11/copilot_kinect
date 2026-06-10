# Copilot Kinect File Map

This document summarizes the current `D:\copilot_kinect` layout after filename
cleanup. The naming standard is:

- Directories and files: lowercase `snake_case` where practical.
- Python files: lowercase `snake_case.py`.
- Runtime/generated local data stays in dedicated folders and is ignored when
  appropriate.
- Common project files such as `README.md` and `AGENTS.md` keep conventional
  uppercase names.

## Root

| Path | Purpose |
| --- | --- |
| `app.py` | Flask app entry point, routes, Kinect streams, attendance APIs, training APIs, CSV export |
| `requirements.txt` | Python dependencies, including CUDA PyTorch, YOLO, ONNX Runtime, InsightFace, PyKinect2 |
| `.gitignore` | Ignore rules for local config, generated data, videos, models, caches |
| `README.md` | Project overview and quickstart |
| `AGENTS.md` | Codex/agent working guide |

The root is now kept for entry points and project-level configuration. Previous
loose files were moved into `docs/`, `clustering/`, or `reels/recordings/`.

## Source

| Path | Purpose |
| --- | --- |
| `src/__init__.py` | Python package marker |
| `src/vision/__init__.py` | Vision package marker |
| `src/vision/kinect_service.py` | Kinect v1/v2/video capture, frame state, MJPEG stream helpers |
| `src/vision/rgb_depth_alignment.py` | RGB/depth alignment profiles and mapping helpers |
| `src/vision/attendance_pipeline.py` | YOLO pose flow, person tracking, face recognition integration, metrics, annotated frames |
| `src/vision/recognition_pipeline.py` | Recognition workflow wrapper |
| `src/vision/face_recognition_db.py` | Student face images, embedding database, frame recognition |
| `src/vision/pose_depth_metrics.py` | Presence, focus, stability, fatigue, posture, desk distance, stillness, hand raise, shared attention |

## UI

| Path | Purpose |
| --- | --- |
| `templates/index.html` | Login/home page template |
| `templates/dashboard.html` | Main classroom dashboard template |
| `static/kinect_v1.png` | Kinect v1 static image |
| `static/kinect_v2.png` | Kinect v2 static image |
| `static/mock_metrics/assignment_score.csv` | Mock metric data |
| `static/mock_metrics/attendance_rate.csv` | Mock metric data |
| `static/mock_metrics/interaction_rate.csv` | Mock metric data |

## Data

Tracked/reference data:

| Path | Purpose |
| --- | --- |
| `data/administrators.example.json` | Example account/course config |
| `data/kinect_alignment_profiles.json` | Kinect RGB/depth alignment profiles |
| `data/runtime_tuning_profile.json` | Runtime tuning profile |
| `data/quad_sample_0000.jpg` | Alignment/calibration sample |
| `data/quad_sample_0300.jpg` | Alignment/calibration sample |
| `data/quad_sample_0900.jpg` | Alignment/calibration sample |
| `data/quad_sample_1800.jpg` | Alignment/calibration sample |
| `data/hand_raise_validation/` | Hand-raise validation images and sources |

Local/generated data, normally ignored:

| Path | Purpose |
| --- | --- |
| `data/administrators.json` | Local account/course config |
| `data/user.json` | Local student/user data |
| `data/embeddings/` | Face embedding database |
| `data/student_faces/` | Training face images |
| `data/recognition_snapshot_compare/` | Recognition comparison snapshots |
| `data/test_videos/` | Local test videos |
| `data/video_tuning/` | Video tuning outputs |
| `data/video_runs/` | Video run outputs |
| `data/presence_records.json` | Runtime presence records |
| `data/_presence_records_tuning.json` | Tuning presence records |

## Models

| Path | Purpose |
| --- | --- |
| `models/yolo/README.md` | YOLO model notes |
| `models/yolo/*.pt` | Local YOLO model weights, ignored by `.gitignore` |

## Scripts

| Path | Purpose |
| --- | --- |
| `scripts/rebuild_face_db.py` | Rebuild face embeddings |
| `scripts/check_gpu_runtime.py` | Check GPU / ONNX Runtime providers |
| `scripts/calibrate_quad_alignment.py` | RGB/depth calibration helper |
| `scripts/eval_rgb_depth_keypoint_fusion.py` | RGB/depth/keypoint fusion evaluation |
| `scripts/evaluate_video_identity_flow.py` | Video identity-flow evaluation |
| `scripts/compare_recognition_snapshots.py` | Compare recognition snapshots |
| `scripts/tune_face_threshold_with_videos.py` | Tune face threshold with videos |
| `scripts/tune_quad_recordings.py` | Tune quad recordings |
| `scripts/tune_with_video.py` | General video tuning helper |

## Reels

| Path | Purpose |
| --- | --- |
| `reels/record_rgb_nir.py` | RGB/NIR side-by-side recording tool |
| `reels/recordings/*.mp4` | Local recordings, ignored by `.gitignore` |

The previous root-level side-by-side video now lives in `reels/recordings/`.

## History

`history/` stores runtime classroom metric exports:

```text
classroom-metrics-<course-name>-YYYY-MM-DD.csv
classroom-metrics-<course-name>-YYYY-MM-DD.json
```

These files are local/generated and ignored by `.gitignore`. They are the main
input for `clustering/k_means.py`.

## Clustering

`clustering/` contains the K-means analysis script and its outputs.

| Path | Purpose |
| --- | --- |
| `clustering/k_means.py` | K-means analysis script |
| `clustering/kmeans_student_features.csv` | Aggregated/generated student features |
| `clustering/cluster_analysis.md` | Cluster analysis report |
| `clustering/kmeans_elbow.png` | Elbow plot |
| `clustering/kmeans_clusters.png` | 2D cluster plot |
| `clustering/kmeans_clusters_3d.png` | 3D cluster plot |
| `clustering/features_raw_cluster_*.png` | Raw feature plots per cluster |
| `clustering/features_relative_cluster_*.png` | Relative feature plots per cluster |
| `clustering/features_relative_mask_cluster_*.png` | Relative mask feature plots per cluster |
| `clustering/features_relative_mask_clusters_combined.png` | Combined relative mask feature plot |
| `clustering/score_cluster_*.png` | Score plots per cluster |

Run:

```powershell
.\.venv\Scripts\python.exe clustering\k_means.py
```

## Docs

| Path | Purpose |
| --- | --- |
| `docs/project_file_map.md` | This file |
| `docs/kinect_v1_python_vscode_setup.pdf` | Kinect v1 Python / VS Code setup notes |
| `docs/notes.txt` | Development notes |
| `docs/web_design.md` | Web design reference link |

## Tests

| Path | Purpose |
| --- | --- |
| `tests/test_pose_depth_metrics.py` | Unit tests for hand-raise logic |

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

## Ignored Local Folders

| Path | Purpose |
| --- | --- |
| `.venv/` | Python virtual environment |
| `libfreenect/` | Local Kinect dependency/source |
| `__pycache__/` | Python bytecode cache |
| `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/` | Tooling caches |

## Quick Lookup

| Task | Start Here |
| --- | --- |
| Run the web app | `app.py`, `templates/`, `static/` |
| Kinect connection and streams | `src/vision/kinect_service.py` |
| RGB/depth alignment | `src/vision/rgb_depth_alignment.py`, `data/kinect_alignment_profiles.json` |
| YOLO / attendance tracking | `src/vision/attendance_pipeline.py` |
| Face recognition | `src/vision/face_recognition_db.py`, `scripts/rebuild_face_db.py` |
| Classroom behavior metrics | `src/vision/pose_depth_metrics.py` |
| Dashboard UI | `templates/dashboard.html`, `app.py` routes |
| K-means analysis | `clustering/k_means.py`, `history/`, `clustering/` |
| Tests | `tests/test_pose_depth_metrics.py` |
