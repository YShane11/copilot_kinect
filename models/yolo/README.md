# YOLO Models

- `yolo26x-pose.pt`: primary pose model used by the attendance pipeline.
- `yolo26l-pose.pt`: fallback pose model kept for high accuracy with lower latency than x.
- `yolo26m-pose.pt`: fallback pose model kept for balanced accuracy and speed.
- `yolo26s-pose.pt`: fallback pose model kept for compatibility and quicker testing.
- `yolo26n-pose.pt`: lightweight fallback pose model.

The app loads models from this folder first before checking legacy files in the project root.
