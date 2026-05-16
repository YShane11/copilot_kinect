# YOLO Models

Place local YOLO model weights in this folder. The weights are ignored by git
because they are large runtime artifacts.

Expected filenames:

- `yolo26x-pose.pt`: primary pose model for the attendance pipeline
- `yolo26l-pose.pt`: high-accuracy fallback
- `yolo26m-pose.pt`: balanced fallback
- `yolo26s-pose.pt`: faster fallback
- `yolo26n-pose.pt`: lightweight fallback for quick testing

The app checks this folder before falling back to legacy model paths in the
project root.
