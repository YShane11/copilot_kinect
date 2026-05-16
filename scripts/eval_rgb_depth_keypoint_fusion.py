import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))


BONES = (
    (5, 6),   # shoulders
    (5, 7),   # left upper arm
    (7, 9),   # left lower arm
    (6, 8),   # right upper arm
    (8, 10),  # right lower arm
    (11, 12), # hips
    (5, 11),  # left torso side
    (6, 12),  # right torso side
    (11, 13), # left upper leg
    (13, 15), # left lower leg
    (12, 14), # right upper leg
    (14, 16), # right lower leg
)


def resolve_model_path(model_arg: str) -> str:
    raw = str(model_arg or "").strip()
    if raw:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = WORKSPACE / candidate
        if candidate.exists():
            return str(candidate)
        return raw

    default_candidates = (
        WORKSPACE / "models" / "yolo" / "yolo26x-pose.pt",
        WORKSPACE / "models" / "yolo" / "yolo26n-pose.pt",
        WORKSPACE / "models" / "yolo" / "yolo26s-pose.pt",
        WORKSPACE / "models" / "yolo" / "yolo11n-pose.pt",
        WORKSPACE / "models" / "yolo" / "yolov8n-pose.pt",
    )
    for path in default_candidates:
        if path.exists():
            return str(path)

    # Fall back to a model alias if local files are not available.
    return "yolo11n-pose.pt"


def resolve_video_paths(video_args: list[str]) -> list[Path]:
    if video_args:
        paths = []
        for video in video_args:
            path = Path(video)
            if not path.is_absolute():
                path = WORKSPACE / video
            paths.append(path)
        return paths

    return [
        WORKSPACE / "data" / "test_videos" / "test_video.mp4",
        WORKSPACE / "data" / "test_videos" / "opencv_vtest.avi",
    ]


def bbox_center(bbox):
    return ((float(bbox[0]) + float(bbox[2])) * 0.5, (float(bbox[1]) + float(bbox[3])) * 0.5)


def bbox_diag(bbox):
    width = max(1.0, float(bbox[2]) - float(bbox[0]))
    height = max(1.0, float(bbox[3]) - float(bbox[1]))
    return math.hypot(width, height)


def iou(left_bbox, right_bbox):
    left = max(float(left_bbox[0]), float(right_bbox[0]))
    top = max(float(left_bbox[1]), float(right_bbox[1]))
    right = min(float(left_bbox[2]), float(right_bbox[2]))
    bottom = min(float(left_bbox[3]), float(right_bbox[3]))
    if right <= left or bottom <= top:
        return 0.0

    inter_area = (right - left) * (bottom - top)
    left_area = max(0.0, float(left_bbox[2]) - float(left_bbox[0])) * max(0.0, float(left_bbox[3]) - float(left_bbox[1]))
    right_area = max(0.0, float(right_bbox[2]) - float(right_bbox[0])) * max(0.0, float(right_bbox[3]) - float(right_bbox[1]))
    union_area = left_area + right_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def point_distance(left, right):
    delta_x = float(left[0]) - float(right[0])
    delta_y = float(left[1]) - float(right[1])
    return math.hypot(delta_x, delta_y)


def build_pseudo_depth(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    depth_norm = cv2.GaussianBlur(gray, (9, 9), 0)
    depth_norm = cv2.normalize(depth_norm, None, 0, 255, cv2.NORM_MINMAX)
    depth_norm = depth_norm.astype(np.uint8)
    depth_raw = (2000.0 - ((depth_norm.astype(np.float32) / 255.0) * 1400.0)).astype(np.uint16)
    return depth_raw


def depth_patch_cm(depth_frame, point, radius=6):
    if depth_frame is None or point is None:
        return None

    height = int(getattr(depth_frame, "shape", [0, 0])[0] or 0)
    width = int(getattr(depth_frame, "shape", [0, 0])[1] or 0)
    if width <= 0 or height <= 0:
        return None

    center_x = int(round(point[0]))
    center_y = int(round(point[1]))
    x1 = max(0, center_x - radius)
    y1 = max(0, center_y - radius)
    x2 = min(width, center_x + radius + 1)
    y2 = min(height, center_y + radius + 1)
    if x2 <= x1 or y2 <= y1:
        return None

    patch = depth_frame[y1:y2, x1:x2]
    if patch is None or patch.size == 0:
        return None

    values_cm = (patch.reshape(-1).astype(np.float32) / 10.0).tolist()
    values_cm = [value for value in values_cm if 20.0 <= value <= 450.0]
    if not values_cm:
        return None

    values_cm.sort()
    trim_count = int(len(values_cm) * 0.18)
    if trim_count * 2 < len(values_cm):
        values_cm = values_cm[trim_count: len(values_cm) - trim_count]
    if not values_cm:
        return None
    return float(values_cm[len(values_cm) // 2])


def decode_pose_detections(model_result):
    detections = []
    keypoints_xy = None
    keypoints_conf = None
    if getattr(model_result, "keypoints", None) is not None:
        keypoints_xy = getattr(model_result.keypoints, "xy", None)
        keypoints_conf = getattr(model_result.keypoints, "conf", None)

    for index, box in enumerate(model_result.boxes):
        bbox = [float(value) for value in box.xyxy[0].tolist()]
        width = max(0.0, bbox[2] - bbox[0])
        height = max(0.0, bbox[3] - bbox[1])
        if width < 24.0 or height < 32.0:
            continue

        points = []
        confs = []
        if keypoints_xy is not None and len(keypoints_xy) > index:
            try:
                points = [[float(item[0]), float(item[1])] for item in keypoints_xy[index].tolist()]
            except Exception:
                points = []
        if keypoints_conf is not None and len(keypoints_conf) > index:
            try:
                confs = [float(value) for value in keypoints_conf[index].tolist()]
            except Exception:
                confs = []

        detections.append(
            {
                "bbox": bbox,
                "keypoints": points,
                "keypoint_conf": confs,
            }
        )

    detections.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return detections


@dataclass
class TrackerState:
    bbox: list[float]
    last_seen_frame: int


class SimpleTracker:
    def __init__(self):
        self._next_track_id = 1
        self._tracks: dict[int, TrackerState] = {}

    def update(self, detections, frame_index):
        pairs = []
        unmatched_detection_indexes = set(range(len(detections)))
        unmatched_track_ids = set(self._tracks.keys())

        candidates = []
        for detection_index, detection in enumerate(detections):
            detection_bbox = detection["bbox"]
            detection_center = bbox_center(detection_bbox)
            detection_diag = bbox_diag(detection_bbox)
            for track_id, state in self._tracks.items():
                overlap = iou(detection_bbox, state.bbox)
                state_center = bbox_center(state.bbox)
                center_dist = point_distance(detection_center, state_center)
                norm_center_dist = center_dist / max(1.0, detection_diag)
                score = (overlap * 3.0) + (1.0 - min(1.0, norm_center_dist))
                if overlap >= 0.1 or norm_center_dist <= 0.55:
                    candidates.append((score, detection_index, track_id))

        for _, detection_index, track_id in sorted(candidates, key=lambda item: item[0], reverse=True):
            if detection_index not in unmatched_detection_indexes or track_id not in unmatched_track_ids:
                continue
            unmatched_detection_indexes.remove(detection_index)
            unmatched_track_ids.remove(track_id)
            pairs.append((track_id, detections[detection_index]))
            self._tracks[track_id] = TrackerState(
                bbox=list(detections[detection_index]["bbox"]),
                last_seen_frame=frame_index,
            )

        for detection_index in sorted(unmatched_detection_indexes):
            track_id = self._next_track_id
            self._next_track_id += 1
            detection = detections[detection_index]
            self._tracks[track_id] = TrackerState(
                bbox=list(detection["bbox"]),
                last_seen_frame=frame_index,
            )
            pairs.append((track_id, detection))

        stale_track_ids = [
            track_id
            for track_id, state in self._tracks.items()
            if frame_index - state.last_seen_frame > 20
        ]
        for track_id in stale_track_ids:
            self._tracks.pop(track_id, None)

        return pairs


@dataclass
class MethodTrackState:
    points: dict[int, tuple[float, float]] = field(default_factory=dict)
    confs: dict[int, float] = field(default_factory=dict)
    depths: dict[int, float] = field(default_factory=dict)
    frame_index: int = -1


class MetricsCollector:
    def __init__(self, name):
        self.name = name
        self.total_keypoints = 0
        self.valid_keypoints = 0
        self.jitter_samples = []
        self.depth_jump_samples = []
        self.jump_outlier_count = 0
        self.jump_sample_count = 0
        self.implausible_bone_count = 0
        self.total_bone_count = 0
        self.bone_lengths: dict[tuple[int, int], list[float]] = {bone: [] for bone in BONES}
        self.track_states: dict[int, MethodTrackState] = {}
        self.frame_count = 0
        self.person_count = 0

    def _normalize_points(self, keypoints, keypoint_conf, min_conf):
        points = {}
        confs = {}
        for index in range(17):
            self.total_keypoints += 1
            if index >= len(keypoints) or index >= len(keypoint_conf):
                continue
            point = keypoints[index]
            conf = float(keypoint_conf[index])
            if point is None or len(point) < 2 or conf < min_conf:
                continue
            points[index] = (float(point[0]), float(point[1]))
            confs[index] = conf
            self.valid_keypoints += 1
        return points, confs

    def update(self, track_id, bbox, keypoints, keypoint_conf, depth_frame, min_conf=0.12):
        diag = bbox_diag(bbox)
        points, confs = self._normalize_points(keypoints, keypoint_conf, min_conf=min_conf)
        depths = {index: depth_patch_cm(depth_frame, point) for index, point in points.items()}

        previous = self.track_states.get(track_id)
        if previous is not None:
            prev_diag = diag
            for index, current_point in points.items():
                previous_point = previous.points.get(index)
                if previous_point is None:
                    continue
                jitter = point_distance(current_point, previous_point) / max(1.0, prev_diag)
                self.jitter_samples.append(jitter)
                self.jump_sample_count += 1
                if jitter > 0.22:
                    self.jump_outlier_count += 1

                current_depth = depths.get(index)
                previous_depth = previous.depths.get(index)
                if current_depth is not None and previous_depth is not None:
                    self.depth_jump_samples.append(abs(current_depth - previous_depth))

        for left_index, right_index in BONES:
            left_point = points.get(left_index)
            right_point = points.get(right_index)
            if left_point is None or right_point is None:
                continue
            bone_length = point_distance(left_point, right_point) / max(1.0, diag)
            self.bone_lengths[(left_index, right_index)].append(bone_length)
            self.total_bone_count += 1
            if bone_length < 0.02 or bone_length > 0.9:
                self.implausible_bone_count += 1

        self.track_states[track_id] = MethodTrackState(
            points=points,
            confs=confs,
            depths={key: value for key, value in depths.items() if value is not None},
            frame_index=self.frame_count,
        )

    def summary(self):
        bone_cvs = []
        for _, lengths in self.bone_lengths.items():
            if len(lengths) < 8:
                continue
            mean_length = statistics.fmean(lengths)
            if mean_length <= 1e-6:
                continue
            bone_cvs.append(statistics.pstdev(lengths) / mean_length)

        valid_ratio = self.valid_keypoints / max(1, self.total_keypoints)
        jitter_mean = statistics.fmean(self.jitter_samples) if self.jitter_samples else 0.0
        jitter_p90 = (
            statistics.quantiles(self.jitter_samples, n=10, method="inclusive")[8]
            if len(self.jitter_samples) >= 2
            else (self.jitter_samples[0] if self.jitter_samples else 0.0)
        )
        depth_jump_mean = statistics.fmean(self.depth_jump_samples) if self.depth_jump_samples else 0.0
        depth_jump_p90 = (
            statistics.quantiles(self.depth_jump_samples, n=10, method="inclusive")[8]
            if len(self.depth_jump_samples) >= 2
            else (self.depth_jump_samples[0] if self.depth_jump_samples else 0.0)
        )
        jump_outlier_ratio = self.jump_outlier_count / max(1, self.jump_sample_count)
        implausible_ratio = self.implausible_bone_count / max(1, self.total_bone_count)
        bone_cv_mean = statistics.fmean(bone_cvs) if bone_cvs else 0.0

        score = (
            100.0
            - (jitter_mean * 220.0)
            - (jump_outlier_ratio * 38.0)
            - (bone_cv_mean * 120.0)
            - (depth_jump_mean * 1.2)
            - (implausible_ratio * 22.0)
            + (valid_ratio * 10.0)
        )

        return {
            "score": round(score, 4),
            "valid_ratio": round(valid_ratio, 4),
            "jitter_mean_norm": round(jitter_mean, 5),
            "jitter_p90_norm": round(jitter_p90, 5),
            "depth_jump_mean_cm": round(depth_jump_mean, 4),
            "depth_jump_p90_cm": round(depth_jump_p90, 4),
            "jump_outlier_ratio": round(jump_outlier_ratio, 4),
            "bone_cv_mean": round(bone_cv_mean, 5),
            "implausible_bone_ratio": round(implausible_ratio, 4),
            "jitter_samples": len(self.jitter_samples),
            "depth_jump_samples": len(self.depth_jump_samples),
            "bone_samples": int(self.total_bone_count),
        }


class DepthFusionFilter:
    def __init__(self):
        self._states: dict[int, MethodTrackState] = {}

    def _torso_depth(self, points, depths):
        candidates = []
        for index in (0, 5, 6, 11, 12):
            point = points.get(index)
            if point is None:
                continue
            depth = depths.get(index)
            if depth is not None:
                candidates.append(depth)
        if not candidates:
            return None
        candidates.sort()
        return float(candidates[len(candidates) // 2])

    def apply(self, track_id, bbox, keypoints, keypoint_conf, depth_frame):
        # Normalize raw points first.
        points = {}
        confs = {}
        depths = {}
        for index in range(17):
            if index >= len(keypoints) or index >= len(keypoint_conf):
                continue
            point = keypoints[index]
            if point is None or len(point) < 2:
                continue
            conf = float(keypoint_conf[index])
            points[index] = (float(point[0]), float(point[1]))
            confs[index] = conf
            depths[index] = depth_patch_cm(depth_frame, points[index])

        torso_depth = self._torso_depth(points, depths)
        prev = self._states.get(track_id)
        diag = bbox_diag(bbox)
        fused_points = {}
        fused_confs = {}
        fused_depths = {}

        for index in range(17):
            point = points.get(index)
            conf = confs.get(index, 0.0)
            depth = depths.get(index)

            is_valid = point is not None and conf >= 0.12
            depth_outlier = False
            if is_valid and depth is not None and torso_depth is not None:
                depth_outlier = abs(depth - torso_depth) > 72.0 and conf < 0.55

            prev_point = prev.points.get(index) if prev else None
            prev_conf = prev.confs.get(index, 0.0) if prev else 0.0
            prev_depth = prev.depths.get(index) if prev else None
            prev_frame = prev.frame_index if prev else -999

            if not is_valid:
                # Brief carry-forward to reduce one-frame dropouts.
                if prev_point is not None and (self._frame_index - prev_frame) <= 2 and prev_conf >= 0.18:
                    fused_points[index] = prev_point
                    fused_confs[index] = max(0.1, prev_conf * 0.85)
                    if prev_depth is not None:
                        fused_depths[index] = prev_depth
                continue

            if prev_point is not None and (self._frame_index - prev_frame) <= 4:
                disp_norm = point_distance(point, prev_point) / max(1.0, diag)
                depth_jump = abs(depth - prev_depth) if depth is not None and prev_depth is not None else None
                alpha = 0.62
                if disp_norm > 0.16:
                    alpha = 0.42
                if depth_jump is not None and depth_jump > 18.0:
                    alpha = min(alpha, 0.35)
                if depth_outlier:
                    alpha = min(alpha, 0.24)
                    conf = max(conf, prev_conf * 0.75)
                point = (
                    (prev_point[0] * (1.0 - alpha)) + (point[0] * alpha),
                    (prev_point[1] * (1.0 - alpha)) + (point[1] * alpha),
                )

            fused_points[index] = point
            fused_confs[index] = conf
            if depth is not None:
                fused_depths[index] = depth

        self._states[track_id] = MethodTrackState(
            points=fused_points,
            confs=fused_confs,
            depths=fused_depths,
            frame_index=self._frame_index,
        )

        output_points = []
        output_confs = []
        for index in range(17):
            if index in fused_points:
                output_points.append([float(fused_points[index][0]), float(fused_points[index][1])])
                output_confs.append(float(fused_confs.get(index, 0.0)))
            else:
                output_points.append([0.0, 0.0])
                output_confs.append(0.0)
        return output_points, output_confs

    def set_frame_index(self, frame_index):
        self._frame_index = int(frame_index)


def evaluate_video(
    model,
    video_path: Path,
    frame_step: int,
    max_frames: int,
    imgsz: int,
    conf: float,
    max_det: int,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    tracker = SimpleTracker()
    baseline_metrics = MetricsCollector("baseline_rgb")
    fusion_metrics = MetricsCollector("rgb_depth_fusion")
    fusion_filter = DepthFusionFilter()

    frame_index = 0
    used_frames = 0
    people_total = 0
    start_at = time.time()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_index += 1
        if frame_index % frame_step != 0:
            continue
        if used_frames >= max_frames:
            break

        used_frames += 1
        depth_frame = build_pseudo_depth(frame)
        result = model(
            frame,
            verbose=False,
            classes=[0],
            imgsz=imgsz,
            conf=conf,
            max_det=max_det,
        )[0]
        detections = decode_pose_detections(result)
        track_pairs = tracker.update(detections, frame_index)
        people_total += len(track_pairs)

        fusion_filter.set_frame_index(frame_index)
        for track_id, detection in track_pairs:
            bbox = detection["bbox"]
            keypoints = detection.get("keypoints") or []
            keypoint_conf = detection.get("keypoint_conf") or []
            if len(keypoints) < 17:
                keypoints = keypoints + ([[0.0, 0.0]] * (17 - len(keypoints)))
            if len(keypoint_conf) < 17:
                keypoint_conf = keypoint_conf + ([0.0] * (17 - len(keypoint_conf)))

            baseline_metrics.update(
                track_id=track_id,
                bbox=bbox,
                keypoints=keypoints,
                keypoint_conf=keypoint_conf,
                depth_frame=depth_frame,
                min_conf=0.12,
            )

            fused_points, fused_confs = fusion_filter.apply(
                track_id=track_id,
                bbox=bbox,
                keypoints=keypoints,
                keypoint_conf=keypoint_conf,
                depth_frame=depth_frame,
            )
            fusion_metrics.update(
                track_id=track_id,
                bbox=bbox,
                keypoints=fused_points,
                keypoint_conf=fused_confs,
                depth_frame=depth_frame,
                min_conf=0.12,
            )

    cap.release()
    elapsed = time.time() - start_at

    baseline_summary = baseline_metrics.summary()
    fusion_summary = fusion_metrics.summary()
    delta = {
        "score_delta": round(fusion_summary["score"] - baseline_summary["score"], 4),
        "jitter_mean_norm_delta": round(
            fusion_summary["jitter_mean_norm"] - baseline_summary["jitter_mean_norm"],
            5,
        ),
        "depth_jump_mean_cm_delta": round(
            fusion_summary["depth_jump_mean_cm"] - baseline_summary["depth_jump_mean_cm"],
            4,
        ),
        "bone_cv_mean_delta": round(
            fusion_summary["bone_cv_mean"] - baseline_summary["bone_cv_mean"],
            5,
        ),
        "valid_ratio_delta": round(
            fusion_summary["valid_ratio"] - baseline_summary["valid_ratio"],
            4,
        ),
    }

    return {
        "video": str(video_path),
        "frames_used": used_frames,
        "people_instances": people_total,
        "elapsed_sec": round(elapsed, 3),
        "baseline": baseline_summary,
        "fusion": fusion_summary,
        "delta": delta,
    }


def aggregate_results(video_results):
    if not video_results:
        return {}

    weights = [max(1, int(item["people_instances"])) for item in video_results]
    total_weight = float(sum(weights))

    def weighted(metric_path):
        value = 0.0
        for weight, item in zip(weights, video_results):
            current = item
            for key in metric_path:
                current = current[key]
            value += float(current) * weight
        return value / total_weight

    return {
        "baseline_score": round(weighted(("baseline", "score")), 4),
        "fusion_score": round(weighted(("fusion", "score")), 4),
        "score_delta": round(weighted(("delta", "score_delta")), 4),
        "baseline_jitter_mean_norm": round(weighted(("baseline", "jitter_mean_norm")), 5),
        "fusion_jitter_mean_norm": round(weighted(("fusion", "jitter_mean_norm")), 5),
        "jitter_mean_norm_delta": round(weighted(("delta", "jitter_mean_norm_delta")), 5),
        "baseline_depth_jump_mean_cm": round(weighted(("baseline", "depth_jump_mean_cm")), 4),
        "fusion_depth_jump_mean_cm": round(weighted(("fusion", "depth_jump_mean_cm")), 4),
        "depth_jump_mean_cm_delta": round(weighted(("delta", "depth_jump_mean_cm_delta")), 4),
        "baseline_bone_cv_mean": round(weighted(("baseline", "bone_cv_mean")), 5),
        "fusion_bone_cv_mean": round(weighted(("fusion", "bone_cv_mean")), 5),
        "bone_cv_mean_delta": round(weighted(("delta", "bone_cv_mean_delta")), 5),
        "baseline_valid_ratio": round(weighted(("baseline", "valid_ratio")), 4),
        "fusion_valid_ratio": round(weighted(("fusion", "valid_ratio")), 4),
        "valid_ratio_delta": round(weighted(("delta", "valid_ratio_delta")), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Offline comparison: RGB keypoints vs RGB+Depth post-fusion keypoints.")
    parser.add_argument("--video", action="append", default=[], help="Video path. Pass multiple --video to evaluate more files.")
    parser.add_argument("--model", default="", help="Pose model path or alias.")
    parser.add_argument("--frame-step", type=int, default=2, help="Evaluate every N-th frame.")
    parser.add_argument("--max-frames", type=int, default=420, help="Max evaluated frames per video.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.30, help="YOLO confidence threshold.")
    parser.add_argument("--max-det", type=int, default=12, help="YOLO max detections.")
    args = parser.parse_args()

    model_ref = resolve_model_path(args.model)
    video_paths = resolve_video_paths(args.video)

    missing = [path for path in video_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing video files: {[str(path) for path in missing]}")

    from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

    model = YOLO(model_ref)
    print(
        json.dumps(
            {
                "stage": "setup",
                "model": model_ref,
                "videos": [str(path) for path in video_paths],
                "frame_step": args.frame_step,
                "max_frames_per_video": args.max_frames,
                "imgsz": args.imgsz,
                "conf": args.conf,
                "max_det": args.max_det,
            },
            ensure_ascii=False,
        )
    )

    results = []
    all_start = time.time()
    for path in video_paths:
        video_start = time.time()
        summary = evaluate_video(
            model=model,
            video_path=path,
            frame_step=max(1, int(args.frame_step)),
            max_frames=max(1, int(args.max_frames)),
            imgsz=max(320, int(args.imgsz)),
            conf=float(args.conf),
            max_det=max(1, int(args.max_det)),
        )
        summary["eval_wall_time_sec"] = round(time.time() - video_start, 3)
        results.append(summary)
        print(json.dumps({"stage": "video_result", **summary}, ensure_ascii=False))

    aggregate = aggregate_results(results)
    output = {
        "stage": "aggregate",
        "video_count": len(results),
        "elapsed_sec": round(time.time() - all_start, 3),
        "aggregate": aggregate,
    }
    print(json.dumps(output, ensure_ascii=False))


if __name__ == "__main__":
    main()
