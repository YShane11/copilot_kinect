import argparse
import json
import math
import random
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from src.vision.attendance_pipeline import RecognitionPipeline
from src.vision.face_recognition_db import FaceRecognitionDB
from src.vision.pose_depth_metrics import PoseDepthMetricEngine


DEFAULT_EXPECTED_PEOPLE_COUNT = 6


class OfflineKinectService:
    def __init__(self):
        self._frame = None
        self._depth = None
        self._depth_visual = None
        self._source_mode = 'video'

    def set_bundle(self, frame, depth_frame=None, depth_visual_frame=None, source_mode='video'):
        if frame is None:
            self._frame = None
            self._depth = None
            self._depth_visual = None
            return
        self._frame = frame.copy()
        if depth_frame is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            depth_frame = (2000.0 - ((gray.astype('float32') / 255.0) * 1400.0)).astype('uint16')
        self._depth = depth_frame.copy()
        self._depth_visual = None if depth_visual_frame is None else depth_visual_frame.copy()
        self._source_mode = str(source_mode or 'video').strip().lower() or 'video'

    def set_frame(self, frame):
        self.set_bundle(frame)

    def get_latest_color_frame(self):
        return None if self._frame is None else self._frame.copy()

    def get_latest_depth_frame(self):
        return None if self._depth is None else self._depth.copy()

    def get_latest_depth_visual_frame(self):
        return None if self._depth_visual is None else self._depth_visual.copy()

    def get_latest_jpeg(self, kind):
        return None

    def get_status(self):
        return {
            'status': 'connected',
            'message': 'offline_video',
            'source_mode': self._source_mode,
        }

    def set_processing_mode(self, mode):
        self._processing_mode = str(mode or 'idle')


@dataclass
class ComboResult:
    params: dict
    score: float
    metrics: dict


def configure_pipeline_pose_model(pipeline: RecognitionPipeline, base_dir: Path, model_ref: str):
    raw_ref = str(model_ref or '').strip()
    if not raw_ref:
        return

    path = Path(raw_ref)
    if not path.is_absolute():
        path = base_dir / raw_ref

    if path.exists():
        display_name = path.stem
        configured_ref = str(path)
        is_local = True
    else:
        display_name = Path(raw_ref).stem or raw_ref
        configured_ref = raw_ref
        is_local = False

    pipeline.MODEL_CANDIDATES = (
        (display_name, configured_ref, is_local),
        *tuple(pipeline.MODEL_CANDIDATES),
    )
    pipeline._yolo_model = None
    pipeline._yolo_error = None
    with pipeline._lock:
        pipeline._status['detector_model'] = 'idle'


def resolve_video_path(path: Path, workspace: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f'Video file not found: {path}')

    try:
        str(path).encode('ascii')
        return path
    except UnicodeEncodeError:
        alias = workspace / 'data' / 'test_videos' / '_aliases' / '_video_tuning_source.mp4'
        src_stat = path.stat()
        need_copy = True
        if alias.exists():
            alias_stat = alias.stat()
            need_copy = (
                alias_stat.st_size != src_stat.st_size
                or int(alias_stat.st_mtime) != int(src_stat.st_mtime)
            )
        if need_copy:
            alias.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, alias)
        return alias


def make_pipeline(base_dir: Path, pose_model_ref: str = ''):
    face_db = FaceRecognitionDB(base_dir)
    offline_kinect = OfflineKinectService()
    pipeline = RecognitionPipeline(base_dir, offline_kinect, face_db)

    # Stop internal polling thread; tuning loop drives the pipeline manually.
    pipeline._running = False
    if pipeline._thread.is_alive():
        pipeline._thread.join(timeout=2.0)
    warmup_thread = getattr(pipeline, '_warmup_thread', None)
    if warmup_thread is not None and warmup_thread.is_alive():
        pipeline._warmup_thread.join(timeout=120.0)

    # Isolate tuning run from production presence records.
    pipeline._presence_file = base_dir / 'data' / '_presence_records_tuning.json'
    with pipeline._lock:
        pipeline._confirmed_people = {}
        pipeline._temporary_people = {}
        pipeline._session_confirmed_ids.clear()
        pipeline._next_temp_number = 1
        pipeline._next_tracking_id = 1
        pipeline._last_person_boxes = []
        pipeline._last_face_person_boxes = []
        pipeline._last_face_person_fallback_at = 0.0
        pipeline._last_person_detect_at = 0.0
    configure_pipeline_pose_model(pipeline, base_dir, pose_model_ref)
    pipeline.set_attendance_mode(True)
    return pipeline, offline_kinect


def reset_pipeline_state(pipeline: RecognitionPipeline):
    with pipeline._lock:
        pipeline._temporary_people = {}
        pipeline._confirmed_people = {}
        pipeline._session_confirmed_ids.clear()
        pipeline._next_temp_number = 1
        pipeline._next_tracking_id = 1
        pipeline._last_person_boxes = []
        pipeline._last_face_person_boxes = []
        pipeline._last_face_person_fallback_at = 0.0
        pipeline._last_person_detect_at = 0.0
        pipeline._announcements = []
    pipeline.set_attendance_mode(True)


def apply_params(pipeline: RecognitionPipeline, params: dict):
    pipeline.MAX_INFERENCE_WIDTH = params['max_inference_width']
    pipeline.YOLO_IMAGE_SIZE = params['yolo_image_size']
    pipeline.YOLO_DETECT_INTERVAL = params['yolo_detect_interval']
    pipeline.YOLO_CONFIDENCE = params['yolo_confidence']
    pipeline.TRACK_IOU_THRESHOLD = params['track_iou_threshold']
    pipeline.TEMP_PERSON_TIMEOUT = params['temp_person_timeout']
    pipeline.CONFIRMED_ABSENT_TIMEOUT = params['confirmed_absent_timeout']
    pipeline.RECOGNITION_THRESHOLD = params['recognition_threshold']
    pipeline.AUTO_RELINK_THRESHOLD = params['auto_relink_threshold']
    pipeline.AUTO_RELINK_INTERVAL = params['auto_relink_interval']
    pipeline.MAX_DETECTIONS = params['max_detections']
    pipeline.MIN_PERSON_BOX_WIDTH = params['min_person_box_width']
    pipeline.MIN_PERSON_BOX_HEIGHT = params['min_person_box_height']
    pipeline.DETECTION_DUPLICATE_IOU_THRESHOLD = params['detection_duplicate_iou_threshold']
    pipeline.DETECTION_DUPLICATE_CENTER_RATIO = params['detection_duplicate_center_ratio']
    pipeline.DETECTION_DUPLICATE_AREA_RATIO = params['detection_duplicate_area_ratio']
    pipeline.TEMPORARY_MERGE_IOU_THRESHOLD = params['temporary_merge_iou_threshold']
    pipeline.TEMPORARY_MERGE_DISTANCE_RATIO = params['temporary_merge_distance_ratio']
    pipeline.FACE_PERSON_FALLBACK_INTERVAL = params.get('face_person_fallback_interval', pipeline.FACE_PERSON_FALLBACK_INTERVAL)
    pipeline.FACE_PERSON_FALLBACK_MIN_SCORE = params.get('face_person_fallback_min_score', pipeline.FACE_PERSON_FALLBACK_MIN_SCORE)
    pipeline.FACE_PERSON_FALLBACK_MIN_SIZE = params.get('face_person_fallback_min_size', pipeline.FACE_PERSON_FALLBACK_MIN_SIZE)
    pipeline.FACE_PERSON_FALLBACK_MAX_WIDTH = params.get('face_person_fallback_max_width', pipeline.FACE_PERSON_FALLBACK_MAX_WIDTH)
    pipeline.FACE_PERSON_FALLBACK_BOX_SCALE_X = params.get('face_person_fallback_box_scale_x', pipeline.FACE_PERSON_FALLBACK_BOX_SCALE_X)
    pipeline.FACE_PERSON_FALLBACK_BOX_TOP_SCALE = params.get('face_person_fallback_box_top_scale', pipeline.FACE_PERSON_FALLBACK_BOX_TOP_SCALE)
    pipeline.FACE_PERSON_FALLBACK_BOX_BOTTOM_SCALE = params.get('face_person_fallback_box_bottom_scale', pipeline.FACE_PERSON_FALLBACK_BOX_BOTTOM_SCALE)


def read_video_meta(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open video: {video_path}')
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return {'fps': fps, 'frames': frames, 'duration': frames / fps if fps > 0 else 0.0}


def quad_backend_slices(backend_name: str):
    normalized = str(backend_name or '').strip().lower()
    if normalized == 'v1':
        return (
            (slice(0, 480), slice(0, 640)),
            (slice(0, 480), slice(640, 1280)),
        )
    if normalized == 'v2':
        return (
            (slice(480, 960), slice(0, 640)),
            (slice(480, 960), slice(640, 1280)),
        )
    raise ValueError(f'Unsupported quad backend: {backend_name}')


def decode_quad_backend_frame(frame, backend_name: str):
    rgb_slice, depth_slice = quad_backend_slices(backend_name)
    rgb = frame[rgb_slice].copy()
    depth_visual = frame[depth_slice].copy()
    return rgb, depth_visual


def decode_side_by_side_frame(frame):
    height, width = frame.shape[:2]
    if width < 2 or height < 1:
        raise ValueError('Invalid side-by-side frame shape.')

    half_width = width // 2
    rgb = frame[:, :half_width].copy()
    depth_visual = frame[:, half_width:half_width * 2].copy()
    return rgb, depth_visual


def depth_from_visual_frame(depth_visual):
    if depth_visual is None:
        return None

    gray = cv2.cvtColor(depth_visual, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    normalized = normalized.astype('uint8')
    return (300.0 + ((255.0 - normalized.astype('float32')) / 255.0) * 4200.0).astype('uint16')


def try_confirm_all_temporaries(pipeline: RecognitionPipeline, max_attempts: int):
    with pipeline._lock:
        temp_ids = [temp_id for temp_id, person in pipeline._temporary_people.items() if person.confirm_status != 'processing']
    confirmed = 0
    failed = 0
    for temp_id in temp_ids[:max(0, max_attempts)]:
        result = pipeline.confirm_temporary_person(temp_id)
        if result.get('status') == 'confirmed':
            confirmed += 1
        elif result.get('status') == 'error':
            failed += 1
    return confirmed, failed


def evaluate_combo(
    pipeline: RecognitionPipeline,
    offline_kinect: OfflineKinectService,
    video_path: Path,
    params: dict,
    loops: int,
    frame_step: int,
    confirm_every_seconds: float,
    max_confirm_attempts: int,
    expected_people_count: int,
    quad_backend: str = '',
    side_by_side: bool = False,
):
    reset_pipeline_state(pipeline)
    apply_params(pipeline, params)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open video during evaluation: {video_path}')

    simulated_now = 0.0
    frame_index = 0
    loop_count = 0
    last_confirm_at = -999.0

    active_counts = []
    present_confirmed_counts = []
    temporary_counts = []
    temp_id_seen = set()
    temp_spawn_events = 0
    confirm_success = 0
    confirm_failed = 0
    confirmed_id_seen = set()
    confirm_promotions = 0
    frames_processed = 0
    classroom_metric_row_counts = []
    pose_detections_total = 0
    pose_detection_frames = 0
    pose_frames_with_detections = 0
    confident_keypoint_counts = []
    skeleton_valid_detections = 0
    duplicate_detection_frames = 0
    duplicate_detection_pairs = 0
    surrogate_metric_engine = PoseDepthMetricEngine()
    face_eval_stride_frames = max(1, int(round(3.0 / max(1e-3, pipeline.LOOP_INTERVAL))))
    face_samples = 0
    face_detected = 0
    face_matched = 0
    face_best_similarity = []

    while loop_count < loops:
        ok, frame = cap.read()
        if not ok or frame is None:
            loop_count += 1
            if loop_count >= loops:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_index += 1
        if frame_index % frame_step != 0:
            continue

        if side_by_side:
            try:
                frame, depth_visual_frame = decode_side_by_side_frame(frame)
            except Exception as exc:  # pylint: disable=broad-except
                cap.release()
                return ComboResult(
                    params=params,
                    score=-1e9,
                    metrics={
                        'error': f'side_by_side_decode_failed: {exc}',
                        'frames_processed': frames_processed,
                    },
                )
            depth_frame = depth_from_visual_frame(depth_visual_frame)
            source_mode = 'video'
            offline_kinect.set_bundle(
                frame,
                depth_frame=depth_frame,
                depth_visual_frame=depth_visual_frame,
                source_mode=source_mode,
            )
        elif quad_backend:
            try:
                frame, depth_visual_frame = decode_quad_backend_frame(frame, quad_backend)
            except Exception as exc:  # pylint: disable=broad-except
                cap.release()
                return ComboResult(
                    params=params,
                    score=-1e9,
                    metrics={
                        'error': f'quad_decode_failed: {exc}',
                        'frames_processed': frames_processed,
                    },
                )
            depth_frame = depth_from_visual_frame(depth_visual_frame)
            source_mode = 'video'
            offline_kinect.set_bundle(
                frame,
                depth_frame=depth_frame,
                depth_visual_frame=depth_visual_frame,
                source_mode=source_mode,
            )
        else:
            offline_kinect.set_frame(frame)
            depth_frame = offline_kinect.get_latest_depth_frame()
            source_mode = offline_kinect.get_status().get('source_mode', 'video')

        frames_processed += 1
        simulated_now += pipeline.LOOP_INTERVAL

        try:
            person_boxes = pipeline._detect_people(
                frame,
                simulated_now,
                depth_frame=depth_frame,
                depth_source_mode=source_mode,
            )
        except Exception as exc:  # pylint: disable=broad-except
            cap.release()
            return ComboResult(
                params=params,
                score=-1e9,
                metrics={
                    'error': f'detect_failed: {exc}',
                    'frames_processed': frames_processed,
                },
            )

        duplicate_pairs_this_frame = 0
        if len(person_boxes) > 1:
            for index in range(len(person_boxes)):
                left_bbox = person_boxes[index]
                for right_index in range(index + 1, len(person_boxes)):
                    right_bbox = person_boxes[right_index]
                    if pipeline._bbox_iou(left_bbox, right_bbox) >= 0.38:
                        duplicate_pairs_this_frame += 1
        if duplicate_pairs_this_frame > 0:
            duplicate_detection_frames += 1
            duplicate_detection_pairs += duplicate_pairs_this_frame

        active_face_bboxes = []
        active_entities = []
        pose_detections = []
        with pipeline._lock:
            pipeline._match_detections_locked(person_boxes, simulated_now)
            pipeline._update_classroom_metrics_locked(
                frame.shape,
                depth_frame,
                simulated_now,
            )
            pipeline._try_auto_relink_locked(frame, simulated_now)

            temporary_ids = list(pipeline._temporary_people.keys())
            for temp_id in temporary_ids:
                if temp_id not in temp_id_seen:
                    temp_id_seen.add(temp_id)
                    temp_spawn_events += 1

            present_confirmed = sum(
                1 for person in pipeline._confirmed_people.values() if person.current_status == 'present'
            )
            active_total = len(pipeline._temporary_people) + present_confirmed

            per_user_metric_rows = []
            for person in pipeline._confirmed_people.values():
                if person.current_status != 'present':
                    continue
                user_metrics = pipeline._metric_engine.get_user_metrics(person.user_id)
                row_count = sum(len(rows) for rows in user_metrics.values())
                per_user_metric_rows.append(row_count)
            if per_user_metric_rows:
                classroom_metric_row_counts.append(statistics.fmean(per_user_metric_rows))

            pose_detections = list(pipeline._last_pose_detections or [])
            pose_detection_frames += 1
            pose_detections_total += len(pose_detections)
            if pose_detections:
                pose_frames_with_detections += 1
            for detection in pose_detections:
                keypoint_conf = detection.get('keypoint_conf') or []
                if not keypoint_conf:
                    confident_count = 0
                else:
                    confident_count = sum(
                        1 for confidence in keypoint_conf
                        if float(confidence) >= float(pipeline.POSE_KEYPOINT_MIN_CONFIDENCE)
                    )
                    confident_keypoint_counts.append(confident_count)
                if confident_count >= 8:
                    skeleton_valid_detections += 1

            for person in pipeline._temporary_people.values():
                if person.current_status != 'tracking' or person.bbox is None:
                    continue
                active_face_bboxes.append(list(person.bbox))
                active_entities.append((f't:{person.tracking_id}', list(person.bbox)))

            for person in pipeline._confirmed_people.values():
                if person.current_status != 'present' or person.bbox is None:
                    continue
                active_face_bboxes.append(list(person.bbox))
                tracking_id = person.current_tracking_id if person.current_tracking_id is not None else -1
                active_entities.append((f'u:{person.user_id}:{tracking_id}', list(person.bbox)))

            active_counts.append(active_total)
            present_confirmed_counts.append(present_confirmed)
            temporary_counts.append(len(pipeline._temporary_people))

        if active_entities:
            used_detection_indexes = set()
            matched_detection_by_entity = {}
            centers_by_entity = {}

            for entity_id, bbox in active_entities:
                detection_index, detection = pipeline._match_pose_detection_to_bbox(
                    bbox,
                    pose_detections,
                    used_indexes=used_detection_indexes,
                )
                if detection is not None and detection_index is not None:
                    used_detection_indexes.add(detection_index)
                else:
                    detection = {'bbox': bbox, 'keypoints': [], 'keypoint_conf': []}
                matched_detection_by_entity[entity_id] = detection
                centers_by_entity[entity_id] = pipeline._bbox_center(detection.get('bbox', bbox))

            for entity_id, bbox in active_entities:
                detection = matched_detection_by_entity[entity_id]
                peer_centers = [
                    center
                    for other_id, center in centers_by_entity.items()
                    if other_id != entity_id
                ]
                surrogate_metric_engine.update_student(
                    user_id=entity_id,
                    frame_shape=frame.shape,
                    bbox=bbox,
                    pose_detection=detection,
                    depth_frame=depth_frame,
                    depth_source_mode=source_mode,
                    peer_centers=peer_centers,
                    now=simulated_now,
                )

        if frames_processed % face_eval_stride_frames == 0 and active_face_bboxes:
            for bbox in active_face_bboxes[:2]:
                analysis = pipeline._analyze_face_inside_bbox(frame, bbox)
                face_samples += 1
                if analysis.get('status') != 'ok':
                    continue
                face_detected += 1
                embedding = analysis.get('embedding')
                if not embedding:
                    continue
                matches = pipeline.face_db.match_embedding(embedding, threshold=-1.0)
                if not matches:
                    continue
                best_similarity = float(matches[0].get('similarity', 0.0))
                face_best_similarity.append(best_similarity)
                if best_similarity >= float(pipeline.RECOGNITION_THRESHOLD):
                    face_matched += 1

        if confirm_every_seconds > 0 and simulated_now - last_confirm_at >= confirm_every_seconds:
            success_count, failed_count = try_confirm_all_temporaries(
                pipeline,
                max_attempts=max_confirm_attempts,
            )
            confirm_success += success_count
            confirm_failed += failed_count
            last_confirm_at = simulated_now
            if success_count:
                with pipeline._lock:
                    for person in pipeline._confirmed_people.values():
                        if person.current_status == 'present':
                            confirmed_id_seen.add(person.user_id)
                    confirm_promotions += success_count

    cap.release()

    if not active_counts:
        return ComboResult(
            params=params,
            score=-1e9,
            metrics={'error': 'no frames processed'},
        )

    expected_count = max(0, int(expected_people_count))
    abs_error = [abs(value - expected_count) for value in active_counts]
    mean_abs_error = statistics.fmean(abs_error)
    over_count = statistics.fmean([max(0, value - expected_count) for value in active_counts])
    under_count = statistics.fmean([max(0, expected_count - value) for value in active_counts])
    mean_active = statistics.fmean(active_counts)
    mean_temporary = statistics.fmean(temporary_counts)
    mean_confirmed = statistics.fmean(present_confirmed_counts)
    max_active = max(active_counts)
    min_active = min(active_counts)

    stable_ratio = sum(1 for value in active_counts if abs(value - expected_count) <= 1) / len(active_counts)
    spike_rate = sum(1 for value in active_counts if value > (expected_count + 2)) / len(active_counts)
    max_active_excess = max(0.0, float(max_active) - float(expected_count + 2))
    churn_rate = temp_spawn_events / max(1, len(active_counts))
    confirmation_rate = confirm_success / max(1, confirm_success + confirm_failed)
    unique_confirmed = len(confirmed_id_seen)
    duplicate_frame_ratio = duplicate_detection_frames / max(1, len(active_counts))
    pose_frame_coverage = pose_frames_with_detections / max(1, pose_detection_frames)
    skeleton_valid_ratio = skeleton_valid_detections / max(1, len(confident_keypoint_counts))
    face_detect_ratio = face_detected / max(1, face_samples)
    face_match_ratio = face_matched / max(1, face_detected)

    indicator_track_row_counts = []
    indicator_alignment_scores = []
    for user_id in list(surrogate_metric_engine._students.keys()):  # pylint: disable=protected-access
        user_metrics = surrogate_metric_engine.get_user_metrics(user_id)
        if not user_metrics:
            continue
        row_lengths = [len(rows) for rows in user_metrics.values()]
        if not row_lengths:
            continue
        indicator_track_row_counts.append(sum(row_lengths))
        spread = max(row_lengths) - min(row_lengths)
        indicator_alignment_scores.append(1.0 if spread <= 1 else 0.0)

    indicator_alignment_ratio = statistics.fmean(indicator_alignment_scores) if indicator_alignment_scores else 0.0

    # Higher is better.
    score = (
        100.0
        - (mean_abs_error * 20.0)
        - (over_count * 9.0)
        - (under_count * 10.0)
        - (churn_rate * 30.0)
        - (duplicate_frame_ratio * 36.0)
        - (spike_rate * 40.0)
        - (max_active_excess * 3.5)
        + (stable_ratio * 32.0)
        + (confirmation_rate * 8.0)
        + (mean_confirmed * 2.5)
        + (unique_confirmed * 1.2)
        + (pose_frame_coverage * 10.0)
        + (skeleton_valid_ratio * 10.0)
        + (face_detect_ratio * 5.0)
        + (face_match_ratio * 7.0)
        + (indicator_alignment_ratio * 7.0)
    )

    metrics = {
        'frames_processed': frames_processed,
        'mean_active': round(mean_active, 3),
        'max_active': max_active,
        'min_active': min_active,
        'mean_temporary': round(mean_temporary, 3),
        'mean_confirmed_present': round(mean_confirmed, 3),
        'mean_abs_error': round(mean_abs_error, 4),
        'stable_ratio_within_1': round(stable_ratio, 4),
        'spike_ratio_above_expected_plus_2': round(spike_rate, 4),
        'max_active_excess_over_expected_plus_2': round(max_active_excess, 3),
        'churn_rate': round(churn_rate, 4),
        'duplicate_frame_ratio': round(duplicate_frame_ratio, 4),
        'duplicate_pairs_total': duplicate_detection_pairs,
        'confirmation_rate': round(confirmation_rate, 4),
        'confirm_success': confirm_success,
        'confirm_failed': confirm_failed,
        'confirm_promotions': confirm_promotions,
        'unique_confirmed_ids': unique_confirmed,
        'temp_spawn_events': temp_spawn_events,
        'face_sample_count': face_samples,
        'face_detected_count': face_detected,
        'face_detect_ratio': round(face_detect_ratio, 4),
        'face_matched_count': face_matched,
        'face_match_ratio': round(face_match_ratio, 4),
        'face_best_similarity_mean': round(statistics.fmean(face_best_similarity), 4) if face_best_similarity else 0.0,
        'face_best_similarity_p90': round(statistics.quantiles(face_best_similarity, n=10, method='inclusive')[8], 4) if len(face_best_similarity) >= 2 else (round(face_best_similarity[0], 4) if face_best_similarity else 0.0),
        'mean_classroom_metric_rows_per_present_user': round(
            statistics.fmean(classroom_metric_row_counts),
            3,
        ) if classroom_metric_row_counts else 0.0,
        'avg_pose_detections_per_frame': round(
            pose_detections_total / max(1, pose_detection_frames),
            3,
        ),
        'pose_frame_coverage': round(pose_frame_coverage, 4),
        'avg_confident_keypoints_per_detection': round(
            statistics.fmean(confident_keypoint_counts),
            3,
        ) if confident_keypoint_counts else 0.0,
        'skeleton_valid_ratio': round(skeleton_valid_ratio, 4),
        'indicator_tracks_seen': len(indicator_track_row_counts),
        'indicator_rows_per_track_mean': round(statistics.fmean(indicator_track_row_counts), 3) if indicator_track_row_counts else 0.0,
        'indicator_alignment_ratio': round(indicator_alignment_ratio, 4),
    }
    return ComboResult(params=params, score=round(score, 4), metrics=metrics)


def build_candidate_params():
    baseline = {
        'max_inference_width': 1600,
        'yolo_image_size': 640,
        'yolo_detect_interval': 0.22,
        'yolo_confidence': 0.32,
        'track_iou_threshold': 0.10,
        'temp_person_timeout': 1.2,
        'confirmed_absent_timeout': 12.0,
        'recognition_threshold': 0.44,
        'auto_relink_threshold': 0.52,
        'auto_relink_interval': 2.8,
        'max_detections': 10,
        'min_person_box_width': 32.0,
        'min_person_box_height': 36.0,
        'detection_duplicate_iou_threshold': 0.64,
        'detection_duplicate_center_ratio': 0.26,
        'detection_duplicate_area_ratio': 2.2,
        'temporary_merge_iou_threshold': 0.50,
        'temporary_merge_distance_ratio': 0.30,
        'face_person_fallback_interval': 0.70,
        'face_person_fallback_min_score': 0.50,
        'face_person_fallback_min_size': 12.0,
        'face_person_fallback_max_width': 960,
        'face_person_fallback_box_scale_x': 3.0,
        'face_person_fallback_box_top_scale': 0.45,
        'face_person_fallback_box_bottom_scale': 2.35,
    }

    grid = {
        'max_inference_width': [960, 1280, 1600],
        'yolo_image_size': [512, 640],
        'yolo_detect_interval': [0.22, 0.28, 0.34],
        'yolo_confidence': [0.30, 0.32, 0.34, 0.36],
        'track_iou_threshold': [0.06, 0.08, 0.10, 0.12],
        'temp_person_timeout': [1.2, 1.6, 2.4, 3.2, 4.5],
        'confirmed_absent_timeout': [10.0, 12.0, 16.0],
        'recognition_threshold': [0.40, 0.44, 0.48, 0.52, 0.56],
        'auto_relink_threshold': [0.48, 0.52, 0.56, 0.60],
        'auto_relink_interval': [1.6, 2.2, 2.8],
        'max_detections': [8, 10, 12],
        'min_person_box_width': [24.0, 28.0, 32.0, 36.0],
        'min_person_box_height': [32.0, 36.0, 44.0, 52.0, 60.0],
        'detection_duplicate_iou_threshold': [0.64, 0.68, 0.72, 0.76],
        'detection_duplicate_center_ratio': [0.18, 0.22, 0.26, 0.30],
        'detection_duplicate_area_ratio': [1.8, 2.2, 2.8],
        'temporary_merge_iou_threshold': [0.50, 0.56, 0.62],
        'temporary_merge_distance_ratio': [0.30, 0.36, 0.40],
        'face_person_fallback_interval': [0.45, 0.70, 0.95],
        'face_person_fallback_min_score': [0.45, 0.50, 0.58],
        'face_person_fallback_min_size': [10.0, 12.0, 14.0],
        'face_person_fallback_max_width': [768, 960],
        'face_person_fallback_box_scale_x': [2.6, 3.0, 3.4],
        'face_person_fallback_box_top_scale': [0.35, 0.45, 0.55],
        'face_person_fallback_box_bottom_scale': [2.0, 2.35, 2.7],
    }
    return baseline, grid


def sample_param_candidates(sample_count: int, seed: int):
    baseline, grid = build_candidate_params()
    rng = random.Random(seed)
    candidates = [baseline]
    seen = {tuple(sorted(baseline.items()))}

    keys = list(grid.keys())
    while len(candidates) < sample_count:
        params = {key: rng.choice(grid[key]) for key in keys}
        signature = tuple(sorted(params.items()))
        if signature in seen:
            continue
        seen.add(signature)
        candidates.append(params)
    return candidates


def main():
    parser = argparse.ArgumentParser(description='Tune attendance and face-recognition parameters with looped video playback.')
    parser.add_argument('--video', default='data/test_videos/test_video.mp4', help='Video filename or absolute path.')
    parser.add_argument('--quad-backend', choices=('v1', 'v2'), default='', help='Treat the video as a V1/V2 quad recording and crop the selected backend pane.')
    parser.add_argument('--sample-count', type=int, default=14, help='Number of parameter sets to evaluate.')
    parser.add_argument('--search-loops', type=int, default=1, help='Video loops for search stage.')
    parser.add_argument('--validate-loops', type=int, default=3, help='Video loops for final validation stage.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for candidate sampling.')
    parser.add_argument('--confirm-every', type=float, default=2.4, help='Seconds between auto-confirm attempts.')
    parser.add_argument('--max-confirm-attempts', type=int, default=2, help='Maximum confirm attempts at each confirm interval.')
    parser.add_argument('--expected-people', type=int, default=DEFAULT_EXPECTED_PEOPLE_COUNT, help='Expected person count in the video scene.')
    parser.add_argument('--pose-model', default='', help='Pose model reference, e.g. models/yolo/yolo26x-pose.pt or yolo26x-pose.pt.')
    parser.add_argument('--side-by-side', action='store_true', help='Treat the video as RGB/NIR side-by-side: left RGB, right NIR/depth visual.')
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[1]
    raw_video_path = Path(args.video)
    if not raw_video_path.is_absolute():
        raw_video_path = workspace / raw_video_path
    video_path = resolve_video_path(raw_video_path, workspace)
    meta = read_video_meta(video_path)

    print(json.dumps({
        'stage': 'video_meta',
        'video': str(video_path),
        'pose_model': args.pose_model or 'auto',
        'quad_backend': args.quad_backend or '',
        'side_by_side': bool(args.side_by_side),
        **meta,
    }, ensure_ascii=False))

    pipeline, offline_kinect = make_pipeline(workspace, pose_model_ref=args.pose_model)
    stride = max(1, int(round(meta['fps'] * pipeline.LOOP_INTERVAL)))
    search_stride = max(stride, 4)

    candidates = sample_param_candidates(max(1, args.sample_count), args.seed)
    search_results = []
    start_time = time.time()
    for index, params in enumerate(candidates, start=1):
        result = evaluate_combo(
            pipeline,
            offline_kinect,
            video_path,
            params=params,
            loops=args.search_loops,
            frame_step=search_stride,
            confirm_every_seconds=args.confirm_every,
            max_confirm_attempts=args.max_confirm_attempts,
            expected_people_count=args.expected_people,
            quad_backend=args.quad_backend,
            side_by_side=args.side_by_side,
        )
        search_results.append(result)
        print(json.dumps({
            'stage': 'search',
            'index': index,
            'total': len(candidates),
            'score': result.score,
            'params': result.params,
            'metrics': result.metrics,
        }, ensure_ascii=False))

    search_results.sort(key=lambda item: item.score, reverse=True)
    finalists = search_results[:3]

    validation_results = []
    for index, candidate in enumerate(finalists, start=1):
        result = evaluate_combo(
            pipeline,
            offline_kinect,
            video_path,
            params=candidate.params,
            loops=args.validate_loops,
            frame_step=stride,
            confirm_every_seconds=args.confirm_every,
            max_confirm_attempts=args.max_confirm_attempts,
            expected_people_count=args.expected_people,
            quad_backend=args.quad_backend,
            side_by_side=args.side_by_side,
        )
        validation_results.append(result)
        print(json.dumps({
            'stage': 'validation',
            'index': index,
            'total': len(finalists),
            'score': result.score,
            'params': result.params,
            'metrics': result.metrics,
        }, ensure_ascii=False))

    validation_results.sort(key=lambda item: item.score, reverse=True)
    best = validation_results[0]
    elapsed = time.time() - start_time

    print(json.dumps({
        'stage': 'best',
        'elapsed_sec': round(elapsed, 2),
        'score': best.score,
        'params': best.params,
        'metrics': best.metrics,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
