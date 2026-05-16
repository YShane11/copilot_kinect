import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.vision.pose_depth_metrics import PoseDepthMetricEngine


def _env_float(name, default):
    raw = os.getenv(name, '')
    if raw is None:
        return float(default)
    raw = str(raw).strip()
    if raw == '':
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_bool(name, default):
    raw = os.getenv(name, '')
    if raw is None:
        return bool(default)
    raw = str(raw).strip().lower()
    if raw == '':
        return bool(default)
    if raw in {'1', 'true', 'yes', 'on'}:
        return True
    if raw in {'0', 'false', 'no', 'off'}:
        return False
    return bool(default)


def _env_int(name, default):
    raw = os.getenv(name, '')
    if raw is None:
        return int(default)
    raw = str(raw).strip()
    if raw == '':
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


@dataclass
class TemporaryPerson:
    temp_id: str
    display_name: str
    bbox: list
    tracking_id: int
    first_seen: float
    last_seen: float
    slot_number: int = 0
    current_status: str = 'tracking'
    confirm_status: str = 'pending'
    confirm_message: str = ''
    face_bbox: list | None = None
    face_embedding: list | None = None
    face_embedding_at: float = 0.0
    last_auto_relink_at: float = 0.0


@dataclass
class ConfirmedPerson:
    user_id: str
    label: str
    name: str
    student_id: str
    department: str
    title: str
    first_confirmed_at: float
    first_seen: float
    last_seen: float
    current_status: str = 'present'
    current_tracking_id: int | None = None
    bbox: list | None = None
    presence_segments: list = field(default_factory=list)
    appearance_count: int = 0
    last_similarity: float = 0.0


class RecognitionPipeline:
    PERSON_CLASS_ID = 0
    LOOP_INTERVAL = 0.08
    IDLE_LOOP_INTERVAL = 0.25
    IDLE_STATUS_REFRESH_INTERVAL = 0.75
    WARMUP_ON_STARTUP = _env_bool('ATTENDANCE_WARMUP_ON_STARTUP', True)
    WARMUP_STARTUP_DELAY = _env_float('ATTENDANCE_WARMUP_STARTUP_DELAY', 0.0)
    RENDER_INTERVAL = _env_float('ATTENDANCE_RENDER_INTERVAL', 0.18)
    SUMMARY_UPDATE_INTERVAL = _env_float('ATTENDANCE_SUMMARY_INTERVAL', 0.60)
    METRIC_UPDATE_INTERVAL = _env_float('ATTENDANCE_METRIC_INTERVAL', 0.60)
    YOLO_DETECT_INTERVAL = 0.18
    TRACKED_YOLO_DETECT_INTERVAL = _env_float('ATTENDANCE_TRACKED_DETECT_INTERVAL', 0.32)
    YOLO_IMAGE_SIZE = 640
    YOLO_CONFIDENCE = 0.22
    YOLO_WARMUP_CONFIDENCE = 0.30
    YOLO_WARMUP_MAX_DETECTIONS = 300
    PRESENCE_SAMPLE_INTERVAL = 1.0
    PRESENCE_WINDOW_SECONDS = 600.0
    MAX_INFERENCE_WIDTH = 1600
    TRACK_IOU_THRESHOLD = 0.08
    TRACK_CENTER_DISTANCE_RATIO = 0.58
    TEMP_PERSON_TIMEOUT = 1.4
    CONFIRMED_ABSENT_TIMEOUT = 12
    RECOGNITION_THRESHOLD = 0.44
    AUTO_RELINK_THRESHOLD = 0.52
    AUTO_RELINK_INTERVAL = 2.8
    AUTO_RELINK_MAX_FACE_CHECKS_PER_TICK = 2
    AUTO_RELINK_MIN_TRACK_SECONDS = 0.5
    MAX_DETECTIONS = 12
    MIN_PERSON_BOX_WIDTH = 24.0
    MIN_PERSON_BOX_HEIGHT = 32.0
    DETECTION_DUPLICATE_IOU_THRESHOLD = 0.64
    DETECTION_DUPLICATE_CENTER_RATIO = 0.22
    DETECTION_DUPLICATE_AREA_RATIO = 1.8
    DETECTION_DUPLICATE_MIN_OVERLAP_RATIO = 0.72
    TEMPORARY_MERGE_IOU_THRESHOLD = 0.56
    TEMPORARY_MERGE_DISTANCE_RATIO = 0.36
    TEMPORARY_MERGE_MIN_OVERLAP_RATIO = 0.74
    TRACK_MIN_OVERLAP_RATIO = 0.38
    TRACK_MAX_AREA_RATIO = 4.5
    UNMATCHED_DUPLICATE_GUARD_IOU_THRESHOLD = 0.46
    UNMATCHED_DUPLICATE_GUARD_OVERLAP_RATIO = 0.64
    POSE_KEYPOINT_MIN_CONFIDENCE = 0.18
    FACE_ANALYSIS_MAX_SIDE = 384
    FACE_REGION_TOP_RATIO = 0.02
    FACE_REGION_BOTTOM_RATIO = 0.64
    FACE_EMBEDDING_CACHE_SECONDS = 2.2
    FACE_PERSON_FALLBACK_ENABLED = _env_bool('ATTENDANCE_FACE_PERSON_FALLBACK_ENABLED', True)
    FACE_PERSON_FALLBACK_INTERVAL = _env_float('ATTENDANCE_FACE_PERSON_FALLBACK_INTERVAL', 0.70)
    FACE_PERSON_FALLBACK_MIN_SCORE = _env_float('ATTENDANCE_FACE_PERSON_FALLBACK_MIN_SCORE', 0.50)
    FACE_PERSON_FALLBACK_MIN_SIZE = _env_float('ATTENDANCE_FACE_PERSON_FALLBACK_MIN_SIZE', 12.0)
    FACE_PERSON_FALLBACK_MAX_WIDTH = _env_int('ATTENDANCE_FACE_PERSON_FALLBACK_MAX_WIDTH', 960)
    FACE_PERSON_FALLBACK_SKIP_UNTIL_READY = _env_bool('ATTENDANCE_FACE_PERSON_FALLBACK_SKIP_UNTIL_READY', True)
    FACE_PERSON_FALLBACK_BOX_SCALE_X = _env_float('ATTENDANCE_FACE_PERSON_FALLBACK_BOX_SCALE_X', 3.0)
    FACE_PERSON_FALLBACK_BOX_TOP_SCALE = _env_float('ATTENDANCE_FACE_PERSON_FALLBACK_BOX_TOP_SCALE', 0.45)
    FACE_PERSON_FALLBACK_BOX_BOTTOM_SCALE = _env_float('ATTENDANCE_FACE_PERSON_FALLBACK_BOX_BOTTOM_SCALE', 2.35)
    JPEG_QUALITY = _env_int('ATTENDANCE_STREAM_JPEG_QUALITY', 72)
    STREAM_PREVIEW_MAX_WIDTH = _env_int('ATTENDANCE_STREAM_MAX_WIDTH', 1280)
    DEPTH_SAMPLE_RADIUS = 7
    DEPTH_SMOOTHING_ALPHA = 0.38
    DEPTH_SMOOTHING_FAST_ALPHA = 0.64
    DEPTH_FAST_JUMP_CM = 45.0
    DEPTH_MIN_VALID_CM = 35.0
    DEPTH_MAX_VALID_CM = 450.0
    DEPTH_DISTANCE_SCALE = _env_float('KINECT_DEPTH_DISTANCE_SCALE', 1.0)
    DEPTH_DISTANCE_BIAS_CM = _env_float('KINECT_DEPTH_DISTANCE_BIAS_CM', 0.0)
    POSE_DEPTH_FUSION_ENABLED = _env_bool('KINECT_POSE_DEPTH_FUSION_ENABLED', True)
    POSE_DEPTH_FUSION_MIN_CONFIDENCE = _env_float('KINECT_POSE_DEPTH_FUSION_MIN_CONFIDENCE', 0.12)
    POSE_DEPTH_FUSION_OUTLIER_CM = _env_float('KINECT_POSE_DEPTH_FUSION_OUTLIER_CM', 72.0)
    POSE_DEPTH_FUSION_OUTLIER_CONF_MAX = _env_float('KINECT_POSE_DEPTH_FUSION_OUTLIER_CONF_MAX', 0.55)
    POSE_DEPTH_FUSION_BASE_ALPHA = _env_float('KINECT_POSE_DEPTH_FUSION_BASE_ALPHA', 0.62)
    POSE_DEPTH_FUSION_FAST_ALPHA = _env_float('KINECT_POSE_DEPTH_FUSION_FAST_ALPHA', 0.42)
    POSE_DEPTH_FUSION_OUTLIER_ALPHA = _env_float('KINECT_POSE_DEPTH_FUSION_OUTLIER_ALPHA', 0.24)
    POSE_DEPTH_FUSION_DEPTH_JUMP_CM = _env_float('KINECT_POSE_DEPTH_FUSION_DEPTH_JUMP_CM', 18.0)
    POSE_DEPTH_FUSION_DEPTH_JUMP_ALPHA = _env_float('KINECT_POSE_DEPTH_FUSION_DEPTH_JUMP_ALPHA', 0.35)
    POSE_DEPTH_FUSION_DISP_NORM_FAST = _env_float('KINECT_POSE_DEPTH_FUSION_DISP_NORM_FAST', 0.16)
    POSE_DEPTH_FUSION_TRACK_MAX_AGE_SECONDS = _env_float('KINECT_POSE_DEPTH_FUSION_TRACK_MAX_AGE_SECONDS', 1.8)
    POSE_DEPTH_FUSION_HOLD_SECONDS = _env_float('KINECT_POSE_DEPTH_FUSION_HOLD_SECONDS', 0.22)
    POSE_DEPTH_FUSION_HOLD_MIN_CONF = _env_float('KINECT_POSE_DEPTH_FUSION_HOLD_MIN_CONF', 0.18)
    POSE_DEPTH_FUSION_HOLD_DECAY = _env_float('KINECT_POSE_DEPTH_FUSION_HOLD_DECAY', 0.85)
    POSE_DEPTH_FUSION_HOLD_FLOOR = _env_float('KINECT_POSE_DEPTH_FUSION_HOLD_FLOOR', 0.10)
    POSE_DEPTH_FUSION_TRACK_MATCH_IOU = _env_float('KINECT_POSE_DEPTH_FUSION_TRACK_MATCH_IOU', 0.10)
    POSE_DEPTH_FUSION_TRACK_MATCH_CENTER_RATIO = _env_float('KINECT_POSE_DEPTH_FUSION_TRACK_MATCH_CENTER_RATIO', 0.55)
    KINECT_RAW_MIN = 300.0
    KINECT_RAW_MAX = 1080.0
    KINECT_RAW_COEFF_A = -0.0030711016
    KINECT_RAW_COEFF_B = 3.3309495161
    POSE_SKELETON_EDGES = (
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),
        (5, 11),
        (6, 12),
        (11, 12),
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
    )
    MODEL_CANDIDATES = (
        ('yolo26x-pose', 'models/yolo/yolo26x-pose.pt', True),
        ('yolo26s-pose', 'models/yolo/yolo26s-pose.pt', True),
        ('yolo26n-pose', 'models/yolo/yolo26n-pose.pt', True),
        ('yolo11n-pose', 'models/yolo/yolo11n-pose.pt', True),
        ('yolov8n-pose', 'models/yolo/yolov8n-pose.pt', True),
        ('yolo11n-pose', 'yolo11n-pose.pt', False),
        ('yolov8n-pose', 'yolov8n-pose.pt', False),
    )
    FONT_CANDIDATES = (
        Path(r'C:\Windows\Fonts\msjh.ttc'),
        Path(r'C:\Windows\Fonts\msyh.ttc'),
        Path(r'C:\Windows\Fonts\mingliu.ttc'),
    )
    RUNTIME_TUNING_FIELDS = {
        'max_inference_width': int,
        'render_interval': float,
        'summary_update_interval': float,
        'metric_update_interval': float,
        'yolo_image_size': int,
        'yolo_detect_interval': float,
        'tracked_yolo_detect_interval': float,
        'yolo_confidence': float,
        'track_iou_threshold': float,
        'temp_person_timeout': float,
        'confirmed_absent_timeout': float,
        'recognition_threshold': float,
        'auto_relink_threshold': float,
        'auto_relink_interval': float,
        'max_detections': int,
        'min_person_box_width': float,
        'min_person_box_height': float,
        'detection_duplicate_iou_threshold': float,
        'detection_duplicate_center_ratio': float,
        'detection_duplicate_area_ratio': float,
        'temporary_merge_iou_threshold': float,
        'temporary_merge_distance_ratio': float,
        'face_person_fallback_interval': float,
        'face_person_fallback_min_score': float,
        'face_person_fallback_min_size': float,
        'face_person_fallback_max_width': int,
        'face_person_fallback_box_scale_x': float,
        'face_person_fallback_box_top_scale': float,
        'face_person_fallback_box_bottom_scale': float,
    }

    def __init__(self, base_dir, kinect_service, face_db):
        self.base_dir = Path(base_dir)
        self.kinect_service = kinect_service
        self.face_db = face_db
        self._presence_file = self.base_dir / 'data' / 'presence_records.json'
        self._runtime_tuning_file = self.base_dir / 'data' / 'runtime_tuning_profile.json'
        self._cv_modules = None
        self._pil_modules = None
        self._yolo_model = None
        self._yolo_error = None
        self._yolo_device = self._resolve_yolo_device()
        self._lock = threading.RLock()
        self._model_lock = threading.Lock()
        self._wake_event = threading.Event()
        self._font_cache = {}
        self._metric_engine = PoseDepthMetricEngine()
        self._running = True
        self._attendance_mode = False
        self._annotated_jpeg = None
        self._annotated_depth_jpeg = None
        self._temporary_people = {}
        self._confirmed_people = self._load_presence_records()
        self._confirmed_people = self._normalize_confirmed_people(self._confirmed_people)
        self._save_presence_records_locked()
        self._session_confirmed_ids = set()
        self._next_temp_number = 1
        self._next_tracking_id = 1
        self._last_person_boxes = []
        self._last_pose_detections = []
        self._last_face_person_boxes = []
        self._last_face_person_fallback_at = 0.0
        self._last_person_detect_at = 0.0
        self._last_render_at = 0.0
        self._last_summary_update_at = 0.0
        self._last_metrics_update_at = 0.0
        self._render_seq = 0
        self._last_idle_status_at = 0.0
        self._distance_smooth_cache = {}
        self._pose_fusion_tracks = {}
        self._pose_fusion_next_track_id = 1
        self._last_source_frame_seq = -1
        self._last_source_frame_timestamp = 0.0
        self._announcements = []
        self._current_course = {
            'course_id': '',
            'course_name': '',
        }
        self._status = {
            'status': 'idle',
            'message': '課堂模式尚未啟動。',
            'attendance_mode': False,
            'temporary_people': [],
            'confirmed_people': [],
            'students': [],
            'temporary_count': 0,
            'confirmed_present_count': 0,
            'confirmed_total_count': len(self._confirmed_people),
            'recognized_count': 0,
            'yolo_person_present': False,
            'detector_model': 'idle',
            'announcement': '尚未開始課堂。',
        }
        self._apply_runtime_tuning_profile()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self._warmup_thread = None
        self._warmup_started = False
        if self.WARMUP_ON_STARTUP:
            self._schedule_startup_warmup()

    def _get_cv_modules(self):
        if self._cv_modules is not None:
            return self._cv_modules

        import cv2  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel

        self._cv_modules = (cv2, np)
        return self._cv_modules

    def _get_pil_modules(self):
        if self._pil_modules is not None:
            return self._pil_modules

        from PIL import Image, ImageDraw, ImageFont  # pylint: disable=import-outside-toplevel

        self._pil_modules = (Image, ImageDraw, ImageFont)
        return self._pil_modules

    def _resolve_yolo_device(self):
        raw_device = str(os.getenv('YOLO_DEVICE', 'auto')).strip()
        if raw_device and raw_device.lower() != 'auto':
            return raw_device

        try:
            import torch  # pylint: disable=import-outside-toplevel
            if torch.cuda.is_available():
                return 0
        except Exception:
            pass
        return 'cpu'

    def _yolo_device_label(self):
        if self._yolo_device == 0:
            return 'cuda:0'
        return str(self._yolo_device)

    def _get_label_font(self, size):
        cached_font = self._font_cache.get(size)
        if cached_font is not None:
            return cached_font

        _, _, image_font = self._get_pil_modules()
        for candidate_path in self.FONT_CANDIDATES:
            if candidate_path.exists():
                font = image_font.truetype(str(candidate_path), size=size)
                self._font_cache[size] = font
                return font

        font = image_font.load_default()
        self._font_cache[size] = font
        return font

    def _apply_runtime_tuning_profile(self):
        if not self._runtime_tuning_file.exists():
            return

        try:
            with self._runtime_tuning_file.open('r', encoding='utf-8') as profile_file:
                payload = json.load(profile_file)
        except Exception:
            return

        params = payload.get('best_params') or payload.get('params') or payload
        if not isinstance(params, dict):
            return

        field_to_attr = {
            'max_inference_width': 'MAX_INFERENCE_WIDTH',
            'render_interval': 'RENDER_INTERVAL',
            'summary_update_interval': 'SUMMARY_UPDATE_INTERVAL',
            'metric_update_interval': 'METRIC_UPDATE_INTERVAL',
            'yolo_image_size': 'YOLO_IMAGE_SIZE',
            'yolo_detect_interval': 'YOLO_DETECT_INTERVAL',
            'tracked_yolo_detect_interval': 'TRACKED_YOLO_DETECT_INTERVAL',
            'yolo_confidence': 'YOLO_CONFIDENCE',
            'track_iou_threshold': 'TRACK_IOU_THRESHOLD',
            'temp_person_timeout': 'TEMP_PERSON_TIMEOUT',
            'confirmed_absent_timeout': 'CONFIRMED_ABSENT_TIMEOUT',
            'recognition_threshold': 'RECOGNITION_THRESHOLD',
            'auto_relink_threshold': 'AUTO_RELINK_THRESHOLD',
            'auto_relink_interval': 'AUTO_RELINK_INTERVAL',
            'max_detections': 'MAX_DETECTIONS',
            'min_person_box_width': 'MIN_PERSON_BOX_WIDTH',
            'min_person_box_height': 'MIN_PERSON_BOX_HEIGHT',
            'detection_duplicate_iou_threshold': 'DETECTION_DUPLICATE_IOU_THRESHOLD',
            'detection_duplicate_center_ratio': 'DETECTION_DUPLICATE_CENTER_RATIO',
            'detection_duplicate_area_ratio': 'DETECTION_DUPLICATE_AREA_RATIO',
            'temporary_merge_iou_threshold': 'TEMPORARY_MERGE_IOU_THRESHOLD',
            'temporary_merge_distance_ratio': 'TEMPORARY_MERGE_DISTANCE_RATIO',
            'face_person_fallback_interval': 'FACE_PERSON_FALLBACK_INTERVAL',
            'face_person_fallback_min_score': 'FACE_PERSON_FALLBACK_MIN_SCORE',
            'face_person_fallback_min_size': 'FACE_PERSON_FALLBACK_MIN_SIZE',
            'face_person_fallback_max_width': 'FACE_PERSON_FALLBACK_MAX_WIDTH',
            'face_person_fallback_box_scale_x': 'FACE_PERSON_FALLBACK_BOX_SCALE_X',
            'face_person_fallback_box_top_scale': 'FACE_PERSON_FALLBACK_BOX_TOP_SCALE',
            'face_person_fallback_box_bottom_scale': 'FACE_PERSON_FALLBACK_BOX_BOTTOM_SCALE',
        }

        for field_name, cast in self.RUNTIME_TUNING_FIELDS.items():
            attr_name = field_to_attr[field_name]
            raw_value = params.get(field_name)
            if raw_value is None:
                continue
            try:
                setattr(self, attr_name, cast(raw_value))
            except Exception:
                continue

    def _warm_models(self):
        try:
            _, np = self._get_cv_modules()
            dummy_frame = np.zeros((self.YOLO_IMAGE_SIZE, self.YOLO_IMAGE_SIZE, 3), dtype=np.uint8)
            self._run_yolo_person_inference(
                dummy_frame,
                conf=self.YOLO_WARMUP_CONFIDENCE,
                max_det=self.YOLO_WARMUP_MAX_DETECTIONS,
            )
            # Preload face analysis model to avoid first-confirm latency spike.
            face_dummy = np.zeros((224, 224, 3), dtype=np.uint8)
            self.face_db.analyze_faces(face_dummy)
        except Exception:
            return

    def _schedule_startup_warmup(self):
        def _delayed_warmup():
            delay = max(0.0, float(self.WARMUP_STARTUP_DELAY))
            if delay > 0:
                time.sleep(delay)
            if self._running:
                self._ensure_warmup_started()

        threading.Thread(target=_delayed_warmup, daemon=True).start()

    def _ensure_warmup_started(self):
        with self._lock:
            if self._warmup_started:
                return
            self._warmup_started = True
            self._warmup_thread = threading.Thread(target=self._warm_models, daemon=True)
            self._warmup_thread.start()

    def _refresh_idle_status(self, now):
        if now < (self._last_idle_status_at + self.IDLE_STATUS_REFRESH_INTERVAL):
            return
        self._last_idle_status_at = now
        kinect_status = self.kinect_service.get_status()
        source_status = str(kinect_status.get('status') or 'idle').strip().lower()
        normalized_status = 'ready' if source_status == 'connected' else source_status
        message = str(kinect_status.get('message') or '').strip() or '待機中'
        with self._lock:
            self._status.update(
                {
                    'status': normalized_status,
                    'message': message,
                    'attendance_mode': False,
                    'temporary_count': 0,
                    'confirmed_present_count': 0,
                    'recognized_count': 0,
                    'yolo_person_present': False,
                    'detector_model': 'idle',
                }
            )

    def _run_yolo_person_inference(self, frame, conf, max_det):
        model = self._get_yolo_model()
        with self._model_lock:
            return model(
                frame,
                verbose=False,
                classes=[self.PERSON_CLASS_ID],
                imgsz=self.YOLO_IMAGE_SIZE,
                conf=conf,
                max_det=max_det,
                device=self._yolo_device,
            )[0]

    def _load_presence_records(self):
        if not self._presence_file.exists():
            return {}

        try:
            with self._presence_file.open('r', encoding='utf-8') as record_file:
                payload = json.load(record_file)
        except Exception:
            return {}

        now = time.time()
        confirmed_people = {}
        for item in payload.get('confirmed_people', []):
            segments = []
            for segment in item.get('presence_segments', []):
                start = float(segment.get('start', now))
                end = segment.get('end')
                if end is not None:
                    end = float(end)
                else:
                    end = float(item.get('last_seen', now))
                segments.append({'start': start, 'end': end})

            user_id = item.get('user_id')
            if not user_id:
                continue

            confirmed_people[user_id] = ConfirmedPerson(
                user_id=user_id,
                label=item.get('label', user_id),
                name=item.get('name', item.get('display_name', user_id)),
                student_id=item.get('student_id', ''),
                department=item.get('department', ''),
                title=item.get('title', ''),
                first_confirmed_at=float(item.get('first_confirmed_at', now)),
                first_seen=float(item.get('first_seen', item.get('first_confirmed_at', now))),
                last_seen=float(item.get('last_seen', item.get('first_confirmed_at', now))),
                current_status='absent',
                current_tracking_id=None,
                bbox=None,
                presence_segments=segments,
                appearance_count=int(item.get('appearance_count', len(segments))),
                last_similarity=float(item.get('last_similarity', 0.0)),
            )
        return confirmed_people

    def _save_presence_records_locked(self):
        self._presence_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'confirmed_people': [
                {
                    'user_id': person.user_id,
                    'label': person.label,
                    'name': person.name,
                    'student_id': person.student_id,
                    'department': person.department,
                    'title': person.title,
                    'first_confirmed_at': person.first_confirmed_at,
                    'first_seen': person.first_seen,
                    'last_seen': person.last_seen,
                    'current_status': person.current_status,
                    'appearance_count': person.appearance_count,
                    'last_similarity': person.last_similarity,
                    'presence_segments': person.presence_segments,
                }
                for person in sorted(self._confirmed_people.values(), key=lambda item: item.name.lower())
            ]
        }
        with self._presence_file.open('w', encoding='utf-8') as record_file:
            json.dump(payload, record_file, ensure_ascii=False, indent=2)

    def _identity_tokens_for_person(self, user_id, person):
        tokens = set()
        for value in (
            user_id,
            getattr(person, 'label', ''),
            getattr(person, 'student_id', ''),
            getattr(person, 'name', ''),
        ):
            normalized = str(value or '').strip().lower()
            if normalized:
                tokens.add(normalized)
        return tokens

    def _merge_presence_segments(self, people):
        segments = []
        for person in people:
            for segment in person.presence_segments:
                start = float(segment.get('start', 0.0))
                end = segment.get('end')
                segments.append(
                    {
                        'start': start,
                        'end': float(end) if end is not None else None,
                    }
                )

        if not segments:
            return []

        segments.sort(key=lambda item: item['start'])
        merged = [segments[0]]
        for segment in segments[1:]:
            current = merged[-1]
            current_end = float('inf') if current['end'] is None else current['end']
            segment_end = float('inf') if segment['end'] is None else segment['end']

            if segment['start'] <= current_end + 1.0:
                current['end'] = None if current['end'] is None or segment['end'] is None else max(current['end'], segment['end'])
            else:
                merged.append(segment)

        return merged

    def _normalize_confirmed_people(self, confirmed_people):
        groups = []

        for user_id, person in confirmed_people.items():
            person_keys = self._identity_tokens_for_person(user_id, person)
            matched_groups = [group for group in groups if group['keys'] & person_keys]

            if not matched_groups:
                groups.append({'keys': set(person_keys), 'items': [(user_id, person)]})
                continue

            primary_group = matched_groups[0]
            primary_group['keys'].update(person_keys)
            primary_group['items'].append((user_id, person))

            for duplicate_group in matched_groups[1:]:
                primary_group['keys'].update(duplicate_group['keys'])
                primary_group['items'].extend(duplicate_group['items'])
                groups.remove(duplicate_group)

        normalized = {}
        for group in groups:
            items = group['items']
            people = [person for _, person in items]
            preferred_user_id, preferred_person = max(
                items,
                key=lambda entry: (
                    1 if entry[1].student_id else 0,
                    1 if entry[1].current_status == 'present' else 0,
                    float(entry[1].last_seen),
                ),
            )

            canonical_user_id = preferred_person.student_id or preferred_user_id or preferred_person.label
            merged_segments = self._merge_presence_segments(people)
            current_present_person = next((person for person in people if person.current_status == 'present'), None)

            canonical_person = ConfirmedPerson(
                user_id=canonical_user_id,
                label=preferred_person.label or canonical_user_id,
                name=preferred_person.name or preferred_person.label or canonical_user_id,
                student_id=preferred_person.student_id,
                department=preferred_person.department,
                title=preferred_person.title,
                first_confirmed_at=min(float(person.first_confirmed_at) for person in people),
                first_seen=min(float(person.first_seen) for person in people),
                last_seen=max(float(person.last_seen) for person in people),
                current_status='present' if current_present_person is not None else 'absent',
                current_tracking_id=current_present_person.current_tracking_id if current_present_person is not None else None,
                bbox=current_present_person.bbox if current_present_person is not None else None,
                presence_segments=merged_segments,
                appearance_count=max(sum(int(person.appearance_count) for person in people), len(merged_segments)),
                last_similarity=max(float(person.last_similarity) for person in people),
            )
            normalized[canonical_user_id] = canonical_person

        return normalized

    def _push_announcement_locked(self, message):
        timestamp = time.time()
        self._announcements.append({'message': message, 'timestamp': timestamp})
        self._announcements = self._announcements[-8:]
        self._status['announcement'] = message

    def _get_yolo_model(self):
        if self._yolo_model is not None:
            return self._yolo_model
        if self._yolo_error is not None:
            raise RuntimeError(self._yolo_error)

        with self._model_lock:
            if self._yolo_model is not None:
                return self._yolo_model
            if self._yolo_error is not None:
                raise RuntimeError(self._yolo_error)

            try:
                from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

                selected_model_ref = None
                selected_model_name = ''
                model_candidates = list(self.MODEL_CANDIDATES)
                model_override = str(os.getenv('YOLO_POSE_MODEL', '')).strip()
                if model_override:
                    override_path = Path(model_override)
                    if not override_path.is_absolute():
                        override_path = self.base_dir / override_path

                    if override_path.exists():
                        model_candidates.insert(0, (override_path.stem, str(override_path), True))
                    else:
                        override_name = Path(model_override).stem or model_override
                        model_candidates.insert(0, (override_name, model_override, False))

                for candidate_name, model_ref, is_local in model_candidates:
                    if is_local:
                        candidate_path = self.base_dir / model_ref
                        if Path(model_ref).is_absolute():
                            candidate_path = Path(model_ref)
                        if not candidate_path.exists():
                            continue
                        selected_model_ref = str(candidate_path)
                        selected_model_name = candidate_name
                        break

                    # Remote model aliases (auto-download by ultralytics).
                    selected_model_ref = str(model_ref)
                    selected_model_name = candidate_name
                    break

                if selected_model_ref is None:
                    raise FileNotFoundError('No YOLO pose model was found.')

                self._yolo_model = YOLO(selected_model_ref)
                with self._lock:
                    self._status['detector_model'] = f'{selected_model_name} ({self._yolo_device_label()})'
            except Exception as exc:  # pylint: disable=broad-except
                self._yolo_error = f'YOLO is not available: {exc}'
                raise RuntimeError(self._yolo_error) from exc
        return self._yolo_model

    def set_attendance_mode(self, enabled):
        now = time.time()
        self.kinect_service.set_processing_mode('attendance' if enabled else 'idle')
        with self._lock:
            self._attendance_mode = enabled
            if not enabled:
                self._session_confirmed_ids.clear()
                self._temporary_people = {}
                self._next_temp_number = 1
                self._last_person_boxes = []
                self._last_pose_detections = []
                self._last_face_person_boxes = []
                self._last_face_person_fallback_at = 0.0
                self._last_person_detect_at = 0.0
                self._last_render_at = 0.0
                self._render_seq = 0
                self._last_source_frame_seq = -1
                self._last_source_frame_timestamp = 0.0
                self._annotated_jpeg = None
                self._annotated_depth_jpeg = None
                self._distance_smooth_cache = {}
                self._last_summary_update_at = 0.0
                self._last_metrics_update_at = 0.0
                self._cleanup_pose_fusion_tracks(now, reset=True)
                self._metric_engine.reset()
                for person in self._confirmed_people.values():
                    if person.current_status == 'present':
                        self._set_confirmed_absent_locked(person, now)
                self._save_presence_records_locked()
                self._push_announcement_locked('課堂模式已停止。')
                self._status.update(
                    {
                        'status': 'idle',
                        'message': '課堂模式尚未啟動。',
                        'attendance_mode': False,
                        'temporary_people': [],
                        'confirmed_people': self._build_confirmed_payload_locked(
                            now,
                            include_presence_points=True,
                            include_metrics=False,
                        ),
                        'students': self._build_confirmed_payload_locked(
                            now,
                            include_presence_points=True,
                            include_metrics=False,
                        ),
                        'temporary_count': 0,
                        'confirmed_present_count': 0,
                        'confirmed_total_count': len(self._confirmed_people),
                        'recognized_count': 0,
                        'yolo_person_present': False,
                        'detector_model': 'idle',
                    }
                )
            else:
                self._session_confirmed_ids.clear()
                self._temporary_people = {}
                self._next_temp_number = 1
                self._last_person_boxes = []
                self._last_pose_detections = []
                self._last_face_person_boxes = []
                self._last_face_person_fallback_at = 0.0
                self._last_person_detect_at = 0.0
                self._last_render_at = 0.0
                self._render_seq = 0
                self._last_source_frame_seq = -1
                self._last_source_frame_timestamp = 0.0
                self._annotated_jpeg = None
                self._annotated_depth_jpeg = None
                self._distance_smooth_cache = {}
                self._last_summary_update_at = 0.0
                self._last_metrics_update_at = 0.0
                self._cleanup_pose_fusion_tracks(now, reset=True)
                self._metric_engine.reset()
                # Start a fresh session timeline on each attendance start.
                for person in self._confirmed_people.values():
                    person.current_status = 'absent'
                    person.current_tracking_id = None
                    person.bbox = None
                    person.presence_segments = []
                    person.appearance_count = 0
                    person.first_seen = now
                    person.last_seen = now
                self._save_presence_records_locked()
                self._push_announcement_locked('課堂模式已啟動。')
                self._status.update(
                    {
                        'status': 'attendance_running',
                        'message': '課堂模式啟動，開始進行課堂追蹤。',
                        'attendance_mode': True,
                    }
                )
        self._wake_event.set()
        if enabled:
            self._ensure_warmup_started()

    def get_attendance_mode(self):
        with self._lock:
            return self._attendance_mode

    def set_current_course(self, course_id, course_name):
        normalized_name = str(course_name or '').strip()
        normalized_id = str(course_id or '').strip()
        if not normalized_name:
            normalized_name = normalized_id
        if not normalized_id:
            normalized_id = normalized_name
        with self._lock:
            self._current_course = {
                'course_id': normalized_id,
                'course_name': normalized_name,
            }

    def get_current_course(self):
        with self._lock:
            return dict(self._current_course)

    def get_latest_color_jpeg(self):
        with self._lock:
            if not self._attendance_mode and self._annotated_jpeg is None:
                return self.kinect_service.get_latest_jpeg('color')
            return self._annotated_jpeg or self.kinect_service.get_latest_jpeg('color')

    def get_latest_depth_jpeg(self):
        with self._lock:
            if not self._attendance_mode and self._annotated_depth_jpeg is None:
                return self.kinect_service.get_latest_jpeg('depth')
            return self._annotated_depth_jpeg or self.kinect_service.get_latest_jpeg('depth')

    def get_status(self, include_metrics=True, metrics_user_id=None):
        with self._lock:
            temporary_people = list(self._status.get('temporary_people') or [])
            confirmed_people = list(self._status.get('confirmed_people') or [])
            if include_metrics:
                requested_user_id = str(metrics_user_id or '').strip()
                include_all_metrics = requested_user_id == ''
                confirmed_with_metrics = []
                for item in confirmed_people:
                    copied_item = dict(item)
                    copied_item['classroom_metrics'] = {}
                    if include_all_metrics or str(copied_item.get('user_id') or '') == requested_user_id or str(copied_item.get('student_id') or '') == requested_user_id:
                        metric_key = str(copied_item.get('user_id') or copied_item.get('student_id') or '').strip()
                        if metric_key:
                            copied_item['classroom_metrics'] = self._metric_engine.get_user_metrics(metric_key)
                    confirmed_with_metrics.append(copied_item)
                confirmed_people = confirmed_with_metrics
            announcement = self._announcements[-1]['message'] if self._announcements else self._status['announcement']
            return {
                'status': self._status['status'],
                'message': self._status['message'],
                'attendance_mode': self._attendance_mode,
                'temporary_people': temporary_people,
                'confirmed_people': confirmed_people,
                'students': confirmed_people,
                'temporary_count': int(self._status.get('temporary_count') or len(temporary_people)),
                'confirmed_present_count': int(self._status.get('confirmed_present_count') or 0),
                'confirmed_total_count': int(self._status.get('confirmed_total_count') or len(confirmed_people)),
                'recognized_count': int(self._status.get('recognized_count') or 0),
                'yolo_person_present': self._status['yolo_person_present'],
                'detector_model': self._status['detector_model'],
                'announcement': announcement,
                'current_course': dict(self._current_course),
            }

    def confirm_temporary_person(self, temp_id):
        with self._lock:
            if not self._attendance_mode:
                return {'status': 'error', 'message': '請先開始課堂。'}

            temp_person = self._temporary_people.get(temp_id)
            if temp_person is None:
                return {'status': 'error', 'message': '找不到這位暫時學生，可能已離開畫面。'}

            bbox = list(temp_person.bbox)
            cached_embedding = None
            cached_bbox = list(temp_person.face_bbox) if temp_person.face_bbox else list(bbox)
            if temp_person.face_embedding and (time.time() - float(temp_person.face_embedding_at)) <= self.FACE_EMBEDDING_CACHE_SECONDS:
                cached_embedding = list(temp_person.face_embedding)
            temp_person.confirm_status = 'processing'
            temp_person.confirm_message = '辨識中...'

        embedding = cached_embedding
        if embedding is None:
            frame = self.kinect_service.get_latest_color_frame()
            if frame is None:
                with self._lock:
                    temp_person = self._temporary_people.get(temp_id)
                    if temp_person is not None:
                        temp_person.confirm_status = 'failed'
                        temp_person.confirm_message = '目前沒有可用影像。'
                return {'status': 'error', 'message': '目前沒有可用影像。'}

            analysis = self._analyze_face_inside_bbox_fast(frame, bbox)
            if analysis['status'] != 'ok':
                with self._lock:
                    temp_person = self._temporary_people.get(temp_id)
                    if temp_person is not None:
                        temp_person.confirm_status = 'failed'
                        temp_person.confirm_message = analysis['message']
                return {'status': 'error', 'message': analysis['message']}

            embedding = analysis['embedding']
            cached_bbox = list(analysis['bbox'])

        matches = self.face_db.match_embedding(
            embedding,
            threshold=self.RECOGNITION_THRESHOLD,
        )
        if not matches:
            with self._lock:
                temp_person = self._temporary_people.get(temp_id)
                if temp_person is not None:
                    temp_person.confirm_status = 'failed'
                    temp_person.confirm_message = '找不到對應身份。'
            return {'status': 'error', 'message': '找不到對應身份。'}

        now = time.time()
        with self._lock:
            temp_person = self._temporary_people.get(temp_id)
            if temp_person is None:
                return {'status': 'error', 'message': '這位暫時學生已離開畫面。'}

            temp_person.face_bbox = list(cached_bbox)
            temp_person.face_embedding = list(embedding)
            temp_person.face_embedding_at = now
            confirmed_person = self._promote_temporary_to_confirmed_locked(temp_id, matches[0], now)
            if confirmed_person is None:
                return {'status': 'error', 'message': '辨識對象已不存在。'}

            self._session_confirmed_ids.add(confirmed_person.user_id)
            self._push_announcement_locked(f'{confirmed_person.name} 已完成身份確認。')
            self._update_summary_locked(now, force=True)

        return {
            'status': 'confirmed',
            'message': f'{confirmed_person.name} 已完成身份確認。',
            'person': {
                'user_id': confirmed_person.user_id,
                'display_name': confirmed_person.name,
                'student_id': confirmed_person.student_id,
            },
        }

    def capture_temporary_person_frames(self, temp_id, count=3, delay=3.0):
        frames = []

        for capture_index in range(count):
            if delay > 0:
                time.sleep(delay)

            with self._lock:
                temp_person = self._temporary_people.get(temp_id)
                if temp_person is None or not temp_person.bbox:
                    raise RuntimeError('Temporary student is no longer available.')
                bbox = list(temp_person.bbox)

            frame = self.kinect_service.get_latest_color_frame()
            if frame is None:
                raise RuntimeError('Unable to capture a Kinect frame.')

            frames.append(self._crop_face_for_training(frame, bbox))

        return frames

    def begin_confirm_temporary_person(self, temp_id):
        with self._lock:
            if not self._attendance_mode:
                return {'status': 'error', 'message': '課堂模式尚未啟動。'}

            temp_person = self._temporary_people.get(temp_id)
            if temp_person is None:
                return {'status': 'error', 'message': 'Temporary student not found.'}

            if temp_person.confirm_status == 'processing':
                return {'status': 'queued', 'message': 'Identity check already in progress.', 'temp_id': temp_id}

            temp_person.confirm_status = 'processing'
            temp_person.confirm_message = '辨識中...'
            temp_person.last_seen = time.time()
            self._update_summary_locked(time.time(), force=True)

        threading.Thread(
            target=self._confirm_temporary_person_worker,
            args=(temp_id,),
            daemon=True,
        ).start()
        return {'status': 'queued', 'message': 'Identity check started.', 'temp_id': temp_id}

    def _confirm_temporary_person_worker(self, temp_id):
        with self._lock:
            temp_person = self._temporary_people.get(temp_id)
            if temp_person is None or not temp_person.bbox:
                return
            bbox = list(temp_person.bbox)
            cached_embedding = None
            cached_bbox = list(temp_person.face_bbox) if temp_person.face_bbox else list(bbox)
            if temp_person.face_embedding and (time.time() - float(temp_person.face_embedding_at)) <= self.FACE_EMBEDDING_CACHE_SECONDS:
                cached_embedding = list(temp_person.face_embedding)

        embedding = cached_embedding
        if embedding is None:
            frame = self.kinect_service.get_latest_color_frame()
            if frame is None:
                with self._lock:
                    temp_person = self._temporary_people.get(temp_id)
                    if temp_person is not None:
                        temp_person.confirm_status = 'failed'
                        temp_person.confirm_message = '目前無法取得 Kinect 畫面。'
                        self._update_summary_locked(time.time(), force=True)
                return

            analysis = self._analyze_face_inside_bbox_fast(frame, bbox)
            if analysis['status'] != 'ok':
                with self._lock:
                    temp_person = self._temporary_people.get(temp_id)
                    if temp_person is not None:
                        temp_person.confirm_status = 'failed'
                        temp_person.confirm_message = analysis['message']
                        self._update_summary_locked(time.time(), force=True)
                return

            embedding = analysis['embedding']
            cached_bbox = list(analysis['bbox'])

        matches = self.face_db.match_embedding(
            embedding,
            threshold=self.RECOGNITION_THRESHOLD,
        )
        if not matches:
            with self._lock:
                temp_person = self._temporary_people.get(temp_id)
                if temp_person is not None:
                    temp_person.confirm_status = 'failed'
                    temp_person.confirm_message = '找不到符合的人臉資料。'
                    self._update_summary_locked(time.time(), force=True)
            return

        now = time.time()
        with self._lock:
            temp_person = self._temporary_people.get(temp_id)
            if temp_person is None:
                return

            temp_person.face_bbox = list(cached_bbox)
            temp_person.face_embedding = list(embedding)
            temp_person.face_embedding_at = now
            confirmed_person = self._promote_temporary_to_confirmed_locked(temp_id, matches[0], now)
            if confirmed_person is None:
                return

            self._session_confirmed_ids.add(confirmed_person.user_id)
            self._push_announcement_locked(f'{confirmed_person.name} 已完成身份確認。')
            self._update_summary_locked(now, force=True)

    def capture_temporary_person_frame(self, temp_id):
        return self.capture_temporary_person_frames(temp_id, count=1, delay=0.0)[0]

    def register_temporary_person(self, temp_id, profile):
        now = time.time()
        synthetic_match = {
            'label': profile['label'],
            'display_name': profile.get('name', profile['label']),
            'student_id': profile.get('student_id', ''),
            'department': profile.get('department', ''),
            'title': profile.get('title', ''),
            'similarity': 1.0,
        }

        with self._lock:
            confirmed_person = self._promote_temporary_to_confirmed_locked(temp_id, synthetic_match, now)
            if confirmed_person is None:
                raise RuntimeError('Unable to register this temporary student.')
            self._session_confirmed_ids.add(confirmed_person.user_id)
            self._push_announcement_locked(f'{confirmed_person.name} 已加入班級。')
            self._update_summary_locked(now, force=True)

        return {
            'status': 'confirmed',
            'message': f'{confirmed_person.name} 已完成加選並建立身份。',
            'person': {
                'user_id': confirmed_person.user_id,
                'display_name': confirmed_person.name,
                'student_id': confirmed_person.student_id,
                'label': confirmed_person.label,
            },
        }

    def mjpeg_stream(self):
        last_token = None
        while True:
            payload, token = self._get_stream_payload_and_token('color')
            if payload is not None and token != last_token:
                last_token = token
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + payload + b'\r\n'
            time.sleep(0.01)

    def depth_mjpeg_stream(self):
        last_token = None
        while True:
            payload, token = self._get_stream_payload_and_token('depth')
            if payload is not None and token != last_token:
                last_token = token
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + payload + b'\r\n'
            time.sleep(0.01)

    def _get_stream_payload_and_token(self, kind):
        with self._lock:
            in_attendance_mode = bool(self._attendance_mode)
            render_seq = int(self._render_seq)
            color_payload = self._annotated_jpeg
            depth_payload = self._annotated_depth_jpeg
            has_annotated_pair = in_attendance_mode and color_payload is not None and depth_payload is not None
            if has_annotated_pair:
                payload = color_payload if kind == 'color' else depth_payload
                return payload, ('attendance', render_seq)

        fallback_payload = self.kinect_service.get_latest_jpeg(kind)
        marker = self.kinect_service.get_latest_frame_marker()
        fallback_seq = int(marker.get('frame_seq') or 0)
        return fallback_payload, ('raw', fallback_seq)

    def _encode_frame(self, frame):
        cv2, _ = self._get_cv_modules()
        prepared_frame = frame
        max_width = max(0, int(self.STREAM_PREVIEW_MAX_WIDTH))
        if max_width and frame is not None and getattr(frame, 'shape', None) is not None:
            frame_width = int(frame.shape[1]) if len(frame.shape) >= 2 else 0
            if frame_width > max_width:
                scale = max_width / float(frame_width)
                target_height = max(1, int(round(frame.shape[0] * scale)))
                prepared_frame = cv2.resize(frame, (max_width, target_height), interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(
            '.jpg',
            prepared_frame,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                int(self.JPEG_QUALITY),
            ],
        )
        if not ok:
            return None
        return buffer.tobytes()

    def _resize_for_inference(self, frame):
        cv2, _ = self._get_cv_modules()
        height, width = frame.shape[:2]
        if width <= self.MAX_INFERENCE_WIDTH:
            return frame, 1.0

        scale = self.MAX_INFERENCE_WIDTH / float(width)
        resized = cv2.resize(frame, (int(width * scale), int(height * scale)))
        return resized, scale

    def _bbox_iou(self, left_bbox, right_bbox):
        left = max(left_bbox[0], right_bbox[0])
        top = max(left_bbox[1], right_bbox[1])
        right = min(left_bbox[2], right_bbox[2])
        bottom = min(left_bbox[3], right_bbox[3])
        if right <= left or bottom <= top:
            return 0.0

        inter_area = (right - left) * (bottom - top)
        left_area = max(0.0, left_bbox[2] - left_bbox[0]) * max(0.0, left_bbox[3] - left_bbox[1])
        right_area = max(0.0, right_bbox[2] - right_bbox[0]) * max(0.0, right_bbox[3] - right_bbox[1])
        union_area = left_area + right_area - inter_area
        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    def _bbox_intersection_over_min_area(self, left_bbox, right_bbox):
        left = max(float(left_bbox[0]), float(right_bbox[0]))
        top = max(float(left_bbox[1]), float(right_bbox[1]))
        right = min(float(left_bbox[2]), float(right_bbox[2]))
        bottom = min(float(left_bbox[3]), float(right_bbox[3]))
        if right <= left or bottom <= top:
            return 0.0

        inter_area = (right - left) * (bottom - top)
        left_area = max(0.0, float(left_bbox[2]) - float(left_bbox[0])) * max(0.0, float(left_bbox[3]) - float(left_bbox[1]))
        right_area = max(0.0, float(right_bbox[2]) - float(right_bbox[0])) * max(0.0, float(right_bbox[3]) - float(right_bbox[1]))
        min_area = min(left_area, right_area)
        if min_area <= 0.0:
            return 0.0
        return inter_area / min_area

    def _boxes_likely_same_person(self, left_bbox, right_bbox, iou_threshold, center_ratio, area_ratio_limit, overlap_ratio):
        iou = self._bbox_iou(left_bbox, right_bbox)
        if iou >= float(iou_threshold):
            return True

        overlap = self._bbox_intersection_over_min_area(left_bbox, right_bbox)
        if overlap >= float(overlap_ratio):
            return True

        center_distance = self._bbox_center_distance(left_bbox, right_bbox)
        min_side = min(self._bbox_max_side(left_bbox), self._bbox_max_side(right_bbox))
        center_limit = max(12.0, min_side * float(center_ratio))
        area_left = max(1.0, self._current_bbox_area(left_bbox))
        area_right = max(1.0, self._current_bbox_area(right_bbox))
        area_ratio = max(area_left, area_right) / min(area_left, area_right)
        return center_distance <= center_limit and area_ratio <= float(area_ratio_limit)

    def _bbox_center(self, bbox):
        return (
            (float(bbox[0]) + float(bbox[2])) / 2.0,
            (float(bbox[1]) + float(bbox[3])) / 2.0,
        )

    def _bbox_torso_point(self, bbox):
        x1, y1, x2, y2 = [float(value) for value in bbox]
        return (
            (x1 + x2) / 2.0,
            y1 + ((y2 - y1) * 0.42),
        )

    def _bbox_center_distance(self, left_bbox, right_bbox):
        left_center = self._bbox_center(left_bbox)
        right_center = self._bbox_center(right_bbox)
        delta_x = left_center[0] - right_center[0]
        delta_y = left_center[1] - right_center[1]
        return (delta_x * delta_x + delta_y * delta_y) ** 0.5

    def _bbox_max_side(self, bbox):
        return max(
            max(0.0, float(bbox[2]) - float(bbox[0])),
            max(0.0, float(bbox[3]) - float(bbox[1])),
        )

    def _detection_candidates_from_result(self, result, factor):
        candidates = []
        keypoints_xy = None
        keypoints_conf = None
        if getattr(result, 'keypoints', None) is not None:
            keypoints_xy = getattr(result.keypoints, 'xy', None)
            keypoints_conf = getattr(result.keypoints, 'conf', None)

        for index, box in enumerate(result.boxes):
            xyxy = [float(value) * factor for value in box.xyxy[0].tolist()]
            confidence = float(box.conf[0].item()) if box.conf is not None and len(box.conf) > 0 else 0.0
            width = max(0.0, xyxy[2] - xyxy[0])
            height = max(0.0, xyxy[3] - xyxy[1])
            if width < self.MIN_PERSON_BOX_WIDTH or height < self.MIN_PERSON_BOX_HEIGHT:
                continue

            pose_points = []
            pose_confidence = []
            if keypoints_xy is not None and len(keypoints_xy) > index:
                try:
                    raw_points = keypoints_xy[index].tolist()
                    for point in raw_points:
                        pose_points.append([float(point[0]) * factor, float(point[1]) * factor])
                except Exception:
                    pose_points = []

            if keypoints_conf is not None and len(keypoints_conf) > index:
                try:
                    pose_confidence = [float(value) for value in keypoints_conf[index].tolist()]
                except Exception:
                    pose_confidence = []

            candidates.append(
                {
                    'bbox': xyxy,
                    'confidence': confidence,
                    'keypoints': pose_points,
                    'keypoint_conf': pose_confidence,
                }
            )
        return candidates

    def _person_bbox_from_face_bbox(self, face_bbox, frame_shape):
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = [float(value) for value in face_bbox]
        face_w = max(1.0, x2 - x1)
        face_h = max(1.0, y2 - y1)
        center_x = (x1 + x2) * 0.5

        box_half_w = max(self.MIN_PERSON_BOX_WIDTH * 0.5, face_w * self.FACE_PERSON_FALLBACK_BOX_SCALE_X * 0.5)
        box_top = y1 - (face_h * self.FACE_PERSON_FALLBACK_BOX_TOP_SCALE)
        box_bottom = y2 + (face_h * self.FACE_PERSON_FALLBACK_BOX_BOTTOM_SCALE)

        bbox = [
            max(0.0, center_x - box_half_w),
            max(0.0, box_top),
            min(float(width), center_x + box_half_w),
            min(float(height), box_bottom),
        ]
        if bbox[2] - bbox[0] < self.MIN_PERSON_BOX_WIDTH or bbox[3] - bbox[1] < self.MIN_PERSON_BOX_HEIGHT:
            return None
        return bbox

    def _face_person_fallback_candidates(self, frame, now, existing_candidates):
        if not self.FACE_PERSON_FALLBACK_ENABLED:
            return []

        if self.FACE_PERSON_FALLBACK_SKIP_UNTIL_READY:
            is_ready = getattr(self.face_db, 'is_analysis_ready', lambda: True)
            if not is_ready():
                return []

        interval = max(0.0, float(self.FACE_PERSON_FALLBACK_INTERVAL))
        if self._last_face_person_boxes and now - self._last_face_person_fallback_at < interval:
            return [
                {
                    'bbox': list(bbox),
                    'confidence': max(0.0, float(self.FACE_PERSON_FALLBACK_MIN_SCORE)),
                    'keypoints': [],
                    'keypoint_conf': [],
                    'source': 'face_fallback',
                }
                for bbox in self._last_face_person_boxes
            ]

        self._last_face_person_fallback_at = now
        self._last_face_person_boxes = []

        try:
            cv2, _ = self._get_cv_modules()
            analyze_frame = frame
            analysis_scale = 1.0
            max_width = max(0, int(self.FACE_PERSON_FALLBACK_MAX_WIDTH))
            if max_width and frame.shape[1] > max_width:
                analysis_scale = max_width / float(frame.shape[1])
                resized_height = max(1, int(round(frame.shape[0] * analysis_scale)))
                analyze_frame = cv2.resize(frame, (max_width, resized_height), interpolation=cv2.INTER_AREA)

            analysis = self.face_db.analyze_faces(analyze_frame)
        except Exception:
            return []
        if analysis.get('status') != 'ok':
            return []

        scale_back = (1.0 / analysis_scale) if analysis_scale > 0 else 1.0
        existing_boxes = [item['bbox'] for item in existing_candidates]
        candidates = []
        for face in analysis.get('faces') or []:
            face_bbox = face.get('bbox') or []
            if len(face_bbox) != 4:
                continue
            if scale_back != 1.0:
                face_bbox = [float(value) * scale_back for value in face_bbox]
            det_score = float(face.get('det_score', 0.0))
            if det_score < float(self.FACE_PERSON_FALLBACK_MIN_SCORE):
                continue

            face_w = max(0.0, float(face_bbox[2]) - float(face_bbox[0]))
            face_h = max(0.0, float(face_bbox[3]) - float(face_bbox[1]))
            if min(face_w, face_h) < float(self.FACE_PERSON_FALLBACK_MIN_SIZE):
                continue

            face_center = self._bbox_center(face_bbox)
            face_inside_existing = False
            for bbox in existing_boxes:
                if (
                    float(bbox[0]) <= face_center[0] <= float(bbox[2])
                    and float(bbox[1]) <= face_center[1] <= float(bbox[3])
                ):
                    face_inside_existing = True
                    break
            if face_inside_existing:
                continue

            person_bbox = self._person_bbox_from_face_bbox(face_bbox, frame.shape)
            if person_bbox is None:
                continue

            existing_boxes.append(person_bbox)
            self._last_face_person_boxes.append(list(person_bbox))
            candidates.append(
                {
                    'bbox': person_bbox,
                    'confidence': det_score,
                    'keypoints': [],
                    'keypoint_conf': [],
                    'source': 'face_fallback',
                }
            )

        return candidates

    def _coerce_pose_arrays(self, detection, min_length=17):
        keypoints = detection.get('keypoints') or []
        keypoint_conf = detection.get('keypoint_conf') or []
        target_length = max(min_length, len(keypoints), len(keypoint_conf))
        normalized_points = []
        normalized_conf = []

        for index in range(target_length):
            point = keypoints[index] if index < len(keypoints) else None
            conf = keypoint_conf[index] if index < len(keypoint_conf) else 0.0

            if point is None or len(point) < 2:
                normalized_points.append([0.0, 0.0])
            else:
                try:
                    normalized_points.append([float(point[0]), float(point[1])])
                except Exception:
                    normalized_points.append([0.0, 0.0])

            try:
                normalized_conf.append(float(conf))
            except Exception:
                normalized_conf.append(0.0)

        return normalized_points, normalized_conf

    def _pose_depth_profile(self, keypoints, keypoint_conf, depth_frame, source_mode):
        if depth_frame is None:
            return {}

        profile = {}
        for index, point in enumerate(keypoints):
            confidence = keypoint_conf[index] if index < len(keypoint_conf) else 0.0
            if float(confidence) < self.POSE_DEPTH_FUSION_MIN_CONFIDENCE:
                continue
            if point is None or len(point) < 2:
                continue
            depth_value = self._depth_patch_distance_cm(
                depth_frame,
                point,
                source_mode=source_mode,
            )
            if depth_value is not None:
                profile[index] = float(depth_value)
        return profile

    def _pose_torso_depth_cm(self, profile):
        candidates = []
        for index in (0, 5, 6, 11, 12):
            depth_value = profile.get(index)
            if depth_value is not None:
                candidates.append(float(depth_value))

        if not candidates:
            return None
        candidates.sort()
        return float(candidates[len(candidates) // 2])

    def _match_pose_fusion_tracks(self, detections, now):
        if not self._pose_fusion_tracks or not detections:
            return {}

        candidates = []
        for detection_index, detection in enumerate(detections):
            detection_bbox = detection.get('bbox')
            if detection_bbox is None:
                continue
            detection_side = max(1.0, self._bbox_max_side(detection_bbox))

            for track_id, state in self._pose_fusion_tracks.items():
                age = float(now) - float(state.get('updated_at', 0.0))
                if age > self.POSE_DEPTH_FUSION_TRACK_MAX_AGE_SECONDS:
                    continue

                track_bbox = state.get('bbox')
                if track_bbox is None:
                    continue

                overlap = self._bbox_iou(detection_bbox, track_bbox)
                center_distance = self._bbox_center_distance(detection_bbox, track_bbox)
                normalized_distance = center_distance / detection_side
                if (
                    overlap < self.POSE_DEPTH_FUSION_TRACK_MATCH_IOU
                    and normalized_distance > self.POSE_DEPTH_FUSION_TRACK_MATCH_CENTER_RATIO
                ):
                    continue

                score = (overlap * 3.0) + (1.0 - min(1.0, normalized_distance))
                # Favor recent tracks to reduce identity flips during crossing.
                score += max(0.0, 0.2 - min(0.2, age * 0.15))
                candidates.append((score, detection_index, track_id))

        matches = {}
        used_detection_indexes = set()
        used_track_ids = set()

        for _, detection_index, track_id in sorted(candidates, key=lambda item: item[0], reverse=True):
            if detection_index in used_detection_indexes or track_id in used_track_ids:
                continue
            used_detection_indexes.add(detection_index)
            used_track_ids.add(track_id)
            matches[detection_index] = track_id

        return matches

    def _fuse_pose_detection(self, detection, track_state, now, depth_frame, source_mode):
        keypoints, keypoint_conf = self._coerce_pose_arrays(detection)
        depth_profile = self._pose_depth_profile(keypoints, keypoint_conf, depth_frame, source_mode)
        torso_depth = self._pose_torso_depth_cm(depth_profile)

        previous_points = track_state.get('keypoints') or []
        previous_conf = track_state.get('keypoint_conf') or []
        previous_depth = track_state.get('depth_by_index') or {}
        previous_updated_at = float(track_state.get('updated_at', 0.0))
        time_delta = max(0.0, float(now) - previous_updated_at)
        bbox = detection.get('bbox') or [0.0, 0.0, 1.0, 1.0]
        bbox_side = max(1.0, self._bbox_max_side(bbox))

        fused_points = []
        fused_conf = []
        fused_depth = {}

        for index, point in enumerate(keypoints):
            confidence = keypoint_conf[index] if index < len(keypoint_conf) else 0.0
            valid_point = point is not None and len(point) >= 2
            is_valid = bool(valid_point and float(confidence) >= self.POSE_DEPTH_FUSION_MIN_CONFIDENCE)
            current_depth = depth_profile.get(index)
            depth_outlier = bool(
                is_valid
                and current_depth is not None
                and torso_depth is not None
                and abs(float(current_depth) - float(torso_depth)) > self.POSE_DEPTH_FUSION_OUTLIER_CM
                and float(confidence) <= self.POSE_DEPTH_FUSION_OUTLIER_CONF_MAX
            )

            previous_point = previous_points[index] if index < len(previous_points) else None
            previous_point_valid = previous_point is not None and len(previous_point) >= 2
            previous_confidence = float(previous_conf[index]) if index < len(previous_conf) else 0.0
            previous_depth_value = previous_depth.get(index)

            if not is_valid:
                if (
                    previous_point_valid
                    and time_delta <= self.POSE_DEPTH_FUSION_HOLD_SECONDS
                    and previous_confidence >= self.POSE_DEPTH_FUSION_HOLD_MIN_CONF
                ):
                    hold_confidence = max(
                        self.POSE_DEPTH_FUSION_HOLD_FLOOR,
                        previous_confidence * self.POSE_DEPTH_FUSION_HOLD_DECAY,
                    )
                    fused_points.append([float(previous_point[0]), float(previous_point[1])])
                    fused_conf.append(float(hold_confidence))
                    if previous_depth_value is not None:
                        fused_depth[index] = float(previous_depth_value)
                else:
                    fused_points.append([0.0, 0.0])
                    fused_conf.append(0.0)
                continue

            fused_x = float(point[0])
            fused_y = float(point[1])
            fused_confidence = float(confidence)
            if (
                previous_point_valid
                and time_delta <= self.POSE_DEPTH_FUSION_TRACK_MAX_AGE_SECONDS
            ):
                displacement = (
                    ((fused_x - float(previous_point[0])) ** 2)
                    + ((fused_y - float(previous_point[1])) ** 2)
                ) ** 0.5
                displacement_norm = displacement / bbox_side

                alpha = float(self.POSE_DEPTH_FUSION_BASE_ALPHA)
                if displacement_norm > self.POSE_DEPTH_FUSION_DISP_NORM_FAST:
                    alpha = min(alpha, float(self.POSE_DEPTH_FUSION_FAST_ALPHA))

                if (
                    current_depth is not None
                    and previous_depth_value is not None
                    and abs(float(current_depth) - float(previous_depth_value)) > self.POSE_DEPTH_FUSION_DEPTH_JUMP_CM
                ):
                    alpha = min(alpha, float(self.POSE_DEPTH_FUSION_DEPTH_JUMP_ALPHA))

                if depth_outlier:
                    alpha = min(alpha, float(self.POSE_DEPTH_FUSION_OUTLIER_ALPHA))
                    fused_confidence = max(fused_confidence, previous_confidence * 0.75)

                fused_x = (float(previous_point[0]) * (1.0 - alpha)) + (fused_x * alpha)
                fused_y = (float(previous_point[1]) * (1.0 - alpha)) + (fused_y * alpha)

            fused_points.append([float(fused_x), float(fused_y)])
            fused_conf.append(float(fused_confidence))
            if current_depth is not None:
                fused_depth[index] = float(current_depth)
            elif previous_depth_value is not None:
                fused_depth[index] = float(previous_depth_value)

        detection['keypoints'] = fused_points
        detection['keypoint_conf'] = fused_conf
        return fused_points, fused_conf, fused_depth

    def _cleanup_pose_fusion_tracks(self, now, active_track_ids=None, reset=False):
        if reset:
            self._pose_fusion_tracks = {}
            self._pose_fusion_next_track_id = 1
            return

        active = set(active_track_ids or [])
        max_age = float(self.POSE_DEPTH_FUSION_TRACK_MAX_AGE_SECONDS)
        self._pose_fusion_tracks = {
            track_id: state
            for track_id, state in self._pose_fusion_tracks.items()
            if (
                track_id in active
                or (float(now) - float(state.get('updated_at', 0.0))) <= max_age
            )
        }

    def _apply_depth_pose_fusion(self, detections, depth_frame, now, source_mode):
        if not detections:
            self._cleanup_pose_fusion_tracks(now, active_track_ids=[], reset=False)
            return detections

        if (not self.POSE_DEPTH_FUSION_ENABLED) or depth_frame is None:
            self._cleanup_pose_fusion_tracks(now, active_track_ids=[], reset=True)
            return detections

        matches = self._match_pose_fusion_tracks(detections, now)
        active_track_ids = []
        for detection_index, detection in enumerate(detections):
            track_id = matches.get(detection_index)
            if track_id is None:
                track_id = int(self._pose_fusion_next_track_id)
                self._pose_fusion_next_track_id += 1

            track_state = self._pose_fusion_tracks.get(track_id, {})
            fused_points, fused_conf, fused_depth = self._fuse_pose_detection(
                detection,
                track_state,
                now=now,
                depth_frame=depth_frame,
                source_mode=source_mode,
            )
            self._pose_fusion_tracks[track_id] = {
                'bbox': list(detection.get('bbox') or [0.0, 0.0, 1.0, 1.0]),
                'keypoints': fused_points,
                'keypoint_conf': fused_conf,
                'depth_by_index': fused_depth,
                'updated_at': float(now),
            }
            active_track_ids.append(track_id)

        self._cleanup_pose_fusion_tracks(now, active_track_ids=active_track_ids, reset=False)
        return detections

    def _deduplicate_detection_boxes(self, candidates):
        if not candidates:
            return []

        prioritized = sorted(
            candidates,
            key=lambda item: (item['confidence'], self._current_bbox_area(item['bbox'])),
            reverse=True,
        )
        kept = []

        for candidate in prioritized:
            bbox = candidate['bbox']
            duplicate_found = False
            for kept_item in kept:
                if self._boxes_likely_same_person(
                    bbox,
                    kept_item['bbox'],
                    iou_threshold=self.DETECTION_DUPLICATE_IOU_THRESHOLD,
                    center_ratio=self.DETECTION_DUPLICATE_CENTER_RATIO,
                    area_ratio_limit=self.DETECTION_DUPLICATE_AREA_RATIO,
                    overlap_ratio=self.DETECTION_DUPLICATE_MIN_OVERLAP_RATIO,
                ):
                    duplicate_found = True
                    break

            if not duplicate_found:
                kept.append(candidate)

        kept.sort(key=lambda item: (item['bbox'][0], item['bbox'][1]))
        return kept

    def _tracking_match_score(self, detection_bbox, entity_bbox):
        iou = self._bbox_iou(detection_bbox, entity_bbox)
        if iou >= self.TRACK_IOU_THRESHOLD:
            return 2.0 + iou

        overlap = self._bbox_intersection_over_min_area(detection_bbox, entity_bbox)
        if overlap >= self.TRACK_MIN_OVERLAP_RATIO:
            return 1.6 + (overlap * 0.6)

        area_left = max(1.0, self._current_bbox_area(detection_bbox))
        area_right = max(1.0, self._current_bbox_area(entity_bbox))
        area_ratio = max(area_left, area_right) / min(area_left, area_right)
        if area_ratio > self.TRACK_MAX_AREA_RATIO:
            return None

        distance = self._bbox_center_distance(detection_bbox, entity_bbox)
        min_side = min(self._bbox_max_side(detection_bbox), self._bbox_max_side(entity_bbox))
        distance_limit = max(18.0, min_side * self.TRACK_CENTER_DISTANCE_RATIO)
        if overlap > 0.0:
            distance_limit *= 1.0 + min(0.42, overlap * 0.45)
        if distance > distance_limit:
            return None

        size_penalty = min(0.35, max(0.0, area_ratio - 1.0) * 0.09)
        return 1.0 - (distance / distance_limit) - size_penalty

    def _current_bbox_area(self, bbox):
        return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

    def _crop_frame_for_training(self, frame, bbox):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in bbox]
        pad_x = max(16, int((x2 - x1) * 0.12))
        pad_y = max(16, int((y2 - y1) * 0.12))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(width, x2 + pad_x)
        y2 = min(height, y2 + pad_y)
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return frame.copy()
        return cropped.copy()

    def _crop_face_for_training(self, frame, person_bbox):
        analysis = self._analyze_face_inside_bbox_fast(frame, person_bbox)
        if analysis.get('status') != 'ok' or not analysis.get('bbox'):
            message = str(analysis.get('message') or '').strip() or 'No face detected in current frame.'
            raise RuntimeError(message)

        face_bbox = [float(value) for value in analysis.get('bbox', [])]
        if len(face_bbox) != 4:
            raise RuntimeError('Invalid face region in current frame.')

        height, width = frame.shape[:2]
        x1, y1, x2, y2 = face_bbox
        face_w = max(1.0, x2 - x1)
        face_h = max(1.0, y2 - y1)
        pad_x = max(12, int(round(face_w * 0.35)))
        pad_y_top = max(14, int(round(face_h * 0.45)))
        pad_y_bottom = max(10, int(round(face_h * 0.22)))

        crop_x1 = max(0, int(round(x1 - pad_x)))
        crop_y1 = max(0, int(round(y1 - pad_y_top)))
        crop_x2 = min(width, int(round(x2 + pad_x)))
        crop_y2 = min(height, int(round(y2 + pad_y_bottom)))
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            raise RuntimeError('Unable to crop face in current frame.')

        cropped = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if cropped is None or cropped.size == 0:
            raise RuntimeError('Unable to crop face in current frame.')
        return cropped.copy()

    def _user_id_for_match(self, match):
        return match.get('student_id') or match['label']

    def _detect_people(self, frame, now, depth_frame=None, depth_source_mode=None):
        detect_interval = self.TRACKED_YOLO_DETECT_INTERVAL if self._last_person_boxes else self.YOLO_DETECT_INTERVAL
        if now - self._last_person_detect_at < detect_interval:
            return list(self._last_person_boxes)

        resized, scale = self._resize_for_inference(frame)
        result = self._run_yolo_person_inference(
            resized,
            conf=self.YOLO_CONFIDENCE,
            max_det=self.MAX_DETECTIONS,
        )

        factor = 1.0 / scale
        candidates = self._detection_candidates_from_result(result, factor)
        candidates.extend(self._face_person_fallback_candidates(frame, now, candidates))
        detections = self._deduplicate_detection_boxes(candidates)
        if depth_source_mode is None:
            depth_source_mode = self._depth_source_mode()
        detections = self._apply_depth_pose_fusion(
            detections,
            depth_frame=depth_frame,
            now=now,
            source_mode=depth_source_mode,
        )
        boxes = [item['bbox'] for item in detections]

        self._last_person_boxes = boxes
        self._last_pose_detections = detections
        self._last_person_detect_at = now
        return boxes

    def _match_pose_detection_to_bbox(self, bbox, detections, used_indexes=None):
        if not detections:
            return None, None

        best_index = None
        best_detection = None
        best_score = None

        for index, detection in enumerate(detections):
            if used_indexes is not None and index in used_indexes:
                continue

            detection_bbox = detection.get('bbox')
            if detection_bbox is None:
                continue

            iou = self._bbox_iou(bbox, detection_bbox)
            if iou > 0:
                score = 2.0 + iou
            else:
                distance = self._bbox_center_distance(bbox, detection_bbox)
                min_side = min(self._bbox_max_side(bbox), self._bbox_max_side(detection_bbox))
                distance_limit = max(18.0, min_side * self.TRACK_CENTER_DISTANCE_RATIO)
                if distance > distance_limit:
                    continue
                score = 1.0 - (distance / distance_limit)

            if best_score is None or score > best_score:
                best_score = score
                best_index = index
                best_detection = detection

        return best_index, best_detection

    def _update_classroom_metrics_locked(self, frame_shape, depth_frame, now, force=False):
        if not force and now < (self._last_metrics_update_at + self.METRIC_UPDATE_INTERVAL):
            return
        if not self._last_pose_detections:
            return

        depth_source_mode = self._depth_source_mode()
        confirmed_present_people = [
            person
            for person in self._confirmed_people.values()
            if person.current_status == 'present' and person.bbox is not None
        ]
        if not confirmed_present_people:
            return

        self._last_metrics_update_at = now

        detections = list(self._last_pose_detections)
        used_detection_indexes = set()
        matched_detection_by_user = {}
        centers_by_user = {}

        for person in confirmed_present_people:
            detection_index, detection = self._match_pose_detection_to_bbox(
                person.bbox,
                detections,
                used_indexes=used_detection_indexes,
            )
            if detection is not None:
                used_detection_indexes.add(detection_index)
            else:
                detection = {'bbox': person.bbox, 'keypoints': [], 'keypoint_conf': []}

            matched_detection_by_user[person.user_id] = detection
            centers_by_user[person.user_id] = self._bbox_center(detection.get('bbox', person.bbox))

        for person in confirmed_present_people:
            detection = matched_detection_by_user.get(person.user_id, {'bbox': person.bbox})
            peer_centers = [
                center
                for user_id, center in centers_by_user.items()
                if user_id != person.user_id
            ]
            self._metric_engine.update_student(
                user_id=person.user_id,
                frame_shape=frame_shape,
                bbox=person.bbox,
                pose_detection=detection,
                depth_frame=depth_frame,
                depth_source_mode=depth_source_mode,
                peer_centers=peer_centers,
                now=now,
            )

    def _draw_pose_overlay(self, overlay, pose_detection, color, bbox=None):
        if not pose_detection:
            return

        keypoints = pose_detection.get('keypoints') or []
        keypoint_conf = pose_detection.get('keypoint_conf') or []
        if not keypoints:
            return

        cv2, _ = self._get_cv_modules()
        if bbox is None:
            bbox = pose_detection.get('bbox')
        bbox_side = max(120.0, self._bbox_max_side(bbox)) if bbox is not None else 180.0
        line_thickness = max(3, min(7, int(round(bbox_side / 135.0))))
        line_shadow_thickness = line_thickness + 2
        point_radius = max(4, min(8, int(round(bbox_side / 165.0))))
        point_shadow_radius = point_radius + 1
        shadow_color = (14, 18, 32)

        def _valid_point(index):
            if index >= len(keypoints):
                return None
            if keypoint_conf and index < len(keypoint_conf):
                if float(keypoint_conf[index]) < self.POSE_KEYPOINT_MIN_CONFIDENCE:
                    return None
            point = keypoints[index]
            if point is None or len(point) < 2:
                return None
            return int(point[0]), int(point[1])

        for start_index, end_index in self.POSE_SKELETON_EDGES:
            start_point = _valid_point(start_index)
            end_point = _valid_point(end_index)
            if start_point is None or end_point is None:
                continue
            cv2.line(overlay, start_point, end_point, shadow_color, line_shadow_thickness, cv2.LINE_AA)
            cv2.line(overlay, start_point, end_point, color, line_thickness, cv2.LINE_AA)

        for point_index in range(len(keypoints)):
            point = _valid_point(point_index)
            if point is None:
                continue
            cv2.circle(overlay, point, point_shadow_radius, shadow_color, -1, cv2.LINE_AA)
            cv2.circle(overlay, point, point_radius, color, -1, cv2.LINE_AA)

    def _pose_point(self, pose_detection, keypoint_index):
        if not pose_detection:
            return None

        keypoints = pose_detection.get('keypoints') or []
        keypoint_conf = pose_detection.get('keypoint_conf') or []
        if keypoint_index >= len(keypoints):
            return None
        if keypoint_conf and keypoint_index < len(keypoint_conf):
            if float(keypoint_conf[keypoint_index]) < self.POSE_KEYPOINT_MIN_CONFIDENCE:
                return None

        point = keypoints[keypoint_index]
        if point is None or len(point) < 2:
            return None
        return float(point[0]), float(point[1])

    def _depth_source_mode(self):
        try:
            status = self.kinect_service.get_status()
        except Exception:
            return 'kinect'
        mode = str(status.get('source_mode') or '').strip().lower()
        return mode if mode in {'kinect', 'kinect_v1_registered', 'kinect_v2', 'video'} else 'kinect'

    def _depth_raw_value_to_cm(self, raw_value, source_mode):
        try:
            value = float(raw_value)
        except Exception:
            return None
        if value <= 0:
            return None

        if source_mode == 'kinect':
            if value < self.KINECT_RAW_MIN or value > self.KINECT_RAW_MAX:
                return None
            denominator = (self.KINECT_RAW_COEFF_A * value) + self.KINECT_RAW_COEFF_B
            if denominator <= 0:
                return None
            distance_cm = 100.0 / denominator
        elif source_mode in {'kinect_v1_registered', 'kinect_v2'}:
            distance_cm = value / 10.0
        else:
            # Offline/video fallback depth map is pseudo-mm and can be treated linearly.
            distance_cm = value / 10.0

        distance_cm = (distance_cm * self.DEPTH_DISTANCE_SCALE) + self.DEPTH_DISTANCE_BIAS_CM
        if distance_cm < self.DEPTH_MIN_VALID_CM or distance_cm > self.DEPTH_MAX_VALID_CM:
            return None
        return float(distance_cm)

    def _depth_patch_distance_cm(self, depth_frame, point, source_mode):
        if depth_frame is None or point is None:
            return None

        height = int(getattr(depth_frame, 'shape', [0, 0])[0] or 0)
        width = int(getattr(depth_frame, 'shape', [0, 0])[1] or 0)
        if width <= 0 or height <= 0:
            return None

        center_x = int(round(point[0]))
        center_y = int(round(point[1]))
        radius = int(self.DEPTH_SAMPLE_RADIUS)
        x1 = max(0, center_x - radius)
        y1 = max(0, center_y - radius)
        x2 = min(width, center_x + radius + 1)
        y2 = min(height, center_y + radius + 1)
        if x2 <= x1 or y2 <= y1:
            return None

        patch = depth_frame[y1:y2, x1:x2]
        if patch is None or getattr(patch, 'size', 0) == 0:
            return None

        values_cm = []
        if len(patch.shape) == 3 and patch.shape[2] >= 3:
            # Safety fallback: a colorized depth frame should not be used for precise distance,
            # but we keep a stable approximation for demo/video mode.
            gray_values = (patch[..., 0] * 0.114) + (patch[..., 1] * 0.587) + (patch[..., 2] * 0.299)
            for item in gray_values.reshape(-1):
                value = float(item)
                if value <= 0:
                    continue
                approx_cm = 30.0 + (value / 255.0) * 160.0
                if self.DEPTH_MIN_VALID_CM <= approx_cm <= self.DEPTH_MAX_VALID_CM:
                    values_cm.append(approx_cm)
        else:
            for item in patch.reshape(-1):
                converted = self._depth_raw_value_to_cm(item, source_mode)
                if converted is not None:
                    values_cm.append(converted)

        if not values_cm:
            return None

        values_cm.sort()
        trim_count = int(len(values_cm) * 0.18)
        if (trim_count * 2) >= len(values_cm):
            trimmed = values_cm
        else:
            trimmed = values_cm[trim_count:len(values_cm) - trim_count]

        if not trimmed:
            return None
        return float(trimmed[len(trimmed) // 2])

    def _estimate_distance_cm(self, depth_frame, bbox, pose_detection=None, smoothing_key=None, source_mode=None):
        if depth_frame is None or bbox is None:
            return None

        if source_mode is None:
            source_mode = self._depth_source_mode()
        sample_points = []

        nose_point = self._pose_point(pose_detection, 0)
        if nose_point is not None:
            sample_points.append((nose_point, 0.45))

        left_shoulder = self._pose_point(pose_detection, 5)
        right_shoulder = self._pose_point(pose_detection, 6)
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_center = (
                (left_shoulder[0] + right_shoulder[0]) * 0.5,
                (left_shoulder[1] + right_shoulder[1]) * 0.5,
            )
            sample_points.append((shoulder_center, 0.35))

        sample_points.append((self._bbox_torso_point(bbox), 0.20))

        weighted_sum = 0.0
        weight_total = 0.0
        for point, weight in sample_points:
            point_distance = self._depth_patch_distance_cm(depth_frame, point, source_mode)
            if point_distance is None:
                continue
            weighted_sum += point_distance * float(weight)
            weight_total += float(weight)

        if weight_total <= 0.0:
            return None

        raw_distance = weighted_sum / weight_total
        if smoothing_key is None:
            return raw_distance

        previous_distance = self._distance_smooth_cache.get(smoothing_key)
        if previous_distance is None:
            smoothed = raw_distance
        else:
            alpha = self.DEPTH_SMOOTHING_ALPHA
            if abs(raw_distance - previous_distance) >= self.DEPTH_FAST_JUMP_CM:
                alpha = self.DEPTH_SMOOTHING_FAST_ALPHA
            smoothed = (previous_distance * (1.0 - alpha)) + (raw_distance * alpha)
        self._distance_smooth_cache[smoothing_key] = smoothed
        return smoothed

    def _cleanup_distance_smoothing_locked(self, active_keys):
        active = {str(item) for item in (active_keys or []) if str(item).strip()}
        self._distance_smooth_cache = {
            key: value
            for key, value in self._distance_smooth_cache.items()
            if key in active
        }

    def _build_entity_index_locked(self):
        entities = []
        for person in self._temporary_people.values():
            if person.current_status == 'tracking':
                entities.append(('temporary', person.temp_id, person.bbox))
        for person in self._confirmed_people.values():
            if person.current_status == 'present' and person.bbox is not None:
                entities.append(('confirmed', person.user_id, person.bbox))
        return entities

    def _temporary_boxes_conflict_locked(self, left_bbox, right_bbox):
        return self._boxes_likely_same_person(
            left_bbox,
            right_bbox,
            iou_threshold=self.TEMPORARY_MERGE_IOU_THRESHOLD,
            center_ratio=self.TEMPORARY_MERGE_DISTANCE_RATIO,
            area_ratio_limit=2.8,
            overlap_ratio=self.TEMPORARY_MERGE_MIN_OVERLAP_RATIO,
        )

    def _merge_duplicate_temporaries_locked(self):
        if len(self._temporary_people) < 2:
            return

        ordered_people = sorted(
            self._temporary_people.values(),
            key=lambda item: (
                item.confirm_status == 'processing',
                item.last_seen,
                self._current_bbox_area(item.bbox),
            ),
            reverse=True,
        )
        kept_people = []
        for person in ordered_people:
            duplicate_found = False
            for existing in kept_people:
                if self._temporary_boxes_conflict_locked(person.bbox, existing.bbox):
                    duplicate_found = True
                    break
            if not duplicate_found:
                kept_people.append(person)

        self._temporary_people = {person.temp_id: person for person in kept_people}

    def _renumber_temporary_people_locked(self):
        student_prefix = f'{chr(0x5B78)}{chr(0x751F)}'
        used_slots = {
            int(person.slot_number)
            for person in self._temporary_people.values()
            if int(getattr(person, 'slot_number', 0) or 0) > 0
        }

        for person in sorted(
            self._temporary_people.values(),
            key=lambda item: (
                item.first_seen,
                item.last_seen,
                item.temp_id,
            ),
        ):
            slot = int(getattr(person, 'slot_number', 0) or 0)
            if slot <= 0:
                slot = self._allocate_temporary_slot_locked()
                person.slot_number = slot
                used_slots.add(slot)
            person.display_name = f'{student_prefix}{slot:06d}'

    def _allocate_temporary_slot_locked(self):
        used_slots = {
            int(person.slot_number)
            for person in self._temporary_people.values()
            if int(getattr(person, 'slot_number', 0) or 0) > 0
        }
        candidate = 1
        while candidate in used_slots:
            candidate += 1
        return candidate

    def _create_temporary_person_locked(self, bbox, now):
        temp_number = self._next_temp_number
        self._next_temp_number += 1
        temp_id = f'temp-{temp_number}'
        person = TemporaryPerson(
            temp_id=temp_id,
            display_name='',
            bbox=bbox,
            tracking_id=self._next_tracking_id,
            slot_number=self._allocate_temporary_slot_locked(),
            first_seen=now,
            last_seen=now,
        )
        self._next_tracking_id += 1
        self._temporary_people[temp_id] = person
        self._renumber_temporary_people_locked()
        self._push_announcement_locked(f'{person.display_name} 已進入畫面。')

    def _set_confirmed_present_locked(self, person, tracking_id, bbox, now):
        if person.current_status != 'present':
            person.presence_segments.append({'start': now, 'end': None})
            person.appearance_count += 1
        person.current_status = 'present'
        person.current_tracking_id = tracking_id
        person.bbox = bbox
        person.last_seen = now

    def _set_confirmed_absent_locked(self, person, now):
        if person.current_status == 'present':
            for segment in reversed(person.presence_segments):
                if segment['end'] is None:
                    segment['end'] = now
                    break
        person.current_status = 'absent'
        person.current_tracking_id = None
        person.bbox = None
        person.last_seen = now

    def _match_detections_locked(self, person_boxes, now):
        entities = self._build_entity_index_locked()
        candidate_pairs = []

        for detection_index, detection_bbox in enumerate(person_boxes):
            for entity_kind, entity_id, entity_bbox in entities:
                score = self._tracking_match_score(detection_bbox, entity_bbox)
                if score is not None:
                    candidate_pairs.append((score, detection_index, entity_kind, entity_id))

        candidate_pairs.sort(reverse=True)
        matched_detections = set()
        matched_entities = set()

        for _, detection_index, entity_kind, entity_id in candidate_pairs:
            if detection_index in matched_detections or (entity_kind, entity_id) in matched_entities:
                continue

            matched_detections.add(detection_index)
            matched_entities.add((entity_kind, entity_id))
            bbox = person_boxes[detection_index]

            if entity_kind == 'temporary':
                person = self._temporary_people.get(entity_id)
                if person is not None:
                    person.bbox = bbox
                    person.last_seen = now
                    person.current_status = 'tracking'
            else:
                person = self._confirmed_people.get(entity_id)
                if person is not None:
                    tracking_id = person.current_tracking_id or self._next_tracking_id
                    if person.current_tracking_id is None:
                        self._next_tracking_id += 1
                    self._set_confirmed_present_locked(person, tracking_id, bbox, now)

        active_entity_boxes = []
        for person in self._temporary_people.values():
            if person.current_status == 'tracking' and person.bbox is not None:
                active_entity_boxes.append(list(person.bbox))
        for person in self._confirmed_people.values():
            if person.current_status == 'present' and person.bbox is not None:
                active_entity_boxes.append(list(person.bbox))

        for detection_index, bbox in enumerate(person_boxes):
            if detection_index not in matched_detections:
                duplicate_shadow = any(
                    self._boxes_likely_same_person(
                        bbox,
                        active_bbox,
                        iou_threshold=self.UNMATCHED_DUPLICATE_GUARD_IOU_THRESHOLD,
                        center_ratio=self.DETECTION_DUPLICATE_CENTER_RATIO * 1.1,
                        area_ratio_limit=max(2.8, self.DETECTION_DUPLICATE_AREA_RATIO),
                        overlap_ratio=self.UNMATCHED_DUPLICATE_GUARD_OVERLAP_RATIO,
                    )
                    for active_bbox in active_entity_boxes
                )
                if duplicate_shadow:
                    continue
                self._create_temporary_person_locked(bbox, now)
                active_entity_boxes.append(list(bbox))

        stale_temporary_ids = [
            temp_id
            for temp_id, person in self._temporary_people.items()
            if person.confirm_status != 'processing' and now - person.last_seen > self.TEMP_PERSON_TIMEOUT
        ]
        for temp_id in stale_temporary_ids:
            self._temporary_people.pop(temp_id, None)

        self._merge_duplicate_temporaries_locked()
        self._renumber_temporary_people_locked()
        if not self._temporary_people:
            self._next_temp_number = 1

        save_needed = False
        for person in self._confirmed_people.values():
            if person.current_status == 'present' and now - person.last_seen > self.CONFIRMED_ABSENT_TIMEOUT:
                self._set_confirmed_absent_locked(person, now)
                self._push_announcement_locked(f'{person.name} 已離開畫面。')
                save_needed = True

        if save_needed:
            self._save_presence_records_locked()

    def _analyze_face_inside_bbox(self, frame, bbox):
        height, width = frame.shape[:2]
        pad_x = int((bbox[2] - bbox[0]) * 0.08)
        pad_y = int((bbox[3] - bbox[1]) * 0.08)
        x1 = max(0, int(bbox[0] - pad_x))
        y1 = max(0, int(bbox[1] - pad_y))
        x2 = min(width, int(bbox[2] + pad_x))
        y2 = min(height, int(bbox[3] + pad_y))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return {'status': 'no_face', 'message': '無法從該學生區域擷取臉部畫面。'}

        analysis = self.face_db.analyze_faces(crop)
        if analysis['status'] != 'ok' or not analysis['faces']:
            return {'status': analysis['status'], 'message': analysis['message']}

        best_face = max(analysis['faces'], key=lambda item: self._current_bbox_area(item['bbox']))
        face_bbox = [
            best_face['bbox'][0] + x1,
            best_face['bbox'][1] + y1,
            best_face['bbox'][2] + x1,
            best_face['bbox'][3] + y1,
        ]
        return {
            'status': 'ok',
            'message': 'Face detected inside the selected person box.',
            'bbox': face_bbox,
            'embedding': best_face['embedding'],
        }

    def _analyze_face_inside_bbox_fast(self, frame, bbox):
        cv2, _ = self._get_cv_modules()
        height, width = frame.shape[:2]

        box_w = max(1.0, float(bbox[2]) - float(bbox[0]))
        box_h = max(1.0, float(bbox[3]) - float(bbox[1]))
        pad_x = int(box_w * 0.08)
        pad_y = int(box_h * 0.08)

        base_x1 = max(0, int(bbox[0] - pad_x))
        base_y1 = max(0, int(bbox[1] - pad_y))
        base_x2 = min(width, int(bbox[2] + pad_x))
        base_y2 = min(height, int(bbox[3] + pad_y))
        if base_x2 <= base_x1 or base_y2 <= base_y1:
            return {'status': 'no_face', 'message': 'Invalid person region for face analysis.'}

        upper_y1 = max(
            base_y1,
            int(float(bbox[1]) + (box_h * self.FACE_REGION_TOP_RATIO)),
        )
        upper_y2 = min(
            base_y2,
            int(float(bbox[1]) + (box_h * self.FACE_REGION_BOTTOM_RATIO)),
        )

        regions = []
        if upper_y2 > upper_y1:
            regions.append((base_x1, upper_y1, base_x2, upper_y2))
        regions.append((base_x1, base_y1, base_x2, base_y2))

        last_failure = {'status': 'no_face', 'message': 'No face detected in selected person box.'}
        for x1, y1, x2, y2 in regions:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            analysis_scale = 1.0
            analyze_frame = crop
            max_side = max(crop.shape[0], crop.shape[1])
            if max_side > self.FACE_ANALYSIS_MAX_SIDE:
                analysis_scale = self.FACE_ANALYSIS_MAX_SIDE / float(max_side)
                resized_w = max(1, int(round(crop.shape[1] * analysis_scale)))
                resized_h = max(1, int(round(crop.shape[0] * analysis_scale)))
                analyze_frame = cv2.resize(crop, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

            analysis = self.face_db.analyze_faces(analyze_frame)
            if analysis['status'] != 'ok' or not analysis['faces']:
                last_failure = {'status': analysis['status'], 'message': analysis['message']}
                continue

            best_face = max(analysis['faces'], key=lambda item: self._current_bbox_area(item['bbox']))
            scale_back = (1.0 / analysis_scale) if analysis_scale > 0 else 1.0
            face_bbox = [
                (best_face['bbox'][0] * scale_back) + x1,
                (best_face['bbox'][1] * scale_back) + y1,
                (best_face['bbox'][2] * scale_back) + x1,
                (best_face['bbox'][3] * scale_back) + y1,
            ]
            return {
                'status': 'ok',
                'message': 'Face detected inside the selected person box.',
                'bbox': face_bbox,
                'embedding': best_face['embedding'],
                'det_score': float(best_face.get('det_score', 0.0)),
            }

        return last_failure

    def _calculate_total_presence_locked(self, person, now):
        total = 0.0
        for segment in person.presence_segments:
            segment_start = float(segment['start'])
            segment_end = float(segment['end'] if segment['end'] is not None else now)
            total += max(0.0, segment_end - segment_start)
        return total

    def _build_presence_points_locked(self, person, now):
        if not person.presence_segments:
            return []

        segments = []
        for segment in person.presence_segments:
            segment_start = float(segment['start'])
            segment_end = float(segment['end'] if segment['end'] is not None else now)
            segments.append((segment_start, segment_end))

        window_start = max(float(person.first_confirmed_at), float(now) - self.PRESENCE_WINDOW_SECONDS)
        sample_time = window_start
        points = []

        while sample_time <= now:
            is_present = any(segment_start <= sample_time <= segment_end for segment_start, segment_end in segments)
            points.append({'t': sample_time, 'v': 1 if is_present else 0})
            sample_time += self.PRESENCE_SAMPLE_INTERVAL

        if not points or points[-1]['t'] < now:
            is_present = any(segment_start <= now <= segment_end for segment_start, segment_end in segments)
            points.append({'t': now, 'v': 1 if is_present else 0})
        return points

    def _build_temporary_payload_locked(self, now):
        payload = []
        for person in sorted(
            self._temporary_people.values(),
            key=lambda item: (
                1 if int(getattr(item, 'slot_number', 0) or 0) <= 0 else 0,
                int(getattr(item, 'slot_number', 0) or 0),
                str(getattr(item, 'display_name', '') or '').lower(),
                float(getattr(item, 'first_seen', 0.0)),
            ),
        ):
            payload.append(
                {
                    'temp_id': person.temp_id,
                    'display_name': person.display_name,
                    'tracking_id': person.tracking_id,
                    'current_status': person.current_status,
                    'first_seen': person.first_seen,
                    'last_seen': person.last_seen,
                    'presence_time': now - person.first_seen,
                    'confirm_status': person.confirm_status,
                    'confirm_message': person.confirm_message,
                }
            )
        return payload

    def _build_confirmed_payload_locked(self, now, include_presence_points=True, include_metrics=True, metrics_user_id=None):
        def _confirmed_sort_key(person):
            student_id = str(getattr(person, 'student_id', '') or '').strip()
            digits = ''.join(char for char in student_id if char.isdigit())
            if digits:
                try:
                    sequence = int(digits)
                except Exception:
                    sequence = 0
                return (
                    0,
                    sequence,
                    student_id.lower(),
                    str(getattr(person, 'name', '') or '').lower(),
                )
            return (
                1,
                str(getattr(person, 'name', '') or '').lower(),
                student_id.lower(),
                str(getattr(person, 'user_id', '') or '').lower(),
            )

        payload = []
        for person in sorted(self._confirmed_people.values(), key=_confirmed_sort_key):
            total_presence_time = self._calculate_total_presence_locked(person, now)
            average_stay_time = (
                total_presence_time / person.appearance_count if person.appearance_count else 0.0
            )
            include_metrics_for_person = bool(include_metrics) and (
                metrics_user_id is None
                or str(metrics_user_id) == ''
                or str(person.user_id) == str(metrics_user_id)
                or str(person.student_id) == str(metrics_user_id)
            )
            payload.append(
                {
                    'user_id': person.user_id,
                    'label': person.label,
                    'display_name': person.name,
                    'name': person.name,
                    'student_id': person.student_id,
                    'department': person.department,
                    'title': person.title,
                    'current_status': person.current_status,
                    'first_confirmed_at': person.first_confirmed_at,
                    'first_seen': person.first_seen,
                    'last_seen': person.last_seen,
                    'total_presence_time': total_presence_time,
                    'appearance_count': person.appearance_count,
                    'average_stay_time': average_stay_time,
                    'presence_segments': person.presence_segments,
                    'presence_points': self._build_presence_points_locked(person, now) if include_presence_points else [],
                    'last_similarity': person.last_similarity,
                    'current_tracking_id': person.current_tracking_id,
                    'session_visible': person.user_id in self._session_confirmed_ids,
                    'classroom_metrics': self._metric_engine.get_user_metrics(person.user_id) if include_metrics_for_person else {},
                }
            )
        return payload

    def _promote_temporary_to_confirmed_locked(self, temp_id, match, now):
        temp_person = self._temporary_people.get(temp_id)
        if temp_person is None:
            return None

        user_id = self._user_id_for_match(match)
        confirmed_person = self._confirmed_people.get(user_id)
        if confirmed_person is None:
            for existing_user_id, existing_person in list(self._confirmed_people.items()):
                same_label = bool(match.get('label')) and existing_person.label == match.get('label')
                same_student_id = bool(match.get('student_id')) and existing_person.student_id == match.get('student_id')
                same_name = bool(match.get('display_name')) and existing_person.name == match.get('display_name')
                if same_label or same_student_id or same_name:
                    confirmed_person = existing_person
                    if existing_user_id != user_id:
                        self._confirmed_people.pop(existing_user_id, None)
                        existing_person.user_id = user_id
                        self._confirmed_people[user_id] = existing_person
                    break

        if confirmed_person is None:
            confirmed_person = ConfirmedPerson(
                user_id=user_id,
                label=match['label'],
                name=match.get('display_name', match['label']),
                student_id=match.get('student_id', ''),
                department=match.get('department', ''),
                title=match.get('title', ''),
                first_confirmed_at=now,
                first_seen=temp_person.first_seen,
                last_seen=now,
                current_status='present',
                current_tracking_id=temp_person.tracking_id,
                bbox=temp_person.bbox,
                presence_segments=[{'start': now, 'end': None}],
                appearance_count=1,
                last_similarity=float(match['similarity']),
            )
            self._confirmed_people[user_id] = confirmed_person
        else:
            confirmed_person.label = match['label']
            confirmed_person.name = match.get('display_name', confirmed_person.name)
            confirmed_person.student_id = match.get('student_id', confirmed_person.student_id)
            confirmed_person.department = match.get('department', confirmed_person.department)
            confirmed_person.title = match.get('title', confirmed_person.title)
            confirmed_person.first_seen = min(confirmed_person.first_seen, temp_person.first_seen)
            confirmed_person.last_similarity = float(match['similarity'])
            self._set_confirmed_present_locked(
                confirmed_person,
                temp_person.tracking_id,
                temp_person.bbox,
                now,
            )

        self._temporary_people.pop(temp_id, None)
        self._save_presence_records_locked()
        return confirmed_person

    def _try_auto_relink_locked(self, frame, now):
        absent_confirmed_ids = {
            user_id
            for user_id, person in self._confirmed_people.items()
            if person.current_status == 'absent'
        }
        if not absent_confirmed_ids:
            return

        face_checks = 0
        for temp_person in sorted(self._temporary_people.values(), key=lambda item: item.first_seen):
            if face_checks >= self.AUTO_RELINK_MAX_FACE_CHECKS_PER_TICK:
                break
            if now - temp_person.first_seen < self.AUTO_RELINK_MIN_TRACK_SECONDS:
                continue
            if now - temp_person.last_auto_relink_at < self.AUTO_RELINK_INTERVAL:
                continue

            temp_person.last_auto_relink_at = now
            face_checks += 1
            analysis = self._analyze_face_inside_bbox_fast(frame, temp_person.bbox)
            if analysis['status'] != 'ok':
                continue

            temp_person.face_bbox = list(analysis['bbox'])
            temp_person.face_embedding = list(analysis['embedding'])
            temp_person.face_embedding_at = now
            matches = self.face_db.match_embedding(
                analysis['embedding'],
                threshold=self.AUTO_RELINK_THRESHOLD,
            )
            if not matches:
                continue

            selected_match = None
            for match in matches:
                if self._user_id_for_match(match) in absent_confirmed_ids:
                    selected_match = match
                    break

            if selected_match is None:
                continue

            temp_person.face_bbox = analysis['bbox']
            confirmed_person = self._promote_temporary_to_confirmed_locked(
                temp_person.temp_id,
                selected_match,
                now,
            )
            if confirmed_person is not None:
                self._push_announcement_locked(f'{confirmed_person.name} 已重新回到畫面中。')
            break

    def _draw_label_pills(self, frame, label_specs):
        if not label_specs:
            return frame

        cv2, np = self._get_cv_modules()
        image, image_draw, _ = self._get_pil_modules()

        pil_image = image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        drawer = image_draw.Draw(pil_image)
        frame_width, frame_height = pil_image.size
        base_font_size = max(24, min(32, int(round(frame_width / 44.0))))

        for label_spec in label_specs:
            text = str(label_spec.get('text', '')).strip()
            if not text:
                continue

            bbox = label_spec.get('bbox')
            font_size = int(label_spec.get('font_size') or 0)
            if font_size <= 0:
                font_size = base_font_size
                if bbox is not None and len(bbox) == 4:
                    box_height = max(1, int(bbox[3]) - int(bbox[1]))
                    font_size = max(font_size, min(36, int(round(box_height * 0.18))))
            font = self._get_label_font(font_size)
            x1, y1 = label_spec['anchor']
            text_box = drawer.textbbox((0, 0), text, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            pad_x = max(16, int(round(font_size * 0.58)))
            pad_y = max(9, int(round(font_size * 0.34)))
            pill_height = text_height + pad_y * 2
            pill_width = text_width + pad_x * 2

            pill_left = max(0, min(frame_width - pill_width, x1))
            pill_top = max(0, y1 - pill_height - 8)
            if pill_top == 0:
                pill_top = max(0, min(frame_height - pill_height, y1 + 8))
            pill_right = min(frame_width, pill_left + pill_width)
            pill_bottom = min(frame_height, pill_top + pill_height)

            drawer.rounded_rectangle(
                (pill_left, pill_top, pill_right, pill_bottom),
                radius=max(16, int(round(font_size * 0.65))),
                fill=label_spec['background'],
            )
            drawer.text(
                (pill_left + pad_x, pill_top + pad_y - 1),
                text,
                font=font,
                fill=(244, 247, 255),
            )

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _annotate_frame_locked(self, frame):
        cv2, _ = self._get_cv_modules()
        annotated = frame.copy()
        overlay = annotated.copy()
        label_specs = []
        detections = list(self._last_pose_detections)

        for person in self._temporary_people.values():
            if person.bbox is None:
                continue
            x1, y1, x2, y2 = [int(value) for value in person.bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (61, 126, 255), 2)
            label_specs.append(
                {
                    'anchor': (x1, y1),
                    'bbox': (x1, y1, x2, y2),
                    'text': person.display_name,
                    'background': (11, 15, 25),
                }
            )

        for person in self._confirmed_people.values():
            if person.current_status != 'present' or person.bbox is None:
                continue
            _, pose_detection = self._match_pose_detection_to_bbox(person.bbox, detections)
            self._draw_pose_overlay(overlay, pose_detection, (112, 245, 255), bbox=person.bbox)
            x1, y1, x2, y2 = [int(value) for value in person.bbox]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (31, 199, 212), 2)
            label_specs.append(
                {
                    'anchor': (x1, y1),
                    'bbox': (x1, y1, x2, y2),
                    'text': person.student_id or person.name,
                    'background': (7, 35, 39),
                }
            )

        cv2.addWeighted(overlay, 0.34, annotated, 0.66, 0, annotated)
        return self._draw_label_pills(annotated, label_specs)

    def _scale_bbox_between_frames(self, bbox, source_shape, target_shape):
        if bbox is None:
            return None

        source_height = int(source_shape[0]) if source_shape else 0
        source_width = int(source_shape[1]) if source_shape else 0
        target_height = int(target_shape[0]) if target_shape else 0
        target_width = int(target_shape[1]) if target_shape else 0
        if source_width <= 0 or source_height <= 0 or target_width <= 0 or target_height <= 0:
            return None

        scale_x = target_width / float(source_width)
        scale_y = target_height / float(source_height)

        x1 = int(round(float(bbox[0]) * scale_x))
        y1 = int(round(float(bbox[1]) * scale_y))
        x2 = int(round(float(bbox[2]) * scale_x))
        y2 = int(round(float(bbox[3]) * scale_y))

        x1 = max(0, min(target_width - 1, x1))
        y1 = max(0, min(target_height - 1, y1))
        x2 = max(x1 + 1, min(target_width, x2))
        y2 = max(y1 + 1, min(target_height, y2))
        return [x1, y1, x2, y2]

    def _draw_distance_label(self, frame, bbox, text, color):
        cv2, _ = self._get_cv_modules()
        x1, y1, x2, y2 = [int(value) for value in bbox]
        if x2 <= x1 or y2 <= y1:
            return

        box_width = max(1, x2 - x1)
        box_height = max(1, y2 - y1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.44, min(0.86, min(box_width / 220.0, box_height / 132.0)))
        thickness = max(1, int(round(font_scale * 2.0)))
        value_text = str(text or '-- cm')
        text_size, baseline = cv2.getTextSize(value_text, font, font_scale, thickness)

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        pad_x = max(8, int(round(12 * font_scale)))
        pad_y = max(5, int(round(8 * font_scale)))
        panel_width = text_size[0] + (pad_x * 2)
        panel_height = text_size[1] + baseline + (pad_y * 2)

        panel_x1 = int(center_x - (panel_width / 2))
        panel_y1 = int(center_y - (panel_height / 2))
        panel_x2 = panel_x1 + panel_width
        panel_y2 = panel_y1 + panel_height

        panel_x1 = max(2, min(frame.shape[1] - panel_width - 2, panel_x1))
        panel_y1 = max(2, min(frame.shape[0] - panel_height - 2, panel_y1))
        panel_x2 = panel_x1 + panel_width
        panel_y2 = panel_y1 + panel_height

        text_x = panel_x1 + pad_x
        text_y = panel_y1 + pad_y + text_size[1]

        overlay = frame.copy()
        panel_fill_color = (8, 17, 30)
        panel_edge_color = (138, 189, 255)
        cv2.rectangle(
            overlay,
            (panel_x1, panel_y1),
            (panel_x2, panel_y2),
            panel_fill_color,
            -1,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            overlay,
            (panel_x1, panel_y1),
            (panel_x2, panel_y2),
            panel_edge_color,
            max(1, thickness),
            cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, 0.54, frame, 0.46, 0.0, frame)

        shadow_color = (0, 0, 0)
        cv2.putText(
            frame,
            value_text,
            (text_x + 1, text_y + 1),
            font,
            font_scale,
            shadow_color,
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            value_text,
            (text_x, text_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _annotate_depth_frame_locked(self, depth_visual_frame, depth_frame, color_shape):
        if depth_visual_frame is None:
            return None

        cv2, _ = self._get_cv_modules()
        target_height = int(color_shape[0]) if color_shape is not None and len(color_shape) >= 2 else 0
        target_width = int(color_shape[1]) if color_shape is not None and len(color_shape) >= 2 else 0
        if target_width > 0 and target_height > 0:
            annotated = cv2.resize(
                depth_visual_frame,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )
        else:
            annotated = depth_visual_frame.copy()
        active_distance_keys = []
        label_specs = []
        detections = list(self._last_pose_detections or [])
        used_detection_indexes = set()
        depth_source_mode = self._depth_source_mode()

        def _draw_depth_person_box(
            bbox,
            box_color,
            label_text,
            pose_detection=None,
            smoothing_key=None,
            label_background=(7, 32, 43),
            show_distance=True,
        ):
            if bbox is None:
                return

            mapped_bbox = self._scale_bbox_between_frames(bbox, color_shape, annotated.shape)
            if mapped_bbox is None:
                return

            x1, y1, x2, y2 = [int(value) for value in mapped_bbox]
            box_width = max(1, x2 - x1)
            box_height = max(1, y2 - y1)
            border_thickness = max(2, min(4, int(round(min(box_width, box_height) / 130.0))))
            corner_size = max(10, min(26, int(round(min(box_width, box_height) * 0.2))))

            box_overlay = annotated.copy()
            cv2.rectangle(
                box_overlay,
                (x1, y1),
                (x2, y2),
                box_color,
                -1,
                cv2.LINE_AA,
            )
            cv2.addWeighted(box_overlay, 0.08, annotated, 0.92, 0.0, annotated)

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                box_color,
                border_thickness,
                cv2.LINE_AA,
            )
            cv2.line(annotated, (x1, y1), (x1 + corner_size, y1), box_color, border_thickness + 1, cv2.LINE_AA)
            cv2.line(annotated, (x1, y1), (x1, y1 + corner_size), box_color, border_thickness + 1, cv2.LINE_AA)
            cv2.line(annotated, (x2, y1), (x2 - corner_size, y1), box_color, border_thickness + 1, cv2.LINE_AA)
            cv2.line(annotated, (x2, y1), (x2, y1 + corner_size), box_color, border_thickness + 1, cv2.LINE_AA)
            cv2.line(annotated, (x1, y2), (x1 + corner_size, y2), box_color, border_thickness + 1, cv2.LINE_AA)
            cv2.line(annotated, (x1, y2), (x1, y2 - corner_size), box_color, border_thickness + 1, cv2.LINE_AA)
            cv2.line(annotated, (x2, y2), (x2 - corner_size, y2), box_color, border_thickness + 1, cv2.LINE_AA)
            cv2.line(annotated, (x2, y2), (x2, y2 - corner_size), box_color, border_thickness + 1, cv2.LINE_AA)

            if show_distance:
                distance_bbox = bbox
                if depth_frame is not None and hasattr(depth_frame, 'shape'):
                    scaled_depth_bbox = self._scale_bbox_between_frames(bbox, color_shape, depth_frame.shape)
                    if scaled_depth_bbox is not None:
                        distance_bbox = scaled_depth_bbox
                distance_cm = self._estimate_distance_cm(
                    depth_frame,
                    distance_bbox,
                    pose_detection=pose_detection,
                    smoothing_key=smoothing_key,
                    source_mode=depth_source_mode,
                )
                distance_text = '-- cm' if distance_cm is None else f'{distance_cm:.1f} cm'
                self._draw_distance_label(annotated, mapped_bbox, distance_text, (233, 241, 255))

            clean_label = str(label_text or '').strip()
            if clean_label:
                label_specs.append(
                    {
                        'anchor': (x1, y1),
                        'bbox': (x1, y1, x2, y2),
                        'text': clean_label,
                        'background': label_background,
                    }
                )

        for person in self._confirmed_people.values():
            if person.current_status != 'present' or person.bbox is None:
                continue
            depth_label = person.student_id or person.name
            smoothing_key = str(person.user_id or person.student_id or person.label or '').strip()
            detection_index, pose_detection = self._match_pose_detection_to_bbox(
                person.bbox,
                detections,
                used_indexes=used_detection_indexes,
            )
            if detection_index is not None:
                used_detection_indexes.add(detection_index)
            if smoothing_key:
                active_distance_keys.append(smoothing_key)
            _draw_depth_person_box(
                person.bbox,
                (31, 199, 212),
                depth_label,
                pose_detection=pose_detection,
                smoothing_key=smoothing_key,
                label_background=(7, 32, 43),
                show_distance=True,
            )

        for person in self._temporary_people.values():
            if person.bbox is None:
                continue
            depth_label = person.display_name
            smoothing_key = str(person.temp_id or person.tracking_id or '').strip()
            detection_index, pose_detection = self._match_pose_detection_to_bbox(
                person.bbox,
                detections,
                used_indexes=used_detection_indexes,
            )
            if detection_index is not None:
                used_detection_indexes.add(detection_index)
            _draw_depth_person_box(
                person.bbox,
                (61, 126, 255),
                depth_label,
                pose_detection=pose_detection,
                smoothing_key=smoothing_key,
                label_background=(11, 15, 25),
                show_distance=False,
            )

        self._cleanup_distance_smoothing_locked(active_distance_keys)

        return self._draw_label_pills(annotated, label_specs)

    def _update_summary_locked(self, now, force=False):
        if not force and now < (self._last_summary_update_at + self.SUMMARY_UPDATE_INTERVAL):
            return
        self._last_summary_update_at = now
        temporary_payload = self._build_temporary_payload_locked(now)
        confirmed_payload = self._build_confirmed_payload_locked(
            now,
            include_presence_points=True,
            include_metrics=False,
        )
        confirmed_present_count = sum(
            1 for item in confirmed_payload if item['current_status'] == 'present'
        )

        if temporary_payload or confirmed_present_count:
            message = (
                f'課堂追蹤進行中，目前偵測 {len(temporary_payload)} 位暫時學生，'
                f'已確認 {confirmed_present_count} 位在場學生。'
            )
            status = 'attendance_running'
        else:
            message = '課堂模式已啟動，正在等待學生進入畫面。'
            status = 'attendance_waiting'

        self._status.update(
            {
                'status': status,
                'message': message,
                'attendance_mode': True,
                'temporary_people': temporary_payload,
                'confirmed_people': confirmed_payload,
                'students': confirmed_payload,
                'temporary_count': len(temporary_payload),
                'confirmed_present_count': confirmed_present_count,
                'confirmed_total_count': len(confirmed_payload),
                'recognized_count': confirmed_present_count,
                'yolo_person_present': bool(self._last_person_boxes),
                'announcement': self._announcements[-1]['message'] if self._announcements else '課堂模式已啟動。',
            }
        )

    def _process_frame(self, frame, depth_frame=None, depth_visual_frame=None, frame_timestamp=None):
        now = float(frame_timestamp) if frame_timestamp is not None else time.time()
        kinect_status = self.kinect_service.get_status()
        if kinect_status['status'] != 'connected':
            with self._lock:
                self._temporary_people = {}
                self._next_temp_number = 1
                for person in self._confirmed_people.values():
                    if person.current_status == 'present':
                        self._set_confirmed_absent_locked(person, now)
                self._save_presence_records_locked()
                self._annotated_jpeg = None
                self._annotated_depth_jpeg = None
                self._last_render_at = 0.0
                self._render_seq = 0
                self._distance_smooth_cache = {}
                self._cleanup_pose_fusion_tracks(now, reset=True)
                confirmed_payload = self._build_confirmed_payload_locked(
                    now,
                    include_presence_points=False,
                    include_metrics=False,
                )
                self._status.update(
                    {
                        'status': 'camera_unavailable',
                        'message': kinect_status['message'],
                        'attendance_mode': self._attendance_mode,
                        'temporary_people': [],
                        'confirmed_people': confirmed_payload,
                        'students': confirmed_payload,
                        'temporary_count': 0,
                        'confirmed_present_count': 0,
                        'confirmed_total_count': len(confirmed_payload),
                        'recognized_count': 0,
                        'yolo_person_present': False,
                        'detector_model': 'idle',
                    }
                )
            return

        if not self.get_attendance_mode():
            with self._lock:
                self._annotated_jpeg = None
                self._annotated_depth_jpeg = None
                self._last_render_at = 0.0
                self._distance_smooth_cache = {}
                self._cleanup_pose_fusion_tracks(now, reset=True)
            return

        if self._warmup_started and self._yolo_model is None and self._yolo_error is None:
            with self._lock:
                self._status.update(
                    {
                        'status': 'model_warming',
                        'message': '模型暖機中，稍後開始偵測學生。',
                        'attendance_mode': True,
                        'detector_model': 'warming_up',
                    }
                )
            return

        try:
            if depth_frame is None and hasattr(self.kinect_service, 'get_latest_depth_frame'):
                depth_frame = self.kinect_service.get_latest_depth_frame()
            if depth_frame is None:
                depth_frame = None
            if depth_visual_frame is None and hasattr(self.kinect_service, 'get_latest_depth_visual_frame'):
                depth_visual_frame = self.kinect_service.get_latest_depth_visual_frame()
            if depth_visual_frame is None:
                depth_visual_frame = None
            depth_source_mode = self._depth_source_mode()
            person_boxes = self._detect_people(
                frame,
                now,
                depth_frame=depth_frame,
                depth_source_mode=depth_source_mode,
            )
        except RuntimeError as exc:
            with self._lock:
                confirmed_payload = self._build_confirmed_payload_locked(
                    now,
                    include_presence_points=False,
                    include_metrics=False,
                )
                self._annotated_jpeg = self._encode_frame(frame)
                self._annotated_depth_jpeg = None
                self._last_render_at = 0.0
                self._render_seq = 0
                self._distance_smooth_cache = {}
                self._cleanup_pose_fusion_tracks(now, reset=True)
                self._status.update(
                    {
                        'status': 'unavailable',
                        'message': str(exc),
                        'temporary_people': [],
                        'confirmed_people': confirmed_payload,
                        'students': confirmed_payload,
                        'temporary_count': 0,
                        'confirmed_present_count': 0,
                        'confirmed_total_count': len(confirmed_payload),
                        'recognized_count': 0,
                        'yolo_person_present': False,
                    }
                )
            return

        with self._lock:
            self._match_detections_locked(person_boxes, now)
            self._update_classroom_metrics_locked(frame.shape, depth_frame, now)
            self._update_summary_locked(now)
            should_render = (
                self._annotated_jpeg is None
                or (now - self._last_render_at) >= self.RENDER_INTERVAL
            )
            if should_render:
                annotated = self._annotate_frame_locked(frame)
                self._annotated_jpeg = self._encode_frame(annotated)
                depth_annotated = self._annotate_depth_frame_locked(
                    depth_visual_frame,
                    depth_frame,
                    frame.shape,
                )
                self._annotated_depth_jpeg = self._encode_frame(depth_annotated) if depth_annotated is not None else None
                self._last_render_at = now
                self._render_seq += 1

    def _loop(self):
        while self._running:
            if not self.get_attendance_mode():
                self._refresh_idle_status(time.time())
                self._wake_event.wait(self.IDLE_LOOP_INTERVAL)
                self._wake_event.clear()
                continue

            frame = None
            depth_frame = None
            depth_visual_frame = None
            frame_timestamp = None

            frame_bundle = None
            frame_seq = 0
            frame_timestamp_candidate = 0.0
            if hasattr(self.kinect_service, 'get_latest_frame_marker'):
                frame_marker = self.kinect_service.get_latest_frame_marker()
                frame_seq = int(frame_marker.get('frame_seq') or 0)
                frame_timestamp_candidate = float(frame_marker.get('timestamp') or 0.0)
                if (
                    frame_seq <= self._last_source_frame_seq
                    and frame_timestamp_candidate <= (self._last_source_frame_timestamp + 1e-6)
                ):
                    self._wake_event.wait(self.LOOP_INTERVAL * 0.5)
                    self._wake_event.clear()
                    continue

            if hasattr(self.kinect_service, 'get_latest_frame_bundle'):
                frame_bundle = self.kinect_service.get_latest_frame_bundle()

            if frame_bundle is not None:
                frame_seq = int(frame_bundle.get('frame_seq') or frame_seq)
                frame_timestamp_candidate = float(frame_bundle.get('timestamp') or frame_timestamp_candidate)
                if (
                    frame_seq <= self._last_source_frame_seq
                    and frame_timestamp_candidate <= (self._last_source_frame_timestamp + 1e-6)
                ):
                    self._wake_event.wait(self.LOOP_INTERVAL * 0.5)
                    self._wake_event.clear()
                    continue

                frame = frame_bundle.get('color_frame')
                depth_frame = frame_bundle.get('depth_raw_frame')
                depth_visual_frame = frame_bundle.get('depth_visual_frame')
                frame_timestamp = frame_timestamp_candidate
                self._last_source_frame_seq = frame_seq
                self._last_source_frame_timestamp = frame_timestamp_candidate
            else:
                frame = self.kinect_service.get_latest_color_frame()

            if frame is None:
                self._wake_event.wait(self.LOOP_INTERVAL)
                self._wake_event.clear()
                continue

            try:
                self._process_frame(
                    frame,
                    depth_frame=depth_frame,
                    depth_visual_frame=depth_visual_frame,
                    frame_timestamp=frame_timestamp,
                )
            except Exception as exc:  # pylint: disable=broad-except
                with self._lock:
                    confirmed_payload = self._build_confirmed_payload_locked(
                        time.time(),
                        include_presence_points=False,
                        include_metrics=False,
                    )
                    self._annotated_jpeg = None
                    self._annotated_depth_jpeg = None
                    self._distance_smooth_cache = {}
                    self._cleanup_pose_fusion_tracks(time.time(), reset=True)
                    self._status.update(
                        {
                            'status': 'error',
                            'message': str(exc),
                            'temporary_people': [],
                            'confirmed_people': confirmed_payload,
                            'students': confirmed_payload,
                            'temporary_count': 0,
                            'confirmed_present_count': 0,
                            'confirmed_total_count': len(confirmed_payload),
                            'recognized_count': 0,
                            'yolo_person_present': False,
                        }
                    )
            self._wake_event.wait(self.LOOP_INTERVAL)
            self._wake_event.clear()
