import math
import os
import time
from dataclasses import dataclass, field


def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def _distance(left, right):
    delta_x = float(left[0]) - float(right[0])
    delta_y = float(left[1]) - float(right[1])
    return (delta_x * delta_x + delta_y * delta_y) ** 0.5


def _format_time_label(timestamp):
    return time.strftime('%H:%M:%S', time.localtime(timestamp))


def _trim_time_series(points, now, window_seconds):
    threshold = float(now) - float(window_seconds)
    return [item for item in points if float(item[0]) >= threshold]


def _trim_metric_rows(rows, now, window_seconds):
    threshold = float(now) - float(window_seconds)
    return [item for item in rows if float(item.get('t', 0.0)) >= threshold]


def _safe_mean(values):
    if not values:
        return 0.0
    return sum(float(item) for item in values) / float(len(values))


def _safe_std(values):
    if len(values) <= 1:
        return 0.0
    mean_value = _safe_mean(values)
    variance = _safe_mean([(float(item) - mean_value) ** 2 for item in values])
    return variance ** 0.5


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


@dataclass
class StudentMetricState:
    last_emit_at: float = 0.0
    last_update_at: float = 0.0
    head_pose_window: list = field(default_factory=list)
    stillness_window: list = field(default_factory=list)
    hand_raise_events: list = field(default_factory=list)
    hand_raise_active: bool = False
    hand_raise_started_at: float = 0.0
    hand_raise_counted_active: bool = False
    hand_raise_pending_events: int = 0
    previous_keypoints: dict = field(default_factory=dict)
    previous_bbox_center: tuple | None = None
    previous_frame_at: float = 0.0
    depth_baseline_cm: float | None = None
    metric_rows: dict = field(
        default_factory=lambda: {
            'focus-ratio': [],
            'head-stability': [],
            'fatigue': [],
            'posture-angle': [],
            'desk-distance': [],
            'stillness': [],
            'hand-raise': [],
            'shared-attention': [],
        }
    )


class PoseDepthMetricEngine:
    HISTORY_SECONDS = 300.0
    EMIT_INTERVAL_SECONDS = 1.0
    HEAD_WINDOW_SECONDS = 12.0
    STILLNESS_WINDOW_SECONDS = 90.0
    HAND_RAISE_EVENT_COOLDOWN_SECONDS = 2.0
    HAND_RAISE_MIN_HOLD_SECONDS = _env_float('KINECT_HAND_RAISE_MIN_HOLD_SECONDS', 0.60)
    HAND_RAISE_KEYPOINT_MIN_CONFIDENCE = _env_float('KINECT_HAND_RAISE_KEYPOINT_MIN_CONFIDENCE', 0.20)
    HAND_RAISE_WRIST_SHOULDER_MARGIN = _env_float('KINECT_HAND_RAISE_WRIST_SHOULDER_MARGIN', 0.18)
    HAND_RAISE_STRICT_WRIST_SHOULDER_MARGIN = _env_float('KINECT_HAND_RAISE_STRICT_WRIST_SHOULDER_MARGIN', 0.32)
    HAND_RAISE_WRIST_ELBOW_MARGIN = _env_float('KINECT_HAND_RAISE_WRIST_ELBOW_MARGIN', 0.08)
    HAND_RAISE_MAX_ELBOW_BELOW_SHOULDER = _env_float('KINECT_HAND_RAISE_MAX_ELBOW_BELOW_SHOULDER', 0.12)
    HAND_RAISE_MIN_FOREARM_VERTICAL_RATIO = _env_float('KINECT_HAND_RAISE_MIN_FOREARM_VERTICAL_RATIO', 0.42)
    HAND_RAISE_MAX_WRIST_SHOULDER_X_RATIO = _env_float('KINECT_HAND_RAISE_MAX_WRIST_SHOULDER_X_RATIO', 1.35)
    DEPTH_SAMPLE_RADIUS = 7
    DEPTH_MIN_VALID_CM = 35.0
    DEPTH_MAX_VALID_CM = 450.0
    DEPTH_DISTANCE_SCALE = _env_float('KINECT_DEPTH_DISTANCE_SCALE', 1.0)
    DEPTH_DISTANCE_BIAS_CM = _env_float('KINECT_DEPTH_DISTANCE_BIAS_CM', 0.0)
    DESK_DISTANCE_SCALE = _env_float('KINECT_DESK_DISTANCE_SCALE', 0.8)
    DESK_DISTANCE_MIN_CM = _env_float('KINECT_DESK_DISTANCE_MIN_CM', 30.0)
    DESK_DISTANCE_MAX_CM = _env_float('KINECT_DESK_DISTANCE_MAX_CM', 95.0)
    STILLNESS_VELOCITY_THRESHOLD = _env_float('KINECT_STILLNESS_VELOCITY_THRESHOLD', 0.16)
    STILLNESS_SCORE_SCALE = _env_float('KINECT_STILLNESS_SCORE_SCALE', 76.0)
    KINECT_RAW_MIN = 300.0
    KINECT_RAW_MAX = 1080.0
    KINECT_RAW_COEFF_A = -0.0030711016
    KINECT_RAW_COEFF_B = 3.3309495161

    COCO_NOSE = 0
    COCO_LEFT_EYE = 1
    COCO_RIGHT_EYE = 2
    COCO_LEFT_SHOULDER = 5
    COCO_RIGHT_SHOULDER = 6
    COCO_LEFT_ELBOW = 7
    COCO_RIGHT_ELBOW = 8
    COCO_LEFT_WRIST = 9
    COCO_RIGHT_WRIST = 10
    COCO_LEFT_HIP = 11
    COCO_RIGHT_HIP = 12

    def __init__(self):
        self._students = {}

    def reset(self):
        self._students = {}

    def _get_state(self, user_id):
        state = self._students.get(user_id)
        if state is None:
            state = StudentMetricState()
            self._students[user_id] = state
        return state

    def _point_from_keypoint(self, keypoints, keypoint_confidence, index, min_confidence=0.12):
        if keypoints is None or index >= len(keypoints):
            return None
        point = keypoints[index]
        if point is None or len(point) < 2:
            return None
        if keypoint_confidence and index < len(keypoint_confidence):
            if float(keypoint_confidence[index]) < min_confidence:
                return None
        x_value = float(point[0])
        y_value = float(point[1])
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            return None
        return (x_value, y_value)

    def _mean_point(self, points):
        valid = [item for item in points if item is not None]
        if not valid:
            return None
        return (
            _safe_mean([item[0] for item in valid]),
            _safe_mean([item[1] for item in valid]),
        )

    def _bbox_center(self, bbox):
        return (
            (float(bbox[0]) + float(bbox[2])) / 2.0,
            (float(bbox[1]) + float(bbox[3])) / 2.0,
        )

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
            distance_cm = value / 10.0

        distance_cm = (distance_cm * self.DEPTH_DISTANCE_SCALE) + self.DEPTH_DISTANCE_BIAS_CM
        if distance_cm < self.DEPTH_MIN_VALID_CM or distance_cm > self.DEPTH_MAX_VALID_CM:
            return None
        return float(distance_cm)

    def _depth_at(self, depth_frame, point, source_mode='kinect'):
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
            # Video fallback depth is pseudo-colored. Convert to a relative gray level.
            patch = patch[..., 0] * 0.114 + patch[..., 1] * 0.587 + patch[..., 2] * 0.299
            for item in patch.reshape(-1):
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

    def _append_metric(self, rows, now, label, value):
        rows.append(
            {
                't': float(now),
                'time': _format_time_label(now),
                'label': label,
                'value': value,
            }
        )

    def _derive_shared_attention_label(self, focus_score, yaw_proxy, nose_point, peer_centers, frame_width):
        if nose_point is None:
            return '看別處'

        if peer_centers:
            nearest_peer = min(peer_centers, key=lambda item: _distance(item, nose_point))
            peer_distance = _distance(nearest_peer, nose_point)
            looking_towards_peer = (
                (nearest_peer[0] > nose_point[0] and yaw_proxy > 0.22)
                or (nearest_peer[0] < nose_point[0] and yaw_proxy < -0.22)
            )
            if looking_towards_peer and peer_distance <= frame_width * 0.35:
                return '看同學'

        if focus_score >= 55.0:
            return '看老師'
        return '看別處'

    def _calculate_stillness_flag(self, state, keypoint_map, bbox_center, torso_length, now):
        if state.previous_frame_at <= 0.0:
            state.previous_keypoints = dict(keypoint_map)
            state.previous_bbox_center = bbox_center
            state.previous_frame_at = now
            return False

        time_delta = max(1e-3, float(now) - float(state.previous_frame_at))
        tracked_indices = [
            self.COCO_NOSE,
            self.COCO_LEFT_SHOULDER,
            self.COCO_RIGHT_SHOULDER,
            self.COCO_LEFT_HIP,
            self.COCO_RIGHT_HIP,
            self.COCO_LEFT_WRIST,
            self.COCO_RIGHT_WRIST,
        ]
        displacement_values = []
        for keypoint_index in tracked_indices:
            current_point = keypoint_map.get(keypoint_index)
            previous_point = state.previous_keypoints.get(keypoint_index)
            if current_point is None or previous_point is None:
                continue
            displacement_values.append(_distance(current_point, previous_point))

        keypoint_displacement = _safe_mean(displacement_values)
        center_displacement = (
            _distance(bbox_center, state.previous_bbox_center)
            if state.previous_bbox_center is not None
            else 0.0
        )
        if len(displacement_values) < 3:
            state.previous_keypoints = dict(keypoint_map)
            state.previous_bbox_center = bbox_center
            state.previous_frame_at = now
            return False

        normalized_motion = (keypoint_displacement + (center_displacement * 0.8)) / max(18.0, torso_length)
        normalized_velocity = normalized_motion / time_delta

        state.previous_keypoints = dict(keypoint_map)
        state.previous_bbox_center = bbox_center
        state.previous_frame_at = now
        return normalized_velocity < float(self.STILLNESS_VELOCITY_THRESHOLD)

    def _is_hand_raised(self, wrist_point, elbow_point, shoulder_point, shoulder_center, torso_length):
        if wrist_point is None or elbow_point is None or shoulder_point is None or shoulder_center is None:
            return False

        body_scale = max(24.0, float(torso_length))
        shoulder_y = shoulder_point[1]
        wrist_y = wrist_point[1]

        if abs(float(wrist_point[0]) - float(shoulder_point[0])) > body_scale * float(self.HAND_RAISE_MAX_WRIST_SHOULDER_X_RATIO):
            return False

        wrist_above_shoulder = wrist_y < (
            shoulder_y - (body_scale * float(self.HAND_RAISE_WRIST_SHOULDER_MARGIN))
        )
        if not wrist_above_shoulder:
            return False

        wrist_clearly_high = wrist_y < (
            shoulder_y - (body_scale * float(self.HAND_RAISE_STRICT_WRIST_SHOULDER_MARGIN))
        )
        elbow_y = elbow_point[1]
        wrist_above_elbow = wrist_y < (
            elbow_y - (body_scale * float(self.HAND_RAISE_WRIST_ELBOW_MARGIN))
        )
        if not wrist_above_elbow:
            return False

        forearm_length = max(1.0, _distance(wrist_point, elbow_point))
        forearm_vertical_ratio = (float(elbow_y) - float(wrist_y)) / forearm_length
        if (
            forearm_vertical_ratio < float(self.HAND_RAISE_MIN_FOREARM_VERTICAL_RATIO)
            and not wrist_clearly_high
        ):
            return False

        elbow_not_drooping = elbow_y < (
            shoulder_y + (body_scale * float(self.HAND_RAISE_MAX_ELBOW_BELOW_SHOULDER))
        )
        return bool(elbow_not_drooping or wrist_clearly_high)

    def update_student(self, user_id, frame_shape, bbox, pose_detection, depth_frame, depth_source_mode, peer_centers, now):
        state = self._get_state(user_id)
        state.last_update_at = float(now)

        keypoints = pose_detection.get('keypoints') if pose_detection else None
        keypoint_confidence = pose_detection.get('keypoint_conf') if pose_detection else None
        keypoint_map = {}

        nose_point = self._point_from_keypoint(keypoints, keypoint_confidence, self.COCO_NOSE)
        left_eye_point = self._point_from_keypoint(keypoints, keypoint_confidence, self.COCO_LEFT_EYE)
        right_eye_point = self._point_from_keypoint(keypoints, keypoint_confidence, self.COCO_RIGHT_EYE)
        left_shoulder_point = self._point_from_keypoint(keypoints, keypoint_confidence, self.COCO_LEFT_SHOULDER)
        right_shoulder_point = self._point_from_keypoint(keypoints, keypoint_confidence, self.COCO_RIGHT_SHOULDER)
        left_elbow_point = self._point_from_keypoint(keypoints, keypoint_confidence, self.COCO_LEFT_ELBOW)
        right_elbow_point = self._point_from_keypoint(keypoints, keypoint_confidence, self.COCO_RIGHT_ELBOW)
        left_wrist_point = self._point_from_keypoint(keypoints, keypoint_confidence, self.COCO_LEFT_WRIST)
        right_wrist_point = self._point_from_keypoint(keypoints, keypoint_confidence, self.COCO_RIGHT_WRIST)
        left_hip_point = self._point_from_keypoint(keypoints, keypoint_confidence, self.COCO_LEFT_HIP)
        right_hip_point = self._point_from_keypoint(keypoints, keypoint_confidence, self.COCO_RIGHT_HIP)

        for keypoint_index, point in (
            (self.COCO_NOSE, nose_point),
            (self.COCO_LEFT_SHOULDER, left_shoulder_point),
            (self.COCO_RIGHT_SHOULDER, right_shoulder_point),
            (self.COCO_LEFT_HIP, left_hip_point),
            (self.COCO_RIGHT_HIP, right_hip_point),
            (self.COCO_LEFT_WRIST, left_wrist_point),
            (self.COCO_RIGHT_WRIST, right_wrist_point),
        ):
            if point is not None:
                keypoint_map[keypoint_index] = point

        shoulder_center = self._mean_point([left_shoulder_point, right_shoulder_point])
        hip_center = self._mean_point([left_hip_point, right_hip_point])
        if shoulder_center is None:
            shoulder_center = (
                (float(bbox[0]) + float(bbox[2])) / 2.0,
                float(bbox[1]) + ((float(bbox[3]) - float(bbox[1])) * 0.28),
            )
        if hip_center is None:
            hip_center = (
                (float(bbox[0]) + float(bbox[2])) / 2.0,
                float(bbox[1]) + ((float(bbox[3]) - float(bbox[1])) * 0.72),
            )
        if nose_point is None:
            nose_point = (
                (float(bbox[0]) + float(bbox[2])) / 2.0,
                float(bbox[1]) + ((float(bbox[3]) - float(bbox[1])) * 0.18),
            )

        shoulder_width = (
            _distance(left_shoulder_point, right_shoulder_point)
            if left_shoulder_point is not None and right_shoulder_point is not None
            else max(20.0, float(bbox[2]) - float(bbox[0]))
        )
        torso_length = max(24.0, _distance(shoulder_center, hip_center))
        bbox_center = self._bbox_center(bbox)

        if left_eye_point is not None and right_eye_point is not None:
            eye_center = self._mean_point([left_eye_point, right_eye_point])
            eye_distance = max(8.0, _distance(left_eye_point, right_eye_point))
            yaw_proxy = (nose_point[0] - eye_center[0]) / eye_distance
        else:
            yaw_proxy = (nose_point[0] - shoulder_center[0]) / max(10.0, shoulder_width * 0.5)

        pitch_proxy = (shoulder_center[1] - nose_point[1]) / max(14.0, torso_length)
        focus_target = (frame_shape[1] * 0.5, frame_shape[0] * 0.2)
        focus_distance_ratio = _clamp(_distance(nose_point, focus_target) / max(1.0, frame_shape[1] * 0.55), 0.0, 1.0)
        focus_orientation_penalty = _clamp(abs(yaw_proxy), 0.0, 1.0)
        focus_score = _clamp((1.0 - (0.55 * focus_distance_ratio + 0.45 * focus_orientation_penalty)) * 100.0, 0.0, 100.0)

        state.head_pose_window.append((float(now), float(yaw_proxy), float(pitch_proxy)))
        state.head_pose_window = _trim_time_series(state.head_pose_window, now, self.HEAD_WINDOW_SECONDS)
        head_yaw_std = _safe_std([item[1] for item in state.head_pose_window])
        head_pitch_std = _safe_std([item[2] for item in state.head_pose_window])
        head_stability = _clamp((head_yaw_std + head_pitch_std) * 55.0, 0.0, 40.0)

        trunk_dx = shoulder_center[0] - hip_center[0]
        trunk_dy = max(1.0, hip_center[1] - shoulder_center[1])
        posture_angle = _clamp(abs(math.degrees(math.atan2(trunk_dx, trunk_dy))), 0.0, 45.0)
        head_drop_ratio = _clamp((nose_point[1] - (shoulder_center[1] - torso_length * 0.58)) / max(10.0, torso_length * 0.92), 0.0, 1.0)

        depth_cm = self._depth_at(depth_frame, nose_point, source_mode=depth_source_mode)
        if depth_cm is None:
            bbox_height = max(16.0, float(bbox[3]) - float(bbox[1]))
            depth_cm = _clamp(8200.0 / bbox_height, 28.0, 120.0)
        posture_cos = _clamp(math.cos(head_drop_ratio * math.pi * 0.30), 0.78, 1.0)
        calibrated_depth_cm = float(depth_cm) * float(self.DESK_DISTANCE_SCALE) * posture_cos
        desk_distance = _clamp(calibrated_depth_cm, float(self.DESK_DISTANCE_MIN_CM), float(self.DESK_DISTANCE_MAX_CM))
        if state.depth_baseline_cm is None:
            state.depth_baseline_cm = float(desk_distance)
        else:
            state.depth_baseline_cm = (state.depth_baseline_cm * 0.98) + (float(desk_distance) * 0.02)

        still_flag = self._calculate_stillness_flag(state, keypoint_map, bbox_center, torso_length, now)
        low_focus_risk = _clamp((72.0 - float(focus_score)) / 38.0, 0.0, 1.0)
        head_drop_risk = _clamp((float(head_drop_ratio) - 0.38) / 0.42, 0.0, 1.0)
        daze_sample = 0.0
        if still_flag:
            daze_sample = 0.12 + (low_focus_risk * 0.58) + (head_drop_risk * 0.30)
            if focus_score >= 72.0 and head_drop_ratio < 0.40:
                daze_sample *= 0.30
        state.stillness_window.append((float(now), _clamp(daze_sample, 0.0, 1.0)))
        state.stillness_window = _trim_time_series(state.stillness_window, now, self.STILLNESS_WINDOW_SECONDS)
        stillness_ratio = _safe_mean([item[1] for item in state.stillness_window])
        stillness_score = _clamp((stillness_ratio ** 1.15) * float(self.STILLNESS_SCORE_SCALE), 0.0, 100.0)

        fatigue_score = _clamp((head_drop_ratio * 0.65 + (stillness_score / 100.0) * 0.35) * 100.0, 0.0, 100.0)

        hand_confidence = float(self.HAND_RAISE_KEYPOINT_MIN_CONFIDENCE)
        hand_left_shoulder_point = self._point_from_keypoint(
            keypoints,
            keypoint_confidence,
            self.COCO_LEFT_SHOULDER,
            min_confidence=hand_confidence,
        )
        hand_right_shoulder_point = self._point_from_keypoint(
            keypoints,
            keypoint_confidence,
            self.COCO_RIGHT_SHOULDER,
            min_confidence=hand_confidence,
        )
        hand_left_elbow_point = self._point_from_keypoint(
            keypoints,
            keypoint_confidence,
            self.COCO_LEFT_ELBOW,
            min_confidence=hand_confidence,
        )
        hand_right_elbow_point = self._point_from_keypoint(
            keypoints,
            keypoint_confidence,
            self.COCO_RIGHT_ELBOW,
            min_confidence=hand_confidence,
        )
        hand_left_wrist_point = self._point_from_keypoint(
            keypoints,
            keypoint_confidence,
            self.COCO_LEFT_WRIST,
            min_confidence=hand_confidence,
        )
        hand_right_wrist_point = self._point_from_keypoint(
            keypoints,
            keypoint_confidence,
            self.COCO_RIGHT_WRIST,
            min_confidence=hand_confidence,
        )

        left_hand_raised = self._is_hand_raised(
            wrist_point=hand_left_wrist_point,
            elbow_point=hand_left_elbow_point,
            shoulder_point=hand_left_shoulder_point,
            shoulder_center=shoulder_center,
            torso_length=torso_length,
        )
        right_hand_raised = self._is_hand_raised(
            wrist_point=hand_right_wrist_point,
            elbow_point=hand_right_elbow_point,
            shoulder_point=hand_right_shoulder_point,
            shoulder_center=shoulder_center,
            torso_length=torso_length,
        )
        hand_raise_active = bool(left_hand_raised or right_hand_raised)
        last_event_at = state.hand_raise_events[-1] if state.hand_raise_events else 0.0
        if hand_raise_active:
            if not state.hand_raise_active:
                state.hand_raise_started_at = float(now)
                state.hand_raise_counted_active = False
            held_seconds = float(now) - float(state.hand_raise_started_at or now)
            if (
                (not state.hand_raise_counted_active)
                and held_seconds >= float(self.HAND_RAISE_MIN_HOLD_SECONDS)
                and (float(now) - float(last_event_at) >= self.HAND_RAISE_EVENT_COOLDOWN_SECONDS)
            ):
                state.hand_raise_events.append(float(now))
                state.hand_raise_pending_events = int(state.hand_raise_pending_events) + 1
                state.hand_raise_counted_active = True
        else:
            state.hand_raise_started_at = 0.0
            state.hand_raise_counted_active = False
        state.hand_raise_active = hand_raise_active
        state.hand_raise_events = [item for item in state.hand_raise_events if float(item) >= float(now) - 60.0]

        shared_attention_label = self._derive_shared_attention_label(
            focus_score=focus_score,
            yaw_proxy=yaw_proxy,
            nose_point=nose_point,
            peer_centers=peer_centers,
            frame_width=frame_shape[1],
        )

        if float(now) - float(state.last_emit_at) < self.EMIT_INTERVAL_SECONDS:
            return

        state.last_emit_at = float(now)
        time_label = _format_time_label(now)

        self._append_metric(state.metric_rows['focus-ratio'], now, time_label, round(focus_score, 1))
        self._append_metric(state.metric_rows['head-stability'], now, time_label, round(head_stability, 1))
        self._append_metric(state.metric_rows['fatigue'], now, time_label, round(fatigue_score, 1))
        self._append_metric(state.metric_rows['posture-angle'], now, time_label, round(posture_angle, 1))
        self._append_metric(state.metric_rows['desk-distance'], now, time_label, round(desk_distance, 1))
        self._append_metric(state.metric_rows['stillness'], now, time_label, round(stillness_score, 1))
        hand_raise_count = int(state.hand_raise_pending_events)
        self._append_metric(state.metric_rows['hand-raise'], now, time_label, hand_raise_count)
        state.hand_raise_pending_events = 0
        self._append_metric(state.metric_rows['shared-attention'], now, shared_attention_label, 1)

        for key in tuple(state.metric_rows.keys()):
            state.metric_rows[key] = _trim_metric_rows(state.metric_rows[key], now, self.HISTORY_SECONDS)

    def get_user_metrics(self, user_id):
        state = self._students.get(user_id)
        if state is None:
            return {}
        return {
            key: [
                {
                    'time': row['time'],
                    'label': row['label'],
                    'value': row['value'],
                }
                for row in value
            ]
            for key, value in state.metric_rows.items()
        }
