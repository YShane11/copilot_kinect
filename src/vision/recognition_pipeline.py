import threading
import time
import os
from pathlib import Path


class RecognitionPipeline:
    PERSON_CLASS_ID = 0
    TRACK_IOU_THRESHOLD = 0.22
    TRACK_LOST_TIMEOUT = 1.2
    RECOGNITION_THRESHOLD = 0.45
    LOOP_INTERVAL = 0.12
    FACE_ANALYSIS_INTERVAL = 0.2
    YOLO_TRACK_INTERVAL = 0.45
    MAX_INFERENCE_WIDTH = 640
    GUIDE_MATCH_THRESHOLD = 0.62
    MODEL_CANDIDATES = (
        ('yolo26s', 'models/yolo/yolo26s.pt'),
        ('yolov8n', 'models/yolo/yolov8n.pt'),
        ('legacy_yolo26s', 'yolo26s.pt'),
        ('legacy_yolov8n', 'yolov8n.pt'),
    )

    def __init__(self, base_dir, kinect_service, face_db):
        self.base_dir = Path(base_dir)
        self.kinect_service = kinect_service
        self.face_db = face_db
        self._cv_modules = None
        self._yolo_model = None
        self._yolo_error = None
        self._yolo_device = self._resolve_yolo_device()
        self._lock = threading.Lock()
        self._running = True
        self._attendance_mode = False
        self._annotated_jpeg = None
        self._tracks = []
        self._next_track_id = 1
        self._last_faces = []
        self._last_face_analysis_at = 0.0
        self._last_person_boxes = []
        self._last_person_detect_at = 0.0
        self._status = {
            'status': 'idle',
            'message': 'Attendance mode is off.',
            'students': [],
            'recognized_count': 0,
            'attendance_mode': False,
            'yolo_person_present': False,
            'detector_model': 'idle',
        }
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _get_cv_modules(self):
        if self._cv_modules is not None:
            return self._cv_modules

        import cv2  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel

        self._cv_modules = (cv2, np)
        return self._cv_modules

    def _get_yolo_model(self):
        if self._yolo_model is not None:
            return self._yolo_model
        if self._yolo_error is not None:
            raise RuntimeError(self._yolo_error)

        try:
            from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

            selected_model_path = None
            selected_model_name = ''
            for candidate_name, relative_path in self.MODEL_CANDIDATES:
                candidate_path = self.base_dir / relative_path
                if candidate_path.exists():
                    selected_model_path = candidate_path
                    selected_model_name = candidate_name
                    break

            if selected_model_path is None:
                raise FileNotFoundError('No YOLO model file was found.')

            self._yolo_model = YOLO(str(selected_model_path))
            self._update_status(detector_model=f'{selected_model_name} ({self._yolo_device_label()})')
        except Exception as exc:  # pylint: disable=broad-except
            self._yolo_error = f'YOLO is not available: {exc}'
            raise RuntimeError(self._yolo_error) from exc
        return self._yolo_model

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

    def set_attendance_mode(self, enabled):
        with self._lock:
            self._attendance_mode = enabled
            self._status['attendance_mode'] = enabled

        if not enabled:
            self._tracks = []
            self._last_faces = []
            self._last_person_boxes = []
            self._last_face_analysis_at = 0.0
            self._last_person_detect_at = 0.0
            self._annotated_jpeg = None
            self._update_status(
                status='idle',
                message='Attendance mode is off.',
                students=[],
                recognized_count=0,
                yolo_person_present=False,
                detector_model='idle',
            )
        else:
            self._update_status(
                status='attendance_ready',
                message='Attendance mode is on. Align a face inside the guide frame.',
            )

    def get_attendance_mode(self):
        with self._lock:
            return self._attendance_mode

    def get_latest_color_jpeg(self):
        with self._lock:
            if not self._attendance_mode and self._annotated_jpeg is None:
                return self.kinect_service.get_latest_jpeg('color')
            return self._annotated_jpeg or self.kinect_service.get_latest_jpeg('color')

    def get_status(self):
        with self._lock:
            return {
                'status': self._status['status'],
                'message': self._status['message'],
                'students': [item.copy() for item in self._status['students']],
                'recognized_count': self._status['recognized_count'],
                'attendance_mode': self._status['attendance_mode'],
                'yolo_person_present': self._status['yolo_person_present'],
                'detector_model': self._status['detector_model'],
            }

    def mjpeg_stream(self):
        while True:
            payload = self.get_latest_color_jpeg()
            if payload is not None:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + payload + b'\r\n'
            time.sleep(0.08)

    def _encode_frame(self, frame):
        cv2, _ = self._get_cv_modules()
        ok, buffer = cv2.imencode('.jpg', frame)
        if not ok:
            return None
        return buffer.tobytes()

    def _update_status(self, **kwargs):
        with self._lock:
            self._status.update(kwargs)

    def _set_annotated_frame(self, frame):
        payload = self._encode_frame(frame)
        with self._lock:
            self._annotated_jpeg = payload

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

    def _guide_box(self, frame_shape):
        height, width = frame_shape[:2]
        guide_width = int(width * 0.28)
        guide_height = int(height * 0.42)
        left = (width - guide_width) // 2
        top = (height - guide_height) // 2 - 10
        return [left, top, left + guide_width, top + guide_height]

    def _face_matches_guide(self, face_bbox, guide_bbox):
        overlap = self._bbox_iou(face_bbox, guide_bbox)
        center_x = (face_bbox[0] + face_bbox[2]) / 2.0
        center_y = (face_bbox[1] + face_bbox[3]) / 2.0
        center_inside = (
            guide_bbox[0] <= center_x <= guide_bbox[2]
            and guide_bbox[1] <= center_y <= guide_bbox[3]
        )
        return overlap >= self.GUIDE_MATCH_THRESHOLD or center_inside

    def _get_face_analysis(self, frame, now):
        if now - self._last_face_analysis_at < self.FACE_ANALYSIS_INTERVAL:
            return self._last_faces

        resized, scale = self._resize_for_inference(frame)
        analysis = self.face_db.analyze_faces(resized)
        if analysis['status'] == 'unavailable':
            raise RuntimeError(analysis['message'])

        faces = analysis['faces']
        if scale != 1.0:
            for face in faces:
                face['bbox'] = [value / scale for value in face['bbox']]

        self._last_faces = faces
        self._last_face_analysis_at = now
        return self._last_faces

    def _detect_people_for_tracks(self, frame, now):
        if not self._tracks:
            self._last_person_boxes = []
            return []
        if now - self._last_person_detect_at < self.YOLO_TRACK_INTERVAL:
            return self._last_person_boxes

        model = self._get_yolo_model()
        resized, scale = self._resize_for_inference(frame)
        result = model(
            resized,
            verbose=False,
            classes=[self.PERSON_CLASS_ID],
            imgsz=256,
            conf=0.35,
            max_det=4,
            device=self._yolo_device,
        )[0]
        boxes = []
        factor = 1.0 / scale
        for box in result.boxes:
            xyxy = [float(value) * factor for value in box.xyxy[0].tolist()]
            boxes.append(xyxy)

        self._last_person_boxes = boxes
        self._last_person_detect_at = now
        self._update_status(detector_model=self._status.get('detector_model', 'yolo'))
        return boxes

    def _match_tracks(self, faces, now):
        matched_faces = []
        assigned_track_ids = set()

        for face in faces:
            best_track = None
            best_iou = 0.0
            for track in self._tracks:
                if track['id'] in assigned_track_ids:
                    continue
                iou = self._bbox_iou(face['bbox'], track['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            if best_track is not None and best_iou >= self.TRACK_IOU_THRESHOLD:
                best_track['bbox'] = face['bbox']
                best_track['last_seen'] = now
                assigned_track_ids.add(best_track['id'])
                matched_faces.append(
                    {
                        'bbox': face['bbox'],
                        'display_name': best_track['display_name'],
                        'person_bbox': best_track.get('person_bbox'),
                    }
                )

        self._tracks = [track for track in self._tracks if now - track['last_seen'] <= self.TRACK_LOST_TIMEOUT]
        return matched_faces

    def _add_track(self, face, match, now):
        track = {
            'id': self._next_track_id,
            'label': match['label'],
            'display_name': match['display_name'],
            'student_id': match.get('student_id', ''),
            'department': match.get('department', ''),
            'title': match.get('title', ''),
            'similarity': match['similarity'],
            'bbox': face['bbox'],
            'person_bbox': None,
            'last_seen': now,
        }
        self._next_track_id += 1
        self._tracks.append(track)
        return track

    def _attach_person_boxes(self, person_boxes):
        for track in self._tracks:
            track['person_bbox'] = None
            face_center_x = (track['bbox'][0] + track['bbox'][2]) / 2.0
            face_center_y = (track['bbox'][1] + track['bbox'][3]) / 2.0
            for person_bbox in person_boxes:
                if (
                    person_bbox[0] <= face_center_x <= person_bbox[2]
                    and person_bbox[1] <= face_center_y <= person_bbox[3]
                ):
                    track['person_bbox'] = person_bbox
                    break

    def _annotate_frame(self, frame, guide_bbox, matched_faces, aligned_face):
        cv2, _ = self._get_cv_modules()
        annotated = frame.copy()

        overlay = annotated.copy()
        guide_color = (86, 98, 134) if aligned_face is None else (78, 161, 255)
        cv2.rectangle(
            overlay,
            (guide_bbox[0], guide_bbox[1]),
            (guide_bbox[2], guide_bbox[3]),
            guide_color,
            2,
        )
        cv2.addWeighted(overlay, 0.36, annotated, 0.64, 0, annotated)

        for item in matched_faces:
            if item.get('person_bbox') is not None:
                px1, py1, px2, py2 = [int(value) for value in item['person_bbox']]
                cv2.rectangle(annotated, (px1, py1), (px2, py2), (44, 92, 176), 1)

            x1, y1, x2, y2 = [int(value) for value in item['bbox']]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (78, 161, 255), 2)
            cv2.rectangle(annotated, (x1, max(0, y1 - 28)), (min(annotated.shape[1] - 1, x1 + 220), y1), (13, 20, 36), -1)
            cv2.putText(
                annotated,
                item['display_name'],
                (x1 + 8, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (244, 247, 255),
                2,
            )

        return annotated

    def _build_student_payload(self):
        return [
            {
                'label': track['label'],
                'display_name': track['display_name'],
                'student_id': track['student_id'],
                'department': track['department'],
                'title': track['title'],
                'similarity': track['similarity'],
            }
            for track in sorted(self._tracks, key=lambda item: item['display_name'])
        ]

    def _process_frame(self, frame):
        now = time.time()
        kinect_status = self.kinect_service.get_status()
        if kinect_status['status'] != 'connected':
            self._tracks = []
            self._last_faces = []
            self._last_person_boxes = []
            self._annotated_jpeg = None
            self._update_status(
                status='camera_unavailable',
                message=kinect_status['message'],
                students=[],
                recognized_count=0,
                yolo_person_present=False,
                detector_model='idle',
            )
            return

        if not self.get_attendance_mode():
            self._tracks = []
            self._last_faces = []
            self._last_person_boxes = []
            self._annotated_jpeg = None
            self._update_status(
                status='idle',
                message='Attendance mode is off.',
                students=[],
                recognized_count=0,
                yolo_person_present=False,
                detector_model='idle',
            )
            return

        guide_bbox = self._guide_box(frame.shape)
        try:
            faces = self._get_face_analysis(frame, now)
        except RuntimeError as exc:
            self._tracks = []
            self._last_faces = []
            self._set_annotated_frame(frame)
            self._update_status(
                status='unavailable',
                message=str(exc),
                students=[],
                recognized_count=0,
                yolo_person_present=False,
            )
            return

        matched_faces = self._match_tracks(faces, now)
        aligned_face = None
        message = 'Place a face inside the guide frame.'
        status = 'attendance_ready'

        if not self._tracks:
            for face in faces:
                if self._face_matches_guide(face['bbox'], guide_bbox):
                    aligned_face = face
                    break

            if aligned_face is not None:
                matches = self.face_db.match_embedding(aligned_face['embedding'], threshold=self.RECOGNITION_THRESHOLD)
                if matches:
                    track = self._add_track(aligned_face, matches[0], now)
                    matched_faces.append(
                        {
                            'bbox': aligned_face['bbox'],
                            'display_name': track['display_name'],
                            'person_bbox': None,
                        }
                    )
                    message = f"{track['display_name']} recognized. Tracking started."
                    status = 'tracking'
                else:
                    message = 'Face is aligned, but no matching student was found.'
                    status = 'unknown'
            elif faces:
                message = 'Align the face with the guide frame to start attendance.'
                status = 'aligning'
        else:
            try:
                person_boxes = self._detect_people_for_tracks(frame, now)
            except RuntimeError:
                person_boxes = []

            self._attach_person_boxes(person_boxes)
            for item in matched_faces:
                for track in self._tracks:
                    if track['display_name'] == item['display_name']:
                        item['person_bbox'] = track.get('person_bbox')
                        break

            if self._tracks:
                message = 'Recognized students are being tracked.'
                status = 'tracking'
            else:
                message = 'Tracking lost. Align the face with the guide frame again.'
                status = 'attendance_ready'

        annotated = self._annotate_frame(frame, guide_bbox, matched_faces, aligned_face)
        self._set_annotated_frame(annotated)
        self._update_status(
            status=status,
            message=message,
            students=self._build_student_payload(),
            recognized_count=len(self._tracks),
            yolo_person_present=bool(self._last_person_boxes),
            attendance_mode=True,
        )

    def _loop(self):
        while self._running:
            frame = self.kinect_service.get_latest_color_frame()
            if frame is None:
                time.sleep(self.LOOP_INTERVAL)
                continue

            try:
                self._process_frame(frame)
            except Exception as exc:  # pylint: disable=broad-except
                self._tracks = []
                self._annotated_jpeg = None
                self._update_status(
                    status='error',
                    message=str(exc),
                    students=[],
                    recognized_count=0,
                    yolo_person_present=False,
                )
            time.sleep(self.LOOP_INTERVAL)
