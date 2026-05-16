import json
import os
import threading
from pathlib import Path

import atexit
import csv
import math
import time
from datetime import datetime, timedelta

import requests
from flask import Flask, Response, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from src.vision.face_recognition_db import FaceRecognitionDB
from src.vision.kinect_service import KinectService
from src.vision.attendance_pipeline import RecognitionPipeline

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
USERS_FILE = BASE_DIR / 'data' / 'administrators.json'
FACE_DB = FaceRecognitionDB(BASE_DIR)
KINECT_SERVICE = KinectService(BASE_DIR)
RECOGNITION_PIPELINE = RecognitionPipeline(BASE_DIR, KINECT_SERVICE, FACE_DB)
PENDING_TRAINING_CAPTURES = {}
PENDING_TRAINING_LOCK = threading.Lock()
POWER_AUTOMATE_UPLOAD_URL = os.environ.get('POWER_AUTOMATE_UPLOAD_URL', '').strip()


def load_users():
    with USERS_FILE.open(encoding='utf-8-sig') as users_file:
        return json.load(users_file)


def _parse_bool_query(value, default=False):
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if text in {'0', 'false', 'no', 'n', 'off'}:
        return False
    return default


def _normalize_course_item(item):
    if isinstance(item, str):
        name = item.strip()
        if not name:
            return None
        return {'id': name, 'name': name}

    if isinstance(item, dict):
        course_id = str(
            item.get('id')
            or item.get('course_id')
            or item.get('code')
            or item.get('課程代碼')
            or ''
        ).strip()
        course_name = str(
            item.get('name')
            or item.get('title')
            or item.get('course_name')
            or item.get('課程名稱')
            or ''
        ).strip()
        if not course_name:
            course_name = course_id
        if not course_name:
            return None
        if not course_id:
            course_id = course_name
        return {'id': course_id, 'name': course_name}

    return None


def extract_manager_courses(user):
    raw_fields = [
        user.get('courses'),
        user.get('課程'),
        user.get('課程列表'),
        user.get('負責課程'),
        user.get('classes'),
    ]

    normalized = []
    for field in raw_fields:
        if field is None:
            continue
        if isinstance(field, str):
            chunks = [
                token.strip()
                for token in field.replace('、', ',').replace('，', ',').replace('\n', ',').split(',')
                if token.strip()
            ]
            for chunk in chunks:
                item = _normalize_course_item(chunk)
                if item is not None:
                    normalized.append(item)
            continue
        if isinstance(field, list):
            for entry in field:
                item = _normalize_course_item(entry)
                if item is not None:
                    normalized.append(item)
            continue

        item = _normalize_course_item(field)
        if item is not None:
            normalized.append(item)

    deduped = []
    seen = set()
    for item in normalized:
        key = (item['id'].strip().lower(), item['name'].strip().lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    if deduped:
        return deduped

    return [{'id': 'default-course', 'name': '未指定課程'}]


def extract_default_course(user, courses):
    candidates = [
        user.get('default_course'),
        user.get('預設課程'),
    ]
    for candidate in candidates:
        normalized = _normalize_course_item(candidate)
        if normalized is None:
            continue
        for course in courses:
            if course['id'] == normalized['id'] or course['name'] == normalized['name']:
                return course
    return courses[0] if courses else {'id': '', 'name': ''}


def get_face_recognition_snapshot(include_training=False, include_metrics=True, metrics_user_id=None):
    snapshot = RECOGNITION_PIPELINE.get_status(
        include_metrics=include_metrics,
        metrics_user_id=metrics_user_id,
    )
    if include_training:
        training_students = FACE_DB.get_training_overview()
        snapshot['training_students'] = training_students
        snapshot['trained_labels'] = [
            item['label']
            for item in training_students
            if int(item.get('image_count', 0) or 0) > 0
        ]
    return snapshot


def get_training_snapshot():
    return {'students': FACE_DB.get_training_overview()}


def reset_pending_captures(temp_id):
    with PENDING_TRAINING_LOCK:
        if temp_id:
            PENDING_TRAINING_CAPTURES.pop(temp_id, None)


class MinuteStudentCsvExporter:
    INTERVAL_SECONDS = 60.0
    CSV_HEADERS = [
        'course_id',
        'course_name',
        'student_id',
        'student_name',
        'metric_key',
        'metric_name',
        'chart_type',
        'recorded_at',
        'label',
        'value',
    ]
    METRIC_METADATA = {
        'presence': {'name': '存在時間', 'chart_type': 'line'},
        'assignment-score': {'name': '作業成績', 'chart_type': 'bar'},
        'attendance-rate': {'name': '出席狀況', 'chart_type': 'bar'},
        'submission-punctuality': {'name': '作業繳交準時率', 'chart_type': 'pie'},
        'focus-ratio': {'name': '專注度（看老師比例）', 'chart_type': 'line'},
        'head-stability': {'name': '專注穩定度（頭部穩定）', 'chart_type': 'line'},
        'fatigue': {'name': '疲勞度（低頭）', 'chart_type': 'line'},
        'posture-angle': {'name': '上課投入度（身體前傾）', 'chart_type': 'line'},
        'desk-distance': {'name': '專心距離（頭與桌距離）', 'chart_type': 'line'},
        'stillness': {'name': '發呆指數（長時間不動）', 'chart_type': 'bar'},
        'hand-raise': {'name': '參與度（舉手次數）', 'chart_type': 'bar'},
        'shared-attention': {'name': '互動模式（看誰比較多）', 'chart_type': 'pie'},
    }
    HISTORY_METRIC_KEYS = (
        'assignment-score',
        'attendance-rate',
        'submission-punctuality',
    )

    def __init__(self, base_dir, pipeline):
        self.base_dir = Path(base_dir)
        self.pipeline = pipeline
        self.history_dir = self.base_dir / 'history'
        self._history_metric_rows_by_key = self._load_history_metric_rows()
        self._last_metric_recorded_at = {}
        self._next_export_at = 0.0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _sanitize_file_segment(self, value, fallback='未指定課程'):
        normalized = str(value or '').strip()
        if not normalized:
            normalized = fallback
        safe_chars = []
        for char in normalized:
            if char in '<>:"/\\|?*':
                safe_chars.append('-')
            elif ord(char) < 32:
                safe_chars.append('-')
            elif char.isspace():
                safe_chars.append('-')
            else:
                safe_chars.append(char)
        safe = ''.join(safe_chars)
        while '--' in safe:
            safe = safe.replace('--', '-')
        safe = safe.strip('-')
        return safe or fallback

    def _build_output_path(self, course_name, now):
        safe_course = self._sanitize_file_segment(course_name or '未指定課程')
        date_tag = datetime.fromtimestamp(float(now)).strftime('%Y-%m-%d')
        return self.history_dir / f'classroom-metrics-{safe_course}-{date_tag}.csv'

    def _load_history_metric_rows(self):
        rows_by_key = {}
        metrics_dir = self.base_dir / 'static' / 'mock_metrics'
        for metric_key in self.HISTORY_METRIC_KEYS:
            metric_rows = []
            csv_path = metrics_dir / f'{metric_key}.csv'
            if csv_path.exists() and csv_path.is_file():
                try:
                    with csv_path.open('r', encoding='utf-8-sig', newline='') as csv_file:
                        reader = csv.DictReader(csv_file)
                        for row in reader:
                            label = str(row.get('label') or row.get('time') or '').strip()
                            value_text = str(row.get('value') or '').strip()
                            if not label and not value_text:
                                continue
                            value = value_text
                            try:
                                numeric_value = float(value_text)
                                value = int(numeric_value) if numeric_value.is_integer() else numeric_value
                            except Exception:
                                value = value_text
                            metric_rows.append(
                                {
                                    'label': label,
                                    'value': value,
                                }
                            )
                except Exception:
                    metric_rows = []
            rows_by_key[metric_key] = metric_rows
        return rows_by_key

    def _stable_seed(self, value):
        seed = 2166136261
        for char in str(value or ''):
            seed ^= ord(char)
            seed = (seed * 16777619) & 0xFFFFFFFF
        return seed

    def _pseudo_random(self, seed, index=0):
        raw_value = math.sin((float(seed) + 1.0) * 12.9898 + (float(index) + 1.0) * 78.233) * 43758.5453
        return raw_value - math.floor(raw_value)

    def _recent_week_labels(self, now, count):
        current = datetime.fromtimestamp(float(now))
        safe_count = max(1, int(count or 1))
        labels = []
        for offset in reversed(range(safe_count)):
            date_value = current - timedelta(days=offset * 7)
            labels.append(date_value.strftime('%m/%d'))
        return labels

    def _build_student_attendance_rows(self, student_key, now, count):
        safe_count = max(4, int(count or 6))
        seed = self._stable_seed(f'{student_key}:attendance-rate:{datetime.fromtimestamp(float(now)).date()}')
        labels = self._recent_week_labels(now, safe_count)
        base_present_chance = 0.82 + (self._pseudo_random(seed, 2) * 0.14)
        rows = []
        for index, label in enumerate(labels):
            if index == safe_count - 1:
                value = 100
            else:
                present_chance = max(0.72, min(0.96, base_present_chance + ((self._pseudo_random(seed, index + 11) - 0.5) * 0.08)))
                value = 100 if self._pseudo_random(seed, index + 31) <= present_chance else 0
            rows.append({'label': label, 'value': value})

        absences = [index for index, row in enumerate(rows[:-1]) if int(row['value']) <= 0]
        while len(absences) > 2:
            flip_index = absences.pop(0)
            rows[flip_index]['value'] = 100
        return rows

    def _build_student_assignment_rows(self, student_key, now, count):
        safe_count = max(4, int(count or 6))
        seed = self._stable_seed(f'{student_key}:assignment-score:{datetime.fromtimestamp(float(now)).date()}')
        labels = self._recent_week_labels(now, safe_count)
        attendance_rows = self._build_student_attendance_rows(student_key, now, safe_count)
        attendance_rate = sum(float(row.get('value') or 0.0) for row in attendance_rows) / max(1.0, safe_count * 100.0)
        base_score = 72.0 + (attendance_rate * 15.0) + (self._pseudo_random(seed, 3) * 7.0)
        rows = []
        for index, label in enumerate(labels):
            progress = index / max(1, safe_count - 1)
            trend = (progress - 0.5) * (self._pseudo_random(seed, 7) * 6.0)
            noise = (self._pseudo_random(seed, index + 17) - 0.5) * 9.0
            absence_penalty = 4.0 if int(attendance_rows[index].get('value') or 0) <= 0 else 0.0
            score = max(68.0, min(98.0, base_score + trend + noise - absence_penalty))
            rows.append({'label': label, 'value': int(round(score))})
        return rows

    def _build_student_history_rows(self, metric_key, student_id, student_name, now, fallback_rows):
        student_key = f'{student_id}:{student_name}'
        fallback_count = len(fallback_rows or [])
        if metric_key == 'attendance-rate':
            return self._build_student_attendance_rows(student_key, now, fallback_count or 6)
        if metric_key == 'assignment-score':
            return self._build_student_assignment_rows(student_key, now, fallback_count or 6)
        return list(fallback_rows or [])

    def _coerce_json_integer(self, value):
        if isinstance(value, bool):
            return int(value)
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return 0
        if not math.isfinite(numeric_value):
            return 0
        return int(round(numeric_value))

    def _normalize_json_row(self, row):
        return {
            'course_id': str(row.get('course_id') or '').strip(),
            'course_name': str(row.get('course_name') or '').strip(),
            'student_id': str(row.get('student_id') or '').strip(),
            'student_name': str(row.get('student_name') or '').strip(),
            'metric_key': str(row.get('metric_key') or '').strip(),
            'metric_name': str(row.get('metric_name') or '').strip(),
            'chart_type': str(row.get('chart_type') or '').strip(),
            'recorded_at': str(row.get('recorded_at') or '').strip(),
            'label': str(row.get('label') or '').strip(),
            'value': self._coerce_json_integer(row.get('value')),
        }

    def _build_json_payload(self, rows, snapshot, now):
        normalized_rows = [
            self._normalize_json_row(row)
            for row in (rows or [])
            if isinstance(row, dict)
        ]
        if not normalized_rows:
            return None

        course = snapshot.get('current_course') or {}
        generated_at = datetime.fromtimestamp(float(now))
        course_id = str(course.get('course_id') or normalized_rows[0].get('course_id') or '').strip()
        course_name = str(
            course.get('course_name') or normalized_rows[0].get('course_name') or '未指定課程'
        ).strip()
        return {
            'generated_at': generated_at.isoformat(timespec='seconds'),
            'date': generated_at.strftime('%Y-%m-%d'),
            'course_id': course_id,
            'course_name': course_name,
            'row_count': len(normalized_rows),
            'rows': normalized_rows,
        }

    def _write_rows(self, output_path, rows):
        if not rows:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (not output_path.exists()) or output_path.stat().st_size == 0
        expected_header_line = ','.join(self.CSV_HEADERS)

        if output_path.exists() and output_path.stat().st_size > 0:
            try:
                with output_path.open('r', encoding='utf-8-sig', newline='') as csv_file:
                    current_header_line = (csv_file.readline() or '').strip('\r\n')
            except Exception:
                current_header_line = ''

            if current_header_line and current_header_line != expected_header_line:
                legacy_path = output_path.with_name(f'{output_path.stem}-legacy-format{output_path.suffix}')
                if legacy_path.exists():
                    timestamp = datetime.now().strftime('%H%M%S')
                    legacy_path = output_path.with_name(
                        f'{output_path.stem}-legacy-format-{timestamp}{output_path.suffix}'
                    )
                output_path.rename(legacy_path)
                write_header = True

        with output_path.open('a', encoding='utf-8-sig', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.CSV_HEADERS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _write_json_rows(self, output_path, rows, snapshot, now):
        if not rows:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        existing_rows = []
        if output_path.exists() and output_path.stat().st_size > 0:
            try:
                with output_path.open('r', encoding='utf-8') as json_file:
                    existing_payload = json.load(json_file)
                if isinstance(existing_payload, dict):
                    existing_rows = existing_payload.get('rows') or []
                elif isinstance(existing_payload, list):
                    existing_rows = existing_payload
            except Exception:
                existing_rows = []

        existing_rows = [row for row in existing_rows if isinstance(row, dict)]
        combined_rows = existing_rows + rows
        payload = self._build_json_payload(combined_rows, snapshot, now)
        if not payload:
            return None

        temp_path = output_path.with_name(f'{output_path.name}.tmp')
        with temp_path.open('w', encoding='utf-8') as json_file:
            json.dump(payload, json_file, ensure_ascii=False, indent=2)
            json_file.write('\n')
        temp_path.replace(output_path)
        return payload

    def _post_json_payload(self, payload):
        if not payload or not payload.get('rows'):
            return False
        if not POWER_AUTOMATE_UPLOAD_URL:
            return False
        try:
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            response = requests.post(
                POWER_AUTOMATE_UPLOAD_URL,
                data=body,
                headers={'Content-Type': 'application/json'},
                timeout=15,
            )
            response.raise_for_status()
            return True
        except Exception as exc:
            print(f'Power Automate metric upload failed: {exc}', flush=True)
            return False

    def _should_emit_metric_row(self, course_id, student_id, metric_key, day_tag, recorded_at):
        normalized_recorded_at = str(recorded_at or '').strip()
        if not normalized_recorded_at:
            return False
        normalized_day_tag = str(day_tag or '').strip()
        marker_key = (
            str(course_id or '').strip(),
            str(student_id or '').strip(),
            str(metric_key or '').strip(),
        )
        last_marker = self._last_metric_recorded_at.get(marker_key)
        if last_marker:
            last_day_tag, last_recorded_at = last_marker
            if normalized_day_tag == last_day_tag and normalized_recorded_at <= last_recorded_at:
                return False
        self._last_metric_recorded_at[marker_key] = (normalized_day_tag, normalized_recorded_at)
        return True

    def _build_export_rows(self, snapshot, now):
        confirmed_people = [
            person
            for person in (snapshot.get('confirmed_people') or [])
            if str(person.get('current_status') or '').strip().lower() == 'present'
        ]
        if not confirmed_people:
            return []

        course = snapshot.get('current_course') or {}
        course_id = str(course.get('course_id') or '').strip()
        course_name = str(course.get('course_name') or '').strip() or '未指定課程'
        day_tag = datetime.fromtimestamp(float(now)).strftime('%Y-%m-%d')
        rows = []

        for person in confirmed_people:
            student_id = str(person.get('student_id') or person.get('user_id') or '--').strip() or '--'
            student_name = str(person.get('display_name') or person.get('name') or '學生').strip() or '學生'

            for point in (person.get('presence_points') or []):
                try:
                    point_ts = float(point.get('t') or 0.0)
                except Exception:
                    point_ts = 0.0
                recorded_at = datetime.fromtimestamp(point_ts).strftime('%H:%M:%S') if point_ts > 0 else ''
                if not self._should_emit_metric_row(course_id, student_id, 'presence', day_tag, recorded_at):
                    continue
                point_value = 1 if float(point.get('v') or 0.0) > 0 else 0
                rows.append(
                    {
                        'course_id': course_id,
                        'course_name': course_name,
                        'student_id': student_id,
                        'student_name': student_name,
                        'metric_key': 'presence',
                        'metric_name': self.METRIC_METADATA['presence']['name'],
                        'chart_type': self.METRIC_METADATA['presence']['chart_type'],
                        'recorded_at': recorded_at,
                        'label': '在場中' if point_value > 0 else '已離開',
                        'value': point_value,
                    }
                )

            for metric_key, metric_rows in (person.get('classroom_metrics') or {}).items():
                metadata = self.METRIC_METADATA.get(
                    metric_key,
                    {'name': str(metric_key), 'chart_type': 'line'},
                )
                for metric_row in (metric_rows or []):
                    recorded_at = str(metric_row.get('time') or '').strip()
                    if not self._should_emit_metric_row(course_id, student_id, metric_key, day_tag, recorded_at):
                        continue
                    rows.append(
                        {
                            'course_id': course_id,
                            'course_name': course_name,
                            'student_id': student_id,
                            'student_name': student_name,
                            'metric_key': metric_key,
                            'metric_name': metadata['name'],
                            'chart_type': metadata['chart_type'],
                            'recorded_at': recorded_at,
                            'label': metric_row.get('label', ''),
                            'value': metric_row.get('value', ''),
                        }
                    )

            for history_metric_key in self.HISTORY_METRIC_KEYS:
                history_rows = self._build_student_history_rows(
                    history_metric_key,
                    student_id,
                    student_name,
                    now,
                    self._history_metric_rows_by_key.get(history_metric_key) or [],
                )
                if not history_rows:
                    continue
                metadata = self.METRIC_METADATA.get(
                    history_metric_key,
                    {'name': str(history_metric_key), 'chart_type': 'line'},
                )
                for index, history_row in enumerate(history_rows, start=1):
                    label_text = str(history_row.get('label') or '').strip()
                    recorded_at = label_text or day_tag
                    dedupe_metric_key = f'{history_metric_key}#{label_text or index}'
                    if not self._should_emit_metric_row(
                        course_id,
                        student_id,
                        dedupe_metric_key,
                        day_tag,
                        day_tag,
                    ):
                        continue
                    rows.append(
                        {
                            'course_id': course_id,
                            'course_name': course_name,
                            'student_id': student_id,
                            'student_name': student_name,
                            'metric_key': history_metric_key,
                            'metric_name': metadata['name'],
                            'chart_type': metadata['chart_type'],
                            'recorded_at': recorded_at,
                            'label': label_text,
                            'value': history_row.get('value', ''),
                        }
                    )

        return rows

    def _export_once(self, now):
        snapshot = self.pipeline.get_status(include_metrics=True, metrics_user_id=None)
        if not snapshot.get('attendance_mode'):
            return

        rows = self._build_export_rows(snapshot, now)
        if not rows:
            return

        current_course = snapshot.get('current_course') or {}
        output_path = self._build_output_path(current_course.get('course_name') or '', now)
        self._write_rows(output_path, rows)
        upload_payload = self._build_json_payload(rows, snapshot, now)
        self._write_json_rows(output_path.with_suffix('.json'), rows, snapshot, now)
        self._post_json_payload(upload_payload)

    def _loop(self):
        while not self._stop_event.is_set():
            now = time.time()
            if now >= self._next_export_at:
                self._next_export_at = now + self.INTERVAL_SECONDS
                try:
                    self._export_once(now)
                except Exception:
                    # Auto export should never interrupt the dashboard runtime.
                    pass
            self._stop_event.wait(1.0)


AUTO_CSV_EXPORTER = MinuteStudentCsvExporter(BASE_DIR, RECOGNITION_PIPELINE)
atexit.register(AUTO_CSV_EXPORTER.stop)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()

    user = next(
        (
            item for item in load_users()
            if item['帳號'] == username and item['password'] == password
        ),
        None,
    )

    if user is None:
        error_message = 'Invalid account or password. Please try again.'
        return render_template('index.html', error_message=error_message, username=username)

    manager_courses = extract_manager_courses(user)
    default_course = extract_default_course(user, manager_courses)
    RECOGNITION_PIPELINE.set_current_course(
        default_course.get('id', ''),
        default_course.get('name', ''),
    )
    FACE_DB.sync_user_profiles()
    return render_template(
        'dashboard.html',
        user=user,
        training_data=get_training_snapshot(),
        manager_courses=manager_courses,
        default_course_id=default_course.get('id', ''),
        default_course_name=default_course.get('name', ''),
    )


@app.route('/logout', methods=['POST'])
def logout():
    return redirect(url_for('index'))


@app.route('/kinect/color_feed')
def kinect_color_feed():
    return Response(
        RECOGNITION_PIPELINE.mjpeg_stream(),
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0',
            'X-Accel-Buffering': 'no',
        },
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.route('/kinect/depth_feed')
def kinect_depth_feed():
    return Response(
        RECOGNITION_PIPELINE.depth_mjpeg_stream(),
        headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Pragma': 'no-cache',
            'Expires': '0',
            'X-Accel-Buffering': 'no',
        },
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.route('/api/kinect/connect', methods=['POST'])
def connect_kinect():
    KINECT_SERVICE.connect()
    return jsonify({'status': 'connecting', 'message': 'Kinect reconnect requested.'})


@app.route('/api/kinect/disconnect', methods=['POST'])
def disconnect_kinect():
    KINECT_SERVICE.disconnect(manual=True)
    return jsonify({'status': 'disconnected', 'message': 'Kinect disconnected.'})


@app.route('/api/kinect/status')
def kinect_status():
    return jsonify(KINECT_SERVICE.get_status())


@app.route('/api/face-recognition/status')
def face_recognition_status():
    include_training = _parse_bool_query(request.args.get('include_training'), default=True)
    include_metrics = _parse_bool_query(request.args.get('include_metrics'), default=True)
    metrics_user_id = (request.args.get('metrics_user_id') or '').strip() or None
    return jsonify(
        get_face_recognition_snapshot(
            include_training=include_training,
            include_metrics=include_metrics,
            metrics_user_id=metrics_user_id,
        )
    )


@app.route('/api/attendance/status')
def attendance_status():
    include_metrics = _parse_bool_query(request.args.get('include_metrics'), default=False)
    metrics_user_id = (request.args.get('metrics_user_id') or '').strip() or None
    return jsonify(
        get_face_recognition_snapshot(
            include_training=False,
            include_metrics=include_metrics,
            metrics_user_id=metrics_user_id,
        )
    )


@app.route('/api/attendance/course', methods=['POST'])
def update_attendance_course():
    payload = request.get_json(silent=True) or {}
    course_id = str(payload.get('course_id') or '').strip()
    course_name = str(payload.get('course_name') or '').strip()
    if not course_name:
        course_name = course_id
    if not course_id:
        course_id = course_name

    if not course_name:
        return jsonify({'status': 'error', 'message': '請先選擇課程。'}), 400

    RECOGNITION_PIPELINE.set_current_course(course_id, course_name)
    return jsonify(
        {
            'status': 'ready',
            'current_course': RECOGNITION_PIPELINE.get_current_course(),
            'message': f'已切換目前課程：{course_name}',
        }
    )


@app.route('/api/attendance/start', methods=['POST'])
def start_attendance():
    payload = request.get_json(silent=True) or {}
    course_id = str(payload.get('course_id') or '').strip()
    course_name = str(payload.get('course_name') or '').strip()
    if not course_name:
        current_course = RECOGNITION_PIPELINE.get_current_course()
        course_id = course_id or str(current_course.get('course_id') or '').strip()
        course_name = str(current_course.get('course_name') or '').strip()

    if not course_name:
        return jsonify({'status': 'error', 'message': '請先選擇課程再開始課堂。'}), 400

    if not course_id:
        course_id = course_name
    RECOGNITION_PIPELINE.set_current_course(course_id, course_name)
    RECOGNITION_PIPELINE.set_attendance_mode(True)
    return jsonify(
        {
            'status': 'ready',
            'attendance_mode': True,
            'current_course': RECOGNITION_PIPELINE.get_current_course(),
            'message': f'已開始課堂，課程：{course_name}',
        }
    )


@app.route('/api/attendance/stop', methods=['POST'])
def stop_attendance():
    RECOGNITION_PIPELINE.set_attendance_mode(False)
    course = RECOGNITION_PIPELINE.get_current_course()
    return jsonify(
        {
            'status': 'ready',
            'attendance_mode': False,
            'current_course': course,
            'message': f'已結束課堂（{course.get("course_name") or "未指定課程"}）',
        }
    )


@app.route('/api/attendance/confirm', methods=['POST'])
def confirm_attendance_person():
    payload = request.get_json(silent=True) or {}
    temp_id = payload.get('temp_id') or request.form.get('temp_id', '').strip()
    if not temp_id:
        return jsonify({'status': 'error', 'message': 'temp_id is required.'}), 400

    result = RECOGNITION_PIPELINE.begin_confirm_temporary_person(temp_id)
    status_code = 200 if result.get('status') != 'error' else 400
    return jsonify(result), status_code


@app.route('/api/attendance/toggle', methods=['POST'])
def toggle_attendance():
    enabled = not RECOGNITION_PIPELINE.get_attendance_mode()
    if enabled:
        current_course = RECOGNITION_PIPELINE.get_current_course()
        if not str(current_course.get('course_name') or '').strip():
            return jsonify({'status': 'error', 'message': '請先選擇課程再開始課堂。'}), 400
    RECOGNITION_PIPELINE.set_attendance_mode(enabled)
    current_course = RECOGNITION_PIPELINE.get_current_course()
    return jsonify(
        {
            'status': 'ready',
            'attendance_mode': enabled,
            'current_course': current_course,
            'message': (
                f'已開始課堂，課程：{current_course.get("course_name") or "未指定課程"}'
                if enabled else
                f'已結束課堂（{current_course.get("course_name") or "未指定課程"}）'
            ),
        }
    )


@app.route('/api/training/students')
def training_students():
    return jsonify(get_training_snapshot())


@app.route('/api/training/upload', methods=['POST'])
def training_upload():
    label = request.form.get('label', '').strip()
    image = request.files.get('image')

    if not label:
        return jsonify({'status': 'error', 'message': 'Please choose a student folder.'}), 400
    if image is None or not image.filename:
        return jsonify({'status': 'error', 'message': 'Please choose an image file.'}), 400

    filename = secure_filename(image.filename)
    if not filename:
        return jsonify({'status': 'error', 'message': 'Invalid filename.'}), 400

    try:
        saved_path = FACE_DB.save_training_image(label, filename, image.read())
        FACE_DB.build_database()
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({'status': 'error', 'message': str(exc)}), 400

    return jsonify(
        {
            'status': 'ready',
            'message': f'Image uploaded to {label}. Embedding DB rebuilt automatically.',
            'saved_name': saved_path.name,
            'students': FACE_DB.get_training_overview(),
        }
    )


@app.route('/api/training/capture', methods=['POST'])
def training_capture():
    payload = request.get_json(silent=True) or {}
    temp_id = (payload.get('temp_id') or '').strip()
    name = (payload.get('name') or '').strip()
    student_id = (payload.get('student_id') or '').strip()
    college = (payload.get('college') or '').strip()
    department = (payload.get('department') or '').strip()

    if not temp_id:
        return jsonify({'status': 'error', 'message': 'temp_id is required.'}), 400
    if not name:
        return jsonify({'status': 'error', 'message': 'Student name is required.'}), 400
    if not student_id:
        return jsonify({'status': 'error', 'message': 'Student ID is required.'}), 400
    if not college:
        return jsonify({'status': 'error', 'message': 'College is required.'}), 400
    if not department:
        return jsonify({'status': 'error', 'message': 'Department is required.'}), 400

    try:
        with PENDING_TRAINING_LOCK:
            frames = list(PENDING_TRAINING_CAPTURES.get(temp_id, []))
        if len(frames) < 3:
            return jsonify({'status': 'error', 'message': 'Please capture 3 training photos first.'}), 400
        profile = FACE_DB.upsert_student_with_captures(
            name=name,
            student_id=student_id,
            frames=frames,
            college=college,
            department=department,
        )
        FACE_DB.build_database()
        register_result = RECOGNITION_PIPELINE.register_temporary_person(temp_id, profile)
        reset_pending_captures(temp_id)
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({'status': 'error', 'message': str(exc)}), 400

    return jsonify(
        {
            'status': 'ready',
            'message': f"{register_result['message']} 已完成 3 張訓練照片拍攝。",
            'profile': profile,
            'attendance': get_face_recognition_snapshot(),
            'students': FACE_DB.get_training_overview(),
        }
    )


@app.route('/api/training/capture-frame', methods=['POST'])
def training_capture_frame():
    payload = request.get_json(silent=True) or {}
    temp_id = (payload.get('temp_id') or '').strip()
    if not temp_id:
        return jsonify({'status': 'error', 'message': 'temp_id is required.'}), 400

    try:
        frame = RECOGNITION_PIPELINE.capture_temporary_person_frame(temp_id)
        with PENDING_TRAINING_LOCK:
            frames = PENDING_TRAINING_CAPTURES.setdefault(temp_id, [])
            if len(frames) >= 3:
                return jsonify({'status': 'ready', 'count': len(frames), 'remaining': 0, 'message': '3 photos already captured.'})
            frames.append(frame)
            count = len(frames)
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({'status': 'error', 'message': str(exc)}), 400

    return jsonify(
        {
            'status': 'ready',
            'count': count,
            'remaining': max(0, 3 - count),
            'message': f'Captured photo {count}/3.',
        }
    )


@app.route('/api/training/capture/reset', methods=['POST'])
def training_capture_reset():
    payload = request.get_json(silent=True) or {}
    temp_id = (payload.get('temp_id') or '').strip()
    reset_pending_captures(temp_id)
    return jsonify({'status': 'ready', 'message': 'Capture buffer cleared.'})


@app.route('/api/face-recognition/rebuild', methods=['POST'])
def rebuild_face_database():
    try:
        FACE_DB.sync_user_profiles()
        students = FACE_DB.build_database()
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({'status': 'error', 'message': str(exc), 'students': []}), 500

    return jsonify(
        {
            'status': 'ready',
            'message': 'Embedding database rebuilt successfully.',
            'students': [
                {
                    'label': item['label'],
                    'image_count': item['image_count'],
                    'matched_images': item['matched_images'],
                }
                for item in students
            ],
        }
    )


if __name__ == '__main__':
    app.run(debug=False, threaded=True)
