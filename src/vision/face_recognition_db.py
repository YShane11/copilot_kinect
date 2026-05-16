import json
import math
import os
import threading
from pathlib import Path


class FaceRecognitionDB:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / 'data'
        self.photo_root = self.data_dir / 'student_faces'
        self.embedding_dir = self.data_dir / 'embeddings'
        self.embedding_file = self.embedding_dir / 'face_embeddings.json'
        self.user_file = self.data_dir / 'user.json'
        self._app = None
        self._app_error = None
        self._app_providers = []
        self._app_ctx_id = None
        self._app_load_lock = threading.Lock()
        self._lock = threading.Lock()
        self._embedding_cache = []
        self._embedding_cache_mtime = None

    def _available_onnx_providers(self):
        try:
            import onnxruntime as ort  # pylint: disable=import-outside-toplevel
            return list(ort.get_available_providers())
        except Exception:
            return []

    def _resolve_insightface_runtime(self):
        available = self._available_onnx_providers()
        available_set = set(available)

        raw_providers = str(os.getenv('INSIGHTFACE_PROVIDERS', '')).strip()
        if raw_providers:
            providers = [item.strip() for item in raw_providers.split(',') if item.strip()]
        else:
            providers = []
            for provider_name in ('CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider'):
                if provider_name in available_set:
                    providers.append(provider_name)
            if not providers:
                providers = ['CPUExecutionProvider']

        if 'CPUExecutionProvider' not in providers:
            providers.append('CPUExecutionProvider')

        raw_ctx_id = str(os.getenv('INSIGHTFACE_CTX_ID', 'auto')).strip().lower()
        if raw_ctx_id in {'', 'auto'}:
            ctx_id = 0 if 'CUDAExecutionProvider' in providers else -1
        else:
            try:
                ctx_id = int(raw_ctx_id)
            except Exception:
                ctx_id = -1

        if ctx_id >= 0 and 'CUDAExecutionProvider' not in providers:
            ctx_id = -1

        return providers, ctx_id

    def get_analysis_runtime_info(self):
        return {
            'providers': list(self._app_providers),
            'ctx_id': self._app_ctx_id,
            'ready': self._app is not None,
        }

    def get_photo_root(self):
        self.photo_root.mkdir(parents=True, exist_ok=True)
        return self.photo_root

    def ensure_storage(self):
        self.get_photo_root()
        self.embedding_dir.mkdir(parents=True, exist_ok=True)

    def list_student_labels(self):
        self.get_photo_root()
        return sorted(path.name for path in self.photo_root.iterdir() if path.is_dir())

    def next_student_label(self):
        max_index = 0
        prefix = f'{chr(0x5B78)}{chr(0x751F)}'
        for label in self.list_student_labels():
            suffix_digits = ''.join(character for character in label if character.isdigit())
            if suffix_digits:
                max_index = max(max_index, int(suffix_digits))
        return f'{prefix}{max_index + 1}'

    def load_user_profiles(self):
        if not self.user_file.exists():
            return {}

        with self.user_file.open('r', encoding='utf-8-sig') as user_file:
            payload = json.load(user_file)

        profiles = {}
        for item in payload.get('students', []):
            label = item.get('label')
            if label:
                profiles[label] = item
        return profiles

    def save_user_profiles(self, profiles):
        ordered_labels = self.list_student_labels()
        students = []

        for label in ordered_labels:
            item = profiles.get(label, {})
            students.append(
                {
                    'label': label,
                    'name': item.get('name', label),
                    'student_id': item.get('student_id', ''),
                    'college': item.get('college', ''),
                    'department': item.get('department', ''),
                    'title': item.get('title', ''),
                }
            )

        with self.user_file.open('w', encoding='utf-8') as user_file:
            json.dump({'students': students}, user_file, ensure_ascii=False, indent=2)

        return students

    def _temporary_label_prefix(self):
        return f'{chr(0x5B78)}{chr(0x751F)}'

    def _is_temporary_label(self, label):
        return str(label or '').startswith(self._temporary_label_prefix())

    def _sanitize_profile_label(self, name, student_id=''):
        label = ''.join(character for character in str(name or '').strip() if character not in '<>:"/\\|?*').strip()
        if not label:
            label = str(student_id or '').strip() or self.next_student_label()
        return label

    def _resolve_profile_label(self, profiles, student_id, name):
        normalized_student_id = (student_id or '').strip()
        normalized_name = (name or '').strip()
        desired_label = self._sanitize_profile_label(normalized_name, normalized_student_id)

        for label, item in profiles.items():
            if normalized_student_id and item.get('student_id', '').strip() == normalized_student_id:
                return label, desired_label

        for label, item in profiles.items():
            if normalized_name and item.get('name', '').strip() == normalized_name:
                return label, desired_label

        return None, desired_label

    def sync_user_profiles(self):
        profiles = self.load_user_profiles()
        labels = self.list_student_labels()

        synced = {}
        for label in labels:
            item = profiles.get(label, {})
            synced[label] = {
                'label': label,
                'name': item.get('name', label),
                'student_id': item.get('student_id', ''),
                'college': item.get('college', ''),
                'department': item.get('department', ''),
                'title': item.get('title', ''),
            }

        self.save_user_profiles(synced)
        return synced

    def _load_insightface_app(self):
        if self._app is not None:
            return self._app

        if self._app_error is not None:
            raise RuntimeError(self._app_error)

        with self._app_load_lock:
            if self._app is not None:
                return self._app
            if self._app_error is not None:
                raise RuntimeError(self._app_error)

            try:
                from insightface.app import FaceAnalysis  # pylint: disable=import-outside-toplevel
            except Exception as exc:  # pylint: disable=broad-except
                self._app_error = f'InsightFace is not available: {exc}'
                raise RuntimeError(self._app_error) from exc

            providers, ctx_id = self._resolve_insightface_runtime()
            app = FaceAnalysis(providers=providers)
            app.prepare(ctx_id=ctx_id)
            self._app = app
            self._app_providers = providers
            self._app_ctx_id = ctx_id
            return app

    def is_analysis_ready(self):
        return self._app is not None

    def _cosine_similarity(self, left, right):
        numerator = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)

    def _extract_embedding(self, image):
        app = self._load_insightface_app()
        faces = app.get(image)
        if not faces:
            return None
        return faces[0].embedding.tolist()

    def analyze_faces(self, image):
        try:
            app = self._load_insightface_app()
        except RuntimeError as exc:
            return {'status': 'unavailable', 'message': str(exc), 'faces': []}

        faces = app.get(image)
        payload = []
        for face in faces:
            bbox = [float(value) for value in face.bbox.tolist()]
            payload.append(
                {
                    'bbox': bbox,
                    'embedding': face.embedding.tolist(),
                    'det_score': float(getattr(face, 'det_score', 0.0)),
                }
            )

        if payload:
            return {'status': 'ok', 'message': 'Faces detected.', 'faces': payload}
        return {'status': 'no_face', 'message': 'No face detected in current frame.', 'faces': []}

    def _read_image(self, image_path):
        import cv2  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel

        image = cv2.imread(str(image_path))
        if image is None:
            try:
                data = np.fromfile(str(image_path), dtype=np.uint8)
                if data.size > 0:
                    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            except Exception:
                image = None
        if image is None:
            raise RuntimeError(f'Unable to read image: {image_path.name}')
        return image

    def _invalidate_embedding_cache(self):
        self._embedding_cache = []
        self._embedding_cache_mtime = None

    def load_database(self):
        with self._lock:
            if not self.embedding_file.exists():
                self._invalidate_embedding_cache()
                return []

            try:
                mtime = self.embedding_file.stat().st_mtime
            except OSError:
                self._invalidate_embedding_cache()
                return []

            if self._embedding_cache_mtime == mtime:
                return self._embedding_cache

            with self.embedding_file.open('r', encoding='utf-8') as db_file:
                payload = json.load(db_file)

            students = payload.get('students', [])
            self._embedding_cache = students if isinstance(students, list) else []
            self._embedding_cache_mtime = mtime
            return self._embedding_cache

    def get_training_overview(self):
        profiles = self.sync_user_profiles()
        overview = []
        for label in self.list_student_labels():
            student_dir = self.photo_root / label
            images = [
                path.name
                for path in sorted(student_dir.iterdir())
                if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
            ]
            item = profiles.get(label, {})
            overview.append(
                {
                    'label': label,
                    'name': item.get('name', label),
                    'student_id': item.get('student_id', ''),
                    'college': item.get('college', ''),
                    'department': item.get('department', ''),
                    'title': item.get('title', ''),
                    'image_count': len(images),
                    'images': images,
                }
            )
        return overview

    def save_captured_frames(self, label, frames, replace=False):
        import cv2  # pylint: disable=import-outside-toplevel

        self.ensure_storage()
        student_dir = self.photo_root / label
        student_dir.mkdir(parents=True, exist_ok=True)
        existing = [
            path
            for path in student_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
        ]
        if replace:
            for path in existing:
                path.unlink()
            next_index = 1
        else:
            next_index = len(existing) + 1
        saved_files = []

        for frame in frames:
            target_path = student_dir / f'{next_index:02d}.jpg'
            ok, buffer = cv2.imencode('.jpg', frame)
            if not ok:
                raise RuntimeError('Unable to encode captured frame.')
            target_path.write_bytes(buffer.tobytes())
            saved_files.append(target_path)
            next_index += 1

        return saved_files

    def upsert_student_with_captures(self, name, student_id, frames, college='', department='', title=''):
        if not name.strip():
            raise RuntimeError('Student name is required.')
        if not student_id.strip():
            raise RuntimeError('Student ID is required.')
        if len(frames) < 3:
            raise RuntimeError('At least 3 captured frames are required.')

        self.ensure_storage()
        profiles = self.load_user_profiles()
        existing_label, desired_label = self._resolve_profile_label(profiles, student_id, name)
        label = desired_label
        student_dir = self.photo_root / label
        student_dir.mkdir(parents=True, exist_ok=True)

        if existing_label and existing_label != label:
            existing_dir = self.photo_root / existing_label
            if existing_dir.exists() and self._is_temporary_label(existing_label):
                for path in existing_dir.iterdir():
                    if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                        path.unlink()
                existing_dir.rmdir()
            profiles.pop(existing_label, None)

        profiles[label] = {
            'label': label,
            'name': name.strip(),
            'student_id': student_id.strip(),
            'college': college.strip(),
            'department': department.strip(),
            'title': title.strip(),
        }
        self.save_user_profiles(profiles)
        saved_files = self.save_captured_frames(label, frames, replace=True)

        return {
            'label': label,
            'name': profiles[label]['name'],
            'student_id': profiles[label]['student_id'],
            'college': profiles[label]['college'],
            'department': profiles[label]['department'],
            'title': profiles[label]['title'],
            'saved_files': [path.name for path in saved_files],
            'image_count': len(
                [
                    path
                    for path in student_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
                ]
            ),
        }

    def save_training_image(self, label, filename, payload):
        self.ensure_storage()
        valid_labels = set(self.list_student_labels())
        if label not in valid_labels:
            raise RuntimeError(f'Unknown student label: {label}')

        suffix = Path(filename).suffix.lower()
        if suffix not in {'.jpg', '.jpeg', '.png', '.bmp'}:
            raise RuntimeError('Only jpg, jpeg, png, and bmp files are supported.')

        student_dir = self.photo_root / label
        existing = [path for path in student_dir.iterdir() if path.is_file()]
        next_index = len(existing) + 1
        target_path = student_dir / f'{next_index:02d}{suffix}'
        target_path.write_bytes(payload)
        return target_path

    def match_embedding(self, embedding, threshold=0.45):
        candidates = []
        for item in self.load_database():
            similarity = self._cosine_similarity(embedding, item['embedding'])
            if similarity >= threshold:
                candidates.append(
                    {
                        'label': item['label'],
                        'display_name': item.get('display_name', item['label']),
                        'student_id': item.get('student_id', ''),
                        'college': item.get('college', ''),
                        'department': item.get('department', ''),
                        'title': item.get('title', ''),
                        'similarity': round(similarity, 4),
                    }
                )

        candidates.sort(key=lambda item: item['similarity'], reverse=True)
        return candidates

    def build_database(self):
        self.ensure_storage()
        students = []
        profiles = self.load_user_profiles()

        with self._lock:
            for student_dir in sorted(path for path in self.photo_root.iterdir() if path.is_dir()):
                embeddings = []
                image_count = 0

                for image_path in sorted(student_dir.iterdir()):
                    if image_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
                        continue

                    image = self._read_image(image_path)
                    embedding = self._extract_embedding(image)
                    image_count += 1
                    if embedding is not None:
                        embeddings.append(embedding)

                if embeddings:
                    embedding_size = len(embeddings[0])
                    mean_embedding = []
                    for index in range(embedding_size):
                        mean_embedding.append(sum(item[index] for item in embeddings) / len(embeddings))

                    profile = profiles.get(student_dir.name, {})

                    students.append(
                        {
                            'label': student_dir.name,
                            'display_name': profile.get('name', student_dir.name),
                            'student_id': profile.get('student_id', ''),
                            'college': profile.get('college', ''),
                            'department': profile.get('department', ''),
                            'title': profile.get('title', ''),
                            'image_count': image_count,
                            'matched_images': len(embeddings),
                            'embedding': mean_embedding,
                        }
                    )

            payload = {
                'students': students,
                'photo_root': str(self.photo_root),
            }
            with self.embedding_file.open('w', encoding='utf-8') as db_file:
                json.dump(payload, db_file, ensure_ascii=False, indent=2)
            self._embedding_cache = students
            try:
                self._embedding_cache_mtime = self.embedding_file.stat().st_mtime
            except OSError:
                self._embedding_cache_mtime = None

        return students

    def recognize_from_frame(self, frame, threshold=0.45):
        analysis = self.analyze_faces(frame)
        if analysis['status'] == 'unavailable':
            return {'status': 'unavailable', 'message': analysis['message'], 'students': []}
        if not analysis['faces']:
            return {'status': 'no_face', 'message': analysis['message'], 'students': []}

        students = self.match_embedding(analysis['faces'][0]['embedding'], threshold=threshold)
        if students:
            return {'status': 'matched', 'message': 'Recognized students found.', 'students': students[:5]}
        return {'status': 'unknown', 'message': 'Face detected but not matched in embedding DB.', 'students': []}
