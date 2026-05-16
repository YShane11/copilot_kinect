import os
import shutil
import threading
import time
from pathlib import Path

from src.vision.rgb_depth_alignment import RgbDepthAligner


def _env_float(name, default):
    raw = os.getenv(name, '').strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name, default):
    raw = os.getenv(name, '').strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_bool(name, default):
    raw = os.getenv(name, '').strip().lower()
    if not raw:
        return bool(default)
    if raw in {'1', 'true', 'yes', 'on'}:
        return True
    if raw in {'0', 'false', 'no', 'off'}:
        return False
    return bool(default)


class KinectService:
    DISCONNECTED_MESSAGE = 'Please connect Kinect camera'
    CONNECTING_MESSAGE = 'Connecting Kinect...'
    CONNECTED_MESSAGE = 'Kinect connected'
    MANUAL_DISCONNECT_MESSAGE = 'Kinect disconnected'
    VIDEO_MODE_MESSAGE = 'Video test mode connected'
    VIDEO_DISCONNECT_MESSAGE = 'Video test source disconnected'
    VIDEO_CAPTURE_RETRY_SECONDS = 0.5
    KINECT_CAPTURE_RETRY_SECONDS = 0.08
    KINECT_V2_WAIT_COLOR_MESSAGE = 'Kinect v2 is warming up (color)...'
    KINECT_V2_WAIT_DEPTH_MESSAGE = 'Kinect v2 is warming up (depth)...'
    KINECT_V2_FALLBACK_MESSAGE = 'Kinect v2 warmup timeout, trying Kinect v1...'
    KINECT_BACKEND_AUTO = 'auto'
    KINECT_BACKEND_V1 = 'kinect_v1'
    KINECT_BACKEND_V2 = 'kinect_v2'
    KINECT_V2_WARMUP_TIMEOUT_SECONDS = 3.0
    KINECT_V2_RETRY_COOLDOWN_SECONDS = 8.0
    KINECT_IDLE_PREVIEW_FPS = _env_float('KINECT_IDLE_PREVIEW_FPS', 15.0)
    KINECT_ATTENDANCE_FPS = _env_float('KINECT_ATTENDANCE_FPS', 12.0)
    KINECT_COLOR_MAX_WIDTH = _env_int('KINECT_COLOR_MAX_WIDTH', 1280)
    KINECT_V2_ALIGN_DEPTH_IN_ATTENDANCE = _env_bool('KINECT_V2_ALIGN_DEPTH_IN_ATTENDANCE', True)
    KINECT_V2_DEPTH_ALIGN_INTERVAL = _env_float('KINECT_V2_DEPTH_ALIGN_INTERVAL', 0.85)
    STREAM_PREVIEW_MAX_WIDTH = _env_int('KINECT_STREAM_PREVIEW_MAX_WIDTH', 1280)
    STREAM_JPEG_QUALITY = _env_int('KINECT_STREAM_JPEG_QUALITY', 74)

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self._cv_modules = None
        self._freenect_module = None
        self._freenect_error = None
        self._kinect_v2_runtime = None
        self._kinect_v2_modules = None
        self._kinect_v2_error = None
        self._kinect_v2_color_frame = None
        self._kinect_v2_depth_frame = None
        self._kinect_v2_aligned_depth_frame = None
        self._kinect_v2_aligned_depth_at = 0.0
        self._kinect_v2_aligned_depth_color_shape = None
        self._kinect_v2_selected_at = 0.0
        self._kinect_v2_first_frame_received = False
        self._kinect_v2_disabled_until = 0.0
        self._kinect_backend = None
        self._kinect_backend_preference = self._resolve_kinect_backend_preference()
        self._video_source_path = self._resolve_video_source()
        self._video_capture = None
        self._video_capture_path = None
        self._video_alias_path = self.base_dir / 'data' / 'test_videos' / '_aliases' / '_video_test_source.mp4'
        self._video_fps = 30.0
        self._video_loop_enabled = str(os.getenv('KINECT_VIDEO_LOOP', '1')).strip().lower() not in {'0', 'false', 'no'}
        self._aligner = RgbDepthAligner(self.base_dir)
        self._active_source_mode = 'video' if self._video_source_path is not None else 'kinect'
        self._processing_mode = 'idle'
        self._lock = threading.Lock()
        self._running = True
        self._desired_connected = True
        self._status = 'connecting'
        if self._video_source_path is None:
            self._message = self.CONNECTING_MESSAGE
        else:
            self._message = f'Connecting video source: {self._video_source_path.name}'
        self._color_frame = None
        self._depth_frame = None
        self._depth_raw_frame = None
        self._color_jpeg = None
        self._depth_jpeg = None
        self._frame_seq = 0
        self._frame_timestamp = 0.0
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _resolve_video_source(self):
        for env_key in ('KINECT_TEST_VIDEO', 'KINECT_VIDEO_SOURCE'):
            raw = os.getenv(env_key, '').strip()
            if not raw:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = self.base_dir / path
            if path.exists():
                return path
        return None

    def _resolve_kinect_backend_preference(self):
        raw = str(os.getenv('KINECT_BACKEND', self.KINECT_BACKEND_AUTO)).strip().lower()
        if raw in {'v2', 'kinect2', 'kinect_v2', self.KINECT_BACKEND_V2}:
            return self.KINECT_BACKEND_V2
        if raw in {'v1', 'kinect1', 'kinect_v1', self.KINECT_BACKEND_V1}:
            return self.KINECT_BACKEND_V1
        return self.KINECT_BACKEND_AUTO

    def _kinect_backend_candidates(self):
        now = time.time()
        v2_allowed = now >= float(self._kinect_v2_disabled_until or 0.0)
        if self._kinect_backend_preference == self.KINECT_BACKEND_V2:
            return [self.KINECT_BACKEND_V2, self.KINECT_BACKEND_V1]
        if self._kinect_backend_preference == self.KINECT_BACKEND_V1:
            return [self.KINECT_BACKEND_V1, self.KINECT_BACKEND_V2]
        if v2_allowed:
            return [self.KINECT_BACKEND_V2, self.KINECT_BACKEND_V1]
        return [self.KINECT_BACKEND_V1, self.KINECT_BACKEND_V2]

    def configure_kinect_dlls(self):
        dll_paths = [
            self.base_dir / 'libfreenect' / 'build' / 'lib' / 'Release',
            Path(r'C:\vcpkg\installed\x64-windows\bin'),
            self.base_dir / '.venv' / 'Lib' / 'site-packages',
        ]

        for dll_path in dll_paths:
            if dll_path.exists():
                os.add_dll_directory(str(dll_path))

    def get_cv_modules(self):
        if self._cv_modules is not None:
            return self._cv_modules

        import cv2  # pylint: disable=import-outside-toplevel
        import numpy as np  # pylint: disable=import-outside-toplevel

        self._cv_modules = (cv2, np)
        return self._cv_modules

    def get_freenect_module(self):
        if self._freenect_module is not None:
            return self._freenect_module

        try:
            self.configure_kinect_dlls()
            import cython0 as freenect  # pylint: disable=import-outside-toplevel
        except Exception as exc:  # pylint: disable=broad-except
            self._freenect_error = f'Kinect v1 modules failed to load: {exc}'
            raise RuntimeError(self._freenect_error) from exc

        self._freenect_module = freenect
        self._freenect_error = None
        return self._freenect_module

    def get_kinect_v2_runtime(self):
        if self._kinect_v2_runtime is not None:
            return self._kinect_v2_runtime

        try:
            if not hasattr(time, 'clock'):
                time.clock = time.perf_counter

            import comtypes  # pylint: disable=import-outside-toplevel

            if hasattr(comtypes, '_check_version'):
                comtypes._check_version = lambda *args, **kwargs: None

            from pykinect2 import PyKinectRuntime, PyKinectV2  # pylint: disable=import-outside-toplevel
        except Exception as exc:  # pylint: disable=broad-except
            self._kinect_v2_error = f'Kinect v2 modules failed to load: {exc}'
            raise RuntimeError(self._kinect_v2_error) from exc

        try:
            flags = PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth
            runtime = PyKinectRuntime.PyKinectRuntime(flags)
        except Exception as exc:  # pylint: disable=broad-except
            self._kinect_v2_error = f'Kinect v2 runtime failed to initialize: {exc}'
            raise RuntimeError(self._kinect_v2_error) from exc

        self._kinect_v2_runtime = runtime
        self._kinect_v2_modules = (PyKinectRuntime, PyKinectV2)
        self._kinect_v2_error = None
        return self._kinect_v2_runtime

    def reset_freenect_module(self):
        if self._freenect_module is not None:
            try:
                if hasattr(self._freenect_module, 'sync_stop'):
                    self._freenect_module.sync_stop()
            except Exception:
                pass

        self._freenect_module = None
        self._freenect_error = None

    def reset_kinect_v2_runtime(self):
        if self._kinect_v2_runtime is not None:
            try:
                self._kinect_v2_runtime.close()
            except Exception:
                pass

        self._kinect_v2_runtime = None
        self._kinect_v2_modules = None
        self._kinect_v2_error = None
        self._kinect_v2_color_frame = None
        self._kinect_v2_depth_frame = None
        self._kinect_v2_aligned_depth_frame = None
        self._kinect_v2_aligned_depth_at = 0.0
        self._kinect_v2_aligned_depth_color_shape = None
        self._kinect_v2_selected_at = 0.0
        self._kinect_v2_first_frame_received = False

    def _reset_kinect_backends(self):
        self.reset_kinect_v2_runtime()
        self.reset_freenect_module()
        self._kinect_v2_disabled_until = 0.0
        self._kinect_backend = None

    def _reset_active_kinect_backend(self):
        active = self._kinect_backend
        if active == self.KINECT_BACKEND_V2:
            self.reset_kinect_v2_runtime()
        elif active == self.KINECT_BACKEND_V1:
            self.reset_freenect_module()
        else:
            self._reset_kinect_backends()
        self._kinect_backend = None

    def _ensure_kinect_backend(self):
        if self._kinect_backend in {self.KINECT_BACKEND_V1, self.KINECT_BACKEND_V2}:
            return self._kinect_backend

        errors = []
        for backend in self._kinect_backend_candidates():
            try:
                if backend == self.KINECT_BACKEND_V2:
                    self.get_kinect_v2_runtime()
                    self._kinect_v2_selected_at = time.time()
                    self._kinect_v2_first_frame_received = False
                else:
                    self.get_freenect_module()
                self._kinect_backend = backend
                return backend
            except Exception as exc:  # pylint: disable=broad-except
                errors.append(f'{backend}: {exc}')

        joined = ' | '.join(errors) if errors else 'no backend details'
        raise RuntimeError(f'No supported Kinect backend available. {joined}')

    def connect(self):
        with self._lock:
            self._desired_connected = True
            self._status = 'connecting'
            if self._video_source_path is None:
                self._message = self.CONNECTING_MESSAGE
            else:
                self._message = f'Connecting video source: {self._video_source_path.name}'

        if self._video_source_path is None:
            self._reset_kinect_backends()
        self._close_video_capture()

    def disconnect(self, manual=True):
        with self._lock:
            self._desired_connected = not manual
            self._status = 'disconnected'
            if self._video_source_path is None:
                self._message = self.MANUAL_DISCONNECT_MESSAGE if manual else self.DISCONNECTED_MESSAGE
            else:
                self._message = self.VIDEO_DISCONNECT_MESSAGE if manual else self.DISCONNECTED_MESSAGE

        if self._video_source_path is None:
            self._reset_kinect_backends()
        self._close_video_capture()
        self._update_placeholders(self._message)

    def close(self):
        self._running = False
        self._close_video_capture()
        self._reset_kinect_backends()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def get_status(self):
        with self._lock:
            return {
                'status': self._status,
                'message': self._message,
                'desired_connected': self._desired_connected,
                'processing_mode': self._processing_mode,
                'source_mode': self._active_source_mode,
                'kinect_backend': self._kinect_backend or '',
                'video_source': str(self._video_source_path) if self._video_source_path is not None else '',
            }

    def set_processing_mode(self, mode):
        normalized_mode = 'attendance' if str(mode or '').strip().lower() == 'attendance' else 'idle'
        with self._lock:
            if self._processing_mode != normalized_mode:
                self._kinect_v2_aligned_depth_frame = None
                self._kinect_v2_aligned_depth_at = 0.0
                self._kinect_v2_aligned_depth_color_shape = None
            self._processing_mode = normalized_mode

    def _is_attendance_processing_mode(self):
        with self._lock:
            return self._processing_mode == 'attendance'

    def get_latest_color_frame(self):
        with self._lock:
            return None if self._color_frame is None else self._color_frame.copy()

    def get_latest_depth_visual_frame(self):
        with self._lock:
            return None if self._depth_frame is None else self._depth_frame.copy()

    def get_latest_depth_frame(self):
        with self._lock:
            return None if self._depth_raw_frame is None else self._depth_raw_frame.copy()

    def get_latest_jpeg(self, kind):
        with self._lock:
            return self._color_jpeg if kind == 'color' else self._depth_jpeg

    def get_latest_frame_marker(self):
        with self._lock:
            return {
                'frame_seq': int(self._frame_seq),
                'timestamp': float(self._frame_timestamp),
            }

    def get_latest_frame_bundle(self):
        with self._lock:
            return {
                'frame_seq': int(self._frame_seq),
                'timestamp': float(self._frame_timestamp),
                'color_frame': None if self._color_frame is None else self._color_frame.copy(),
                'depth_visual_frame': None if self._depth_frame is None else self._depth_frame.copy(),
                'depth_raw_frame': None if self._depth_raw_frame is None else self._depth_raw_frame.copy(),
                'color_jpeg': self._color_jpeg,
                'depth_jpeg': self._depth_jpeg,
            }

    def _should_reset(self, message):
        lowered = message.lower()
        return (
            self.DISCONNECTED_MESSAGE.lower() in lowered
            or 'device disconnected' in lowered
            or 'marked dead' in lowered
            or 'control transfer failed' in lowered
            or 'send_cmd' in lowered
            or 'write_register' in lowered
            or 'stopping streams' in lowered
            or 'no supported kinect backend available' in lowered
            or 'kinect v2 runtime failed' in lowered
            or 'kinect v2 modules failed to load' in lowered
            or 'kinect v1 modules failed to load' in lowered
            or 'wrong version' in lowered
            or 'dll load failed' in lowered
            or 'comerror' in lowered
        )

    def _build_placeholder_frame(self, message):
        cv2, np = self.get_cv_modules()

        frame = np.zeros((720, 960, 3), dtype=np.uint8)
        frame[:] = (248, 250, 253)
        cv2.rectangle(frame, (34, 34), (926, 686), (205, 216, 232), 2)
        text = message[:48]
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = max(48, (960 - text_size[0]) // 2)
        text_y = (720 + text_size[1]) // 2
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (68, 82, 108), 2)
        return frame

    def _encode_frame(self, frame):
        cv2, _ = self.get_cv_modules()
        prepared_frame = frame
        max_width = max(0, int(self.STREAM_PREVIEW_MAX_WIDTH))
        if max_width and frame is not None and getattr(frame, 'shape', None) is not None:
            frame_width = int(frame.shape[1]) if len(frame.shape) >= 2 else 0
            if frame_width > max_width:
                scale = max_width / float(frame_width)
                target_height = max(1, int(round(frame.shape[0] * scale)))
                prepared_frame = cv2.resize(frame, (max_width, target_height), interpolation=cv2.INTER_AREA)
        success, buffer = cv2.imencode(
            '.jpg',
            prepared_frame,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                int(self.STREAM_JPEG_QUALITY),
            ],
        )
        if not success:
            return None
        return buffer.tobytes()

    def _update_placeholders(self, message):
        color = self._build_placeholder_frame(message)
        depth = self._build_placeholder_frame(message)
        placeholder_ts = time.time()
        with self._lock:
            self._color_frame = color
            self._depth_frame = depth
            self._depth_raw_frame = None
            self._color_jpeg = self._encode_frame(color)
            self._depth_jpeg = self._encode_frame(depth)
            self._frame_seq += 1
            self._frame_timestamp = placeholder_ts
            self._active_source_mode = 'video' if self._video_source_path is not None else 'kinect'

    def _depth_visual_from_raw(self, depth_raw, clip_min, clip_max):
        cv2, np = self.get_cv_modules()
        depth_clipped = np.clip(depth_raw, clip_min, clip_max)
        depth_norm = cv2.normalize(depth_clipped, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype(np.uint8)
        depth_norm = 255 - depth_norm
        depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
        return depth_vis

    def _capture_color_v1(self):
        cv2, _ = self.get_cv_modules()
        freenect = self.get_freenect_module()
        result = freenect.sync_get_video()
        if result is None:
            raise RuntimeError(self.DISCONNECTED_MESSAGE)
        frame, _ = result
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def _capture_depth_v1(self):
        _, np = self.get_cv_modules()
        freenect = self.get_freenect_module()
        depth_raw = None

        registered_format = getattr(freenect, 'DEPTH_REGISTERED', None)
        if registered_format is not None:
            try:
                result = freenect.sync_get_depth(format=registered_format)
                if result is not None:
                    depth_raw, _ = result
                    self._active_source_mode = 'kinect_v1_registered'
            except Exception:
                depth_raw = None

        if depth_raw is None:
            result = freenect.sync_get_depth()
            if result is None:
                raise RuntimeError(self.DISCONNECTED_MESSAGE)
            depth_raw, _ = result
            self._active_source_mode = 'kinect'

        depth_raw = self._aligner.align_v1_depth(depth_raw.astype(np.uint16), (480, 640, 3))
        clip_min, clip_max = (500, 4500) if self._active_source_mode == 'kinect_v1_registered' else (400, 2000)
        depth_vis = self._depth_visual_from_raw(depth_raw, clip_min=clip_min, clip_max=clip_max)
        return depth_raw, depth_vis

    def _capture_color_v2(self):
        _, np = self.get_cv_modules()
        runtime = self.get_kinect_v2_runtime()

        if runtime.has_new_color_frame():
            color_flat = runtime.get_last_color_frame()
            if color_flat is not None and getattr(color_flat, 'size', 0) > 0:
                width = int(runtime.color_frame_desc.Width)
                height = int(runtime.color_frame_desc.Height)
                color_bgra = color_flat.reshape((height, width, 4)).astype(np.uint8)
                self._kinect_v2_color_frame = color_bgra[:, :, :3].copy()
                self._kinect_v2_first_frame_received = True

        if self._kinect_v2_color_frame is None:
            raise RuntimeError(self.KINECT_V2_WAIT_COLOR_MESSAGE)

        return self._kinect_v2_color_frame.copy()

    def _capture_depth_v2(self, color_frame=None):
        _, np = self.get_cv_modules()
        runtime = self.get_kinect_v2_runtime()

        if runtime.has_new_depth_frame():
            depth_flat = runtime.get_last_depth_frame()
            if depth_flat is not None and getattr(depth_flat, 'size', 0) > 0:
                width = int(runtime.depth_frame_desc.Width)
                height = int(runtime.depth_frame_desc.Height)
                self._kinect_v2_depth_frame = depth_flat.reshape((height, width)).astype(np.uint16)
                self._kinect_v2_first_frame_received = True

        if self._kinect_v2_depth_frame is None:
            raise RuntimeError(self.KINECT_V2_WAIT_DEPTH_MESSAGE)

        if color_frame is None:
            color_frame = self._kinect_v2_color_frame

        if not self._is_attendance_processing_mode():
            depth_raw = self._kinect_v2_depth_frame.copy()
            depth_vis = self._depth_visual_from_raw(depth_raw, clip_min=500, clip_max=4500)
            self._active_source_mode = 'kinect_v2_preview'
            return depth_raw, depth_vis

        # Keep the preview on raw depth; aligned v2 depth is expensive and visually noisy.
        raw_depth_vis = self._depth_visual_from_raw(self._kinect_v2_depth_frame, clip_min=500, clip_max=4500)
        self._active_source_mode = 'kinect_v2'

        if not self.KINECT_V2_ALIGN_DEPTH_IN_ATTENDANCE:
            self._kinect_v2_aligned_depth_frame = None
            self._kinect_v2_aligned_depth_at = 0.0
            self._kinect_v2_aligned_depth_color_shape = None
            return None, raw_depth_vis

        if color_frame is None:
            color_height = int(runtime.color_frame_desc.Height)
            color_width = int(runtime.color_frame_desc.Width)
            color_shape = (color_height, color_width, 3)
        else:
            color_shape = color_frame.shape

        color_shape_key = tuple(int(value) for value in color_shape[:2])
        now = time.time()
        align_interval = max(0.1, float(self.KINECT_V2_DEPTH_ALIGN_INTERVAL))
        depth_raw = self._kinect_v2_aligned_depth_frame
        should_refresh = (
            depth_raw is None
            or self._kinect_v2_aligned_depth_color_shape != color_shape_key
            or now >= (float(self._kinect_v2_aligned_depth_at or 0.0) + align_interval)
        )

        if should_refresh:
            pykinect_v2_module = None if self._kinect_v2_modules is None else self._kinect_v2_modules[1]
            aligned_depth = self._aligner.align_v2_depth(
                runtime,
                self._kinect_v2_depth_frame,
                color_shape,
                pykinect_v2_module=pykinect_v2_module,
            )
            if aligned_depth is not None:
                self._kinect_v2_aligned_depth_frame = aligned_depth.copy()
                self._kinect_v2_aligned_depth_at = now
                self._kinect_v2_aligned_depth_color_shape = color_shape_key
                depth_raw = aligned_depth
        else:
            depth_raw = depth_raw.copy()

        return depth_raw, raw_depth_vis

    def _capture_color(self):
        backend = self._ensure_kinect_backend()
        if backend == self.KINECT_BACKEND_V2:
            return self._capture_color_v2()
        return self._capture_color_v1()

    def _capture_depth(self, color_frame=None):
        backend = self._ensure_kinect_backend()
        if backend == self.KINECT_BACKEND_V2:
            return self._capture_depth_v2(color_frame=color_frame)
        return self._capture_depth_v1()

    def _resize_color_for_processing(self, frame):
        cv2, _ = self.get_cv_modules()
        max_width = max(0, int(self.KINECT_COLOR_MAX_WIDTH))
        if not max_width or frame is None:
            return frame
        height, width = frame.shape[:2]
        if width <= max_width:
            return frame
        scale = max_width / float(width)
        target_height = max(1, int(round(height * scale)))
        return cv2.resize(frame, (max_width, target_height), interpolation=cv2.INTER_AREA)

    def _connected_message_for_active_backend(self):
        if self._kinect_backend == self.KINECT_BACKEND_V2:
            return f'{self.CONNECTED_MESSAGE} (v2)'
        if self._kinect_backend == self.KINECT_BACKEND_V1:
            return f'{self.CONNECTED_MESSAGE} (v1)'
        return self.CONNECTED_MESSAGE

    def _close_video_capture(self):
        capture = self._video_capture
        self._video_capture = None
        self._video_capture_path = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass

    def _resolve_video_capture_path(self):
        if self._video_source_path is None:
            return None

        try:
            str(self._video_source_path).encode('ascii')
            return self._video_source_path
        except UnicodeEncodeError:
            source_stat = self._video_source_path.stat()
            needs_copy = True
            if self._video_alias_path.exists():
                alias_stat = self._video_alias_path.stat()
                needs_copy = (
                    alias_stat.st_size != source_stat.st_size
                    or int(alias_stat.st_mtime) != int(source_stat.st_mtime)
                )
            if needs_copy:
                self._video_alias_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self._video_source_path, self._video_alias_path)
            return self._video_alias_path

    def _open_video_capture(self):
        cv2, _ = self.get_cv_modules()
        capture_path = self._resolve_video_capture_path()
        if capture_path is None:
            raise RuntimeError('Video source path is not configured.')

        capture = cv2.VideoCapture(str(capture_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f'Unable to open video source: {capture_path.name}')

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps > 0:
            self._video_fps = fps
        self._video_capture = capture
        self._video_capture_path = capture_path

    def _capture_color_from_video(self):
        cv2, _ = self.get_cv_modules()
        if self._video_capture is None:
            self._open_video_capture()

        ok, frame = self._video_capture.read()
        if not ok or frame is None:
            if not self._video_loop_enabled:
                raise RuntimeError('Video source reached end-of-file.')
            self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._video_capture.read()
            if not ok or frame is None:
                raise RuntimeError('Unable to read frame from video source.')
        return frame

    def _build_depth_from_color(self, frame):
        cv2, _ = self.get_cv_modules()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        depth_norm = cv2.GaussianBlur(gray, (9, 9), 0)
        depth_norm = cv2.normalize(depth_norm, None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = depth_norm.astype('uint8')
        # Offline-video fallback: generate a pseudo raw depth map in millimeters.
        depth_raw = (2000.0 - ((depth_norm.astype('float32') / 255.0) * 1400.0)).astype('uint16')
        depth_vis = cv2.applyColorMap(255 - depth_norm, cv2.COLORMAP_TURBO)
        return depth_raw, depth_vis

    def _video_interval_seconds(self):
        if self._video_fps <= 0:
            return 1.0 / 30.0
        return 1.0 / min(self._video_fps, 60.0)

    def _capture_loop(self):
        self._update_placeholders(self.DISCONNECTED_MESSAGE)

        while self._running:
            with self._lock:
                desired = self._desired_connected

            if not desired:
                time.sleep(0.2)
                continue

            frame_interval = self._video_interval_seconds() if self._video_source_path is not None else self.KINECT_CAPTURE_RETRY_SECONDS

            try:
                if self._video_source_path is None:
                    backend = self._ensure_kinect_backend()
                    color = self._capture_color()
                    color = self._resize_color_for_processing(color)
                    depth_raw, depth = self._capture_depth(color_frame=color)
                    connected_message = self._connected_message_for_active_backend()
                    if backend == self.KINECT_BACKEND_V2:
                        if self._is_attendance_processing_mode():
                            frame_interval = 1.0 / max(1.0, float(self.KINECT_ATTENDANCE_FPS))
                        else:
                            frame_interval = 1.0 / max(1.0, float(self.KINECT_IDLE_PREVIEW_FPS))
                else:
                    color = self._capture_color_from_video()
                    depth_raw, depth = self._build_depth_from_color(color)
                    connected_message = self.VIDEO_MODE_MESSAGE
                    self._active_source_mode = 'video'

                color_jpeg = self._encode_frame(color)
                depth_jpeg = self._encode_frame(depth)
                capture_ts = time.time()

                with self._lock:
                    self._color_frame = color
                    self._depth_frame = depth
                    self._depth_raw_frame = depth_raw
                    self._color_jpeg = color_jpeg
                    self._depth_jpeg = depth_jpeg
                    self._frame_seq += 1
                    self._frame_timestamp = capture_ts
                    self._status = 'connected'
                    self._message = connected_message
            except Exception as exc:  # pylint: disable=broad-except
                message = str(exc)
                if (
                    self._video_source_path is None
                    and self._kinect_backend == self.KINECT_BACKEND_V2
                    and self._kinect_backend_preference == self.KINECT_BACKEND_AUTO
                    and not self._kinect_v2_first_frame_received
                    and message in {self.KINECT_V2_WAIT_COLOR_MESSAGE, self.KINECT_V2_WAIT_DEPTH_MESSAGE}
                ):
                    warmup_elapsed = time.time() - float(self._kinect_v2_selected_at or 0.0)
                    if warmup_elapsed >= self.KINECT_V2_WARMUP_TIMEOUT_SECONDS:
                        self.reset_kinect_v2_runtime()
                        self._kinect_backend = None
                        self._kinect_v2_disabled_until = time.time() + self.KINECT_V2_RETRY_COOLDOWN_SECONDS
                        message = self.KINECT_V2_FALLBACK_MESSAGE
                if self._video_source_path is None and self._should_reset(message):
                    self._reset_active_kinect_backend()
                    message = self.DISCONNECTED_MESSAGE

                if self._video_source_path is not None:
                    self._close_video_capture()

                is_connecting_state = message in {
                    self.KINECT_V2_WAIT_COLOR_MESSAGE,
                    self.KINECT_V2_WAIT_DEPTH_MESSAGE,
                    self.KINECT_V2_FALLBACK_MESSAGE,
                }
                self._update_placeholders(message)
                with self._lock:
                    self._status = 'connecting' if is_connecting_state else 'disconnected'
                    self._message = message
                time.sleep(self.VIDEO_CAPTURE_RETRY_SECONDS if self._video_source_path is not None else self.KINECT_CAPTURE_RETRY_SECONDS)
                continue

            time.sleep(frame_interval)

    def mjpeg_stream(self, kind):
        while True:
            payload = self.get_latest_jpeg(kind)
            if payload is not None:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + payload + b'\r\n'
            time.sleep(0.08)
