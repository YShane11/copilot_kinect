import argparse
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np


PREVIEW_WINDOW_NAME = 'Kinect Recorder Preview'
PREVIEW_MAX_SIZE = (960, 540)
PREVIEW_BOOT_SIZE = (720, 405)
_PREVIEW_WINDOW_CREATED = False
_PREVIEW_STATE = {
    'backend': None,
    'tk_root': None,
    'tk_label': None,
    'tk_photo': None,
    'tk_modules': None,
    'tk_closed': False,
    'size': None,
    'target_size': None,
}


def add_project_root_to_path(project_root: Path) -> None:
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)


def create_writer(path: Path, size: tuple[int, int], fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError(f'Unable to open video writer: {path}')
    return writer


def configure_kinect_dlls(base_dir: Path) -> None:
    add_dll_directory = getattr(os, 'add_dll_directory', None)
    if add_dll_directory is None:
        return

    dll_paths = [
        base_dir / 'libfreenect' / 'build' / 'lib' / 'Release',
        Path(r'C:\vcpkg\installed\x64-windows\bin'),
        base_dir / '.venv' / 'Lib' / 'site-packages',
    ]
    for dll_path in dll_paths:
        if dll_path.exists():
            add_dll_directory(str(dll_path))


@contextmanager
def suppress_stderr_fd():
    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        yield
        return

    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, 'w', encoding='utf-8') as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        try:
            os.dup2(saved_fd, stderr_fd)
        finally:
            os.close(saved_fd)


def is_highgui_available() -> bool:
    try:
        cv2.namedWindow('__cv_test__', cv2.WINDOW_NORMAL)
        test_frame = np.zeros((120, 200, 3), dtype=np.uint8)
        cv2.imshow('__cv_test__', test_frame)
        cv2.waitKey(1)
        cv2.destroyWindow('__cv_test__')
        return True
    except Exception:
        return False


def safe_destroy_windows() -> None:
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    root = _PREVIEW_STATE.get('tk_root')
    if root is not None:
        try:
            root.destroy()
        except Exception:
            pass
    reset_preview_window_state()


def clamp_text(text: str, limit: int) -> str:
    clean = ' '.join(str(text or '').split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + '...'


def probe_v1_available(base_dir: Path) -> bool:
    try:
        configure_kinect_dlls(base_dir)
        with suppress_stderr_fd():
            import cython0 as freenect
            result = freenect.sync_get_video(index=0, format=freenect.VIDEO_RGB)
            try:
                if hasattr(freenect, 'sync_stop'):
                    freenect.sync_stop()
            except Exception:
                pass
        if result is None:
            return False
        frame, _ = result
        return frame is not None
    except Exception:
        return False


def fit_to_pane(frame, pane_size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = pane_size
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = min(float(target_w) / float(w), float(target_h) / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def draw_pane_label(frame: np.ndarray, label: str) -> np.ndarray:
    out = frame.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (18, 18, 18), -1)
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
    return out


def placeholder_pane(pane_size: tuple[int, int], label: str, message: str) -> np.ndarray:
    target_w, target_h = pane_size
    frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    frame[:] = (20, 24, 34)
    cv2.rectangle(frame, (14, 14), (target_w - 14, target_h - 14), (56, 72, 98), 2)
    cv2.putText(frame, label, (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 240), 2, cv2.LINE_AA)
    brief = (message or 'waiting').strip()[:48]
    cv2.putText(frame, brief, (22, min(target_h - 24, 84)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (178, 196, 220), 1, cv2.LINE_AA)
    return frame


def format_stream_status_line(streams) -> str:
    status_parts = []
    for stream in streams.values():
        status = stream.get('status', {})
        backend_label = str(stream.get('backend', '?')).upper()
        state = str(status.get('status', 'unknown')).upper()
        message = clamp_text(status.get('message', ''), 34)
        if message:
            status_parts.append(f'{backend_label}: {state} - {message}')
        else:
            status_parts.append(f'{backend_label}: {state}')
    return ' | '.join(status_parts)


def resize_to_box(frame: np.ndarray, box_size: tuple[int, int], allow_upscale: bool = False) -> np.ndarray:
    box_w, box_h = box_size
    height, width = frame.shape[:2]
    if box_w <= 0 or box_h <= 0:
        return frame
    if not allow_upscale and width <= box_w and height <= box_h:
        return frame

    scale = min(float(box_w) / float(width), float(box_h) / float(height))
    if not allow_upscale:
        scale = min(scale, 1.0)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (new_w, new_h), interpolation=interpolation)


def scale_to_fit(frame: np.ndarray, max_size: tuple[int, int]) -> np.ndarray:
    return resize_to_box(frame, max_size, allow_upscale=False)


def build_preview_boot_frame(message: str) -> np.ndarray:
    width, height = PREVIEW_BOOT_SIZE
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (17, 21, 30)
    cv2.rectangle(frame, (20, 20), (width - 20, height - 20), (62, 84, 113), 2)
    cv2.putText(frame, PREVIEW_WINDOW_NAME, (28, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (240, 243, 248), 2, cv2.LINE_AA)
    cv2.putText(frame, clamp_text(message, 58), (28, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (181, 198, 220), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Waiting for Kinect stream...', (28, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (155, 173, 195), 1, cv2.LINE_AA)
    cv2.putText(frame, 'Press q to stop preview recording', (28, height - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (143, 158, 180), 1, cv2.LINE_AA)
    return frame


def ensure_preview_window() -> None:
    global _PREVIEW_WINDOW_CREATED
    if _PREVIEW_WINDOW_CREATED:
        return

    if hasattr(cv2, 'startWindowThread'):
        try:
            cv2.startWindowThread()
        except Exception:
            pass

    window_flags = cv2.WINDOW_NORMAL
    if hasattr(cv2, 'WINDOW_KEEPRATIO'):
        window_flags |= cv2.WINDOW_KEEPRATIO
    cv2.namedWindow(PREVIEW_WINDOW_NAME, window_flags)
    cv2.resizeWindow(PREVIEW_WINDOW_NAME, PREVIEW_BOOT_SIZE[0], PREVIEW_BOOT_SIZE[1])
    try:
        cv2.moveWindow(PREVIEW_WINDOW_NAME, 80, 80)
    except Exception:
        pass
    _PREVIEW_WINDOW_CREATED = True


def _on_tk_preview_close() -> None:
    _PREVIEW_STATE['tk_closed'] = True


def _on_tk_preview_key(event) -> None:
    if str(getattr(event, 'keysym', '')).lower() in {'q', 'escape'}:
        _PREVIEW_STATE['tk_closed'] = True


def _on_tk_preview_resize(event) -> None:
    if event.widget != _PREVIEW_STATE.get('tk_root'):
        return
    _PREVIEW_STATE['target_size'] = (max(1, int(event.width)), max(1, int(event.height)))


def ensure_tk_preview_window() -> None:
    if _PREVIEW_STATE.get('tk_root') is not None:
        return

    import tkinter as tk  # pylint: disable=import-outside-toplevel
    from PIL import Image, ImageTk  # pylint: disable=import-outside-toplevel

    root = tk.Tk()
    root.title(PREVIEW_WINDOW_NAME)
    root.geometry(f'{PREVIEW_BOOT_SIZE[0]}x{PREVIEW_BOOT_SIZE[1]}+80+80')
    root.configure(bg='#10141c')
    label = tk.Label(root, bg='#10141c', bd=0, highlightthickness=0)
    label.pack(fill='both', expand=True)
    root.protocol('WM_DELETE_WINDOW', _on_tk_preview_close)
    root.bind('<Configure>', _on_tk_preview_resize)
    root.bind('<Key>', _on_tk_preview_key)
    label.bind('<Key>', _on_tk_preview_key)
    root.focus_force()

    _PREVIEW_STATE['backend'] = 'tk'
    _PREVIEW_STATE['tk_root'] = root
    _PREVIEW_STATE['tk_label'] = label
    _PREVIEW_STATE['tk_modules'] = (Image, ImageTk)
    _PREVIEW_STATE['tk_closed'] = False
    _PREVIEW_STATE['size'] = None
    _PREVIEW_STATE['target_size'] = PREVIEW_BOOT_SIZE


def reset_preview_window_state() -> None:
    global _PREVIEW_WINDOW_CREATED
    _PREVIEW_WINDOW_CREATED = False
    _PREVIEW_STATE['backend'] = None
    _PREVIEW_STATE['tk_root'] = None
    _PREVIEW_STATE['tk_label'] = None
    _PREVIEW_STATE['tk_photo'] = None
    _PREVIEW_STATE['tk_modules'] = None
    _PREVIEW_STATE['tk_closed'] = False
    _PREVIEW_STATE['size'] = None
    _PREVIEW_STATE['target_size'] = None


def build_preview_frame(
    combined_frame: np.ndarray,
    streams,
    frame_count: int,
    started_at: float,
) -> np.ndarray:
    scaled_combined = scale_to_fit(combined_frame, PREVIEW_MAX_SIZE)
    footer_height = max(88, int(round(scaled_combined.shape[1] * 0.08)))
    width = scaled_combined.shape[1]
    footer = np.zeros((footer_height, width, 3), dtype=np.uint8)
    footer[:] = (16, 20, 28)

    elapsed = max(0.0, time.time() - started_at)
    actual_fps = float(frame_count) / elapsed if elapsed > 0.0 else 0.0

    rec_font = max(0.72, width / 1200.0)
    main_font = max(0.6, width / 1550.0)
    sub_font = max(0.5, width / 1750.0)
    rec_y = max(30, int(round(footer_height * 0.36)))
    line2_y = max(54, int(round(footer_height * 0.67)))
    line3_y = max(74, int(round(footer_height * 0.9)))

    cv2.putText(footer, 'REC', (18, rec_y), cv2.FONT_HERSHEY_SIMPLEX, rec_font, (40, 90, 255), 2, cv2.LINE_AA)
    cv2.putText(
        footer,
        f'Frames {frame_count}   Time {elapsed:0.1f}s   FPS {actual_fps:0.1f}',
        (int(round(width * 0.1)), rec_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        main_font,
        (236, 240, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        clamp_text(format_stream_status_line(streams), max(56, int(width / 10))),
        (18, line2_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        sub_font,
        (172, 188, 209),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        'Press q or Esc to stop preview recording',
        (18, line3_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        sub_font,
        (143, 158, 180),
        1,
        cv2.LINE_AA,
    )

    return cv2.vconcat([scaled_combined, footer])


def _show_preview_frame_cv2(preview_frame: np.ndarray) -> bool:
    ensure_preview_window()
    cv2.imshow(PREVIEW_WINDOW_NAME, preview_frame)
    key = cv2.waitKey(1) & 0xFF
    return key == ord('q')


def _show_preview_frame_tk(preview_frame: np.ndarray) -> bool:
    ensure_tk_preview_window()
    root = _PREVIEW_STATE['tk_root']
    label = _PREVIEW_STATE['tk_label']
    image_module, imagetk_module = _PREVIEW_STATE['tk_modules']

    if _PREVIEW_STATE.get('tk_closed'):
        return True

    target_size = _PREVIEW_STATE.get('target_size') or PREVIEW_BOOT_SIZE
    fitted_frame = resize_to_box(preview_frame, target_size, allow_upscale=True)
    canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    y0 = max(0, (target_size[1] - fitted_frame.shape[0]) // 2)
    x0 = max(0, (target_size[0] - fitted_frame.shape[1]) // 2)
    canvas[y0:y0 + fitted_frame.shape[0], x0:x0 + fitted_frame.shape[1]] = fitted_frame

    rgb_frame = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    image = image_module.fromarray(rgb_frame)
    photo = imagetk_module.PhotoImage(image=image)
    label.configure(image=photo)
    label.image = photo
    _PREVIEW_STATE['tk_photo'] = photo

    size = target_size
    if _PREVIEW_STATE.get('size') != size:
        root.geometry(f'{size[0]}x{size[1]}+80+80')
        _PREVIEW_STATE['size'] = size

    root.update_idletasks()
    root.update()
    return bool(_PREVIEW_STATE.get('tk_closed'))


def show_preview_frame(preview_frame: np.ndarray) -> bool:
    backend = _PREVIEW_STATE.get('backend')
    if backend == 'tk':
        return _show_preview_frame_tk(preview_frame)

    try:
        _PREVIEW_STATE['backend'] = 'cv2'
        return _show_preview_frame_cv2(preview_frame)
    except Exception:
        safe_destroy_windows()
        ensure_tk_preview_window()
        return _show_preview_frame_tk(preview_frame)


def with_backend_env(backend_name: str):
    previous = os.getenv('KINECT_BACKEND')
    if backend_name == 'v1':
        os.environ['KINECT_BACKEND'] = 'kinect_v1'
    elif backend_name == 'v2':
        os.environ['KINECT_BACKEND'] = 'kinect_v2'
    else:
        os.environ.pop('KINECT_BACKEND', None)
    return previous


def restore_backend_env(previous):
    if previous is None:
        os.environ.pop('KINECT_BACKEND', None)
    else:
        os.environ['KINECT_BACKEND'] = previous


def load_kinect_service():
    from src.vision.kinect_service import KinectService

    class StrictKinectService(KinectService):
        def __init__(self, base_dir, forced_backend):
            self._forced_backend = forced_backend
            super().__init__(base_dir)

        def _kinect_backend_candidates(self):
            if self._forced_backend == 'v1':
                return [self.KINECT_BACKEND_V1]
            if self._forced_backend == 'v2':
                return [self.KINECT_BACKEND_V2]
            return super()._kinect_backend_candidates()

    return StrictKinectService


def create_stream(base_dir: Path, backend_name: str):
    KinectService = load_kinect_service()
    previous_env = with_backend_env(backend_name)
    try:
        service = KinectService(base_dir, forced_backend=backend_name)
    finally:
        restore_backend_env(previous_env)

    return {
        'backend': backend_name,
        'service': service,
        'last_seq': -1,
        'status': {'status': 'connecting', 'message': 'Connecting...'},
        'latest_rgb': None,
        'latest_nir': None,
    }


def correct_stream_orientation(stream, rgb, nir):
    if stream.get('backend') != 'v2':
        return rgb, nir

    corrected_rgb = None if rgb is None else cv2.flip(rgb, 1)
    corrected_nir = None if nir is None else cv2.flip(nir, 1)
    return corrected_rgb, corrected_nir


def poll_stream(stream) -> bool:
    service = stream['service']
    status = service.get_status()
    stream['status'] = status

    bundle = service.get_latest_frame_bundle()
    seq = int(bundle.get('frame_seq', 0))
    rgb = bundle.get('color_frame')
    nir = bundle.get('depth_visual_frame')
    rgb, nir = correct_stream_orientation(stream, rgb, nir)

    if status.get('status') != 'connected':
        return False
    if rgb is None or nir is None:
        return False
    if seq == stream['last_seq']:
        return False

    stream['last_seq'] = seq
    stream['latest_rgb'] = rgb
    stream['latest_nir'] = nir
    return True


def build_single_frame(stream, pane_size: tuple[int, int]) -> np.ndarray:
    backend_label = stream['backend'].upper()
    rgb = stream.get('latest_rgb')
    nir = stream.get('latest_nir')
    if rgb is None:
        rgb = placeholder_pane(pane_size, f'{backend_label} RGB', stream['status'].get('message', 'waiting'))
    else:
        rgb = draw_pane_label(fit_to_pane(rgb, pane_size), f'{backend_label} RGB')

    if nir is None:
        nir = placeholder_pane(pane_size, f'{backend_label} NIR', stream['status'].get('message', 'waiting'))
    else:
        nir = draw_pane_label(fit_to_pane(nir, pane_size), f'{backend_label} NIR')

    return cv2.hconcat([rgb, nir])


def build_quad_frame(stream_v1, stream_v2, pane_size: tuple[int, int]) -> np.ndarray:
    v1_rgb = stream_v1.get('latest_rgb')
    v1_nir = stream_v1.get('latest_nir')
    v2_rgb = stream_v2.get('latest_rgb')
    v2_nir = stream_v2.get('latest_nir')

    pane_v1_rgb = draw_pane_label(fit_to_pane(v1_rgb, pane_size), 'V1 RGB') if v1_rgb is not None else placeholder_pane(
        pane_size,
        'V1 RGB',
        stream_v1['status'].get('message', 'waiting'),
    )
    pane_v1_nir = draw_pane_label(fit_to_pane(v1_nir, pane_size), 'V1 NIR') if v1_nir is not None else placeholder_pane(
        pane_size,
        'V1 NIR',
        stream_v1['status'].get('message', 'waiting'),
    )
    pane_v2_rgb = draw_pane_label(fit_to_pane(v2_rgb, pane_size), 'V2 RGB') if v2_rgb is not None else placeholder_pane(
        pane_size,
        'V2 RGB',
        stream_v2['status'].get('message', 'waiting'),
    )
    pane_v2_nir = draw_pane_label(fit_to_pane(v2_nir, pane_size), 'V2 NIR') if v2_nir is not None else placeholder_pane(
        pane_size,
        'V2 NIR',
        stream_v2['status'].get('message', 'waiting'),
    )

    top = cv2.hconcat([pane_v1_rgb, pane_v1_nir])
    bottom = cv2.hconcat([pane_v2_rgb, pane_v2_nir])
    return cv2.vconcat([top, bottom])


def build_combined_frame(streams, pane_size: tuple[int, int]) -> np.ndarray:
    if 'v1' in streams and 'v2' in streams:
        return build_quad_frame(streams['v1'], streams['v2'], pane_size)

    backend_name = next(iter(streams.keys()))
    return build_single_frame(streams[backend_name], pane_size)


def wait_until_any_stream_ready(streams, pane_size: tuple[int, int], preview: bool) -> bool:
    last_print_at = 0.0
    preview_started_at = time.time()
    while True:
        ready = []
        for stream in streams.values():
            poll_stream(stream)
            if stream.get('latest_rgb') is not None and stream.get('latest_nir') is not None:
                ready.append(stream['backend'])
        if ready:
            return True

        if preview:
            try:
                combined_frame = build_combined_frame(streams, pane_size)
                preview_frame = build_preview_frame(
                    combined_frame=combined_frame,
                    streams=streams,
                    frame_count=0,
                    started_at=preview_started_at,
                )
                if show_preview_frame(preview_frame):
                    return False
            except Exception as exc:
                preview = False
                print(f'[WARN] Preview disabled while waiting for stream: {exc}')

        now = time.time()
        if now - last_print_at >= 1.0:
            status_text = ' | '.join(
                f"{stream['backend']}:{stream['status'].get('status', 'unknown')} ({stream['status'].get('message', '')})"
                for stream in streams.values()
            )
            print(f'[INFO] Waiting stream: {status_text}', flush=True)
            last_print_at = now
        time.sleep(0.05)


def record_loop(
    streams,
    output_dir: Path,
    fps: float,
    preview: bool,
) -> tuple[Path | None, int]:
    pane_size = (640, 480)
    if not wait_until_any_stream_ready(streams, pane_size, preview):
        return None, 0

    dual_mode = 'v1' in streams and 'v2' in streams
    if dual_mode:
        backend_name = 'v1_v2'
        layout_name = 'quad'
    else:
        backend_name = next(iter(streams.keys()))
        layout_name = 'side_by_side'

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    combined_path = output_dir / f'rgb_nir_{layout_name}_{timestamp}_{backend_name}.mp4'

    combined_writer = None
    frame_count = 0
    last_print_at = time.time()
    started_at = time.time()
    latest_combined_frame = None
    next_write_at = None
    frame_interval = 1.0 / max(1.0, float(fps))

    try:
        while True:
            changed = False
            for stream in streams.values():
                changed = poll_stream(stream) or changed

            if changed or latest_combined_frame is None:
                latest_combined_frame = build_combined_frame(streams, pane_size)

            if latest_combined_frame is None:
                time.sleep(0.002)
                continue

            if combined_writer is None:
                out_h, out_w = latest_combined_frame.shape[:2]
                combined_writer = create_writer(combined_path, (out_w, out_h), fps)
                started_at = time.time()
                next_write_at = started_at
                print(f'[INFO] Start recording ({layout_name})')
                print(f'[INFO] Output video -> {combined_path}')
                print(f'[INFO] Output FPS {fps:0.2f}; duplicate frames preserve real recording duration.')
                if dual_mode:
                    print('[INFO] Panes: V1 RGB | V1 NIR | V2 RGB | V2 NIR')
                else:
                    print(f'[INFO] Left = {backend_name.upper()} RGB, Right = {backend_name.upper()} NIR')
                print('[INFO] Press Ctrl+C to stop')
                if preview:
                    print('[INFO] Preview enabled, press q to stop')

            wrote_frame = False
            now = time.time()
            while next_write_at is not None and now >= next_write_at:
                combined_writer.write(latest_combined_frame)
                frame_count += 1
                wrote_frame = True
                next_write_at += frame_interval

            if preview and (changed or wrote_frame):
                try:
                    preview_frame = build_preview_frame(
                        combined_frame=latest_combined_frame,
                        streams=streams,
                        frame_count=frame_count,
                        started_at=started_at,
                    )
                    if show_preview_frame(preview_frame):
                        break
                except Exception as exc:
                    preview = False
                    print(f'[WARN] Preview disabled during recording: {exc}')

            now = time.time()
            if now - last_print_at >= 1.0:
                print(f'[INFO] Recorded frames: {frame_count}', flush=True)
                last_print_at = now

            if not changed and not wrote_frame and next_write_at is not None:
                sleep_seconds = max(0.001, min(0.01, next_write_at - now))
                time.sleep(sleep_seconds)
    finally:
        if combined_writer is not None:
            combined_writer.release()
        if preview:
            safe_destroy_windows()

    return combined_path, frame_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Record system RGB/NIR stream to reels/recordings/*.mp4 (side-by-side or quad)')
    parser.add_argument('--backend', choices=('auto', 'v2', 'v1'), default='auto', help='Kinect backend selection')
    parser.add_argument('--fps', type=float, default=30.0, help='Output video FPS; frames are duplicated as needed to preserve real recording duration.')
    parser.add_argument('--no-preview', action='store_true', help='Disable realtime preview window')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reels_dir = Path(__file__).resolve().parent
    output_dir = reels_dir / 'recordings'
    base_dir = reels_dir.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    add_project_root_to_path(base_dir)
    configure_kinect_dlls(base_dir)

    preview_enabled = not bool(args.no_preview)
    if preview_enabled and not is_highgui_available():
        print('[WARN] OpenCV GUI backend unavailable. Will try tkinter preview fallback.')
    if preview_enabled:
        try:
            show_preview_frame(build_preview_boot_frame('Initializing preview window...'))
            preview_backend = _PREVIEW_STATE.get('backend') or 'unknown'
            print(f'[INFO] Preview window ready: {PREVIEW_WINDOW_NAME} ({preview_backend})')
        except Exception as exc:
            preview_enabled = False
            reset_preview_window_state()
            print(f'[WARN] Preview initialization failed: {exc}')

    try:
        streams = {}
        if args.backend == 'auto':
            v1_available = probe_v1_available(base_dir)
            if v1_available:
                try:
                    streams['v1'] = create_stream(base_dir=base_dir, backend_name='v1')
                except Exception as exc:  # pylint: disable=broad-except
                    print(f'[WARN] Kinect v1 stream init failed: {exc}')
            else:
                print('[INFO] Kinect v1 not detected, skip v1 stream.')
            try:
                streams['v2'] = create_stream(base_dir=base_dir, backend_name='v2')
            except Exception as exc:  # pylint: disable=broad-except
                print(f'[WARN] Kinect v2 stream init failed: {exc}')
            if not streams:
                # Last fallback: even if probe misses v1, try once.
                try:
                    streams['v1'] = create_stream(base_dir=base_dir, backend_name='v1')
                except Exception:
                    pass
        elif args.backend == 'v1':
            streams['v1'] = create_stream(base_dir=base_dir, backend_name='v1')
        elif args.backend == 'v2':
            streams['v2'] = create_stream(base_dir=base_dir, backend_name='v2')
        if not streams:
            print('[ERROR] No available Kinect backend.', file=sys.stderr)
            return 1
    except Exception as exc:  # pylint: disable=broad-except
        print(f'[ERROR] Unable to initialize Kinect streams: {exc}', file=sys.stderr)
        return 1

    try:
        combined_path, frame_count = record_loop(
            streams=streams,
            output_dir=output_dir,
            fps=max(1.0, float(args.fps)),
            preview=preview_enabled,
        )
        if combined_path is None:
            print('[INFO] Recording cancelled before first frame.')
            return 0
        print(f'[INFO] Saved video: {combined_path}')
        print(f'[INFO] Total recorded frames: {frame_count}')
        return 0
    except KeyboardInterrupt:
        print('\n[INFO] Recording stopped by user')
        return 0
    except Exception as exc:  # pylint: disable=broad-except
        print(f'[ERROR] Recording failed: {exc}', file=sys.stderr)
        return 1
    finally:
        for stream in streams.values():
            try:
                stream['service'].close()
            except Exception:
                pass
        safe_destroy_windows()
        reset_preview_window_state()


if __name__ == '__main__':
    raise SystemExit(main())
