import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from src.vision.rgb_depth_alignment import RgbDepthAligner  # noqa: E402


def resolve_video_paths(video_args):
    if video_args:
        paths = []
        for raw in video_args:
            path = Path(raw)
            if not path.is_absolute():
                path = WORKSPACE / path
            paths.append(path)
        return paths

    return sorted((WORKSPACE / 'reels' / 'recordings').glob('*v1_v2*.mp4'))


def quad_backend_slices(backend_name):
    normalized = str(backend_name or '').strip().lower()
    if normalized == 'v1':
        return (slice(0, 480), slice(0, 640)), (slice(0, 480), slice(640, 1280))
    if normalized == 'v2':
        return (slice(480, 960), slice(0, 640)), (slice(480, 960), slice(640, 1280))
    raise ValueError(f'Unsupported backend: {backend_name}')


def split_quad_backend_frame(frame, backend_name):
    rgb_slice, depth_slice = quad_backend_slices(backend_name)
    return frame[rgb_slice].copy(), frame[depth_slice].copy()


def non_empty_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return (gray > 8).astype(np.uint8)


def preprocess_for_ecc(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    return magnitude


def estimate_translation(rgb_frame, depth_frame):
    rgb_frame = rgb_frame[48:, :, :]
    depth_frame = depth_frame[48:, :, :]

    rgb_prepared = preprocess_for_ecc(rgb_frame)
    depth_prepared = preprocess_for_ecc(depth_frame)
    mask = (non_empty_mask(rgb_frame) & non_empty_mask(depth_frame))[48:, :]
    if int(mask.sum()) <= 0:
        return None

    def brute_force_translation():
        rgb_edges = cv2.Canny(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY), 40, 120)
        depth_edges = cv2.Canny(cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY), 40, 120)
        rgb_edges = cv2.dilate(rgb_edges, np.ones((3, 3), np.uint8), iterations=1)
        depth_edges = cv2.dilate(depth_edges, np.ones((3, 3), np.uint8), iterations=1)
        rgb_edges = (rgb_edges > 0).astype(np.float32)
        depth_edges = (depth_edges > 0).astype(np.float32)

        best_score = None
        best_shift = None
        for shift_y in range(-16, 17):
            for shift_x in range(-32, 33):
                if shift_x >= 0:
                    rgb_x = slice(shift_x, rgb_edges.shape[1])
                    depth_x = slice(0, rgb_edges.shape[1] - shift_x)
                else:
                    rgb_x = slice(0, rgb_edges.shape[1] + shift_x)
                    depth_x = slice(-shift_x, rgb_edges.shape[1])

                if shift_y >= 0:
                    rgb_y = slice(shift_y, rgb_edges.shape[0])
                    depth_y = slice(0, rgb_edges.shape[0] - shift_y)
                else:
                    rgb_y = slice(0, rgb_edges.shape[0] + shift_y)
                    depth_y = slice(-shift_y, rgb_edges.shape[0])

                rgb_view = rgb_edges[rgb_y, rgb_x]
                depth_view = depth_edges[depth_y, depth_x]
                if rgb_view.size == 0 or depth_view.size == 0:
                    continue

                intersection = float((rgb_view * depth_view).sum())
                union = float(rgb_view.sum() + depth_view.sum() - intersection)
                score = intersection / max(1.0, union)
                if best_score is None or score > best_score:
                    best_score = score
                    best_shift = (shift_x, shift_y)

        if best_shift is None:
            return None
        return {
            'shift_x': float(best_shift[0]),
            'shift_y': float(best_shift[1]),
            'correlation': float(best_score or 0.0),
        }

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        80,
        1e-5,
    )
    try:
        correlation, warp = cv2.findTransformECC(
            rgb_prepared,
            depth_prepared,
            warp,
            cv2.MOTION_TRANSLATION,
            criteria,
            mask,
            5,
        )
    except cv2.error:
        return brute_force_translation()

    result = {
        'shift_x': float(warp[0, 2]),
        'shift_y': float(warp[1, 2]),
        'correlation': float(correlation),
    }
    if result['correlation'] < 0.18:
        fallback = brute_force_translation()
        if fallback is not None and fallback['correlation'] > result['correlation']:
            return fallback
    return result


def sample_video_alignment(video_path, backend_name, sample_count):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open video: {video_path}')

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        cap.release()
        return []

    usable_start = max(0, min(frame_count - 1, int(round(frame_count * 0.08))))
    usable_end = max(usable_start, min(frame_count - 1, int(round(frame_count * 0.92))))
    indexes = np.linspace(usable_start, usable_end, num=max(1, sample_count), dtype=np.int32).tolist()

    results = []
    for index in indexes:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        rgb_frame, depth_frame = split_quad_backend_frame(frame, backend_name)
        estimate = estimate_translation(rgb_frame, depth_frame)
        if estimate is None:
            continue
        results.append(
            {
                'frame_index': int(index),
                **estimate,
            }
        )

    cap.release()
    return results


def summarize_backend(estimates):
    if not estimates:
        return None

    shift_x_values = [item['shift_x'] for item in estimates]
    shift_y_values = [item['shift_y'] for item in estimates]
    correlation_values = [item['correlation'] for item in estimates]
    return {
        'shift_x': round(float(statistics.median(shift_x_values)), 3),
        'shift_y': round(float(statistics.median(shift_y_values)), 3),
        'correlation_mean': round(float(statistics.fmean(correlation_values)), 4),
        'correlation_median': round(float(statistics.median(correlation_values)), 4),
        'samples': len(estimates),
    }


def write_profiles(summary_by_backend):
    aligner = RgbDepthAligner(WORKSPACE)
    payload = aligner.default_payload()
    mapping = {
        'v1': 'kinect_v1',
        'v2': 'kinect_v2',
    }
    for backend_name, summary in summary_by_backend.items():
        target_key = mapping[backend_name]
        if summary is None:
            continue
        payload[target_key]['shift_x'] = summary['shift_x']
        payload[target_key]['shift_y'] = summary['shift_y']

    aligner.profile_file.parent.mkdir(parents=True, exist_ok=True)
    with aligner.profile_file.open('w', encoding='utf-8') as profile_file:
        json.dump(payload, profile_file, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Estimate fine RGB/depth translation offsets from V1/V2 quad recordings.')
    parser.add_argument('--video', action='append', default=[], help='Quad recording path. Can be used multiple times.')
    parser.add_argument('--samples', type=int, default=10, help='Sampled frames per video/backend.')
    parser.add_argument('--write', action='store_true', help='Write median shifts into data/kinect_alignment_profiles.json.')
    args = parser.parse_args()

    video_paths = resolve_video_paths(args.video)
    if not video_paths:
        raise RuntimeError('No quad recordings found.')

    all_estimates = {
        'v1': [],
        'v2': [],
    }
    for video_path in video_paths:
        for backend_name in ('v1', 'v2'):
            estimates = sample_video_alignment(video_path, backend_name, sample_count=max(1, args.samples))
            all_estimates[backend_name].extend(estimates)
            print(json.dumps({
                'stage': 'video_backend',
                'video': str(video_path),
                'backend': backend_name,
                'samples': len(estimates),
                'estimates': estimates[:5],
            }, ensure_ascii=False))

    summary_by_backend = {
        backend_name: summarize_backend(estimates)
        for backend_name, estimates in all_estimates.items()
    }
    print(json.dumps({
        'stage': 'summary',
        'summary': summary_by_backend,
    }, ensure_ascii=False))

    if args.write:
        write_profiles(summary_by_backend)
        print(json.dumps({
            'stage': 'written',
            'profile_file': str(WORKSPACE / 'data' / 'kinect_alignment_profiles.json'),
        }, ensure_ascii=False))


if __name__ == '__main__':
    main()
