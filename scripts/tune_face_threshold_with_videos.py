import argparse
import json
import statistics
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[1]

import sys

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from src.vision.face_recognition_db import FaceRecognitionDB


def read_image_unicode(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def jitter_variants(image):
    variants = [('orig', image)]

    for alpha in (0.82, 1.18):
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        variants.append((f'bright_{alpha:.2f}', adjusted))

    blur = cv2.GaussianBlur(image, (3, 3), 0)
    variants.append(('blur_3x3', blur))
    blur_heavy = cv2.GaussianBlur(image, (5, 5), 0)
    variants.append(('blur_5x5', blur_heavy))

    height, width = image.shape[:2]
    for angle in (-8.0, 8.0):
        matrix = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        variants.append((f'rotate_{angle:+.0f}', rotated))

    scale = 0.45
    resized_small = cv2.resize(image, (max(24, int(width * scale)), max(24, int(height * scale))), interpolation=cv2.INTER_AREA)
    restored = cv2.resize(resized_small, (width, height), interpolation=cv2.INTER_LINEAR)
    variants.append(('downscale_45pct', restored))

    kernel = np.zeros((9, 9), dtype=np.float32)
    kernel[4, :] = 1.0 / 9.0
    motion = cv2.filter2D(image, -1, kernel)
    variants.append(('motion_blur', motion))

    noise = np.random.normal(0, 9, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    variants.append(('gaussian_noise', noisy))

    crop_margin = int(min(height, width) * 0.08)
    if crop_margin > 0:
        cropped = image[crop_margin:height - crop_margin, crop_margin:width - crop_margin]
        if cropped.size > 0:
            cropped = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
            variants.append(('crop_8pct', cropped))

    encode_ok, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 58])
    if encode_ok:
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is not None:
            variants.append(('jpeg_q58', decoded))

    return variants


def extract_largest_face(analysis):
    faces = analysis.get('faces') or []
    if not faces:
        return None
    return max(
        faces,
        key=lambda item: (float(item['bbox'][2]) - float(item['bbox'][0])) * (float(item['bbox'][3]) - float(item['bbox'][1])),
    )


def collect_positive_embeddings(face_db: FaceRecognitionDB):
    positives = []
    for label in face_db.list_student_labels():
        folder = face_db.photo_root / label
        for image_path in sorted(folder.iterdir()):
            if image_path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp'}:
                continue
            image = read_image_unicode(image_path)
            if image is None:
                continue
            for variant_name, variant in jitter_variants(image):
                analysis = face_db.analyze_faces(variant)
                face = extract_largest_face(analysis)
                if face is None:
                    continue
                positives.append(
                    {
                        'label': label,
                        'source': f'{label}/{image_path.name}',
                        'variant': variant_name,
                        'embedding': face['embedding'],
                    }
                )
    return positives


def list_negative_video_paths(base_dir: Path):
    def pick_video(relative_candidates):
        for relative_path in relative_candidates:
            candidate = base_dir / relative_path
            if candidate.exists():
                return candidate
        return None

    candidates = [
        pick_video(
            [
                Path('data/test_videos/opencv_vtest.avi'),
            ]
        ),
        pick_video(
            [
                Path('data/test_videos/opencv_megamind.avi'),
            ]
        ),
        pick_video(
            [
                Path('data/test_videos/opencv_megamind_bugy.avi'),
            ]
        ),
        base_dir / 'data' / 'test_videos' / 'test_video.mp4',
        base_dir / 'data' / 'test_videos' / '_aliases' / '_video_test_source.mp4',
        base_dir / 'data' / 'test_videos' / '_aliases' / '_video_tuning_source.mp4',
        base_dir / 'test_video.mp4',
        base_dir / '_video_test_source.mp4',
        base_dir / '_video_tuning_source.mp4',
    ]
    return [path for path in candidates if path is not None and path.exists()]


def collect_negative_embeddings(face_db: FaceRecognitionDB, video_paths, step_seconds=1.4, max_faces_per_video=80):
    negatives = []
    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        step_frames = max(1, int(round(fps * step_seconds)))
        frame_index = 0
        collected = 0

        while collected < max_faces_per_video:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_index += 1
            if frame_index % step_frames != 0:
                continue

            analysis = face_db.analyze_faces(frame)
            faces = analysis.get('faces') or []
            if not faces:
                continue

            faces = sorted(
                faces,
                key=lambda item: (float(item['bbox'][2]) - float(item['bbox'][0])) * (float(item['bbox'][3]) - float(item['bbox'][1])),
                reverse=True,
            )
            for face in faces[:2]:
                negatives.append(
                    {
                        'video': str(video_path),
                        'frame': frame_index,
                        'embedding': face['embedding'],
                    }
                )
                collected += 1
                if collected >= max_faces_per_video:
                    break

        cap.release()
    return negatives


def evaluate_threshold(face_db: FaceRecognitionDB, positives, negatives, threshold: float):
    pos_total = len(positives)
    neg_total = len(negatives)

    true_accept = 0
    false_reject = 0
    wrong_identity = 0
    pos_similarities = []

    for item in positives:
        matches = face_db.match_embedding(item['embedding'], threshold=threshold)
        if not matches:
            false_reject += 1
            continue
        top = matches[0]
        pos_similarities.append(float(top.get('similarity', 0.0)))
        if top.get('label') == item['label']:
            true_accept += 1
        else:
            wrong_identity += 1

    false_accept = 0
    neg_similarities = []
    for item in negatives:
        matches = face_db.match_embedding(item['embedding'], threshold=threshold)
        if not matches:
            continue
        false_accept += 1
        neg_similarities.append(float(matches[0].get('similarity', 0.0)))

    tpr = true_accept / max(1, pos_total)
    far = false_accept / max(1, neg_total)
    wrong_rate = wrong_identity / max(1, pos_total)
    precision = true_accept / max(1, true_accept + wrong_identity + false_accept)

    score = (tpr * 100.0) - (far * 260.0) - (wrong_rate * 160.0) + (precision * 40.0)

    return {
        'threshold': round(threshold, 3),
        'score': round(score, 4),
        'positive_total': pos_total,
        'negative_total': neg_total,
        'true_accept': true_accept,
        'false_reject': false_reject,
        'wrong_identity': wrong_identity,
        'false_accept': false_accept,
        'tpr': round(tpr, 4),
        'far': round(far, 4),
        'wrong_rate': round(wrong_rate, 4),
        'precision': round(precision, 4),
        'pos_similarity_mean': round(statistics.fmean(pos_similarities), 4) if pos_similarities else 0.0,
        'neg_similarity_mean': round(statistics.fmean(neg_similarities), 4) if neg_similarities else 0.0,
    }


def select_best(results):
    ordered = sorted(
        results,
        key=lambda item: (
            item['score'],
            -item['far'],
            -item['wrong_rate'],
            item['precision'],
            item['threshold'],
        ),
        reverse=True,
    )
    return ordered[0], ordered


def main():
    parser = argparse.ArgumentParser(description='Tune face recognition threshold with student photos and test videos.')
    parser.add_argument('--video', action='append', default=[], help='Optional negative video path. Can be used multiple times.')
    parser.add_argument('--min-threshold', type=float, default=0.20)
    parser.add_argument('--max-threshold', type=float, default=0.90)
    parser.add_argument('--step', type=float, default=0.02)
    parser.add_argument('--video-step-seconds', type=float, default=1.4)
    parser.add_argument('--max-faces-per-video', type=int, default=90)
    args = parser.parse_args()

    face_db = FaceRecognitionDB(WORKSPACE)

    positives = collect_positive_embeddings(face_db)
    if args.video:
        video_paths = []
        for raw in args.video:
            path = Path(raw)
            if not path.is_absolute():
                path = WORKSPACE / path
            video_paths.append(path)
    else:
        video_paths = list_negative_video_paths(WORKSPACE)
    negatives = collect_negative_embeddings(
        face_db,
        video_paths,
        step_seconds=args.video_step_seconds,
        max_faces_per_video=max(1, args.max_faces_per_video),
    )

    print(json.dumps({
        'stage': 'dataset',
        'positive_embeddings': len(positives),
        'negative_embeddings': len(negatives),
        'negative_videos': [str(path) for path in video_paths],
    }, ensure_ascii=False))

    results = []
    threshold = float(args.min_threshold)
    max_threshold = float(args.max_threshold)
    step = max(0.005, float(args.step))

    while threshold <= max_threshold + 1e-9:
        result = evaluate_threshold(face_db, positives, negatives, threshold)
        results.append(result)
        print(json.dumps({'stage': 'eval', **result}, ensure_ascii=False))
        threshold += step

    best, ordered = select_best(results)
    top3 = ordered[:3]

    auto_relink_threshold = max(best['threshold'], 0.52)
    print(json.dumps({
        'stage': 'best',
        'best': best,
        'top3': top3,
        'recommended_recognition_threshold': best['threshold'],
        'recommended_auto_relink_threshold': round(auto_relink_threshold, 3),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
