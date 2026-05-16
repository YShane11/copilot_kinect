import argparse
import csv
import json
from pathlib import Path

import cv2

WORKSPACE = Path(__file__).resolve().parents[1]

import sys

if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from src.vision.face_recognition_db import FaceRecognitionDB


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def crop_face(frame, bbox):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return frame[y1:y2, x1:x2]


def compare_video(face_db, video_path: Path, output_dir: Path, old_threshold: float, new_threshold: float, step_seconds: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open video: {video_path}')

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    step_frames = max(1, int(round(fps * step_seconds)))

    snapshot_dir = output_dir / video_path.stem
    ensure_dir(snapshot_dir)
    csv_path = output_dir / f'{video_path.stem}_compare.csv'

    rows = []
    frame_index = 0
    saved_faces = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_index += 1
        if frame_index % step_frames != 0:
            continue

        analysis = face_db.analyze_faces(frame)
        if analysis.get('status') != 'ok':
            continue
        faces = analysis.get('faces') or []
        if not faces:
            continue

        for face_index, face in enumerate(faces, start=1):
            embedding = face.get('embedding')
            if not embedding:
                continue

            all_matches = face_db.match_embedding(embedding, threshold=-1.0)
            if all_matches:
                top = all_matches[0]
                top_label = top.get('label', '')
                top_similarity = float(top.get('similarity', 0.0))
            else:
                top_label = ''
                top_similarity = 0.0

            old_pass = top_similarity >= old_threshold
            new_pass = top_similarity >= new_threshold

            face_crop = crop_face(frame, face['bbox'])
            snapshot_name = f'frame_{frame_index:06d}_face_{face_index:02d}.jpg'
            snapshot_path = snapshot_dir / snapshot_name
            if face_crop.size > 0:
                cv2.imwrite(str(snapshot_path), face_crop)
                saved_faces += 1
            else:
                snapshot_path = None

            rows.append(
                {
                    'video': str(video_path),
                    'frame': frame_index,
                    'time_sec': round(frame_index / max(1e-6, fps), 3),
                    'face_index': face_index,
                    'top_label': top_label,
                    'top_similarity': round(top_similarity, 4),
                    'old_threshold': old_threshold,
                    'new_threshold': new_threshold,
                    'old_pass': int(old_pass),
                    'new_pass': int(new_pass),
                    'snapshot': str(snapshot_path) if snapshot_path is not None else '',
                }
            )

    cap.release()

    with csv_path.open('w', encoding='utf-8', newline='') as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                'video',
                'frame',
                'time_sec',
                'face_index',
                'top_label',
                'top_similarity',
                'old_threshold',
                'new_threshold',
                'old_pass',
                'new_pass',
                'snapshot',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    old_pass_count = sum(1 for row in rows if int(row['old_pass']) == 1)
    new_pass_count = sum(1 for row in rows if int(row['new_pass']) == 1)
    changed_count = sum(1 for row in rows if int(row['old_pass']) != int(row['new_pass']))

    return {
        'video': str(video_path),
        'rows': len(rows),
        'saved_faces': saved_faces,
        'old_pass_count': old_pass_count,
        'new_pass_count': new_pass_count,
        'changed_count': changed_count,
        'csv': str(csv_path),
        'snapshot_dir': str(snapshot_dir),
    }


def main():
    parser = argparse.ArgumentParser(description='Compare old/new recognition thresholds with video face snapshots.')
    parser.add_argument('--video', action='append', required=True, help='Video path (absolute or workspace-relative). Can be passed multiple times.')
    parser.add_argument('--old-threshold', type=float, default=0.54)
    parser.add_argument('--new-threshold', type=float, default=0.38)
    parser.add_argument('--step-seconds', type=float, default=0.9)
    parser.add_argument('--output-dir', default='data/recognition_snapshot_compare')
    args = parser.parse_args()

    face_db = FaceRecognitionDB(WORKSPACE)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = WORKSPACE / output_dir
    ensure_dir(output_dir)

    results = []
    for raw_video in args.video:
        video_path = Path(raw_video)
        if not video_path.is_absolute():
            video_path = WORKSPACE / video_path
        result = compare_video(
            face_db,
            video_path=video_path,
            output_dir=output_dir,
            old_threshold=float(args.old_threshold),
            new_threshold=float(args.new_threshold),
            step_seconds=float(args.step_seconds),
        )
        results.append(result)
        print(json.dumps({'stage': 'video_done', **result}, ensure_ascii=False))

    print(json.dumps({'stage': 'done', 'results': results}, ensure_ascii=False))


if __name__ == '__main__':
    main()
