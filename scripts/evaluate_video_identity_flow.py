import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import KMeans

WORKSPACE = Path(__file__).resolve().parents[1]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from scripts import tune_with_video as tuning  # noqa: E402
from src.vision.face_recognition_db import FaceRecognitionDB  # noqa: E402


FOCUSED_BEST_PARAMS = {
    'max_inference_width': 1600,
    'yolo_image_size': 640,
    'yolo_detect_interval': 0.18,
    'yolo_confidence': 0.28,
    'track_iou_threshold': 0.08,
    'temp_person_timeout': 4.5,
    'confirmed_absent_timeout': 12.0,
    'recognition_threshold': 0.44,
    'auto_relink_threshold': 0.52,
    'auto_relink_interval': 2.8,
    'max_detections': 10,
    'min_person_box_width': 32.0,
    'min_person_box_height': 36.0,
    'detection_duplicate_iou_threshold': 0.64,
    'detection_duplicate_center_ratio': 0.26,
    'detection_duplicate_area_ratio': 2.2,
    'temporary_merge_iou_threshold': 0.50,
    'temporary_merge_distance_ratio': 0.30,
    'face_person_fallback_interval': 0.70,
    'face_person_fallback_min_score': 0.45,
    'face_person_fallback_min_size': 12.0,
    'face_person_fallback_box_scale_x': 3.0,
    'face_person_fallback_box_top_scale': 0.45,
    'face_person_fallback_box_bottom_scale': 2.35,
}


@dataclass
class FaceSample:
    frame_index: int
    time_seconds: float
    bbox: list
    det_score: float
    embedding: list
    cluster: int | None = None


def cosine_similarity(left, right):
    left_arr = np.asarray(left, dtype=np.float32)
    right_arr = np.asarray(right, dtype=np.float32)
    left_norm = float(np.linalg.norm(left_arr))
    right_norm = float(np.linalg.norm(right_arr))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return float(np.dot(left_arr, right_arr) / (left_norm * right_norm))


def normalized_mean(embeddings):
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.size == 0:
        return []
    mean = arr.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm > 0.0:
        mean = mean / norm
    return mean.astype(float).tolist()


def resolve_video_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = WORKSPACE / path
    if not path.exists():
        raise FileNotFoundError(f'Video not found: {path}')
    return path


def collect_face_samples(video_path: Path, face_db: FaceRecognitionDB, sample_step_seconds: float, min_det_score: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open video: {video_path}')

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step_frames = max(1, int(round(fps * max(0.05, float(sample_step_seconds)))))
    samples = []
    frame_index = 0

    while True:
        ok, combined = cap.read()
        if not ok or combined is None:
            break
        if frame_index % step_frames != 0:
            frame_index += 1
            continue

        rgb, _ = tuning.decode_side_by_side_frame(combined)
        analysis = face_db.analyze_faces(rgb)
        for face in analysis.get('faces') or []:
            det_score = float(face.get('det_score', 0.0))
            if det_score < float(min_det_score):
                continue
            samples.append(
                FaceSample(
                    frame_index=frame_index,
                    time_seconds=frame_index / max(1e-6, fps),
                    bbox=[float(value) for value in face.get('bbox', [])],
                    det_score=det_score,
                    embedding=list(face.get('embedding') or []),
                )
            )
        frame_index += 1

    cap.release()
    return samples, {'fps': fps, 'frames': total_frames, 'duration': total_frames / fps if fps > 0 else 0.0}


def assign_clusters(samples, expected_people):
    if len(samples) < expected_people:
        raise RuntimeError(f'Not enough face samples for {expected_people} identities.')

    vectors = np.asarray([sample.embedding for sample in samples], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-9)
    model = KMeans(n_clusters=expected_people, random_state=42, n_init=20)
    labels = model.fit_predict(vectors)
    for sample, label in zip(samples, labels):
        sample.cluster = int(label)

    cluster_summaries = []
    for cluster_id in sorted(set(int(item) for item in labels)):
        members = [sample for sample in samples if sample.cluster == cluster_id]
        centers_x = [
            (float(sample.bbox[0]) + float(sample.bbox[2])) * 0.5
            for sample in members
            if len(sample.bbox) == 4
        ]
        cluster_summaries.append(
            {
                'cluster': cluster_id,
                'samples': len(members),
                'first_time': round(min(sample.time_seconds for sample in members), 3),
                'last_time': round(max(sample.time_seconds for sample in members), 3),
                'mean_center_x': round(statistics.fmean(centers_x), 2) if centers_x else 0.0,
                'mean_det_score': round(statistics.fmean(sample.det_score for sample in members), 4),
            }
        )

    ordered = sorted(cluster_summaries, key=lambda item: item['mean_center_x'])
    cluster_to_identity = {
        item['cluster']: f'VideoPerson{index + 1:02d}'
        for index, item in enumerate(ordered)
    }
    return cluster_to_identity, cluster_summaries


def build_gallery(samples, cluster_to_identity, enroll_until_seconds, max_enroll_per_identity):
    gallery = {}
    for cluster_id, identity in cluster_to_identity.items():
        candidates = [
            sample
            for sample in samples
            if sample.cluster == cluster_id and sample.time_seconds <= enroll_until_seconds
        ]
        if len(candidates) < 3:
            candidates = [sample for sample in samples if sample.cluster == cluster_id]
        candidates.sort(key=lambda sample: (-sample.det_score, sample.time_seconds))
        selected = candidates[:max(3, int(max_enroll_per_identity))]
        gallery[identity] = {
            'label': identity,
            'display_name': identity,
            'student_id': identity.replace('VideoPerson', 'VID'),
            'embedding': normalized_mean([sample.embedding for sample in selected]),
            'enroll_samples': len(selected),
            'cluster': cluster_id,
        }
    return gallery


def evaluate_gallery_thresholds(samples, gallery, cluster_to_identity, verify_after_seconds, thresholds):
    verification_samples = [
        sample
        for sample in samples
        if sample.time_seconds >= verify_after_seconds and sample.cluster in cluster_to_identity
    ]
    if not verification_samples:
        raise RuntimeError('No verification samples were collected after the requested split.')

    results = []
    identities = sorted(gallery)
    for threshold in thresholds:
        true_accept = 0
        false_reject = 0
        wrong_accept = 0
        impostor_pairs = 0
        impostor_false_accept = 0
        target_similarities = []
        impostor_similarities = []

        for sample in verification_samples:
            target_identity = cluster_to_identity[sample.cluster]
            similarities = {
                identity: cosine_similarity(sample.embedding, item['embedding'])
                for identity, item in gallery.items()
            }
            top_identity = max(similarities, key=similarities.get)
            top_similarity = similarities[top_identity]
            target_similarity = similarities[target_identity]
            target_similarities.append(float(target_similarity))

            if top_similarity >= threshold and top_identity == target_identity:
                true_accept += 1
            elif top_similarity >= threshold and top_identity != target_identity:
                wrong_accept += 1
            else:
                false_reject += 1

            for identity in identities:
                if identity == target_identity:
                    continue
                impostor_pairs += 1
                impostor_similarity = similarities[identity]
                impostor_similarities.append(float(impostor_similarity))
                if impostor_similarity >= threshold:
                    impostor_false_accept += 1

        total = len(verification_samples)
        tpr = true_accept / max(1, total)
        wrong_rate = wrong_accept / max(1, total)
        pair_far = impostor_false_accept / max(1, impostor_pairs)
        precision = true_accept / max(1, true_accept + wrong_accept + impostor_false_accept)
        score = (tpr * 100.0) - (pair_far * 280.0) - (wrong_rate * 180.0) + (precision * 35.0)
        results.append(
            {
                'threshold': round(float(threshold), 3),
                'score': round(score, 4),
                'verification_samples': total,
                'true_accept': true_accept,
                'false_reject': false_reject,
                'wrong_accept': wrong_accept,
                'impostor_pairs': impostor_pairs,
                'impostor_false_accept': impostor_false_accept,
                'tpr': round(tpr, 4),
                'wrong_rate': round(wrong_rate, 4),
                'pair_far': round(pair_far, 4),
                'precision': round(precision, 4),
                'target_similarity_mean': round(statistics.fmean(target_similarities), 4),
                'target_similarity_p10': round(
                    statistics.quantiles(target_similarities, n=10, method='inclusive')[0],
                    4,
                ) if len(target_similarities) >= 2 else round(target_similarities[0], 4),
                'impostor_similarity_p99': round(
                    statistics.quantiles(impostor_similarities, n=100, method='inclusive')[98],
                    4,
                ) if len(impostor_similarities) >= 2 else 0.0,
            }
        )
    return sorted(results, key=lambda item: (item['score'], item['tpr'], -item['pair_far']), reverse=True)


class VideoGalleryFaceDB:
    def __init__(self, delegate: FaceRecognitionDB, gallery):
        self.delegate = delegate
        self.gallery = gallery

    def analyze_faces(self, image):
        return self.delegate.analyze_faces(image)

    def match_embedding(self, embedding, threshold=0.45):
        matches = []
        for identity, item in self.gallery.items():
            similarity = cosine_similarity(embedding, item['embedding'])
            if similarity >= threshold:
                matches.append(
                    {
                        'label': identity,
                        'display_name': item['display_name'],
                        'student_id': item['student_id'],
                        'college': 'video_eval',
                        'department': 'video_eval',
                        'title': '',
                        'similarity': round(float(similarity), 4),
                    }
                )
        matches.sort(key=lambda item: item['similarity'], reverse=True)
        return matches


def try_confirm_temporaries(pipeline, max_attempts):
    with pipeline._lock:
        temp_ids = [
            temp_id
            for temp_id, person in pipeline._temporary_people.items()
            if person.confirm_status != 'processing'
        ]
    success = 0
    failed = 0
    for temp_id in temp_ids[:max(0, int(max_attempts))]:
        result = pipeline.confirm_temporary_person(temp_id)
        if result.get('status') == 'confirmed':
            success += 1
        elif result.get('status') == 'error':
            failed += 1
    return success, failed


def evaluate_pipeline_with_gallery(
    video_path,
    gallery,
    params,
    pose_model,
    frame_step,
    confirm_every_seconds,
    max_confirm_attempts,
    expected_people,
):
    pipeline, offline = tuning.make_pipeline(WORKSPACE, pose_model_ref=pose_model)
    pipeline.face_db = VideoGalleryFaceDB(pipeline.face_db, gallery)
    params = dict(params)
    params['recognition_threshold'] = params.get('recognition_threshold', 0.44)
    params['auto_relink_threshold'] = params.get('auto_relink_threshold', max(0.52, params['recognition_threshold']))
    tuning.apply_params(pipeline, params)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f'Unable to open video: {video_path}')

    simulated_now = 0.0
    frame_index = 0
    frames_processed = 0
    last_confirm_at = -999.0
    active_counts = []
    confirmed_counts = []
    temp_counts = []
    unique_confirmed = set()
    confirm_success = 0
    confirm_failed = 0
    metric_row_counts = []
    metric_complete_ticks = 0
    metric_ticks = 0

    while True:
        ok, combined = cap.read()
        if not ok or combined is None:
            break
        frame_index += 1
        if frame_index % max(1, int(frame_step)) != 0:
            continue

        frame, depth_visual = tuning.decode_side_by_side_frame(combined)
        depth = tuning.depth_from_visual_frame(depth_visual)
        offline.set_bundle(frame, depth_frame=depth, depth_visual_frame=depth_visual, source_mode='video')
        simulated_now += pipeline.LOOP_INTERVAL
        frames_processed += 1

        person_boxes = pipeline._detect_people(
            frame,
            simulated_now,
            depth_frame=depth,
            depth_source_mode='video',
        )
        with pipeline._lock:
            pipeline._match_detections_locked(person_boxes, simulated_now)
            pipeline._try_auto_relink_locked(frame, simulated_now)
            pipeline._update_classroom_metrics_locked(frame.shape, depth, simulated_now, force=True)

            present_confirmed = [
                person
                for person in pipeline._confirmed_people.values()
                if person.current_status == 'present'
            ]
            for person in present_confirmed:
                unique_confirmed.add(person.user_id)
                user_metrics = pipeline._metric_engine.get_user_metrics(person.user_id)
                if user_metrics:
                    row_lengths = [len(rows) for rows in user_metrics.values()]
                    metric_row_counts.append(sum(row_lengths))
                    metric_ticks += 1
                    if len(row_lengths) == 8 and min(row_lengths) > 0 and max(row_lengths) - min(row_lengths) <= 1:
                        metric_complete_ticks += 1

            active_counts.append(len(pipeline._temporary_people) + len(present_confirmed))
            confirmed_counts.append(len(present_confirmed))
            temp_counts.append(len(pipeline._temporary_people))

        if confirm_every_seconds > 0 and simulated_now - last_confirm_at >= confirm_every_seconds:
            success, failed = try_confirm_temporaries(pipeline, max_confirm_attempts)
            confirm_success += success
            confirm_failed += failed
            last_confirm_at = simulated_now

    cap.release()
    expected = max(0, int(expected_people))
    return {
        'frames_processed': frames_processed,
        'mean_active': round(statistics.fmean(active_counts), 3) if active_counts else 0.0,
        'mean_confirmed_present': round(statistics.fmean(confirmed_counts), 3) if confirmed_counts else 0.0,
        'mean_temporary': round(statistics.fmean(temp_counts), 3) if temp_counts else 0.0,
        'max_active': max(active_counts) if active_counts else 0,
        'min_active': min(active_counts) if active_counts else 0,
        'stable_ratio_within_1': round(sum(1 for value in active_counts if abs(value - expected) <= 1) / max(1, len(active_counts)), 4),
        'mean_abs_error': round(statistics.fmean(abs(value - expected) for value in active_counts), 4) if active_counts else 0.0,
        'confirm_success': confirm_success,
        'confirm_failed': confirm_failed,
        'unique_confirmed_ids': len(unique_confirmed),
        'metric_rows_per_present_user_mean': round(statistics.fmean(metric_row_counts), 3) if metric_row_counts else 0.0,
        'metric_complete_tick_ratio': round(metric_complete_ticks / max(1, metric_ticks), 4),
    }


def threshold_range(min_threshold, max_threshold, step):
    value = float(min_threshold)
    values = []
    while value <= float(max_threshold) + 1e-9:
        values.append(round(value, 6))
        value += max(0.005, float(step))
    return values


def main():
    parser = argparse.ArgumentParser(description='Evaluate side-by-side video enrollment, recognition, rejection, and post-recognition tracking.')
    parser.add_argument('--video', default='reels/recordings/rgb_nir_side_by_side_20260420_174707_v2.mp4')
    parser.add_argument('--expected-people', type=int, default=5)
    parser.add_argument('--sample-step-seconds', type=float, default=1.0)
    parser.add_argument('--min-det-score', type=float, default=0.50)
    parser.add_argument('--enroll-until-seconds', type=float, default=95.0)
    parser.add_argument('--verify-after-seconds', type=float, default=120.0)
    parser.add_argument('--max-enroll-per-identity', type=int, default=45)
    parser.add_argument('--min-threshold', type=float, default=0.25)
    parser.add_argument('--max-threshold', type=float, default=0.75)
    parser.add_argument('--threshold-step', type=float, default=0.02)
    parser.add_argument('--pipeline-frame-step', type=int, default=4)
    parser.add_argument('--confirm-every', type=float, default=1.2)
    parser.add_argument('--max-confirm-attempts', type=int, default=5)
    parser.add_argument('--pose-model', default='models/yolo/yolo26x-pose.pt')
    parser.add_argument('--output', default='data/video_tuning/rgb_nir_side_by_side_20260420_174707_v2_identity_flow.json')
    args = parser.parse_args()

    started_at = time.time()
    video_path = resolve_video_path(args.video)
    face_db = FaceRecognitionDB(WORKSPACE)
    samples, meta = collect_face_samples(
        video_path,
        face_db=face_db,
        sample_step_seconds=args.sample_step_seconds,
        min_det_score=args.min_det_score,
    )
    cluster_to_identity, cluster_summaries = assign_clusters(samples, args.expected_people)
    gallery = build_gallery(
        samples,
        cluster_to_identity,
        enroll_until_seconds=args.enroll_until_seconds,
        max_enroll_per_identity=args.max_enroll_per_identity,
    )
    thresholds = threshold_range(args.min_threshold, args.max_threshold, args.threshold_step)
    threshold_results = evaluate_gallery_thresholds(
        samples,
        gallery,
        cluster_to_identity,
        verify_after_seconds=args.verify_after_seconds,
        thresholds=thresholds,
    )
    zero_false_accept_results = [
        item
        for item in threshold_results
        if int(item.get('wrong_accept', 0)) == 0 and int(item.get('impostor_false_accept', 0)) == 0
    ]
    zero_false_accept_results.sort(key=lambda item: (item['tpr'], item['threshold']), reverse=True)
    selected_threshold_result = zero_false_accept_results[0] if zero_false_accept_results else threshold_results[0]
    best_threshold = selected_threshold_result['threshold']
    params = dict(FOCUSED_BEST_PARAMS)
    params['recognition_threshold'] = best_threshold
    params['auto_relink_threshold'] = max(best_threshold, 0.52)
    pipeline_metrics = evaluate_pipeline_with_gallery(
        video_path,
        gallery=gallery,
        params=params,
        pose_model=args.pose_model,
        frame_step=args.pipeline_frame_step,
        confirm_every_seconds=args.confirm_every,
        max_confirm_attempts=args.max_confirm_attempts,
        expected_people=args.expected_people,
    )

    output = {
        'generated_at_epoch': round(time.time(), 3),
        'elapsed_sec': round(time.time() - started_at, 2),
        'video': str(video_path),
        'video_meta': meta,
        'expected_people': args.expected_people,
        'face_samples': len(samples),
        'cluster_summaries': cluster_summaries,
        'cluster_to_identity': cluster_to_identity,
        'gallery': {
            identity: {
                'student_id': item['student_id'],
                'enroll_samples': item['enroll_samples'],
                'cluster': item['cluster'],
            }
            for identity, item in gallery.items()
        },
        'best_threshold': best_threshold,
        'threshold_selection': 'zero_false_accept' if zero_false_accept_results else 'best_score',
        'selected_threshold_result': selected_threshold_result,
        'best_score_threshold_result': threshold_results[0],
        'top_thresholds': threshold_results[:5],
        'zero_false_accept_top_thresholds': zero_false_accept_results[:5],
        'selected_params': params,
        'pipeline_metrics': pipeline_metrics,
    }
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = WORKSPACE / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(output, file, ensure_ascii=False, indent=2)

    print(json.dumps({'stage': 'identity_flow', **output, 'output': str(output_path)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
