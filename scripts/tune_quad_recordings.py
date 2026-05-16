import argparse
import json
import statistics
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import tune_with_video as tuning  # noqa: E402


def resolve_sources(video_glob: str, backend_filter: str):
    videos = sorted(WORKSPACE.glob(video_glob))
    if not videos:
        raise RuntimeError(f'No videos found for pattern: {video_glob}')

    backends = ['v1', 'v2'] if backend_filter == 'all' else [backend_filter]
    sources = []
    for video_path in videos:
        for backend_name in backends:
            sources.append(
                {
                    'id': f'{video_path.name}:{backend_name}',
                    'video_path': video_path,
                    'backend': backend_name,
                }
            )
    return sources


def aggregate_result_metrics(per_source_results):
    numeric_metrics = {}
    for item in per_source_results:
        metrics = item.get('metrics', {})
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_metrics.setdefault(key, []).append(float(value))

    aggregated = {
        key: round(float(statistics.fmean(values)), 4)
        for key, values in numeric_metrics.items()
        if values
    }
    aggregated['sources_tested'] = len(per_source_results)
    aggregated['per_source'] = per_source_results
    return aggregated


def evaluate_params_across_sources(
    pipeline,
    offline_kinect,
    sources,
    params,
    loops,
    frame_step,
    confirm_every_seconds,
    max_confirm_attempts,
    expected_people_count,
):
    per_source_results = []
    scores = []
    for source in sources:
        result = tuning.evaluate_combo(
            pipeline,
            offline_kinect,
            source['video_path'],
            params=params,
            loops=loops,
            frame_step=frame_step,
            confirm_every_seconds=confirm_every_seconds,
            max_confirm_attempts=max_confirm_attempts,
            expected_people_count=expected_people_count,
            quad_backend=source['backend'],
        )
        per_source_results.append(
            {
                'source': source['id'],
                'score': result.score,
                'metrics': result.metrics,
            }
        )
        scores.append(float(result.score))
        if result.score <= -1e8:
            break

    aggregate_metrics = aggregate_result_metrics(per_source_results)
    return tuning.ComboResult(
        params=params,
        score=round(float(statistics.fmean(scores)) if scores else -1e9, 4),
        metrics=aggregate_metrics,
    )


def main():
    parser = argparse.ArgumentParser(description='Tune attendance parameters across multiple V1/V2 quad recordings.')
    parser.add_argument('--video-glob', default='reels/recordings/*v1_v2*.mp4', help='Glob pattern for quad recordings.')
    parser.add_argument('--backend', choices=('all', 'v1', 'v2'), default='all', help='Which backend panes to tune.')
    parser.add_argument('--sample-count', type=int, default=12, help='Number of parameter sets to evaluate.')
    parser.add_argument('--search-loops', type=int, default=1, help='Video loops for search stage.')
    parser.add_argument('--validate-loops', type=int, default=2, help='Video loops for validation stage.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for candidate sampling.')
    parser.add_argument('--confirm-every', type=float, default=2.4, help='Seconds between auto-confirm attempts.')
    parser.add_argument('--max-confirm-attempts', type=int, default=2, help='Maximum confirm attempts at each confirm interval.')
    parser.add_argument('--expected-people', type=int, default=2, help='Expected person count in the scene.')
    parser.add_argument('--pose-model', default='', help='Pose model reference, e.g. models/yolo/yolo26x-pose.pt.')
    parser.add_argument('--output', default='data/runtime_tuning_profile.json', help='Where to save the final recommendation JSON.')
    args = parser.parse_args()

    sources = resolve_sources(args.video_glob, args.backend)
    metadata = []
    for source in sources:
        meta = tuning.read_video_meta(source['video_path'])
        metadata.append(
            {
                'source': source['id'],
                'video': str(source['video_path']),
                'backend': source['backend'],
                **meta,
            }
        )
    print(json.dumps({
        'stage': 'sources',
        'count': len(sources),
        'sources': metadata,
        'pose_model': args.pose_model or 'auto',
    }, ensure_ascii=False))

    pipeline, offline_kinect = tuning.make_pipeline(WORKSPACE, pose_model_ref=args.pose_model)
    baseline, _ = tuning.build_candidate_params()

    first_meta = tuning.read_video_meta(sources[0]['video_path'])
    stride = max(1, int(round(first_meta['fps'] * pipeline.LOOP_INTERVAL)))
    search_stride = max(stride, 4)
    candidates = tuning.sample_param_candidates(max(1, args.sample_count), args.seed)
    if baseline not in candidates:
        candidates.insert(0, baseline)

    search_results = []
    started_at = time.time()
    for index, params in enumerate(candidates, start=1):
        result = evaluate_params_across_sources(
            pipeline,
            offline_kinect,
            sources=sources,
            params=params,
            loops=args.search_loops,
            frame_step=search_stride,
            confirm_every_seconds=args.confirm_every,
            max_confirm_attempts=args.max_confirm_attempts,
            expected_people_count=args.expected_people,
        )
        search_results.append(result)
        print(json.dumps({
            'stage': 'search',
            'index': index,
            'total': len(candidates),
            'score': result.score,
            'params': result.params,
            'metrics': result.metrics,
        }, ensure_ascii=False))

    search_results.sort(key=lambda item: item.score, reverse=True)
    finalists = search_results[:3]

    validation_results = []
    for index, candidate in enumerate(finalists, start=1):
        result = evaluate_params_across_sources(
            pipeline,
            offline_kinect,
            sources=sources,
            params=candidate.params,
            loops=args.validate_loops,
            frame_step=stride,
            confirm_every_seconds=args.confirm_every,
            max_confirm_attempts=args.max_confirm_attempts,
            expected_people_count=args.expected_people,
        )
        validation_results.append(result)
        print(json.dumps({
            'stage': 'validation',
            'index': index,
            'total': len(finalists),
            'score': result.score,
            'params': result.params,
            'metrics': result.metrics,
        }, ensure_ascii=False))

    validation_results.sort(key=lambda item: item.score, reverse=True)
    best = validation_results[0]
    elapsed = time.time() - started_at
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = WORKSPACE / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        'generated_at_epoch': round(time.time(), 3),
        'elapsed_sec': round(elapsed, 2),
        'sources': metadata,
        'best_score': best.score,
        'best_params': best.params,
        'best_metrics': best.metrics,
    }
    with output_path.open('w', encoding='utf-8') as output_file:
        json.dump(output_payload, output_file, ensure_ascii=False, indent=2)

    print(json.dumps({
        'stage': 'best',
        'elapsed_sec': round(elapsed, 2),
        'output': str(output_path),
        'score': best.score,
        'params': best.params,
        'metrics': best.metrics,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
