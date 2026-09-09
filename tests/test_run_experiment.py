from experiments.run_experiment import aggregate_deployment_summaries


def test_aggregate_deployment_summaries_uses_all_videos():
    metrics = aggregate_deployment_summaries([
        {"total_frames": 100, "elapsed_seconds": 10, "source_fps": 20, "process_cpu_seconds": 5, "peak_rss_mb": 100, "latency": {"yolo": {"avg": 10, "p95": 15, "count": 100}}},
        {"total_frames": 200, "elapsed_seconds": 20, "source_fps": 20, "process_cpu_seconds": 10, "peak_rss_mb": 120, "latency": {"yolo": {"avg": 20, "p95": 30, "count": 200}}},
    ])
    assert metrics["videos"] == 2
    assert metrics["throughput_fps"] == 10.0
    assert metrics["realtime_factor"] == 0.5
    assert metrics["process_cpu_percent"] == 50.0
    assert metrics["peak_rss_mb"] == 120.0
    assert metrics["latency"]["yolo"]["weighted_avg_ms"] == 16.667
    assert metrics["latency"]["yolo"]["mean_video_p95_ms"] == 22.5
