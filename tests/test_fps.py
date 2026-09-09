from pipeline.fps import LatencyMeter


def test_latency_meter_keeps_lifetime_stats_after_rolling_cleanup():
    meter = LatencyMeter(window_seconds=1.0)
    meter.record("entity", 1.0)
    meter.record("entity", 3.0)
    stats = meter.get_all_time_stats()["entity"]
    assert stats["count"] == 2
    assert stats["avg"] == 2.0
    assert stats["p95"] == 3.0
