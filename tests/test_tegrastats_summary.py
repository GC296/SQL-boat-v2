from experiments.summarize_tegrastats import summarize_lines


def test_summarize_tegrastats_reports_resource_and_energy_metrics():
    metrics = summarize_lines([
        "RAM 2048/31919MB CPU [10%@729,20%@729,off] GR3D_FREQ 30% VDD_IN 10000mW",
        "RAM 3072/31919MB CPU [30%@729,50%@729,off] GR3D_FREQ 70% VDD_IN 14000mW",
    ], interval_ms=1000)
    assert metrics["samples"] == 2
    assert metrics["duration_seconds"] == 2.0
    assert metrics["avg_ram_mb"] == 2560.0
    assert metrics["peak_ram_mb"] == 3072.0
    assert metrics["avg_gpu_util_percent"] == 50.0
    assert metrics["avg_cpu_core_util_percent"] == 27.5
    assert metrics["avg_power_w"] == 12.0
    assert metrics["peak_power_w"] == 14.0
    assert metrics["energy_wh"] == 0.006667
