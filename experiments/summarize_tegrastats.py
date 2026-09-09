"""Summarize NVIDIA Jetson tegrastats logs for deployment experiments."""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def summarize_lines(lines: list[str], interval_ms: int = 1000) -> dict[str, Any]:
    ram: list[float] = []
    gpu: list[float] = []
    power: list[float] = []
    cpu: list[float] = []
    for line in lines:
        match = re.search(r"RAM\s+(\d+(?:\.\d+)?)/", line)
        if match:
            ram.append(float(match.group(1)))
        match = re.search(r"GR3D_FREQ\s+(\d+(?:\.\d+)?)%", line)
        if match:
            gpu.append(float(match.group(1)))
        match = re.search(r"VDD_IN\s+(\d+(?:\.\d+)?)mW", line)
        if match:
            power.append(float(match.group(1)))
        percentages = [float(value) for value in re.findall(r"(\d+(?:\.\d+)?)%@", line)]
        if percentages:
            cpu.append(mean(percentages))
    sample_count = max(len(ram), len(gpu), len(power), len(cpu))
    duration_seconds = sample_count * interval_ms / 1000.0
    return {
        "samples": sample_count,
        "duration_seconds": round(duration_seconds, 3),
        "avg_ram_mb": round(mean(ram), 3) if ram else 0.0,
        "peak_ram_mb": round(max(ram), 3) if ram else 0.0,
        "avg_gpu_util_percent": round(mean(gpu), 3) if gpu else 0.0,
        "p95_gpu_util_percent": round(percentile(gpu, 0.95), 3),
        "avg_cpu_core_util_percent": round(mean(cpu), 3) if cpu else 0.0,
        "avg_power_w": round(mean(power) / 1000.0, 3) if power else 0.0,
        "peak_power_w": round(max(power) / 1000.0, 3) if power else 0.0,
        "energy_wh": round(sum(power) * interval_ms / 3_600_000_000.0, 6) if power else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--interval-ms", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = summarize_lines(args.log.read_text(encoding="utf-8", errors="ignore").splitlines(), args.interval_ms)
    output = args.output or args.log.with_suffix(".summary.json")
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
