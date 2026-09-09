"""Edge-cloud execution metrics and deterministic network simulation."""
from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, TypeVar

T = TypeVar("T")


@dataclass
class EdgeCloudRecord:
    action: str
    site: str
    upload_bytes: int
    simulated_network_ms: float
    cloud_roundtrip_ms: float
    total_latency_ms: float
    success: bool
    outcome: str


class EdgeCloudExecutor:
    def __init__(self, config: dict[str, Any] | None = None, seed: int = 42):
        cfg = config or {}
        self.simulate = bool(cfg.get("simulate", False))
        self.latency_ms = float(cfg.get("latency_ms", 0.0))
        self.bandwidth_mbps = max(0.001, float(cfg.get("bandwidth_mbps", 100.0)))
        self.failure_probability = min(1.0, max(0.0, float(cfg.get("failure_probability", 0.0))))
        self.jitter_ms = max(0.0, float(cfg.get("jitter_ms", 0.0)))
        self._rng = random.Random(seed)

    def run(self, action: str, upload_bytes: int, fn: Callable[[], T], outcome_fn: Callable[[T], str] | None = None) -> tuple[T, dict[str, Any]]:
        started = time.perf_counter()
        transfer_ms = (max(0, upload_bytes) * 8.0 / (self.bandwidth_mbps * 1_000_000.0)) * 1000.0
        network_ms = self.latency_ms + transfer_ms + self._rng.uniform(0.0, self.jitter_ms)
        if self.simulate and self._rng.random() < self.failure_probability:
            raise ConnectionError("simulated edge-cloud interruption")
        if self.simulate and network_ms > 0:
            time.sleep(network_ms / 1000.0)
        cloud_started = time.perf_counter()
        result = fn()
        cloud_ms = (time.perf_counter() - cloud_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        record = EdgeCloudRecord(
            action=action, site="cloud", upload_bytes=int(upload_bytes),
            simulated_network_ms=round(network_ms if self.simulate else 0.0, 3),
            cloud_roundtrip_ms=round(cloud_ms, 3), total_latency_ms=round(total_ms, 3),
            success=True, outcome=outcome_fn(result) if outcome_fn else "completed",
        )
        return result, asdict(record)
