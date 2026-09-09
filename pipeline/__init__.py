"""Maritime watchkeeping pipeline package.

Heavy detector and video dependencies are imported lazily so evidence, policy,
and evaluation modules remain usable in lightweight experiment environments.
"""
from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ShipDetector": ("pipeline.detector", "ShipDetector"),
    "AgentResult": ("agent", "AgentResult"),
    "TrackManager": ("pipeline.tracker", "TrackManager"),
    "ShipPipeline": ("pipeline.pipeline", "ShipPipeline"),
    "FPSMeter": ("pipeline.fps", "FPSMeter"),
    "LatencyMeter": ("pipeline.fps", "LatencyMeter"),
    "InputSource": ("pipeline.video_input", "InputSource"),
    "DemoRenderer": ("pipeline.demo", "DemoRenderer"),
    "ScreenshotSaver": ("pipeline.output", "ScreenshotSaver"),
    "create_grounding_provider": ("pipeline.grounding", "create_grounding_provider"),
}

__all__ = list(_EXPORTS)

def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
