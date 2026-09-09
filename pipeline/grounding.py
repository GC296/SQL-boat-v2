from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from pipeline.detector import ShipDetector, Detection


class GroundingProvider(Protocol):
    def detect(self, frame: np.ndarray, frame_id: int = 0) -> list[Detection]: ...
    def cleanup(self) -> None: ...


class YoloGroundingProvider:
    def __init__(self, pipe_cfg: dict[str, Any]):
        self._detector = ShipDetector(
            model_path=pipe_cfg.get("yolo_model", "yolov8n.pt"),
            device=pipe_cfg.get("device", ""),
            conf_threshold=pipe_cfg.get("conf_threshold", 0.25),
            iou_threshold=pipe_cfg.get("iou_threshold", 0.45),
            tracker_type=pipe_cfg.get("tracker", "bytetrack"),
            tracker_params=pipe_cfg.get("tracker_params"),
            classes=pipe_cfg.get("detect_classes", [8]),
        )

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> list[Detection]:
        return self._detector.detect(frame, frame_id)

    def cleanup(self) -> None:
        self._detector.cleanup()


def create_grounding_provider(pipe_cfg: dict[str, Any]) -> GroundingProvider:
    provider = (pipe_cfg.get("grounding_provider") or "yolo").lower()
    if provider in {"yolo", "yolo_tensorrt", "yolo_ultralytics"}:
        return YoloGroundingProvider(pipe_cfg)
    raise ValueError(f"Unsupported grounding provider: {provider}")
