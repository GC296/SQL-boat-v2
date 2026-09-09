"""JSONL experiment logging utilities."""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import cv2


class ExperimentLogger:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.run_name = str(cfg.get("run_name", "default"))
        root = Path(cfg.get("output_dir", "./experiment_outputs"))
        self.run_dir = root / self.run_name
        self.save_jsonl = bool(cfg.get("save_jsonl", True))
        self.save_crops = bool(cfg.get("save_crops", False))
        self.jpeg_quality = min(100, max(1, int(cfg.get("crop_jpeg_quality", 90))))
        self.context = {"video_id": cfg.get("video_id", ""), "split": cfg.get("split", "test")}
        self._lock = threading.Lock()
        if self.enabled:
            self.run_dir.mkdir(parents=True, exist_ok=True)

    def log(self, stream: str, payload: dict[str, Any]) -> None:
        if not self.enabled or not self.save_jsonl:
            return
        record = {"run_name": self.run_name, "timestamp": time.time(), **self.context, **payload}
        path = self.run_dir / f"{stream}.jsonl"
        with self._lock, path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=self._json_default) + "\n")

    def write_summary(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self.run_dir / "summary.json"
        with self._lock, path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=self._json_default)

    def save_image(
        self,
        stream: str,
        image: Any,
        *,
        entity_id: str = "",
        track_id: int = 0,
        frame_id: int = 0,
    ) -> str:
        if not self.enabled or not self.save_crops or image is None or getattr(image, "size", 0) == 0:
            return ""
        safe_stream = re.sub(r"[^A-Za-z0-9_.-]+", "_", stream).strip("_") or "images"
        video_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(self.context.get("video_id", "video"))).strip("_") or "video"
        safe_entity = re.sub(r"[^A-Za-z0-9_.-]+", "_", entity_id).strip("_") or "entity"
        filename = f"{video_id}_{safe_entity}_T{int(track_id):06d}_F{int(frame_id):08d}.jpg"
        path = self.run_dir / "evidence" / safe_stream / filename
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not success:
            return ""
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(encoded.tobytes())
        return path.relative_to(self.run_dir).as_posix()

    @staticmethod
    def _json_default(value: Any) -> Any:
        if hasattr(value, "item"):
            return value.item()
        if hasattr(value, "tolist"):
            return value.tolist()
        return str(value)
