"""Remote Qwen3-VL reranker backend for direct visual archive matching."""
from __future__ import annotations

import base64
import mimetypes
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pipeline.visual_archive import IMAGE_SUFFIXES, aggregate_identity_matches


DEFAULT_INSTRUCTION = (
    "Determine whether the query image and document image depict the same physical vessel. "
    "Use stable identity cues such as hull shape, superstructure, cabin layout, railings, "
    "windows, deck equipment, markings, and relative geometry. Ignore viewpoint, scale, "
    "illumination, water, and background differences."
)


def _image_data_url(data: bytes, suffix: str = ".jpg") -> str:
    media_type = mimetypes.types_map.get(suffix.lower(), "image/jpeg")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _score_url(base_url: str) -> str:
    normalized = str(base_url or "http://localhost:7894").rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return f"{normalized}/score"


def _parse_scores(payload: dict[str, Any], expected_count: int) -> list[float]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("Reranker response does not contain a data list")
    indexed: dict[int, float] = {}
    for position, item in enumerate(data):
        if not isinstance(item, dict) or "score" not in item:
            raise ValueError("Reranker response contains an invalid score item")
        index = int(item.get("index", position))
        indexed[index] = float(item["score"])
    if set(indexed) != set(range(expected_count)):
        raise ValueError(
            f"Reranker returned indices {sorted(indexed)}, expected 0..{expected_count - 1}"
        )
    return [indexed[index] for index in range(expected_count)]


class QwenVisualRerankerIndex:
    """Compare a target crop directly with every visual archive prototype."""

    device = "remote"

    def __init__(self, config: dict[str, Any]):
        self.model = str(config.get("model", "Qwen/Qwen3-VL-Reranker-2B"))
        self.api_key = str(config.get("api_key", "abc123"))
        self.url = _score_url(str(config.get("base_url", "http://localhost:7894")))
        self.instruction = str(config.get("instruction", DEFAULT_INSTRUCTION))
        self.timeout_seconds = max(1.0, float(config.get("request_timeout_seconds", 120)))
        self.max_retries = max(0, int(config.get("max_retries", 1)))
        self.request_batch_size = max(1, int(config.get("request_batch_size", 8)))
        self.jpeg_quality = min(100, max(50, int(config.get("jpeg_quality", 90))))
        self._top_k = max(1, int(config.get("top_k", 3)))
        self.archive_root = Path(
            config.get("archive_root", "data/archive/visual_prototypes")
        ).expanduser().resolve()
        self.records = self._load_archive_records()
        self._thread_state = threading.local()

    @property
    def last_request_count(self) -> int:
        return int(getattr(self._thread_state, "request_count", 0))

    @property
    def last_pair_count(self) -> int:
        return int(getattr(self._thread_state, "pair_count", 0))

    def _load_archive_records(self) -> list[dict[str, Any]]:
        if not self.archive_root.exists():
            raise FileNotFoundError(f"Visual archive does not exist: {self.archive_root}")
        records: list[dict[str, Any]] = []
        for hull_dir in sorted(path for path in self.archive_root.iterdir() if path.is_dir()):
            for image_path in sorted(hull_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                try:
                    image_bytes = image_path.read_bytes()
                except OSError:
                    continue
                if not image_bytes:
                    continue
                records.append(
                    {
                        "hull_number": hull_dir.name,
                        "prototype_id": image_path.stem,
                        "image_path": str(image_path),
                        "data_url": _image_data_url(image_bytes, image_path.suffix),
                    }
                )
        if not records:
            raise RuntimeError(f"No visual prototypes found under: {self.archive_root}")
        return records

    @staticmethod
    def _image_param(data_url: str) -> dict[str, Any]:
        return {
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            ]
        }

    def _request_scores(self, query: dict[str, Any], records: list[dict[str, Any]]) -> list[float]:
        import httpx

        payload = {
            "model": self.model,
            "queries": query,
            "documents": [self._image_param(str(record["data_url"])) for record in records],
            "instruction": self.instruction,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        timeout = httpx.Timeout(
            connect=min(15.0, self.timeout_seconds),
            read=self.timeout_seconds,
            write=self.timeout_seconds,
            pool=min(15.0, self.timeout_seconds),
        )
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                self._thread_state.request_count = self.last_request_count + 1
                response = httpx.post(self.url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                return _parse_scores(response.json(), len(records))
            except (httpx.HTTPError, TypeError, ValueError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(1.0, 0.2 * (attempt + 1)))
        raise RuntimeError(
            f"Qwen visual reranker failed after {self.max_retries + 1} attempt(s): {last_error}"
        ) from last_error

    def search(self, image: np.ndarray, top_k: int | None = None) -> tuple[list[dict[str, Any]], float]:
        if image is None or image.size == 0:
            return [], 0.0
        success, buffer = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not success:
            raise RuntimeError("Unable to encode target crop for visual reranking")
        query = self._image_param(_image_data_url(buffer.tobytes(), ".jpg"))
        started = time.perf_counter()
        scores: list[float] = []
        self._thread_state.request_count = 0
        self._thread_state.pair_count = 0
        for start in range(0, len(self.records), self.request_batch_size):
            batch = self.records[start : start + self.request_batch_size]
            scores.extend(self._request_scores(query, batch))
            self._thread_state.pair_count += len(batch)
        matches = aggregate_identity_matches(scores, self.records, top_k or self._top_k)
        return matches, (time.perf_counter() - started) * 1000.0


_INDEX_CACHE: dict[tuple[Any, ...], QwenVisualRerankerIndex] = {}
_INDEX_CACHE_LOCK = threading.Lock()


def get_qwen_visual_reranker_index(config: dict[str, Any]) -> QwenVisualRerankerIndex:
    key = (
        str(Path(config.get("archive_root", "data/archive/visual_prototypes")).expanduser().resolve()),
        str(config.get("base_url", "http://localhost:7894")),
        str(config.get("model", "Qwen/Qwen3-VL-Reranker-2B")),
        str(config.get("api_key", "abc123")),
        float(config.get("request_timeout_seconds", 120)),
        int(config.get("max_retries", 1)),
        int(config.get("request_batch_size", 8)),
        int(config.get("jpeg_quality", 90)),
        str(config.get("instruction", DEFAULT_INSTRUCTION)),
    )
    with _INDEX_CACHE_LOCK:
        if key not in _INDEX_CACHE:
            _INDEX_CACHE[key] = QwenVisualRerankerIndex(config)
        return _INDEX_CACHE[key]
