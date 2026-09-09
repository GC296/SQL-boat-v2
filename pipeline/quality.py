"""Observation reliability estimation for target crops."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ObservationQuality:
    score: float
    components: dict[str, float]


def _clip01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def compute_observation_quality(
    crop: np.ndarray | None,
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
    confidence: float,
    config: dict[str, Any] | None = None,
) -> ObservationQuality:
    cfg = config or {}
    if crop is None or crop.size == 0:
        return ObservationQuality(0.0, {name: 0.0 for name in ("brightness", "blur", "contrast", "clipping", "scale", "confidence")})
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    mean = float(np.mean(gray))
    brightness = _clip01(1.0 - abs(mean - 127.5) / 127.5)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur = _clip01(lap_var / float(cfg.get("blur_reference", 300.0)))
    contrast = _clip01(float(np.std(gray)) / float(cfg.get("contrast_reference", 64.0)))
    clipped = float(np.mean((gray <= int(cfg.get("dark_clip", 5))) | (gray >= int(cfg.get("bright_clip", 250)))))
    clipping = _clip01(1.0 - clipped / max(1e-6, float(cfg.get("max_clipped_fraction", 0.35))))
    x1, y1, x2, y2 = bbox
    frame_h, frame_w = frame_shape[:2]
    area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, frame_h * frame_w)
    scale = _clip01(area_ratio / float(cfg.get("target_area_reference", 0.08)))
    components = {
        "brightness": brightness, "blur": blur, "contrast": contrast,
        "clipping": clipping, "scale": scale, "confidence": _clip01(float(confidence)),
    }
    weights = {
        "brightness": float(cfg.get("brightness_weight", 0.12)),
        "blur": float(cfg.get("blur_weight", 0.24)),
        "contrast": float(cfg.get("contrast_weight", 0.12)),
        "clipping": float(cfg.get("clipping_weight", 0.07)),
        "scale": float(cfg.get("scale_weight", 0.30)),
        "confidence": float(cfg.get("confidence_weight", 0.15)),
    }
    total_weight = sum(max(0.0, weight) for weight in weights.values()) or 1.0
    score = sum(components[name] * max(0.0, weights[name]) for name in components) / total_weight
    return ObservationQuality(round(_clip01(score), 4), {key: round(value, 4) for key, value in components.items()})
