"""Local DINOv2 visual prototype retrieval for sidecar identity evidence."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def aggregate_identity_matches(scores: list[float], records: list[dict[str, Any]], top_k: int = 3) -> list[dict[str, Any]]:
    best_by_hull: dict[str, dict[str, Any]] = {}
    for score, record in zip(scores, records):
        hull_number = str(record["hull_number"])
        candidate = {
            "hull_number": hull_number,
            "score": round(float(score), 6),
            "prototype_id": str(record["prototype_id"]),
            "image_path": str(record["image_path"]),
        }
        if hull_number not in best_by_hull or candidate["score"] > best_by_hull[hull_number]["score"]:
            best_by_hull[hull_number] = candidate
    ranked = sorted(best_by_hull.values(), key=lambda item: (-float(item["score"]), item["hull_number"]))
    return ranked[: max(1, int(top_k))]


def visual_confirmation_block_reason(
    text_candidate_id: str,
    visual_candidate_id: str,
    visual_score: float,
    visual_margin: float,
    config: dict[str, Any],
) -> str:
    """Return a conservative reason for blocking semantic confirmation."""
    if not bool(config.get("decision_enabled", False)):
        return ""
    if not visual_candidate_id:
        return "" if bool(config.get("fail_open", True)) else "visual_evidence_missing"
    min_support_score = float(config.get("min_support_score", 0.45))
    if float(visual_score) < min_support_score:
        return "visual_evidence_too_weak"
    if (
        text_candidate_id
        and visual_candidate_id != text_candidate_id
        and float(visual_score) >= float(config.get("conflict_min_score", 0.50))
        and float(visual_margin) >= float(config.get("conflict_min_margin", 0.05))
    ):
        return "visual_identity_conflict"
    min_support_margin = float(config.get("min_support_margin", 0.10))
    if float(visual_margin) < min_support_margin:
        return "visual_candidate_ambiguous"
    return ""


def visual_agreement_supports_confirmation(
    text_candidate_id: str,
    text_state: str,
    text_score: float,
    visual_candidate_id: str,
    visual_score: float,
    visual_margin: float,
    config: dict[str, Any],
) -> bool:
    """Return whether independent structure and visual evidence jointly verify one identity."""
    if not bool(config.get("decision_enabled", False)):
        return False
    if not bool(config.get("agreement_confirmation_enabled", True)):
        return False
    if text_state != "uncertain" or not text_candidate_id or text_candidate_id != visual_candidate_id:
        return False
    if float(text_score) < float(config.get("agreement_min_text_score", 0.70)):
        return False
    if float(visual_score) < float(config.get("agreement_min_visual_score", config.get("min_support_score", 0.45))):
        return False
    if float(visual_margin) < float(config.get("agreement_min_visual_margin", config.get("min_support_margin", 0.10))):
        return False
    return True


def visual_consistency_supports_confirmation(
    text_candidate_id: str,
    text_state: str,
    visual_candidate_id: str,
    visual_score: float,
    visual_margin: float,
    consistent_observations: int,
    config: dict[str, Any],
) -> bool:
    """Return whether repeated strong visual retrieval can rescue missing text evidence."""
    if not bool(config.get("decision_enabled", False)):
        return False
    if not bool(config.get("visual_only_confirmation_enabled", True)):
        return False
    if not visual_candidate_id or text_state not in {"unknown", "uncertain", "conflicting", "out_of_archive"}:
        return False
    if text_candidate_id and text_candidate_id != visual_candidate_id and text_state not in {"conflicting", "out_of_archive"}:
        return False
    if int(consistent_observations) < int(config.get("visual_only_min_observations", 2)):
        return False
    score = float(visual_score)
    margin = float(visual_margin)
    if str(config.get("visual_only_decision_rule", "legacy")).strip().lower() == "score_gate":
        return score >= float(config.get("visual_only_min_score", 0.80))
    if score >= float(config.get("visual_only_strong_score", 0.90)):
        return True
    return (
        score >= float(config.get("visual_only_min_score", 0.70))
        and margin >= float(config.get("visual_only_min_margin", 0.03))
    )


def visual_only_rejection_supports_terminal(
    visual_candidate_id: str,
    visual_score: float,
    visual_margin: float,
    observation_count: int,
    low_score_observations: int,
    config: dict[str, Any],
) -> bool:
    """Return whether repeated weak visual evidence can terminate as unknown."""
    if not visual_candidate_id:
        return False
    min_observations = int(config.get("min_out_of_archive_observations", 2))
    if observation_count < min_observations or low_score_observations < min_observations:
        return False
    if str(config.get("visual_only_decision_rule", "legacy")).strip().lower() == "score_gate":
        return float(visual_score) < float(config.get("visual_only_reject_score", 0.80))
    return (
        float(visual_score) < float(config.get("visual_only_reject_score", 0.70))
        and float(visual_margin) < float(config.get("visual_only_reject_margin", 0.08))
    )


def visual_open_set_override(
    text_candidate_id: str,
    visual_candidate_id: str,
    visual_score: float,
    visual_margin: float,
    config: dict[str, Any],
    low_score_observations: int = 1,
    observation_count: int | None = None,
) -> tuple[str, str]:
    """Map visual evidence to a guarded open-set state override."""
    if not bool(config.get("decision_enabled", False)) or not bool(config.get("open_set_enabled", False)):
        return "", ""
    if not visual_candidate_id:
        return "", ""
    if str(config.get("identity_decision_mode", "fused")).strip().lower() == "visual_only":
        weak_score = float(visual_score) < float(config.get("visual_only_reject_score", 0.70))
        weak_margin = float(visual_margin) < float(config.get("visual_only_reject_margin", 0.08))
        strong_score = float(visual_score) >= float(config.get("visual_only_strong_score", 0.90))
        if str(config.get("visual_only_decision_rule", "legacy")).strip().lower() == "score_gate":
            observations = int(low_score_observations) if observation_count is None else int(observation_count)
            if weak_score and visual_only_rejection_supports_terminal(
                visual_candidate_id,
                visual_score,
                visual_margin,
                observations,
                int(low_score_observations),
                config,
            ):
                return "out_of_archive", "visual_only_score_gate_rejected"
            if weak_score:
                return "uncertain", "visual_only_score_gate_requires_more_evidence"
            return "uncertain", "visual_only_score_gate_candidate"
        if weak_score and weak_margin:
            observations = int(low_score_observations) if observation_count is None else int(observation_count)
            if visual_only_rejection_supports_terminal(
                visual_candidate_id,
                visual_score,
                visual_margin,
                observations,
                int(low_score_observations),
                config,
            ):
                return "out_of_archive", "visual_only_repeated_weak_evidence"
            return "uncertain", "visual_only_weak_evidence_requires_reobservation"
        if strong_score:
            return "uncertain", "visual_only_strong_candidate_requires_consistency"
        if weak_score or weak_margin:
            return "uncertain", "visual_only_gray_zone"
        return "uncertain", "visual_candidate_requires_consistency"
    if float(visual_score) < float(config.get("out_of_archive_score", 0.50)):
        if int(low_score_observations) >= int(config.get("min_out_of_archive_observations", 2)):
            return "out_of_archive", "visual_out_of_archive_repeated_low_score"
        return "uncertain", "visual_low_score_requires_reobservation"
    if float(visual_margin) < float(config.get("out_of_archive_margin", 0.20)):
        return "uncertain", "visual_candidate_ambiguous"
    if not text_candidate_id:
        return "uncertain", "visual_candidate_requires_consistency"
    if text_candidate_id and visual_candidate_id != text_candidate_id:
        return "conflicting", "visual_identity_conflict"
    return "", ""


def _unwrap_state_dict(checkpoint: Any) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise TypeError("DINOv2 checkpoint must contain a state dictionary")
    for key in ("state_dict", "model", "teacher"):
        nested = checkpoint.get(key)
        if isinstance(nested, dict):
            checkpoint = nested
            break
    cleaned = {}
    for key, value in checkpoint.items():
        normalized = str(key)
        for prefix in ("module.", "backbone."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        cleaned[normalized] = value
    return cleaned


class VisualPrototypeIndex:
    def __init__(self, config: dict[str, Any]):
        import torch
        import torch.nn.functional as functional

        self._torch = torch
        self._functional = functional
        self._lock = threading.Lock()
        self._input_size = max(14, int(config.get("input_size", 224)))
        self._batch_size = max(1, int(config.get("batch_size", 16)))
        self._top_k = max(1, int(config.get("top_k", 3)))
        requested_device = str(config.get("device", "auto")).lower()
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Visual archive requested CUDA, but CUDA is unavailable")
        self.device = torch.device(requested_device)
        self.archive_root = Path(config.get("archive_root", "data/archive/visual_prototypes")).expanduser().resolve()
        self.model_repo = Path(config.get("model_repo", "models/dinov2_torch_home/hub/dinov2-main")).expanduser().resolve()
        self.weights = Path(config.get("weights", "models/dinov2_torch_home/hub/checkpoints/dinov2_vits14_pretrain.pth")).expanduser().resolve()
        self.records = self._load_archive_records()
        self.model = self._load_model()
        self.features = self._encode([record["image"] for record in self.records])

    def _load_archive_records(self) -> list[dict[str, Any]]:
        if not self.archive_root.exists():
            raise FileNotFoundError(f"Visual archive does not exist: {self.archive_root}")
        records = []
        for hull_dir in sorted(path for path in self.archive_root.iterdir() if path.is_dir()):
            for image_path in sorted(hull_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is not None and image.size:
                    records.append({"hull_number": hull_dir.name, "prototype_id": image_path.stem, "image_path": str(image_path), "image": image})
        if not records:
            raise RuntimeError(f"No readable visual prototypes found under: {self.archive_root}")
        return records

    def _load_model(self):
        torch = self._torch
        if not (self.model_repo / "hubconf.py").exists():
            raise FileNotFoundError(f"DINOv2 hubconf.py not found: {self.model_repo}")
        if not self.weights.exists():
            raise FileNotFoundError(f"DINOv2 weights not found: {self.weights}")
        model = torch.hub.load(str(self.model_repo), "dinov2_vits14", source="local", pretrained=False)
        checkpoint = torch.load(str(self.weights), map_location="cpu", weights_only=True)
        model.load_state_dict(_unwrap_state_dict(checkpoint), strict=True)
        return model.eval().to(self.device)

    def _preprocess(self, image: np.ndarray):
        torch = self._torch
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        scale = min(self._input_size / max(1, width), self._input_size / max(1, height))
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        canvas = np.empty((self._input_size, self._input_size, 3), dtype=np.uint8)
        canvas[...] = np.round(IMAGENET_MEAN * 255).astype(np.uint8)
        left = (self._input_size - resized_width) // 2
        top = (self._input_size - resized_height) // 2
        canvas[top:top + resized_height, left:left + resized_width] = resized
        normalized = (canvas.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(normalized.transpose(2, 0, 1)).contiguous()

    def _encode(self, images: list[np.ndarray]):
        torch = self._torch
        features = []
        with torch.inference_mode():
            for start in range(0, len(images), self._batch_size):
                batch = torch.stack([self._preprocess(image) for image in images[start:start + self._batch_size]]).to(self.device)
                output = self.model(batch)
                if isinstance(output, dict):
                    output = output.get("x_norm_clstoken", output.get("features"))
                features.append(self._functional.normalize(output.float(), dim=1).cpu())
        return torch.cat(features)

    def search(self, image: np.ndarray, top_k: int | None = None) -> tuple[list[dict[str, Any]], float]:
        if image is None or image.size == 0:
            return [], 0.0
        started = time.perf_counter()
        with self._lock:
            feature = self._encode([image])[0]
        scores = self._torch.mv(self.features, feature).tolist()
        matches = aggregate_identity_matches(scores, self.records, top_k or self._top_k)
        return matches, (time.perf_counter() - started) * 1000.0


_INDEX_CACHE: dict[tuple[Any, ...], VisualPrototypeIndex] = {}
_INDEX_CACHE_LOCK = threading.Lock()


def get_visual_prototype_index(config: dict[str, Any]) -> VisualPrototypeIndex:
    key = (
        str(Path(config.get("archive_root", "data/archive/visual_prototypes")).expanduser().resolve()),
        str(Path(config.get("model_repo", "models/dinov2_torch_home/hub/dinov2-main")).expanduser().resolve()),
        str(Path(config.get("weights", "models/dinov2_torch_home/hub/checkpoints/dinov2_vits14_pretrain.pth")).expanduser().resolve()),
        str(config.get("device", "auto")),
        int(config.get("input_size", 224)),
    )
    with _INDEX_CACHE_LOCK:
        if key not in _INDEX_CACHE:
            _INDEX_CACHE[key] = VisualPrototypeIndex(config)
        return _INDEX_CACHE[key]
