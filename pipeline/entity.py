"""Video-local tracklet reconciliation for persistent vessel entities."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np


def aggregate_entity_identity(member_tracks: list[Any]) -> dict[str, Any]:
    """Aggregate persistent identity state across all tracklets of one entity."""
    tracks = [track for track in member_tracks if track is not None]
    if not tracks:
        return {
            "verified_identity": "",
            "archive_candidate_id": "",
            "archive_similarity_score": 0.0,
            "identity_state": "unknown",
            "identity_evidence_source": "none",
            "recognition_attempts": 0,
        }

    accepted = [
        track for track in tracks
        if getattr(track, "identity_state", "unknown") in {"confirmed", "structure_verified"}
        and getattr(track, "verified_identity", "")
    ]
    accepted_identities = {str(track.verified_identity) for track in accepted}
    attempts = max((int(getattr(track, "recognition_attempts", 0)) for track in tracks), default=0)
    if len(accepted_identities) > 1:
        return {
            "verified_identity": "",
            "archive_candidate_id": "",
            "archive_similarity_score": max(float(getattr(track, "archive_similarity_score", 0.0)) for track in accepted),
            "identity_state": "conflicting",
            "identity_evidence_source": "entity_member_conflict",
            "recognition_attempts": attempts,
        }
    review_requested = [track for track in tracks if getattr(track, "identity_state", "unknown") == "review_requested"]
    if accepted and review_requested:
        accepted_identity = next(iter(accepted_identities))
        contradictory_reviews = [
            track for track in review_requested
            if str(getattr(track, "identity_state_before_review", "")) == "conflicting"
            and str(getattr(track, "archive_candidate_id", "")) not in {"", accepted_identity}
        ]
        if not contradictory_reviews:
            review_requested = []
    if review_requested:
        selected = max(review_requested, key=lambda track: int(getattr(track, "review_requested_frame", 0)))
        return {
            "verified_identity": "",
            "archive_candidate_id": str(getattr(selected, "archive_candidate_id", "")),
            "archive_similarity_score": float(getattr(selected, "archive_similarity_score", 0.0)),
            "identity_state": "review_requested",
            "identity_state_before_review": str(getattr(selected, "identity_state_before_review", "")),
            "identity_evidence_source": "operator_escalation",
            "recognition_attempts": attempts,
            "review_requested_frame": int(getattr(selected, "review_requested_frame", 0)),
            "review_action": str(getattr(selected, "review_action", "")),
            "review_reasons": list(getattr(selected, "review_reasons", [])),
        }
    if accepted:
        selected = max(
            accepted,
            key=lambda track: (
                getattr(track, "identity_state", "unknown") == "confirmed",
                float(getattr(track, "archive_similarity_score", 0.0)),
            ),
        )
        identity = str(selected.verified_identity)
        return {
            "verified_identity": identity,
            "archive_candidate_id": identity,
            "archive_similarity_score": float(getattr(selected, "archive_similarity_score", 0.0)),
            "identity_state": str(getattr(selected, "identity_state", "structure_verified")),
            "identity_evidence_source": str(getattr(selected, "identity_evidence_source", "entity_member_identity")),
            "recognition_attempts": attempts,
        }

    for state in ("out_of_archive", "conflicting", "uncertain"):
        candidates = [track for track in tracks if getattr(track, "identity_state", "unknown") == state]
        if candidates:
            selected = max(candidates, key=lambda track: float(getattr(track, "archive_similarity_score", 0.0)))
            return {
                "verified_identity": "",
                "archive_candidate_id": "" if state == "out_of_archive" else str(getattr(selected, "archive_candidate_id", "")),
                "archive_similarity_score": float(getattr(selected, "archive_similarity_score", 0.0)),
                "identity_state": state,
                "identity_evidence_source": str(getattr(selected, "identity_evidence_source", "entity_member_state")),
                "recognition_attempts": attempts,
            }

    return {
        "verified_identity": "",
        "archive_candidate_id": "",
        "archive_similarity_score": 0.0,
        "identity_state": "unknown",
        "identity_evidence_source": "none",
        "recognition_attempts": attempts,
    }


def appearance_descriptor(crop: np.ndarray | None) -> list[float]:
    if crop is None or crop.size == 0:
        return []
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).reshape(-1)
    norm = float(np.linalg.norm(histogram))
    if norm <= 1e-12:
        return []
    return (histogram / norm).astype(np.float32).tolist()


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return float(np.dot(np.asarray(left, dtype=np.float32), np.asarray(right, dtype=np.float32)))


@dataclass
class EntityBinding:
    entity_id: str
    track_id: int
    source_track_id: int | None
    reassociated: bool
    association_score: float
    member_track_ids: list[int]


@dataclass
class EntityRecord:
    entity_id: str
    first_frame: int
    last_frame: int
    latest_track_id: int
    member_track_ids: list[int] = field(default_factory=list)
    bbox: tuple[int, int, int, int] | None = None
    normalized_center: tuple[float, float] = (0.0, 0.0)
    normalized_area: float = 0.0
    appearance: list[float] = field(default_factory=list)
    appearance_bank: list[list[float]] = field(default_factory=list)
    reassociation_count: int = 0
    last_association_score: float = 0.0


class EntityReconciler:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.max_gap_frames = max(1, int(cfg.get("max_gap_frames", 180)))
        self.min_gap_frames = max(1, int(cfg.get("min_gap_frames", 1)))
        self.min_appearance_similarity = float(cfg.get("min_appearance_similarity", 0.78))
        self.max_center_distance = float(cfg.get("max_center_distance", 0.35))
        self.max_log_area_ratio = float(cfg.get("max_log_area_ratio", 1.20))
        self.min_association_score = float(cfg.get("min_association_score", 0.72))
        self.appearance_weight = float(cfg.get("appearance_weight", 0.70))
        self.position_weight = float(cfg.get("position_weight", 0.20))
        self.scale_weight = float(cfg.get("scale_weight", 0.10))
        self.appearance_ema = min(1.0, max(0.0, float(cfg.get("appearance_ema", 0.80))))
        self.max_appearance_prototypes = max(1, int(cfg.get("max_appearance_prototypes", 6)))
        self.appearance_novelty_threshold = float(cfg.get("appearance_novelty_threshold", 0.90))
        self._entities: dict[str, EntityRecord] = {}
        self._track_to_entity: dict[int, str] = {}
        self._next_entity_id = 1

    @staticmethod
    def _geometry(bbox: tuple[int, int, int, int], frame_shape: tuple[int, ...]) -> tuple[tuple[float, float], float]:
        height, width = max(1, frame_shape[0]), max(1, frame_shape[1])
        x1, y1, x2, y2 = bbox
        center = (((x1 + x2) * 0.5) / width, ((y1 + y2) * 0.5) / height)
        area = max(1, x2 - x1) * max(1, y2 - y1) / float(width * height)
        return center, area

    def _new_entity(self, track_id: int, frame_id: int, bbox: tuple[int, int, int, int], center: tuple[float, float], area: float, descriptor: list[float]) -> EntityBinding:
        entity_id = f"E{self._next_entity_id:06d}"
        self._next_entity_id += 1
        self._entities[entity_id] = EntityRecord(
            entity_id=entity_id,
            first_frame=frame_id,
            last_frame=frame_id,
            latest_track_id=track_id,
            member_track_ids=[track_id],
            bbox=bbox,
            normalized_center=center,
            normalized_area=area,
            appearance=descriptor,
            appearance_bank=[descriptor] if descriptor else [],
        )
        self._track_to_entity[track_id] = entity_id
        return EntityBinding(entity_id, track_id, None, False, 1.0, [track_id])

    def _candidate_score(
        self,
        entity: EntityRecord,
        frame_id: int,
        center: tuple[float, float],
        area: float,
        descriptor: list[float],
        visible_track_ids: set[int],
    ) -> float | None:
        if entity.latest_track_id in visible_track_ids:
            return None
        gap = frame_id - entity.last_frame
        if gap < self.min_gap_frames or gap > self.max_gap_frames:
            return None
        prototypes = entity.appearance_bank or ([entity.appearance] if entity.appearance else [])
        appearance = max((cosine_similarity(prototype, descriptor) for prototype in prototypes), default=0.0)
        if appearance < self.min_appearance_similarity:
            return None
        distance = math.dist(entity.normalized_center, center)
        if distance > self.max_center_distance:
            return None
        log_area_ratio = abs(math.log(max(area, 1e-9) / max(entity.normalized_area, 1e-9)))
        if log_area_ratio > self.max_log_area_ratio:
            return None
        position_score = max(0.0, 1.0 - distance / max(self.max_center_distance, 1e-9))
        scale_score = max(0.0, 1.0 - log_area_ratio / max(self.max_log_area_ratio, 1e-9))
        return self.appearance_weight * appearance + self.position_weight * position_score + self.scale_weight * scale_score

    def observe(
        self,
        track_id: int,
        frame_id: int,
        bbox: tuple[int, int, int, int],
        crop: np.ndarray | None,
        frame_shape: tuple[int, ...],
        visible_track_ids: set[int] | None = None,
    ) -> EntityBinding:
        descriptor = appearance_descriptor(crop)
        center, area = self._geometry(bbox, frame_shape)
        visible_track_ids = set(visible_track_ids or ())
        visible_track_ids.discard(track_id)
        existing_id = self._track_to_entity.get(track_id)
        if existing_id:
            entity = self._entities[existing_id]
            self._update_entity(entity, track_id, frame_id, bbox, center, area, descriptor)
            return EntityBinding(existing_id, track_id, None, False, entity.last_association_score, list(entity.member_track_ids))
        if not self.enabled or not descriptor:
            return self._new_entity(track_id, frame_id, bbox, center, area, descriptor)
        candidates: list[tuple[float, EntityRecord]] = []
        for entity in self._entities.values():
            score = self._candidate_score(entity, frame_id, center, area, descriptor, visible_track_ids)
            if score is not None and score >= self.min_association_score:
                candidates.append((score, entity))
        if not candidates:
            return self._new_entity(track_id, frame_id, bbox, center, area, descriptor)
        score, entity = max(candidates, key=lambda item: (item[0], item[1].last_frame))
        source_track_id = entity.latest_track_id
        entity.member_track_ids.append(track_id)
        entity.reassociation_count += 1
        entity.last_association_score = round(score, 6)
        self._track_to_entity[track_id] = entity.entity_id
        self._update_entity(entity, track_id, frame_id, bbox, center, area, descriptor)
        return EntityBinding(entity.entity_id, track_id, source_track_id, True, round(score, 6), list(entity.member_track_ids))

    def _update_entity(self, entity: EntityRecord, track_id: int, frame_id: int, bbox: tuple[int, int, int, int], center: tuple[float, float], area: float, descriptor: list[float]) -> None:
        entity.latest_track_id = track_id
        entity.last_frame = frame_id
        entity.bbox = bbox
        entity.normalized_center = center
        entity.normalized_area = area
        if descriptor:
            prototypes = entity.appearance_bank or ([entity.appearance] if entity.appearance else [])
            best_similarity = max((cosine_similarity(prototype, descriptor) for prototype in prototypes), default=0.0)
            if not prototypes or best_similarity < self.appearance_novelty_threshold:
                if len(entity.appearance_bank) >= self.max_appearance_prototypes:
                    removal_index = 1 if len(entity.appearance_bank) > 1 else 0
                    entity.appearance_bank.pop(removal_index)
                entity.appearance_bank.append(descriptor)
            if entity.appearance and len(entity.appearance) == len(descriptor):
                old = np.asarray(entity.appearance, dtype=np.float32)
                new = np.asarray(descriptor, dtype=np.float32)
                mixed = self.appearance_ema * old + (1.0 - self.appearance_ema) * new
                norm = float(np.linalg.norm(mixed))
                entity.appearance = (mixed / norm).tolist() if norm > 1e-12 else descriptor
            else:
                entity.appearance = descriptor

    def entity_id_for(self, track_id: int) -> str:
        return self._track_to_entity.get(track_id, "")

    def export(self) -> list[dict[str, Any]]:
        return [asdict(entity) for entity in sorted(self._entities.values(), key=lambda item: item.entity_id)]
