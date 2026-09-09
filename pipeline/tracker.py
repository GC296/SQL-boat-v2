"""Track-level temporal evidence ledger and state management."""
from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvidenceRecord:
    frame_id: int
    source: str
    timestamp: float = field(default_factory=time.time)
    bbox: tuple[int, int, int, int] | None = None
    confidence: float = 0.0
    observation_quality: float = 0.0
    quality_components: dict[str, float] = field(default_factory=dict)
    modality: str = "visible"
    hull_number: str = ""
    description: str = ""
    identity_features: dict[str, str] = field(default_factory=dict)
    match_type: str = "none"
    semantic_match_ids: list[str] = field(default_factory=list)
    semantic_matches: list[dict[str, Any]] = field(default_factory=list)
    action: str = ""
    reasons: list[str] = field(default_factory=list)


@dataclass
class TrackInfo:
    track_id: int
    entity_id: str = ""
    member_track_ids: list[int] = field(default_factory=list)
    reassociation_count: int = 0
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    last_recognized_frame: int = 0
    last_recognition_track_id: int = 0
    bbox: tuple[int, int, int, int] | None = None
    confidence: float = 0.0
    observation_quality: float = 0.0
    quality_components: dict[str, float] = field(default_factory=dict)
    modality: str = "visible"
    hull_number: str = ""
    description: str = ""
    identity_features: dict[str, str] = field(default_factory=dict)
    recognized: bool = False
    pending: bool = False
    db_match_id: str = ""
    db_match_desc: str = ""
    db_matched: bool = False
    verified_identity: str = ""
    archive_candidate_id: str = ""
    archive_similarity_score: float = 0.0
    visual_candidate_id: str = ""
    visual_similarity_score: float = 0.0
    visual_margin: float = 0.0
    visual_best_candidate_id: str = ""
    visual_best_similarity_score: float = 0.0
    visual_best_margin: float = 0.0
    visual_observation_count: int = 0
    visual_consistent_observations: int = 0
    visual_low_score_observations: int = 0
    semantic_match_ids: list[str] = field(default_factory=list)
    semantic_matches: list[dict[str, Any]] = field(default_factory=list)
    structure_candidate_scores: dict[str, float] = field(default_factory=dict)
    structure_evidence_count: int = 0
    structure_consistent_observations: int = 0
    structure_conflict_observations: int = 0
    structure_best_single_score: float = 0.0
    structure_min_consistent_score: float = 0.0
    structure_min_consistent_margin: float = 0.0
    identity_evidence_source: str = "none"
    identity_decision_mode: str = "fused"
    identity_state: str = "unknown"
    identity_state_before_review: str = ""
    episode_status: str = "active"
    review_requested_frame: int = 0
    review_action: str = ""
    review_reasons: list[str] = field(default_factory=list)
    hull_candidate_scores: dict[str, float] = field(default_factory=dict)
    recognition_attempts: int = 0
    last_recognition_quality: float = 0.0
    failed_recognition_count: int = 0
    last_uncertainty_score: float = 1.0
    uncertainty_reasons: list[str] = field(default_factory=lambda: ["not_recognized"])
    risk_level: str = "low"
    risk_score: float = 0.0
    risk_reasons: list[str] = field(default_factory=list)
    encounter_explanation: str = ""
    warning_zone_frames: int = 0
    trajectory: list[dict[str, float]] = field(default_factory=list)
    evidence_history: list[EvidenceRecord] = field(default_factory=list)
    event_history: list[dict[str, Any]] = field(default_factory=list)
    action_history: list[dict[str, Any]] = field(default_factory=list)


class TrackManager:
    def __init__(
        self,
        max_stale_frames: int = 300,
        memory_mode: str = "full",
        ledger_decay: float = 0.97,
        ledger_min_support: float = 0.30,
        max_evidence_per_track: int = 60,
        max_events_per_track: int = 120,
    ):
        self._tracks: dict[int, TrackInfo] = {}
        self._completed_tracks: dict[int, TrackInfo] = {}
        self._max_stale_frames = int(max_stale_frames)
        self._memory_mode = memory_mode.lower()
        self._ledger_decay = min(1.0, max(0.0, float(ledger_decay)))
        self._ledger_min_support = max(0.0, float(ledger_min_support))
        self._max_evidence_per_track = int(max_evidence_per_track)
        self._max_events_per_track = int(max_events_per_track)
        self._lock = threading.RLock()

    @staticmethod
    def _normalize_hull(value: str) -> str:
        return "".join((value or "").upper().split())

    def get_or_create(self, track_id: int, frame_id: int) -> TrackInfo:
        with self._lock:
            if track_id not in self._tracks:
                restored = self._completed_tracks.pop(track_id, None)
                self._tracks[track_id] = restored or TrackInfo(track_id=track_id, entity_id=f"T{track_id}", member_track_ids=[track_id], first_seen_frame=frame_id, last_seen_frame=frame_id)
                self._tracks[track_id].last_seen_frame = frame_id
            else:
                self._tracks[track_id].last_seen_frame = frame_id
            return self._tracks[track_id]

    def record_observation(
        self,
        track_id: int,
        frame_id: int,
        bbox: tuple[int, int, int, int],
        confidence: float,
        observation_quality: float,
        quality_components: dict[str, float] | None = None,
        frame_shape: tuple[int, ...] | None = None,
        entity_id: str = "",
        member_track_ids: list[int] | None = None,
    ) -> TrackInfo:
        with self._lock:
            info = self.get_or_create(track_id, frame_id)
            if entity_id:
                info.entity_id = entity_id
            if member_track_ids:
                info.member_track_ids = list(dict.fromkeys(member_track_ids))
            info.bbox = bbox
            info.confidence = float(confidence)
            info.observation_quality = float(observation_quality)
            info.quality_components = dict(quality_components or {})
            x1, y1, x2, y2 = bbox
            width, height = max(1, x2 - x1), max(1, y2 - y1)
            frame_h = float(frame_shape[0]) if frame_shape else 1.0
            frame_w = float(frame_shape[1]) if frame_shape and len(frame_shape) > 1 else 1.0
            item = {
                "frame_id": float(frame_id),
                "cx": ((x1 + x2) / 2.0) / max(1.0, frame_w),
                "cy": ((y1 + y2) / 2.0) / max(1.0, frame_h),
                "scale": (width * height) / max(1.0, frame_w * frame_h),
                "quality": float(observation_quality),
            }
            info.trajectory.append(item)
            info.trajectory[:] = info.trajectory[-60:]
            self._append_evidence_locked(info, EvidenceRecord(
                frame_id=frame_id, source="visual_observation", bbox=bbox,
                confidence=float(confidence), observation_quality=float(observation_quality),
                quality_components=dict(quality_components or {}),
            ))
            return info

    def inherit_entity_state(self, track_id: int, source_track_id: int, entity_id: str, member_track_ids: list[int], frame_id: int) -> TrackInfo:
        with self._lock:
            target = self.get_or_create(track_id, frame_id)
            source = self._tracks.get(source_track_id) or self._completed_tracks.get(source_track_id)
            if source is None or source is target:
                target.entity_id = entity_id
                target.member_track_ids = list(dict.fromkeys(member_track_ids))
                return target
            preserved = {
                "hull_number", "description", "identity_features", "recognized", "db_match_id", "db_match_desc", "db_matched",
                "verified_identity", "archive_candidate_id", "archive_similarity_score", "semantic_match_ids",
                "visual_candidate_id", "visual_similarity_score", "visual_margin", "visual_observation_count",
                "visual_best_candidate_id", "visual_best_similarity_score", "visual_best_margin",
                "visual_consistent_observations", "visual_low_score_observations",
                "semantic_matches", "structure_candidate_scores", "structure_evidence_count",
                "structure_consistent_observations", "structure_conflict_observations", "structure_best_single_score",
                "structure_min_consistent_score", "structure_min_consistent_margin", "identity_evidence_source",
                "identity_decision_mode",
                "identity_state", "identity_state_before_review", "episode_status", "review_requested_frame",
                "review_action", "review_reasons", "hull_candidate_scores", "recognition_attempts", "last_recognition_quality",
                "failed_recognition_count", "last_uncertainty_score", "uncertainty_reasons", "risk_level",
                "risk_score", "risk_reasons", "encounter_explanation", "warning_zone_frames", "evidence_history",
                "event_history", "action_history", "last_recognized_frame", "last_recognition_track_id",
            }
            for name in preserved:
                setattr(target, name, copy.deepcopy(getattr(source, name)))
            target.pending = False
            target.entity_id = entity_id
            target.member_track_ids = list(dict.fromkeys(member_track_ids))
            target.reassociation_count = source.reassociation_count + 1
            self._append_event_locked(target, "tracklet_reassociated", frame_id, {
                "source_track_id": source_track_id,
                "new_track_id": track_id,
                "entity_id": entity_id,
            })
            return target

    def needs_recognition(self, track_id: int) -> bool:
        with self._lock:
            info = self._tracks.get(track_id)
            return info is None or (not info.recognized and not info.pending)

    def needs_refresh(self, track_id: int, frame_id: int, gap_num: int, skip_matched: bool = False) -> bool:
        with self._lock:
            info = self._tracks.get(track_id)
            if info is None or not info.recognized or info.pending or (skip_matched and info.db_matched):
                return False
            anchor = info.last_recognized_frame or info.first_seen_frame
            return frame_id - anchor >= gap_num

    def mark_pending(self, track_id: int) -> None:
        with self._lock:
            if track_id in self._tracks:
                self._tracks[track_id].pending = True

    def cancel_pending(self, track_id: int) -> None:
        with self._lock:
            if track_id in self._tracks:
                self._tracks[track_id].pending = False

    def bind_result(
        self,
        track_id: int,
        hull_number: str,
        description: str,
        frame_id: int = 0,
        identity_features: dict[str, str] | None = None,
        match_type: str = "none",
        semantic_match_ids: list[str] | None = None,
        semantic_matches: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._lock:
            info = self._tracks.get(track_id)
            if info is None:
                return
            info.pending = False
            info.recognized = True
            info.last_recognized_frame = frame_id
            info.last_recognition_track_id = track_id
            info.recognition_attempts += 1
            info.last_recognition_quality = info.observation_quality
            info.description = description or info.description
            info.db_matched = False
            info.db_match_id = ""
            info.semantic_match_ids = list(semantic_match_ids or [])
            info.semantic_matches = list(semantic_matches or [])
            if hull_number or description:
                info.failed_recognition_count = 0
            else:
                info.failed_recognition_count += 1
            record = EvidenceRecord(
                frame_id=frame_id, source="vlm_recognition", bbox=info.bbox,
                confidence=info.confidence, observation_quality=info.observation_quality,
                quality_components=dict(info.quality_components), hull_number=hull_number,
                description=description, identity_features=dict(identity_features or {}), match_type=match_type,
                semantic_match_ids=list(semantic_match_ids or []), semantic_matches=list(semantic_matches or []),
                action="recognize",
            )
            self._append_evidence_locked(info, record)
            self._fuse_identity_locked(info, frame_id)
            self._fuse_structure_locked(info, frame_id)
            self._append_event_locked(info, "recognition", frame_id, {
                "raw_hull_number": hull_number, "fused_hull_number": info.hull_number,
                "identity_state": info.identity_state, "uncertainty": info.last_uncertainty_score,
                "structure_candidate_scores": dict(info.structure_candidate_scores),
            })

    def _fuse_structure_locked(self, info: TrackInfo, frame_id: int) -> None:
        records = [record for record in info.evidence_history if record.source == "vlm_recognition" and record.semantic_matches]
        weighted_scores: dict[str, float] = {}
        weights: dict[str, float] = {}
        observation_bests: list[tuple[str, float, float]] = []
        for record in records:
            age = max(0, frame_id - record.frame_id)
            weight = max(0.05, record.observation_quality) * max(0.10, record.confidence) * math.pow(self._ledger_decay, age)
            ranked_record: list[tuple[str, float]] = []
            for match in record.semantic_matches:
                candidate = self._normalize_hull(str(match.get("hull_number", "")))
                if not candidate:
                    continue
                score = min(1.0, max(0.0, float(match.get("score", 0.0))))
                ranked_record.append((candidate, score))
                weighted_scores[candidate] = weighted_scores.get(candidate, 0.0) + score * weight
                weights[candidate] = weights.get(candidate, 0.0) + weight
            ranked_record.sort(key=lambda item: (-item[1], item[0]))
            if ranked_record:
                second_score = ranked_record[1][1] if len(ranked_record) > 1 else 0.0
                observation_bests.append((ranked_record[0][0], ranked_record[0][1], ranked_record[0][1] - second_score))
        info.structure_candidate_scores = {
            candidate: round(weighted_scores[candidate] / max(weights[candidate], 1e-9), 6)
            for candidate in weighted_scores
        }
        info.structure_evidence_count = len(records)
        if not info.structure_candidate_scores:
            info.structure_consistent_observations = 0
            info.structure_conflict_observations = 0
            info.structure_best_single_score = 0.0
            info.structure_min_consistent_score = 0.0
            info.structure_min_consistent_margin = 0.0
            return
        fused_best = max(info.structure_candidate_scores.items(), key=lambda item: (item[1], item[0]))[0]
        consistent = [(score, margin) for candidate, score, margin in observation_bests if candidate == fused_best]
        info.structure_consistent_observations = len(consistent)
        info.structure_conflict_observations = len(observation_bests) - len(consistent)
        info.structure_best_single_score = round(max((score for score, _ in consistent), default=0.0), 6)
        info.structure_min_consistent_score = round(min((score for score, _ in consistent), default=0.0), 6)
        info.structure_min_consistent_margin = round(min((margin for _, margin in consistent), default=0.0), 6)

    def _fuse_identity_locked(self, info: TrackInfo, frame_id: int) -> None:
        records = [r for r in info.evidence_history if r.source == "vlm_recognition" and self._normalize_hull(r.hull_number)]
        if not records:
            info.hull_number = ""
            info.hull_candidate_scores = {}
            info.identity_state = "unknown"
            info.last_uncertainty_score = 1.0
            info.uncertainty_reasons = ["missing_hull_number"]
            return
        if self._memory_mode in {"none", "single", "single_frame"}:
            scores = {self._normalize_hull(records[-1].hull_number): 1.0}
        else:
            scores: dict[str, float] = {}
            for record in records:
                candidate = self._normalize_hull(record.hull_number)
                age = max(0, frame_id - record.frame_id)
                if self._memory_mode == "majority":
                    weight = 1.0
                else:
                    quality = max(0.05, record.observation_quality)
                    confidence = max(0.10, record.confidence)
                    weight = quality * confidence * math.pow(self._ledger_decay, age)
                scores[candidate] = scores.get(candidate, 0.0) + weight
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        info.hull_candidate_scores = {key: round(value, 6) for key, value in ranked}
        best, best_score = ranked[0]
        total = sum(scores.values())
        support = best_score / max(total, 1e-9)
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = (best_score - second) / max(best_score, 1e-9)
        info.hull_number = best if best_score >= self._ledger_min_support else ""
        if not info.hull_number:
            info.identity_state = "unknown"
        elif len(ranked) > 1 and support < 0.60:
            info.identity_state = "conflicting"
        else:
            info.identity_state = "probable"
        info.last_uncertainty_score = round(min(1.0, max(0.0, 1.0 - 0.65 * support - 0.35 * margin)), 4)
        reasons: list[str] = []
        if len(ranked) > 1:
            reasons.append("conflicting_hull_candidates")
        if support < 0.65:
            reasons.append("weak_temporal_support")
        if info.observation_quality < 0.35:
            reasons.append("low_observation_quality")
        info.uncertainty_reasons = reasons or ["stable_temporal_evidence"]

    def bind_db_match(self, track_id: int, db_match_id: str, db_match_desc: str, evidence_source: str = "exact_hull", confidence: float = 1.0) -> None:
        with self._lock:
            info = self._tracks.get(track_id)
            if info:
                info.db_match_id = db_match_id
                info.db_match_desc = db_match_desc
                info.db_matched = True
                info.verified_identity = db_match_id
                info.archive_candidate_id = db_match_id
                info.archive_similarity_score = float(confidence)
                info.identity_state = "confirmed" if evidence_source == "exact_hull" else "structure_verified"
                info.identity_evidence_source = evidence_source
                info.last_uncertainty_score = min(info.last_uncertainty_score, max(0.0, 1.0 - confidence))

    def bind_controller_identity(
        self,
        track_id: int,
        archive_id: str,
        description: str,
        evidence_source: str = "central_vlm_controller",
        state: str = "confirmed",
    ) -> bool:
        """Record the central controller's terminal identity without visual gating."""
        if state not in {"confirmed", "out_of_archive"}:
            return False
        with self._lock:
            info = self._tracks.get(track_id) or self._completed_tracks.get(track_id)
            if info is None:
                return False
            if state == "confirmed":
                if not archive_id:
                    return False
                info.db_match_id = archive_id
                info.db_match_desc = description
                info.db_matched = True
                info.verified_identity = archive_id
                info.archive_candidate_id = archive_id
            else:
                info.db_match_id = ""
                info.db_match_desc = ""
                info.db_matched = False
                info.verified_identity = ""
                info.archive_candidate_id = ""
                info.archive_similarity_score = 0.0
            info.identity_state = state
            info.recognized = True
            info.pending = False
            info.episode_status = "closed"
            info.identity_evidence_source = evidence_source
            info.uncertainty_reasons = []
            return True

    def bind_semantic_matches(self, track_id: int, match_ids: list[str]) -> None:
        with self._lock:
            info = self._tracks.get(track_id)
            if info:
                info.semantic_match_ids = list(match_ids)

    def update_visual_match(
        self,
        track_id: int,
        candidate_id: str,
        score: float,
        margin: float,
        low_score_threshold: float = 0.50,
    ) -> None:
        with self._lock:
            info = self._tracks.get(track_id)
            if info:
                previous_candidate = info.visual_candidate_id
                info.visual_candidate_id = candidate_id
                info.visual_similarity_score = float(score)
                info.visual_margin = float(margin)
                if candidate_id and (
                    not info.visual_best_candidate_id
                    or float(score) > info.visual_best_similarity_score
                ):
                    info.visual_best_candidate_id = candidate_id
                    info.visual_best_similarity_score = float(score)
                    info.visual_best_margin = float(margin)
                if candidate_id:
                    info.visual_observation_count += 1
                    if previous_candidate == candidate_id:
                        info.visual_consistent_observations += 1
                    else:
                        info.visual_consistent_observations = 1
                else:
                    info.visual_consistent_observations = 0
                if not candidate_id or float(score) < float(low_score_threshold):
                    info.visual_low_score_observations += 1
                else:
                    info.visual_low_score_observations = 0

    def finalize_visual_score_gate(self, score_threshold: float) -> int:
        finalized = 0
        with self._lock:
            all_tracks = {**self._completed_tracks, **self._tracks}
            for info in all_tracks.values():
                if info.identity_evidence_source == "central_vlm_controller":
                    continue
                if info.identity_decision_mode != "visual_only":
                    continue
                if info.identity_state in {"confirmed", "structure_verified", "out_of_archive"}:
                    continue
                if info.visual_observation_count <= 0:
                    continue
                if info.visual_best_similarity_score >= float(score_threshold):
                    continue
                info.verified_identity = ""
                info.archive_candidate_id = ""
                info.archive_similarity_score = info.visual_best_similarity_score
                info.identity_state = "out_of_archive"
                info.identity_evidence_source = "visual_score_gate_finalization"
                info.last_uncertainty_score = min(info.last_uncertainty_score, 0.1)
                info.uncertainty_reasons = ["visual_score_gate_below_acceptance_threshold"]
                finalized += 1
        return finalized

    def set_identity_decision_mode(self, track_id: int, mode: str) -> None:
        with self._lock:
            info = self._tracks.get(track_id)
            if info:
                info.identity_decision_mode = str(mode)

    def apply_identity_decision(
        self,
        track_id: int,
        candidate_identity: str,
        state: str,
        confidence: float,
        evidence_source: str = "structure_semantic",
        preserve_verified_identity: bool = True,
        conflict_observations_to_downgrade: int = 2,
    ) -> bool:
        with self._lock:
            info = self._tracks.get(track_id)
            if not info:
                return False
            if preserve_verified_identity and info.verified_identity:
                same_candidate = not candidate_identity or candidate_identity == info.verified_identity
                strong_repeated_conflict = (
                    state == "conflicting"
                    and not same_candidate
                    and info.structure_conflict_observations >= max(1, conflict_observations_to_downgrade)
                )
                if state not in {"confirmed", "structure_verified"} and not strong_repeated_conflict:
                    info.uncertainty_reasons = ["verified_identity_preserved"]
                    return False
                if state in {"confirmed", "structure_verified"} and not same_candidate:
                    info.uncertainty_reasons = ["conflicting_identity_rejected"]
                    return False
            info.identity_state = state
            info.identity_evidence_source = evidence_source if state != "unknown" else "none"
            info.archive_candidate_id = candidate_identity
            info.archive_similarity_score = float(confidence)
            info.last_uncertainty_score = round(1.0 - min(1.0, max(0.0, confidence)), 4)
            if state not in {"confirmed", "structure_verified"}:
                info.db_matched = False
                info.db_match_id = ""
                info.db_match_desc = ""
                info.verified_identity = ""
            if state in {"uncertain", "conflicting"}:
                info.uncertainty_reasons = ["structure_match_uncertain"]
            elif state == "out_of_archive":
                info.uncertainty_reasons = ["structure_below_archive_threshold"]
            elif state == "unknown":
                info.uncertainty_reasons = ["missing_identity_evidence"]
            return True

    def active_perception_decision(self, track_id: int, frame_id: int, threshold: float = 0.65, min_gap_frames: int = 45) -> tuple[bool, float, list[str]]:
        with self._lock:
            info = self._tracks.get(track_id)
            if info is None:
                return False, 0.0, ["missing_track"]
            reasons = list(info.uncertainty_reasons)
            score = info.last_uncertainty_score
            if info.risk_level == "high":
                score = max(score, 0.85)
                reasons.append("high_risk")
            if info.failed_recognition_count >= 2:
                score = max(score, 0.75)
                reasons.append("repeated_failure")
            enough_gap = frame_id - (info.last_recognized_frame or info.first_seen_frame) >= min_gap_frames
            should_query = not info.pending and (not info.recognized or (enough_gap and score >= threshold))
            return should_query, round(score, 4), list(dict.fromkeys(reasons))

    def record_action(self, track_id: int, frame_id: int, action: str, reasons: list[str], metadata: dict[str, Any] | None = None) -> None:
        with self._lock:
            info = self._tracks.get(track_id)
            if not info:
                return
            item = {"frame_id": frame_id, "timestamp": time.time(), "action": action, "reasons": list(reasons), **(metadata or {})}
            info.action_history.append(item)
            info.action_history[:] = info.action_history[-self._max_events_per_track:]
            self._append_event_locked(info, "policy_action", frame_id, item)

    def request_review(
        self,
        track_id: int,
        frame_id: int,
        action: str,
        reasons: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        with self._lock:
            info = self._tracks.get(track_id) or self._completed_tracks.get(track_id)
            if not info or info.episode_status == "review_requested":
                return False
            if action != "central_vlm_review" and info.identity_state in {"confirmed", "structure_verified"} and info.verified_identity:
                self._append_event_locked(info, "review_request_blocked", frame_id, {
                    "action": action,
                    "reasons": list(dict.fromkeys(reasons)),
                    "identity_state": info.identity_state,
                    "verified_identity": info.verified_identity,
                    **(metadata or {}),
                })
                return False
            info.identity_state_before_review = info.identity_state
            info.identity_state = "review_requested"
            info.identity_evidence_source = "central_vlm_controller" if action == "central_vlm_review" else "operator_escalation"
            info.pending = False
            if action == "central_vlm_review":
                info.recognized = True
                info.verified_identity = ""
                info.db_matched = False
                info.db_match_id = ""
                info.db_match_desc = ""
            info.episode_status = "review_requested"
            info.review_requested_frame = int(frame_id)
            info.review_action = action
            info.review_reasons = list(dict.fromkeys(reasons))
            self._append_event_locked(info, "review_requested", frame_id, {
                "action": action,
                "reasons": info.review_reasons,
                "identity_state_before_review": info.identity_state_before_review,
                **(metadata or {}),
            })
            return True

    def update_risk(self, track_id: int, frame_id: int, risk_level: str, reasons: list[str], explanation: str, score: float = 0.0) -> None:
        with self._lock:
            info = self._tracks.get(track_id)
            if not info:
                return
            info.risk_level = risk_level
            info.risk_score = float(score)
            info.risk_reasons = list(reasons)
            info.encounter_explanation = explanation
            self._append_event_locked(info, "risk_update", frame_id, {"risk_level": risk_level, "risk_score": score, "reasons": reasons})

    def get_display_text(self, track_id: int) -> str:
        with self._lock:
            info = self._tracks.get(track_id)
            if info is None:
                return "(等待识别...)"
            if info.identity_state == "review_requested":
                return "(等待远程复核)"
            if not info.recognized:
                return "(识别中...)" if info.pending else "(等待识别...)"
            if info.db_matched:
                return f"(库内确定id：{info.db_match_id})"
            label = info.hull_number or "未知"
            return f"({info.identity_state}：{label})"

    def cleanup_stale(self, current_frame: int) -> int:
        with self._lock:
            stale = [track_id for track_id, info in self._tracks.items() if current_frame - info.last_seen_frame > self._max_stale_frames]
            for track_id in stale:
                self._completed_tracks[track_id] = self._tracks.pop(track_id)
            return len(stale)

    def get(self, track_id: int) -> TrackInfo | None:
        with self._lock:
            return self._tracks.get(track_id)

    def get_any(self, track_id: int) -> TrackInfo | None:
        with self._lock:
            return self._tracks.get(track_id) or self._completed_tracks.get(track_id)

    def tracks_for_entity(self, entity_id: str) -> list[TrackInfo]:
        with self._lock:
            return [t for t in {**self._completed_tracks, **self._tracks}.values() if t.entity_id == entity_id]

    @property
    def all_tracks(self) -> dict[int, TrackInfo]:
        with self._lock:
            return {**self._completed_tracks, **self._tracks}

    @property
    def active_tracks(self) -> dict[int, TrackInfo]:
        with self._lock:
            return dict(self._tracks)

    def export_memory(self) -> list[dict[str, Any]]:
        with self._lock:
            output = []
            all_tracks = {**self._completed_tracks, **self._tracks}
            for info in all_tracks.values():
                item = asdict(info)
                item["evidence_history"] = [asdict(record) for record in info.evidence_history]
                output.append(item)
            return output

    def export_event_log(self) -> list[dict[str, Any]]:
        with self._lock:
            all_tracks = {**self._completed_tracks, **self._tracks}
            events = [dict(event, track_id=info.track_id) for info in all_tracks.values() for event in info.event_history]
            return sorted(events, key=lambda event: (event.get("timestamp", 0), event.get("track_id", 0)))

    def _append_evidence_locked(self, info: TrackInfo, record: EvidenceRecord) -> None:
        info.evidence_history.append(record)
        info.evidence_history[:] = info.evidence_history[-self._max_evidence_per_track:]

    def _append_event_locked(self, info: TrackInfo, event_type: str, frame_id: int, payload: dict[str, Any]) -> None:
        info.event_history.append({"type": event_type, "frame_id": frame_id, "timestamp": time.time(), **payload})
        info.event_history[:] = info.event_history[-self._max_events_per_track:]

    def __len__(self) -> int:
        with self._lock:
            return len(self._tracks)
