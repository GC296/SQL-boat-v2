"""Small, thread-safe entity episode state used by the central controller."""
from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EntityEpisode:
    session_id: str
    entity_id: str
    episode_id: str
    query_limit: int = 3
    version: int = 0
    observation_version: int = 0
    used: int = 0
    reserved: int = 0
    status: str = "active"
    identity_state: str = "uncertain"
    views: dict[str, dict[str, Any]] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    claims: list[dict[str, Any]] = field(default_factory=list)
    candidate_assessments: list[dict[str, Any]] = field(default_factory=list)
    gaps: list[dict[str, Any]] = field(default_factory=list)
    revisions: list[dict[str, Any]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    belief_proposals: list[dict[str, Any]] = field(default_factory=list)
    tracklet_history: list[int] = field(default_factory=list)
    request_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    visual: dict[str, Any] = field(default_factory=lambda: {
        "candidate_id": "", "score": 0.0, "margin": 0.0,
        "search_count": 0, "observation_count": 0, "low_support_count": 0,
        "searched_view_ids": [],
    })
    out_of_archive_ready: bool = False
    last_result: dict[str, Any] = field(default_factory=dict)
    max_views: int = 64
    control_calls: int = 0
    settlement_calls: int = 0
    last_consumed_observation: int = 0
    feedback: dict[str, Any] = field(default_factory=dict)
    assessment: dict[str, Any] = field(default_factory=dict)
    previous_assessment: dict[str, Any] = field(default_factory=dict)
    terminal_record: dict[str, Any] = field(default_factory=dict)
    preferred_views: list[str] = field(default_factory=list)
    pinned_views: set[str] = field(default_factory=set)
    archive_images: list[dict[str, Any]] = field(default_factory=list)
    read_archive_refs: set[str] = field(default_factory=set)
    tool_counts: dict[str, int] = field(default_factory=dict)
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def add_observation(self, view_id: str, frame_id: int, crop: Any, metadata: dict[str, Any] | None = None) -> bool:
        with self.lock:
            track_id = (metadata or {}).get("track_id")
            if track_id is not None and int(track_id) not in self.tracklet_history:
                self.tracklet_history.append(int(track_id))
            if self.status != "active":
                return False
            view_id = str(view_id).strip()
            if not view_id:
                raise ValueError("view_id is required")
            if view_id in self.views:
                return False
            view = {"view_id": view_id, "frame_id": int(frame_id), "observed_at": time.time(), "image": crop}
            view.update(dict(metadata or {}))
            is_new = view_id not in self.views
            self.views[view_id] = view
            if is_new:
                self.observation_version += 1
            while len(self.views) > max(1, self.max_views):
                protected = set(self.preferred_views) | self.pinned_views
                protected.update(v for e in self.evidence for v in e.get("view_ids", []))
                victim = next((v for v in self.views if v not in protected and v != view_id), None)
                if victim is None:
                    victim = next((v for v in self.views if v not in self.pinned_views and v != view_id), None)
                if victim is None:
                    break
                del self.views[victim]
            self.version += 1
            track_id = view.get("track_id")
            if track_id is not None and int(track_id) not in self.tracklet_history:
                self.tracklet_history.append(int(track_id))
            return is_new and self.status == "active"

    def view(self, view_id: str) -> Any | None:
        with self.lock:
            record = self.views.get(str(view_id))
            return record.get("image") if record else None

    def reserve_query(self) -> bool:
        with self.lock:
            if self.status != "active" or self.used + self.reserved >= self.query_limit:
                return False
            self.reserved += 1
            self.version += 1
            return True

    def settle_query(self, success: bool = True) -> None:
        with self.lock:
            if self.reserved:
                self.reserved -= 1
            if success:
                self.used += 1
            self.version += 1

    def append_action(self, action: dict[str, Any]) -> None:
        with self.lock:
            self.actions.append({"timestamp": time.time(), **dict(action)})
            self.actions[:] = self.actions[-120:]
            self.version += 1

    def append_evidence(self, evidence: dict[str, Any]) -> bool:
        with self.lock:
            request_id = str(evidence.get("request_id", "")).strip()
            if request_id and request_id in self.request_cache:
                return False
            item = {
                "evidence_id": str(evidence.get("evidence_id") or f"{request_id}:evidence"),
                "received_at": time.time(),
                "validity_status": "active",
                **dict(evidence),
            }
            self.evidence.append(item)
            self.evidence[:] = self.evidence[-120:]
            if request_id:
                self.request_cache[request_id] = copy.deepcopy(item)
                while len(self.request_cache) > 120:
                    self.request_cache.pop(next(iter(self.request_cache)))
            self.last_result = copy.deepcopy(item)
            self.version += 1
            return True

    def add_claims(self, claims: list[dict[str, Any]]) -> None:
        with self.lock:
            existing = {str(item.get("claim_id")) for item in self.claims}
            for claim in claims:
                claim_id = str(claim.get("claim_id", "")).strip()
                if claim_id and claim_id not in existing:
                    self.claims.append(copy.deepcopy(claim))
                    existing.add(claim_id)
            self.claims[:] = self.claims[-240:]
            self.version += 1

    def apply_belief_patch(
        self, patch, valid_evidence_ids, valid_claim_ids, valid_archive_ids, valid_view_ids,
        revision_evidence_ids=None,
    ) -> list[str]:
        """Commit a source-linked patch atomically, including same-patch dependencies."""
        with self.lock:
            names = ("claims", "candidate_assessments", "gaps", "revisions", "evidence")
            backup = {name: copy.deepcopy(getattr(self, name)) for name in names}
            old_version = self.version
            try:
                errors = self._apply_belief_patch(
                    patch, set(valid_evidence_ids), set(valid_claim_ids), set(valid_archive_ids),
                    set(valid_view_ids), revision_evidence_ids,
                )
            except (TypeError, ValueError, KeyError, StopIteration, AttributeError) as exc:
                errors = [f"malformed belief patch: {exc}"]
            if errors:
                for name, value in backup.items():
                    setattr(self, name, value)
                self.version = old_version
            return errors

    def _apply_belief_patch(
        self,
        patch: tuple[dict[str, Any], ...],
        valid_evidence_ids: set[str],
        valid_claim_ids: set[str],
        valid_archive_ids: set[str],
        valid_view_ids: set[str],
        revision_evidence_ids: set[str] | None = None,
    ) -> list[str]:
        """Apply only source-linked controller proposals and return rejections."""
        rejected: list[str] = []
        with self.lock:
            for item in patch:
                operation = str(item.get("op", "")).strip().lower()
                if operation == "upsert_claim":
                    claim = copy.deepcopy(item.get("claim"))
                    if not isinstance(claim, dict):
                        rejected.append("upsert_claim requires claim")
                        continue
                    claim_id = str(claim.get("claim_id", "")).strip()
                    evidence_id = str(claim.get("evidence_id", "")).strip()
                    subject = str(claim.get("subject_entity_id", "")).strip()
                    views = claim.get("view_ids", [])
                    required_claim_fields = ("attribute", "observed_value", "visibility", "source_role")
                    if (
                        not claim_id
                        or not evidence_id
                        or evidence_id not in valid_evidence_ids
                        or subject != self.entity_id
                        or any(field not in claim for field in required_claim_fields)
                        or not str(claim.get("attribute", "")).strip()
                        or not str(claim.get("visibility", "")).strip()
                        or not str(claim.get("source_role", "")).strip()
                    ):
                        rejected.append("claim is not linked to this episode")
                        continue
                    if not isinstance(views, list) or any(str(view_id) not in valid_view_ids for view_id in views):
                        rejected.append("claim references an unavailable view")
                        continue
                    origin = next((e for e in self.evidence if e.get("evidence_id") == evidence_id), {})
                    origin_views = origin.get("view_ids", []) or ([origin.get("payload", {}).get("view_id")] if origin.get("payload", {}).get("view_id") else [])
                    if any(v not in origin_views for v in views):
                        rejected.append("claim views do not belong to its evidence")
                        continue
                    existing = next((c for c in self.claims if c.get("claim_id") == claim_id), None)
                    if existing and existing.get("source_role") != "controller_observation":
                        rejected.append("cannot replace an original tool claim")
                        continue
                    claim["source_role"] = "controller_observation"
                    claim["validity_status"] = "active"
                    claim.setdefault("validity_status", "active")
                    claim.setdefault("created_at", time.time())
                    replaced = False
                    for index, existing in enumerate(self.claims):
                        if existing.get("claim_id") == claim_id:
                            self.claims[index] = claim
                            replaced = True
                            break
                    if not replaced:
                        self.claims.append(claim)
                    valid_claim_ids.add(claim_id)
                elif operation == "upsert_candidate_assessment":
                    assessment = copy.deepcopy(item.get("assessment"))
                    if not isinstance(assessment, dict):
                        rejected.append("upsert_candidate_assessment requires assessment")
                        continue
                    archive_id = str(assessment.get("archive_id", "")).strip()
                    supports = assessment.get("support_claim_ids", [])
                    contradicts = assessment.get("contradict_claim_ids", [])
                    gaps = assessment.get("unresolved_gap_ids", [])
                    if archive_id not in valid_archive_ids or not all(isinstance(values, list) for values in (supports, contradicts, gaps)):
                        rejected.append("candidate assessment is invalid")
                        continue
                    active_claim_ids = {
                        str(value.get("claim_id"))
                        for value in self.claims
                        if value.get("validity_status", "active") in {"active", "disputed"}
                    }
                    if any(str(value) not in valid_claim_ids or str(value) not in active_claim_ids for value in supports + contradicts):
                        rejected.append("candidate assessment has an unknown claim")
                        continue
                    if any(str(value) not in {str(g.get("gap_id")) for g in self.gaps} for value in gaps):
                        rejected.append("candidate assessment has an unknown gap")
                        continue
                    assessment.setdefault("assessment_source", "controller")
                    assessment["updated_at_version"] = self.version
                    self._upsert_by_id(self.candidate_assessments, "archive_id", archive_id, assessment)
                elif operation == "upsert_gap":
                    gap = copy.deepcopy(item.get("gap"))
                    if not isinstance(gap, dict):
                        rejected.append("upsert_gap requires gap")
                        continue
                    gap_id = str(gap.get("gap_id", "")).strip()
                    candidate_ids = gap.get("candidate_ids", [])
                    basis_refs = gap.get("basis_refs", gap.get("evidence_refs", []))
                    if not gap_id or not isinstance(candidate_ids, list) or any(str(value) not in valid_archive_ids for value in candidate_ids):
                        rejected.append("gap candidates are invalid")
                        continue
                    if not isinstance(basis_refs, list) or any(not self._valid_source_ref(str(value), valid_evidence_ids, valid_claim_ids, valid_archive_ids) for value in basis_refs):
                        rejected.append("gap has invalid basis references")
                        continue
                    gap.setdefault("status", "unresolved")
                    gap.setdefault("attempt_request_ids", [])
                    gap.setdefault("resolution_refs", [])
                    if gap["status"] == "resolved" and not gap["resolution_refs"]:
                        rejected.append("resolved gap needs resolution references")
                        continue
                    if any(not self._valid_source_ref(str(ref), valid_evidence_ids, valid_claim_ids, valid_archive_ids) for ref in gap["resolution_refs"]):
                        rejected.append("gap has invalid resolution references")
                        continue
                    self._upsert_by_id(self.gaps, "gap_id", gap_id, gap)
                elif operation == "resolve_gap":
                    gap_id = str(item.get("gap_id", "")).strip()
                    refs = item.get("resolution_refs", [])
                    if not gap_id or not isinstance(refs, list) or not refs or any(not self._valid_source_ref(str(value), valid_evidence_ids, valid_claim_ids, valid_archive_ids) for value in refs):
                        rejected.append("gap resolution needs valid source references")
                        continue
                    gap = next((value for value in self.gaps if value.get("gap_id") == gap_id), None)
                    if gap is None:
                        rejected.append("cannot resolve an unknown gap")
                        continue
                    gap["resolution_refs"] = list(dict.fromkeys(str(value) for value in refs))
                    gap["status"] = "resolved"
                elif operation == "reopen_gap":
                    gap_id = str(item.get("gap_id", "")).strip()
                    gap = next((value for value in self.gaps if value.get("gap_id") == gap_id), None)
                    if gap is None:
                        rejected.append("cannot reopen an unknown gap")
                        continue
                    gap["status"] = "unresolved"
                    gap["resolution_refs"] = []
                elif operation == "set_evidence_status":
                    evidence_id = str(item.get("evidence_id", "")).strip()
                    status = str(item.get("status", "")).strip().lower()
                    reason = str(item.get("reason", "")).strip()
                    known_evidence_ids = revision_evidence_ids if revision_evidence_ids is not None else valid_evidence_ids
                    if evidence_id not in known_evidence_ids or status not in {"active", "disputed", "retracted", "superseded"} or not reason:
                        rejected.append("evidence status revision is invalid")
                        continue
                    record = next(value for value in self.evidence if value.get("evidence_id") == evidence_id)
                    record["validity_status"] = status
                    if status in {"active", "disputed"}:
                        valid_evidence_ids.add(evidence_id)
                    else:
                        valid_evidence_ids.discard(evidence_id)
                    self.revisions.append({"evidence_id": evidence_id, "status": status, "reason": reason, "source": "controller", "created_at": time.time()})
                    for claim in self.claims:
                        if claim.get("evidence_id") == evidence_id:
                            if status != "active":
                                claim.setdefault("status_before_source_revision", claim.get("validity_status", "active"))
                                claim["validity_status"] = status
                            else:
                                claim["validity_status"] = claim.pop("status_before_source_revision", "active")
                            if claim["validity_status"] in {"active", "disputed"}:
                                valid_claim_ids.add(str(claim.get("claim_id")))
                            else:
                                valid_claim_ids.discard(str(claim.get("claim_id")))
                elif operation:
                    rejected.append(f"unsupported belief patch operation: {operation}")
                else:
                    rejected.append("belief patch operation is missing")
            self._recompute_belief_locked()
            self.revisions[:] = self.revisions[-120:]
            self.version += 1
        return rejected

    @staticmethod
    def _upsert_by_id(items: list[dict[str, Any]], key: str, value: str, item: dict[str, Any]) -> None:
        for index, existing in enumerate(items):
            if str(existing.get(key, "")) == value:
                items[index] = item
                return
        items.append(item)

    def _valid_source_ref(
        self,
        value: str,
        evidence_ids: set[str],
        claim_ids: set[str],
        archive_ids: set[str],
    ) -> bool:
        if value in evidence_ids or value in claim_ids:
            return True
        if not value.startswith("archive:"):
            return False
        parts = value.split(":", 2)
        return len(parts) >= 2 and parts[1] in archive_ids and value in self.read_archive_refs

    def _recompute_belief_locked(self) -> None:
        valid_evidence = {item.get("evidence_id") for item in self.evidence if item.get("validity_status", "active") == "active"}
        invalid_evidence = {
            item.get("evidence_id")
            for item in self.evidence
            if item.get("validity_status", "active") != "active"
        }
        invalid_claims = {
            item.get("claim_id")
            for item in self.claims
            if item.get("validity_status", "active") != "active"
            or item.get("evidence_id") in invalid_evidence
        }
        for claim in self.claims:
            if claim.get("evidence_id") not in valid_evidence and claim.get("validity_status", "active") == "active":
                claim["validity_status"] = "disputed"
        for gap in self.gaps:
            if any(ref in invalid_evidence or ref in invalid_claims for ref in gap.get("resolution_refs", [])):
                gap["status"] = "unresolved"
                gap["resolution_refs"] = []
        invalid_claims.update(c.get("claim_id") for c in self.claims if c.get("validity_status", "active") != "active")
        for candidate in self.candidate_assessments:
            for key in ("support_claim_ids", "contradict_claim_ids"):
                candidate[f"effective_{key}"] = [r for r in candidate.get(key, []) if r not in invalid_claims]

    def append_belief_proposal(self, patch: tuple[dict[str, Any], ...]) -> None:
        with self.lock:
            for item in patch:
                self.belief_proposals.append(copy.deepcopy(item))
            self.belief_proposals[:] = self.belief_proposals[-60:]
            self.version += 1

    def update_visual(self, payload: dict[str, Any], min_score: float, min_observations: int) -> None:
        with self.lock:
            matches = list(payload.get("matches", []) or [])
            view_id = str(payload.get("view_id", "")).strip()
            is_new_view = bool(view_id and view_id not in self.visual["searched_view_ids"])
            if is_new_view:
                self.visual["searched_view_ids"].append(view_id)
                self.visual["search_count"] += 1
            candidate = str(matches[0].get("hull_number", "")) if matches else ""
            score = float(matches[0].get("score", 0.0)) if matches else 0.0
            second = float(matches[1].get("score", 0.0)) if len(matches) > 1 else 0.0
            if candidate and is_new_view:
                self.visual["observation_count"] += 1
            if is_new_view and (not candidate or score < float(min_score)):
                self.visual["low_support_count"] += 1
            elif is_new_view:
                self.visual["low_support_count"] = 0
            self.visual.update({"candidate_id": candidate, "score": score, "margin": score - second})
            self.out_of_archive_ready = (
                self.visual["search_count"] >= int(min_observations)
                and self.visual["low_support_count"] >= int(min_observations)
            )
            self.version += 1

    def recompute_visual(self, min_score: float, min_observations: int) -> None:
        """Rebuild visual counters from active SearchArchive evidence."""
        with self.lock:
            records: list[dict[str, Any]] = []
            self.visual.update({"candidate_id": "", "score": 0.0, "margin": 0.0})
            seen_views: set[str] = set()
            for evidence in self.evidence:
                if evidence.get("tool") != "SearchArchive" or evidence.get("status") != "success":
                    continue
                if evidence.get("validity_status", "active") != "active":
                    continue
                payload = evidence.get("payload", {}) or {}
                view_id = str(payload.get("view_id", "")).strip()
                if not view_id or view_id in seen_views:
                    continue
                seen_views.add(view_id)
                records.append(payload)
            self.visual["search_count"] = len(records)
            self.visual["observation_count"] = 0
            self.visual["low_support_count"] = 0
            self.visual["searched_view_ids"] = list(seen_views)
            for payload in records:
                matches = list(payload.get("matches", []) or [])
                if matches:
                    self.visual["observation_count"] += 1
                if not matches or float(matches[0].get("score", 0.0)) < float(min_score):
                    self.visual["low_support_count"] += 1
                else:
                    self.visual["low_support_count"] = 0
                second = float(matches[1].get("score", 0.0)) if len(matches) > 1 else 0.0
                self.visual.update({
                    "candidate_id": str(matches[0].get("hull_number", "")) if matches else "",
                    "score": float(matches[0].get("score", 0.0)) if matches else 0.0,
                    "margin": (float(matches[0].get("score", 0.0)) - second) if matches else 0.0,
                })
            self.out_of_archive_ready = (
                self.visual["search_count"] >= int(min_observations)
                and self.visual["low_support_count"] >= int(min_observations)
            )
            self.version += 1

    def close(self, status: str, archive_id: str = "") -> None:
        with self.lock:
            if status not in {"known", "out_of_archive", "review", "failed"}:
                raise ValueError(f"invalid episode close status: {status}")
            self.status = status
            self.identity_state = "confirmed" if status == "known" else status
            if not self.terminal_record:
                self.terminal_record = {"decision": status, "archive_id": archive_id}
            # Closed episodes retain evidence metadata; full crops are no longer
            # needed for inference. Diagnostic images are stored separately.
            for view in self.views.values():
                view["image"] = None
            self.pinned_views.clear()
            self.version += 1

    def snapshot(self, include_images: bool = False, evidence_limit: int = 120) -> dict[str, Any]:
        with self.lock:
            views = []
            for view in self.views.values():
                item = {key: value for key, value in view.items() if key != "image"}
                if include_images:
                    item["image"] = view.get("image")
                views.append(item)
            return {
                "session_id": self.session_id,
                "entity_id": self.entity_id,
                "episode_id": self.episode_id,
                "version": self.version,
                "observation_version": self.observation_version,
                "status": self.status,
                "identity_state": self.identity_state,
                "control_calls": self.control_calls,
                "settlement_calls": self.settlement_calls,
                "last_validation_feedback": copy.deepcopy(self.feedback),
                "assessment": copy.deepcopy(self.assessment),
                "previous_assessment": copy.deepcopy(self.previous_assessment),
                "terminal_record": copy.deepcopy(self.terminal_record),
                "preferred_views": list(self.preferred_views),
                "archive_images": copy.deepcopy(self.archive_images),
                "tool_counts": dict(self.tool_counts),
                "resources": {"verification_limit": self.query_limit, "used": self.used, "reserved": self.reserved},
                "visual": copy.deepcopy(self.visual),
                "out_of_archive_ready": self.out_of_archive_ready,
                "views": views,
                "evidence": copy.deepcopy(self.evidence[-max(1, evidence_limit):]),
                "claims": copy.deepcopy(self.claims[-80:]),
                "candidate_assessments": copy.deepcopy(self.candidate_assessments),
                "gaps": copy.deepcopy(self.gaps),
                "revisions": copy.deepcopy(self.revisions[-40:]),
                "actions": copy.deepcopy(self.actions[-20:]),
                "belief_proposals": copy.deepcopy(self.belief_proposals[-20:]),
                "tracklet_history": list(self.tracklet_history),
                "last_result": copy.deepcopy(self.last_result),
            }


class EntityEpisodeStore:
    def __init__(self, session_id: str = "session"):
        self.session_id = str(session_id or "session")
        self._episodes: dict[str, EntityEpisode] = {}
        self._lock = threading.RLock()

    def get_or_create(self, entity_id: str, query_limit: int = 3) -> EntityEpisode:
        entity_id = str(entity_id).strip()
        if not entity_id:
            raise ValueError("entity_id is required")
        with self._lock:
            if entity_id not in self._episodes:
                self._episodes[entity_id] = EntityEpisode(
                    session_id=self.session_id,
                    entity_id=entity_id,
                    episode_id=f"{entity_id}:{uuid.uuid4().hex[:8]}",
                    query_limit=max(0, int(query_limit)),
                )
            return self._episodes[entity_id]

    def get(self, entity_id: str) -> EntityEpisode | None:
        with self._lock:
            return self._episodes.get(str(entity_id))

    def snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            return [episode.snapshot() for episode in self._episodes.values()]
