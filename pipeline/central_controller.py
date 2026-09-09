"""Belief-driven central VLM controller for online entity episodes."""
from __future__ import annotations

import base64
import copy
import logging
import json
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
import threading
import time
import uuid
from typing import Any, Callable

from pipeline.agent_state import EntityEpisode, EntityEpisodeStore
from pipeline.controller_schema import ControllerDecision, ControllerProtocolError, parse_controller_output, validate_tool_args
from pipeline.agent_diagnostics import AgentDiagnostics
from pipeline.identity_tools import IdentityToolService, tool_result_to_agent_result

logger = logging.getLogger(__name__)


TOOL_SPECS = [
    {"name": "SearchArchive", "args": {"view_id": "string", "top_k": "integer"}, "description": "Search the enrolled visual archive using one supplied view."},
    {"name": "ReadArchive", "args": {"candidate_ids": "string[]", "fields": "string[]"}, "description": "Read stored descriptions for selected archive candidates."},
    {"name": "ReadHistory", "args": {"view_ids": "string[]"}, "description": "Read already observed views and evidence for this entity."},
    {"name": "VerifyEvidence", "args": {"view_id": "string", "fields": "string[]", "question": "string"}, "description": "Extract requested semantic identity evidence from one supplied view."},
]


class CentralVLMController:
    def __init__(
        self,
        database: Any,
        visual_index: Any = None,
        visual_config: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        image_encoder: Callable[[Any], str] | None = None,
        controller_infer: Callable[[dict[str, Any], list[str]], dict[str, Any]] | None = None,
        verify_fn: Callable[[Any, str, list[str]], dict[str, Any]] | None = None,
        on_agent_result: Callable[[str, int, Any], None] | None = None,
        on_verify_result: Callable[[str, int, Any], None] | None = None,
        on_visual_result: Callable[[str, int, Any], None] | None = None,
        on_finish: Callable[[str, int, int, str, str, list[str], str], dict[str, Any] | None] | None = None,
        on_log: Callable[[str, dict[str, Any]], None] | None = None,
        session_id: str = "session",
    ):
        cfg = dict(config or {})
        self.query_limit = max(0, int(cfg.get("query_limit", 3)))
        self.min_ooa_observations = max(1, int(cfg.get("min_ooa_observations", 2)))
        self.min_ooa_score = float(cfg.get("min_ooa_score", 0.50))
        self.max_steps = max(1, int(cfg.get("max_control_steps", 12)))
        self.max_context_views = max(1, int(cfg.get("max_context_views", 4)))
        self.max_context_evidence = max(1, int(cfg.get("max_context_evidence", 20)))
        self.max_tokens = max(128, int(cfg.get("max_tokens", 1536)))
        self.charge_failed_queries = bool(cfg.get("charge_failed_queries", True))
        self.terminal_policy = str(cfg.get("terminal_policy", "multimodal_evidence")).strip().lower()
        if self.terminal_policy not in {"legacy_visual_only", "multimodal_evidence"}:
            raise ValueError("terminal_policy must be legacy_visual_only or multimodal_evidence")
        self.max_episode_steps = max(1, int(cfg.get("max_episode_steps", 48)))
        self.settlement_limit = max(1, int(cfg.get("settlement_steps", 3)))
        self.max_memory_views = max(1, int(cfg.get("max_memory_views", 64)))
        self.max_context_chars = max(8000, int(cfg.get("max_context_chars", 32000)))
        self._diagnostics = AgentDiagnostics(cfg)
        self.max_workers = max(1, int(cfg.get("max_workers", 4)))
        self.session_id = str(session_id or "session")
        self.store = EntityEpisodeStore(self.session_id)
        self._active: set[str] = set()
        self._lock = threading.RLock()
        self._futures: dict[str, Future] = {}
        self._job_tokens: dict[str, str] = {}
        self._idle = threading.Condition(self._lock)
        self._pending_observations: dict[str, tuple[int, int, Any, Any, bool]] = {}
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="central-agent")
        self._executor_shutdown = False
        self._request_counter = 0
        self._controller_infer = controller_infer
        self._on_agent_result = on_agent_result
        self._on_verify_result = on_verify_result
        self._on_visual_result = on_visual_result
        self._on_finish = on_finish
        self._on_log = on_log
        self._tools = IdentityToolService(
            database,
            visual_index=visual_index,
            visual_config=visual_config,
            image_encoder=image_encoder,
            verify_fn=verify_fn,
        )

    def reset_session(self, session_id: str) -> None:
        self.wait_for_idle()
        with self._lock:
            if self._executor_shutdown:
                self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="central-agent")
                self._executor_shutdown = False
        with self._lock:
            self.session_id = str(session_id or "session")
            self.store = EntityEpisodeStore(self.session_id)
            self._active.clear()
            self._pending_observations.clear()
            self._tools.reset()

    def _next_request_id(self, entity_id: str) -> str:
        with self._lock:
            self._request_counter += 1
            return f"{self.session_id}:{entity_id}:r{self._request_counter:06d}:{uuid.uuid4().hex[:6]}"

    def _log(self, event_type: str, entity_id: str, frame_id: int, **payload: Any) -> None:
        event = {"event": event_type, "session_id": self.session_id, "entity_id": entity_id, "frame_id": int(frame_id), "timestamp": time.time(), **payload}
        self._diagnostics.write(event)
        if self._on_log:
            self._on_log(event_type, event)
        else:
            logger.info("central_vlm %s: %s", event_type, event)

    def observe(self, entity_id: str, track_id: int, frame_id: int, crop: Any, info: Any = None, eof: bool = False) -> dict[str, Any]:
        """Append an observation and run one serial decision chain if idle."""
        episode = self.store.get_or_create(entity_id, self.query_limit)
        episode.max_views = self.max_memory_views
        view_id = f"{entity_id}:f{int(frame_id)}"
        is_new_observation = episode.add_observation(view_id, frame_id, crop, {
            "track_id": int(track_id),
            "bbox": list(getattr(info, "bbox", ()) or ()),
            "quality": float(getattr(info, "observation_quality", 0.0)),
            "identity_state": str(getattr(info, "identity_state", "unknown")),
        })
        if episode.status != "active":
            self._log("observation_after_close", entity_id, frame_id, episode_id=episode.episode_id)
            return episode.snapshot()
        if not is_new_observation and not eof:
            self._log("observation_unchanged", entity_id, frame_id, episode_id=episode.episode_id, observation_version=episode.observation_version)
            return episode.snapshot()
        with self._lock:
            if entity_id in self._active:
                self._log("observation_cached_while_busy", entity_id, frame_id, episode_id=episode.episode_id)
                return episode.snapshot()
            self._active.add(entity_id)
        try:
            return self._run_chain(episode, track_id, frame_id, info, eof=eof)
        finally:
            with self._idle:
                self._active.discard(entity_id)
                self._schedule_pending_locked()
                self._idle.notify_all()

    def submit_observation(self, entity_id: str, track_id: int, frame_id: int, crop: Any, info: Any = None, eof: bool = False) -> dict[str, Any]:
        episode = self.store.get_or_create(entity_id, self.query_limit)
        episode.max_views = self.max_memory_views
        is_new = episode.add_observation(f"{entity_id}:f{int(frame_id)}", frame_id, crop, {
            "track_id": int(track_id), "bbox": list(getattr(info, "bbox", ()) or []),
            "quality": float(getattr(info, "observation_quality", 0.0)),
        })
        if episode.status != "active" or (not is_new and not eof):
            return episode.snapshot()
        with self._lock:
            self._pending_observations[entity_id] = (track_id, frame_id, info, None, eof)
            self._schedule_pending_locked()
        return episode.snapshot()

    def _schedule_pending_locked(self) -> None:
        if self._executor_shutdown:
            return
        for entity_id in list(self._pending_observations):
            if len(self._active) >= self.max_workers:
                break
            if entity_id in self._active:
                continue
            track_id, frame_id, info, _, eof = self._pending_observations.pop(entity_id)
            episode = self.store.get(entity_id)
            if episode is None or episode.status != "active":
                continue
            token = uuid.uuid4().hex
            self._active.add(entity_id)
            self._job_tokens[entity_id] = token
            self._futures[entity_id] = self._executor.submit(self._background_chain, entity_id, track_id, frame_id, info, eof, token)

    def _background_chain(self, entity_id: str, track_id: int, frame_id: int, info: Any, eof: bool, token: str = "") -> None:
        episode = self.store.get(entity_id)
        try:
            if episode is not None:
                self._run_chain(episode, track_id, frame_id, info, eof=eof)
        except Exception as exc:
            logger.exception("central agent background chain failed for %s", entity_id)
            if episode is not None:
                self._record_protocol_error(episode, frame_id, f"background_error: {exc}")
                self._complete(episode, track_id, frame_id, "review", "", [], "background execution failure", system_fallback=True)
        finally:
            with self._idle:
                if self._job_tokens.get(entity_id) == token:
                    self._active.discard(entity_id)
                    self._futures.pop(entity_id, None)
                    self._job_tokens.pop(entity_id, None)
                    if episode is None or episode.status != "active" or episode.observation_version <= episode.last_consumed_observation:
                        pending = self._pending_observations.get(entity_id)
                        if pending is None or not pending[-1]:
                            self._pending_observations.pop(entity_id, None)
                    self._schedule_pending_locked()
                    self._idle.notify_all()

    def wait_for_idle(self, timeout: float | None = None) -> None:
        with self._idle:
            if not self._idle.wait_for(lambda: not self._active and not self._pending_observations, timeout):
                raise TimeoutError("central tasks did not settle before timeout")

    def shutdown(self, wait_for_tasks: bool = True) -> None:
        if self._executor_shutdown:
            return
        if wait_for_tasks:
            self.wait_for_idle()
        self._executor.shutdown(wait=wait_for_tasks)
        self._executor_shutdown = True

    def _default_controller_infer(self, context: dict[str, Any], images: list[str]) -> dict[str, Any]:
        from tools import _controller_infer

        return _controller_infer(context, images)

    def _prompt(self, snapshot: dict[str, Any], eof: bool) -> str:
        template = (Path(__file__).resolve().parent / "prompts" / "central_controller.txt").read_text(encoding="utf-8")
        return template.replace("{{tools}}", json.dumps(self._tools.capabilities(), ensure_ascii=False)).replace("{{context}}", json.dumps(snapshot, ensure_ascii=False, default=str)).replace("{{stream_ended}}", str(bool(eof)).lower())

    def _context(self, episode: EntityEpisode, info: Any, eof: bool) -> tuple[dict[str, Any], list[str]]:
        # One consistent snapshot; image objects are retained by reference during encoding.
        with episode.lock:
            snapshot = episode.snapshot(include_images=True)
            # Preserve these crops until the response has been applied. Producers
            # may keep accepting new frames while image encoding/inference runs.
            episode.pinned_views = {v["view_id"] for v in snapshot["views"]}
        all_views = snapshot.pop("views")
        by_id = {v["view_id"]: v for v in all_views}
        last = snapshot.get("last_result", {})
        priority = list(snapshot.pop("preferred_views", [])) if last.get("tool") == "ReadHistory" else []
        candidates = [("observation", by_id[v]) for v in priority if v in by_id]
        if all_views:
            candidates.append(("observation", all_views[-1]))
        candidates.extend(("archive", v) for v in snapshot.pop("archive_images", []))
        candidates.extend(("observation", v) for v in reversed(all_views))
        images, manifest, seen, errors = [], [], set(), []
        for role, view in candidates:
            image_id = str(view.get("image_id") or view.get("view_id"))
            if image_id in seen:
                continue
            seen.add(image_id)
            if len(images) >= self.max_context_views:
                continue
            try:
                image = view.get("image") if role == "observation" else self._tools.reference_image(image_id)
                if image is None or self._tools.image_encoder is None:
                    raise ValueError("image unavailable or encoder not configured")
                encoded = self._tools.image_encoder(image)
                item = {k: v for k, v in view.items() if k != "image"}
                item.update({"image_id": image_id, "role": role, "image_index": len(images), **self._diagnostics.image(encoded)})
                images.append(encoded)
                manifest.append(item)
            except Exception as exc:
                errors.append({"image_id": image_id, "error": str(exc)})
        snapshot["views"] = [{k:v for k,v in view.items() if k != "image"} for view in all_views]
        snapshot["current_view_id"] = all_views[-1]["view_id"] if all_views else ""
        snapshot["image_manifest"] = manifest
        snapshot["image_view_ids"] = [m["image_id"] for m in manifest]
        snapshot["context_view_ids"] = [m["image_id"] for m in manifest if m["role"] == "observation"]
        snapshot["image_errors"] = errors
        evidence = snapshot["evidence"]
        searches = [e for e in evidence if e.get("tool") == "SearchArchive" and e.get("status") == "success" and e.get("validity_status", "active") not in {"retracted", "superseded"}]
        snapshot["evidence_count"] = len(evidence)
        snapshot["successful_search_count"] = len(searches)
        snapshot["active_evidence_count"] = sum(e.get("status") == "success" and e.get("validity_status", "active") == "active" for e in evidence)
        snapshot["semantic_claim_count"] = sum(c.get("source_role") == "VerifyEvidence" for c in snapshot["claims"])
        snapshot["successful_searches"] = [{"evidence_id": e["evidence_id"], "view_id": e.get("payload", {}).get("view_id"), "matches": e.get("payload", {}).get("matches", [])} for e in searches][-self.max_context_evidence:]
        snapshot["evidence"] = evidence[-self.max_context_evidence:]
        snapshot["resources"].update({"remaining": max(0, episode.query_limit - snapshot["resources"]["used"] - snapshot["resources"]["reserved"]), "control_limit": self.max_episode_steps, "control_used": snapshot["control_calls"], "settlement_limit": self.settlement_limit, "settlement_used": snapshot["settlement_calls"]})
        snapshot["stream_ended"] = bool(eof)
        snapshot["terminal_policy"] = self.terminal_policy
        snapshot["settlement_only"] = snapshot["control_calls"] >= self.max_episode_steps
        if self.terminal_policy == "multimodal_evidence":
            snapshot.pop("out_of_archive_ready", None)
        snapshot["decision_target"] = "initial_search" if not searches else "select_next_evidence_or_terminal_action"
        snapshot["context_truncation"] = {"total_evidence_count": len(evidence), "included_evidence_count": len(snapshot["evidence"]), "omitted_image_ids": sorted(seen - {m["image_id"] for m in manifest}), "max_context_chars": self.max_context_chars}
        # Never silently truncate strings/JSON. Drop whole optional records and disclose omissions.
        removed = {}
        for key in ("belief_proposals", "revisions", "actions", "views", "evidence", "claims", "successful_searches", "candidate_assessments", "gaps"):
            while len(json.dumps(snapshot, ensure_ascii=False, default=str)) > self.max_context_chars and snapshot.get(key):
                snapshot[key].pop(0)
                removed[key] = removed.get(key, 0) + 1
        if len(json.dumps(snapshot, ensure_ascii=False, default=str)) > self.max_context_chars:
            snapshot["last_result"] = {k:v for k,v in last.items() if k != "payload"}
            removed["last_result_payload"] = 1
        snapshot["context_truncation"].update({"dropped_records": removed, "included_evidence_count": len(snapshot["evidence"])})
        return {"prompt": self._prompt(snapshot, eof), "snapshot": snapshot, "max_tokens": self.max_tokens}, images

    def _valid_view(self, episode: EntityEpisode, args: dict[str, Any], key: str = "view_id") -> bool:
        view_id = str(args.get(key, "")).strip()
        return bool(view_id and view_id in episode.views)

    def _valid_references(self, episode: EntityEpisode, refs: tuple[str, ...]) -> bool:
        if not refs:
            return True
        active_evidence = [item for item in episode.evidence if item.get("validity_status", "active") in {"active", "disputed"}]
        known = {ref for item in active_evidence for ref in item.get("evidence_refs", []) or []}
        known.update(str(item.get("request_id", "")) for item in active_evidence)
        known.update(str(item.get("evidence_id", "")) for item in active_evidence)
        known.update(
            str(item.get("claim_id", ""))
            for item in episode.claims
            if item.get("validity_status", "active") in {"active", "disputed"}
        )
        return all(ref in known for ref in refs)

    @staticmethod
    def _has_active_evidence(episode: EntityEpisode, refs: list[str]) -> bool:
        active = [item for item in episode.evidence if item.get("validity_status", "active") in {"active", "disputed"} and item.get("status") == "success"]
        if not active:
            return False
        if not refs:
            return True
        active_claims = {
            str(claim.get("claim_id")): str(claim.get("evidence_id"))
            for claim in episode.claims
            if claim.get("validity_status", "active") in {"active", "disputed"}
        }
        return all(
            any(
                ref == item.get("evidence_id")
                or ref == item.get("request_id")
                or ref in (item.get("evidence_refs") or [])
                or active_claims.get(ref) == item.get("evidence_id")
                for item in active
            )
            for ref in refs
        )

    def _archive_ids(self, episode: EntityEpisode) -> set[str]:
        values: set[str] = set()
        for evidence in episode.evidence:
            for match in (evidence.get("payload", {}) or {}).get("matches", []) or []:
                archive_id = str(match.get("hull_number", "")).strip()
                if archive_id:
                    values.add(archive_id)
        for item in episode.candidate_assessments:
            archive_id = str(item.get("archive_id", "")).strip()
            if archive_id:
                values.add(archive_id)
        records = getattr(self._tools.database, "_data", None)
        if isinstance(records, dict):
            values.update(str(key).strip() for key in records if str(key).strip())
        elif isinstance(records, list):
            for record in records:
                if isinstance(record, dict):
                    archive_id = str(record.get("hull_number") or record.get("archive_id") or "").strip()
                    if archive_id:
                        values.add(archive_id)
        return values

    def _apply_belief_patch(self, episode: EntityEpisode, patch: tuple[dict[str, Any], ...], frame_id: int) -> bool:
        if not patch:
            return True
        before = {
            "claims": len(episode.claims),
            "candidates": len(episode.candidate_assessments),
            "gaps": len(episode.gaps),
        }
        evidence_ids = {
            str(item.get("evidence_id", ""))
            for item in episode.evidence
            if item.get("validity_status", "active") in {"active", "disputed"}
        }
        claim_ids = {
            str(item.get("claim_id", ""))
            for item in episode.claims
            if item.get("validity_status", "active") in {"active", "disputed"}
        }
        all_evidence_ids = {str(item.get("evidence_id", "")) for item in episode.evidence}
        rejected = episode.apply_belief_patch(
            patch,
            evidence_ids,
            claim_ids,
            self._archive_ids(episode),
            set(episode.views),
            all_evidence_ids,
        )
        for message in rejected:
            self._record_protocol_error(episode, frame_id, f"belief patch rejected: {message}")
        episode.recompute_visual(self.min_ooa_score, self.min_ooa_observations)
        after = {
            "claims": len(episode.claims),
            "candidates": len(episode.candidate_assessments),
            "gaps": len(episode.gaps),
        }
        changes = {
            key: f"{before[key]}->{after[key]}"
            for key in before
            if before[key] != after[key]
        }
        self._log(
            "belief_update",
            episode.entity_id,
            frame_id,
            episode_id=episode.episode_id,
            changes=changes,
            accepted_patch_count=0 if rejected else len(patch),
            rejected_patch_count=len(rejected),
            state_version=episode.version,
        )
        return not rejected

    def _claims_from_result(self, episode: EntityEpisode, tool: str, result: dict[str, Any]) -> list[dict[str, Any]]:
        if result.get("status") != "success":
            return []
        evidence_id = f"{result.get('request_id', '')}:evidence"
        payload = result.get("payload", {}) or {}
        claims: list[dict[str, Any]] = []
        for raw in payload.get("claims", []) or []:
            if not isinstance(raw, dict):
                continue
            claim = dict(raw)
            if any(
                not str(claim.get(field, "")).strip()
                for field in ("attribute", "visibility", "source_role")
            ) or "observed_value" not in claim:
                continue
            claim.update({"claim_id": f"{evidence_id}:c{len(claims) + 1}", "evidence_id": evidence_id,
                          "view_ids": [payload["view_id"]] if payload.get("view_id") else [],
                          "subject_entity_id": episode.entity_id, "source_role": tool, "validity_status": "active"})
            claims.append(claim)
        if tool == "VerifyEvidence" and not claims:
            for field, value in dict(payload.get("observed", {}) or {}).items():
                claims.append({
                    "claim_id": f"{evidence_id}:{field}", "evidence_id": evidence_id,
                    "view_ids": [str(payload.get("view_id", ""))], "subject_entity_id": episode.entity_id,
                    "attribute": str(field), "observed_value": value,
                    "visibility": payload.get("visibility", "unknown"), "source_role": "VerifyEvidence",
                    "created_at": time.time(), "validity_status": "active",
                })
        if tool == "SearchArchive" and not claims:
            view_id = str(payload.get("view_id", ""))
            for index, match in enumerate(payload.get("matches", []) or []):
                archive_id = str(match.get("hull_number", "")).strip()
                if not archive_id:
                    continue
                claims.append({
                    "claim_id": f"{evidence_id}:candidate:{index}", "evidence_id": evidence_id,
                    "view_ids": [view_id] if view_id else [], "subject_entity_id": episode.entity_id,
                    "attribute": "visual_archive_match", "observed_value": archive_id,
                    "score": match.get("score"), "visibility": "visible_view",
                    "source_role": "SearchArchive", "created_at": time.time(), "validity_status": "active",
                })
        return claims

    def _tool_cost(self, tool: str) -> int:
        return 1 if tool in {"SearchArchive", "VerifyEvidence"} else 0

    @staticmethod
    def _successful_duplicate_search(episode: EntityEpisode, decision: ControllerDecision) -> bool:
        """Prevent charging the same successful archive search repeatedly."""
        if decision.tool != "SearchArchive":
            return False
        requested_view = str(decision.args.get("view_id", "")).strip()
        requested_top_k = int(decision.args.get("top_k", 3) or 3)
        proposed = {
            str(action.get("request_id", "")): action
            for action in episode.actions
            if action.get("phase") == "proposed" and action.get("tool") == "SearchArchive"
        }
        for evidence in episode.evidence:
            if evidence.get("tool") != "SearchArchive" or evidence.get("status") != "success":
                continue
            action = proposed.get(str(evidence.get("request_id", "")))
            if not action:
                continue
            args = action.get("args", {}) or {}
            if str(args.get("view_id", "")).strip() == requested_view and int(args.get("top_k", 3) or 3) == requested_top_k:
                return True
        return False

    def _dispatch(self, episode: EntityEpisode, track_id: int, frame_id: int, decision: ControllerDecision) -> bool:
        try:
            validate_tool_args(decision.tool, decision.args)
        except ControllerProtocolError as exc:
            self._record_protocol_error(episode, frame_id, str(exc))
            return False
        capability = next(c for c in self._tools.capabilities() if c["name"] == decision.tool)
        if not capability["available"]:
            self._record_protocol_error(episode, frame_id, f"{decision.tool} unavailable: {capability['unavailable_reason']}")
            return False
        if decision.tool in {"SearchArchive", "VerifyEvidence"} and not self._valid_view(episode, decision.args):
            self._record_protocol_error(episode, frame_id, "tool requires an existing view_id")
            return False
        if self._successful_duplicate_search(episode, decision):
            self._record_protocol_error(
                episode,
                frame_id,
                "SearchArchive already succeeded for this view and top_k; use the returned matches, another tool, or a newer view",
            )
            return False
        if decision.tool == "ReadHistory":
            requested = decision.args.get("view_ids")
            if requested is not None and (not isinstance(requested, list) or any(str(item) not in episode.views for item in requested)):
                self._record_protocol_error(episode, frame_id, "ReadHistory references an unavailable view")
                return False
        cost = self._tool_cost(decision.tool)
        before = {
            "evidence": len(episode.evidence),
            "claims": len(episode.claims),
            "candidates": len(episode.candidate_assessments),
            "gaps": len(episode.gaps),
            "used": episode.used,
            "reserved": episode.reserved,
        }
        if cost and not episode.reserve_query():
            self._log("budget_blocked", episode.entity_id, frame_id, episode_id=episode.episode_id, tool=decision.tool)
            self._record_protocol_error(episode, frame_id, "Tool budget exhausted. Use available free tools or finish using existing evidence; depletion is not OOA.")
            return False
        request_id = self._next_request_id(episode.entity_id)
        with episode.lock:
            request_snapshot = episode.snapshot(include_images=True)
        request = {
            "request_id": request_id,
            "session_id": episode.session_id,
            "entity_id": episode.entity_id,
            "episode_id": episode.episode_id,
            "state_version": episode.version,
            "view_id": str(decision.args.get("view_id", "")),
            "views": {view["view_id"]: view.get("image") for view in request_snapshot["views"]},
            "history": [e for e in request_snapshot["evidence"] if e.get("tool") != "ReadHistory"],
            "view_records": [{key: value for key, value in view.items() if key != "image"} for view in request_snapshot["views"]],
        }
        episode.append_action({"request_id": request_id, "phase": "proposed", "tool": decision.tool, "args": copy.deepcopy(decision.args), "state_version": request["state_version"], "cost": cost})
        self._log("tool_request", episode.entity_id, frame_id, request_id=request_id, episode_id=episode.episode_id, tool=decision.tool, args=decision.args, state_version=request["state_version"])
        try:
            result = self._tools.execute(request_id, decision.tool, decision.args, request)
        except Exception as exc:
            result = {"request_id": request_id, "status": "error", "payload": {"message": str(exc)}, "evidence_refs": [], "service_time": time.time(), "error_code": "dispatch_exception"}
        if cost:
            episode.settle_query(success=result.get("status") == "success" or self.charge_failed_queries)
        # Results retain owning request metadata; never accept model-supplied ownership.
        result_item = {**result, "request_id": request_id, "tool": decision.tool, "state_version": request["state_version"],
                       "session_id": episode.session_id, "entity_id": episode.entity_id, "episode_id": episode.episode_id,
                       "view_ids": [request["view_id"]] if request["view_id"] else []}
        if decision.tool == "ReadHistory":
            # Store references rather than recursively copying prior tool results.
            result_item = copy.deepcopy(result_item)
            result_item["payload"]["history_refs"] = [e["evidence_id"] for e in result_item["payload"].pop("history", [])]
        episode.append_evidence(result_item)
        with episode.lock:
            episode.tool_counts[decision.tool] = episode.tool_counts.get(decision.tool, 0) + 1
            episode.feedback = {}
            if decision.tool == "ReadHistory" and result.get("status") == "success":
                episode.preferred_views = list(result.get("payload", {}).get("selected_view_ids", []))
            if decision.tool == "ReadArchive" and result.get("status") == "success":
                records = result.get("payload", {}).get("records", [])
                episode.archive_images = [r for record in records for r in record.get("reference_views", [])]
                episode.read_archive_refs.update(r["archive_ref"] for r in records)
        episode.add_claims(self._claims_from_result(episode, decision.tool, result))
        episode.append_action({"request_id": request_id, "phase": "returned", "tool": decision.tool, "status": result.get("status"), "error_code": result.get("error_code", ""), "cost": cost})
        payload = result.get("payload", {}) or {}
        if decision.tool == "SearchArchive" and result.get("status") == "success":
            episode.update_visual(payload, self.min_ooa_score, self.min_ooa_observations)
        adapted = tool_result_to_agent_result(decision.tool, result)
        if adapted is not None:
            if decision.tool == "SearchArchive":
                callback = self._on_visual_result
            else:
                callback = self._on_verify_result if decision.tool == "VerifyEvidence" else self._on_agent_result
            if callback:
                adapted.entity_id = episode.entity_id
                adapted.episode_id = episode.episode_id
                adapted.session_id = episode.session_id
                callback(track_id, frame_id, adapted)
        after = {
            "evidence": len(episode.evidence),
            "claims": len(episode.claims),
            "candidates": len(episode.candidate_assessments),
            "gaps": len(episode.gaps),
            "used": episode.used,
            "reserved": episode.reserved,
        }
        state_changes = {
            key: f"{before[key]}->{after[key]}"
            for key in before
            if before[key] != after[key]
        }
        self._log(
            "tool_result",
            episode.entity_id,
            frame_id,
            request_id=request_id,
            episode_id=episode.episode_id,
            tool=decision.tool,
            status=result.get("status"),
            evidence_refs=result.get("evidence_refs", []),
            error_code=result.get("error_code", ""),
            used=episode.used,
            reserved=episode.reserved,
            query_limit=self.query_limit,
            state_version_after=episode.version,
            state_changes=state_changes,
            payload_summary=self._tool_payload_summary(decision.tool, result),
        )
        return True

    @staticmethod
    def _tool_payload_summary(tool: str, result: dict[str, Any]) -> dict[str, Any]:
        """Return a small terminal summary without dumping images or payloads."""
        payload = result.get("payload", {}) or {}
        summary: dict[str, Any] = {}
        if tool == "SearchArchive":
            matches = list(payload.get("matches", []) or [])
            summary["matches"] = [
                {
                    "archive_id": item.get("hull_number", item.get("archive_id", "")),
                    "score": item.get("score"),
                }
                for item in matches[:3]
                if isinstance(item, dict)
            ]
        elif tool == "ReadArchive":
            summary["records"] = len(payload.get("records", []) or [])
            summary["requested_fields"] = list(payload.get("requested_fields", []) or [])
        elif tool == "ReadHistory":
            summary["views"] = len(payload.get("views", []) or [])
            summary["history_items"] = len(payload.get("history", []) or [])
        elif tool == "VerifyEvidence":
            summary["observed"] = dict(payload.get("observed", {}) or {})
            summary["visibility"] = payload.get("visibility", "unknown")
            summary["claims"] = len(payload.get("claims", []) or [])
        if result.get("status") != "success" and result.get("error_code"):
            summary["error"] = result.get("error_code")
            message = str(payload.get("message", "") or "").strip()
            if message:
                summary["error_message"] = message[:360]
        return summary

    def _record_protocol_error(self, episode: EntityEpisode, frame_id: int, message: str) -> None:
        with episode.lock:
            episode.feedback = {"message": message, "state_version": episode.version}
        episode.append_action({"phase": "protocol_error", "message": message})
        self._log("protocol_error", episode.entity_id, frame_id, episode_id=episode.episode_id, error=message)

    def _complete(
        self,
        episode: EntityEpisode,
        track_id: int,
        frame_id: int,
        decision: str,
        archive_id: str,
        refs: list[str],
        reason: str,
        system_fallback: bool = False,
    ) -> bool:
        """Commit a terminal proposal after structural validation only.

        In multimodal mode, evidence sufficiency and the identity judgment belong
        to the controller. The callback may still reject an invalid archive ID,
        stale track, or an uncommittable state; that rejection is fed back to the
        controller instead of being silently converted into another identity.
        """
        if decision in {"known", "out_of_archive"} and not system_fallback and not refs:
            message = f"{decision} finish requires explicit evidence_refs"
            episode.append_action({"phase": "validation_rejected", "decision": decision, "reason": message})
            self._log("finish_rejected", episode.entity_id, frame_id, episode_id=episode.episode_id, decision=decision, reason=message)
            return False
        if decision in {"known", "out_of_archive"} and not system_fallback:
            if not self._has_active_evidence(episode, refs):
                message = f"{decision} finish has no active successful evidence"
                episode.append_action({"phase": "validation_rejected", "decision": decision, "reason": message})
                self._log("finish_rejected", episode.entity_id, frame_id, episode_id=episode.episode_id, decision=decision, reason=message)
                return False
        if (
            self.terminal_policy == "legacy_visual_only"
            and decision == "out_of_archive"
            and not system_fallback
            and not episode.out_of_archive_ready
        ):
            message = "legacy visual-only OOA guard not satisfied"
            episode.append_action({"phase": "validation_rejected", "decision": decision, "reason": message})
            self._log("finish_rejected", episode.entity_id, frame_id, episode_id=episode.episode_id, decision=decision, reason=message)
            return False
        if decision == "known" and not system_fallback:
            if not archive_id or self._tools.database.lookup(archive_id) is None:
                message = "archive identity is not present in the enrolled archive"
                episode.append_action({"phase": "validation_rejected", "decision": decision, "reason": message})
                self._log("finish_rejected", episode.entity_id, frame_id, episode_id=episode.episode_id, decision=decision, reason=message)
                return False
            if not self._has_active_evidence(episode, refs):
                message = "known finish has no active successful evidence"
                episode.append_action({"phase": "validation_rejected", "decision": decision, "reason": message})
                self._log("finish_rejected", episode.entity_id, frame_id, episode_id=episode.episode_id, decision=decision, reason=message)
                return False

        callback_result = None
        if self._on_finish:
            callback_result = self._on_finish(
                episode.entity_id,
                track_id,
                frame_id,
                decision,
                archive_id if decision == "known" else "",
                refs,
                reason,
            )
            if callback_result and callback_result.get("status") == "rejected":
                rejection = callback_result.get("reason", "terminal proposal rejected")
                episode.append_action({"phase": "validation_rejected", "decision": decision, "reason": rejection})
                self._log("finish_rejected", episode.entity_id, frame_id, episode_id=episode.episode_id, decision=decision, reason=rejection)
                self._record_protocol_error(episode, frame_id, f"terminal commit rejected: {rejection}")
                if system_fallback:
                    episode.terminal_record = {"decision": "failed", "proposed_decision": decision, "commit_status": "failed", "reason": rejection, "system_fallback": True}
                    episode.close("failed")
                return False
        episode.terminal_record = {"decision": decision, "archive_id": archive_id if decision == "known" else "",
                                   "evidence_refs": list(refs), "reason": reason, "system_fallback": system_fallback,
                                   "decision_source": "system_fallback" if system_fallback else "vlm",
                                   "commit_status": "committed", "committed_at": time.time(),
                                   "decision_snapshot_version": getattr(episode, "decision_snapshot_version", episode.version)}
        episode.close(decision, archive_id)
        episode.append_action({"phase": "finish", "decision": decision, "archive_id": archive_id if decision == "known" else "", "evidence_refs": list(refs), "reason": reason, "system_fallback": system_fallback, "callback": callback_result or {}})
        self._log("episode_closed", episode.entity_id, frame_id, episode_id=episode.episode_id, decision=decision, archive_id=archive_id if decision == "known" else "", reason=reason, system_fallback=system_fallback, terminal_record=episode.terminal_record, resources=episode.snapshot()["resources"], control_calls=episode.control_calls, tool_counts=dict(episode.tool_counts))
        return True

    def _run_chain(self, episode: EntityEpisode, track_id: int, frame_id: int, info: Any, eof: bool) -> dict[str, Any]:
        for step in range(self.max_steps + self.settlement_limit):
            if episode.status != "active":
                break
            settlement = episode.control_calls >= self.max_episode_steps or step >= self.max_steps
            if settlement and episode.settlement_calls >= self.settlement_limit:
                break
            if settlement:
                self._record_protocol_error(episode, frame_id, "Control budget exhausted: finish using existing evidence; no more tool execution.")
            context, images = self._context(episode, info, eof)
            snapshot = context["snapshot"]
            if settlement:
                snapshot["settlement_only"] = True
                context["prompt"] = self._prompt(snapshot, eof)
            with episode.lock:
                episode.last_consumed_observation = snapshot["observation_version"]
                episode.decision_snapshot_version = snapshot["version"]
                episode.control_calls += 1
                episode.pinned_views = set(snapshot["context_view_ids"])
                if settlement:
                    episode.settlement_calls += 1
            self._log("controller_infer_start", episode.entity_id, frame_id,
                      episode_id=episode.episode_id, state_version=snapshot["version"],
                      control_step=step + 1, current_view_id=snapshot["current_view_id"],
                      observation_count=len(snapshot["views"]), evidence_count=snapshot["evidence_count"],
                      claim_count=len(snapshot["claims"]), candidate_assessment_count=len(snapshot["candidate_assessments"]),
                      visual_candidate_id=snapshot["visual"]["candidate_id"], visual_score=snapshot["visual"]["score"],
                      gap_count=len(snapshot["gaps"]), budget_used=episode.used, budget_reserved=episode.reserved,
                      query_limit=self.query_limit, request_context=context)
            raw = None
            started = time.perf_counter()
            try:
                raw = (self._controller_infer or self._default_controller_infer)(context, images)
                decision = parse_controller_output(raw)
            except Exception as exc:
                with episode.lock:
                    episode.pinned_views.clear()
                self._log("controller_invalid_response", episode.entity_id, frame_id, episode_id=episode.episode_id,
                          control_step=step + 1, error=str(exc), raw_response=raw)
                self._record_protocol_error(episode, frame_id, str(exc))
                continue
            self._log("controller_call", episode.entity_id, frame_id, episode_id=episode.episode_id,
                      state_version=snapshot["version"], control_step=step + 1, decision=raw,
                      elapsed_ms=round((time.perf_counter()-started)*1000, 3), settlement=settlement)
            episode.append_action({"phase": "controller", "kind": decision.kind, "tool": decision.tool,
                                   "args": copy.deepcopy(decision.args), "decision": decision.decision,
                                   "decision_target": decision.decision_target, "operation_purpose": decision.operation_purpose,
                                   "state_version": snapshot["version"], "reason": decision.reason})
            if decision.belief_patch:
                episode.append_belief_proposal(decision.belief_patch)
                if not self._apply_belief_patch(episode, decision.belief_patch, frame_id):
                    continue
            if not self._valid_references(episode, decision.evidence_refs):
                self._record_protocol_error(episode, frame_id, "controller references evidence outside this episode or withdrawn evidence")
                continue
            if decision.assessment:
                refs = tuple(decision.assessment.get("basis_refs", []))
                candidate = decision.assessment.get("leading_candidate", "")
                gaps = {g["gap_id"] for g in episode.gaps}
                if not self._valid_references(episode, refs) or (candidate and self._tools.database.lookup(candidate) is None) or any(g not in gaps for g in decision.assessment.get("blocking_gap_ids", [])):
                    self._record_protocol_error(episode, frame_id, "assessment has unavailable references")
                    continue
                with episode.lock:
                    episode.previous_assessment = copy.deepcopy(episode.assessment)
                    episode.assessment = copy.deepcopy(decision.assessment)
            if decision.kind == "finish":
                if self._complete(episode, track_id, frame_id, decision.decision, decision.archive_id, list(decision.evidence_refs), decision.reason):
                    return episode.snapshot()
                continue
            if settlement:
                self._record_protocol_error(episode, frame_id, "Only finish is executable in final settlement.")
                continue
            if decision.kind == "tool":
                self._dispatch(episode, track_id, frame_id, decision)
                with episode.lock:
                    episode.pinned_views.clear()
                continue
            if eof or episode.used >= episode.query_limit:
                self._record_protocol_error(episode, frame_id, "Cannot wait for more evidence at EOF or exhausted tool budget. Use available free tools or finish.")
                continue
            self._log("continue_wait", episode.entity_id, frame_id, episode_id=episode.episode_id,
                      observation_version=episode.last_consumed_observation)
            with episode.lock:
                episode.pinned_views.clear()
            return episode.snapshot()
        if episode.status == "active":
            self._complete(episode, track_id, frame_id, "review", "", [], "controller execution/settlement limit reached", system_fallback=True)
        return episode.snapshot()

    def snapshots(self) -> list[dict[str, Any]]:
        return self.store.snapshots()

    def finish_eof(self) -> list[dict[str, Any]]:
        """Give each still-active episode one final settlement decision."""
        results = []
        for snapshot in self.store.snapshots():
            if snapshot.get("status") != "active":
                results.append(snapshot)
                continue
            episode = self.store.get(snapshot["entity_id"])
            if episode is None or not episode.views:
                continue
            view = next(reversed(episode.views.values()))
            self._run_chain(
                episode,
                int(view.get("track_id", 0) or 0),
                int(view.get("frame_id", 0) or 0),
                None,
                eof=True,
            )
            results.append(episode.snapshot())
        return results
