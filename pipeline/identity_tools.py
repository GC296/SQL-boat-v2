"""Identity evidence tools exposed to the central controller.

The service only reads views supplied by the owning episode. It does not let
the controller choose arbitrary files or mutate the archive.
"""
from __future__ import annotations

import time
import copy
import threading
import hashlib
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable

from agent import AgentResult
from pipeline.controller_schema import validate_tool_args, ControllerProtocolError, TOOL_ARGUMENTS


@dataclass(frozen=True)
class ToolSpec:
    name: str
    handler_name: str
    cost: int
    description: str


class IdentityToolService:
    def __init__(
        self,
        database: Any,
        visual_index: Any = None,
        visual_config: dict[str, Any] | None = None,
        image_encoder: Callable[[Any], str] | None = None,
        verify_fn: Callable[[Any, str, list[str]], dict[str, Any]] | None = None,
    ):
        self.database = database
        self.visual_index = visual_index
        self.visual_config = dict(visual_config or {})
        self.image_encoder = image_encoder
        self.verify_fn = verify_fn
        self._request_cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._request_locks = [threading.Lock() for _ in range(64)]
        self._archive_records = self._load_reference_records()
        self._specs = {
            spec.name: spec for spec in (
                ToolSpec("SearchArchive", "_search", 1, "Search the enrolled visual archive using one supplied view."),
                ToolSpec("ReadArchive", "_read_archive", 0, "Read archived hull numbers, descriptions and reference images for selected archive candidates."),
                ToolSpec("ReadHistory", "_read_history", 0, "Read already observed views and evidence for this entity."),
                ToolSpec("VerifyEvidence", "_verify", 1, "Extract requested semantic identity evidence from one supplied view."),
            )
        }

    @property
    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def reset(self) -> None:
        """Called only after in-flight requests have settled."""
        with self._lock:
            self._request_cache.clear()

    def capabilities(self) -> list[dict[str, Any]]:
        unavailable = {"SearchArchive": self.visual_index is None, "VerifyEvidence": self.verify_fn is None}
        return [{"name": s.name, "description": s.description, "cost": s.cost,
                 "available": not unavailable.get(s.name, False),
                 "unavailable_reason": "service_not_configured" if unavailable.get(s.name, False) else "",
                 "arguments": sorted(TOOL_ARGUMENTS[s.name])} for s in self.specs]

    def _load_reference_records(self) -> list[dict[str, Any]]:
        records = getattr(self.visual_index, "records", None)
        if records is not None:
            return [{k: v for k, v in r.items() if k != "data_url"} for r in records]
        root = Path(self.visual_config.get("archive_root", "data/archive/visual_prototypes")).resolve()
        if not root.is_dir():
            return []
        return [{"hull_number": folder.name, "prototype_id": p.stem, "image_path": str(p.resolve())}
                for folder in sorted(root.iterdir()) if folder.is_dir()
                for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

    def reference_image(self, image_id: str) -> Any:
        record = next((r for r in self._archive_records if self._reference_id(r) == image_id), None)
        if record is None:
            return None
        import cv2
        import numpy as np
        try:
            return cv2.imdecode(np.frombuffer(Path(record["image_path"]).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        except (OSError, KeyError):
            return None

    @staticmethod
    def _reference_id(record: dict[str, Any]) -> str:
        return f"archive-image:{record.get('hull_number', record.get('archive_id', ''))}:{record.get('prototype_id', '')}"

    @staticmethod
    def _response(request_id: str, status: str, payload: dict[str, Any] | None = None, evidence_refs: list[str] | None = None, error_code: str = "") -> dict[str, Any]:
        return {
            "request_id": request_id,
            "status": status,
            "payload": payload or {},
            "evidence_refs": list(evidence_refs or []),
            "service_time": time.time(),
            "error_code": error_code,
        }

    @staticmethod
    def _view(request: dict[str, Any], view_id: str) -> Any | None:
        return (request.get("views") or {}).get(view_id)

    def execute(self, request_id: str, tool: str, args: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            cached = self._request_cache.get(str(request_id))
        if cached is not None:
            return copy.deepcopy(cached)
        with self._lock:
            request_lock = self._request_locks[hash(str(request_id)) % len(self._request_locks)]
        with request_lock:
            with self._lock:
                cached = self._request_cache.get(str(request_id))
            if cached is not None:
                return copy.deepcopy(cached)
            started = time.perf_counter()
            spec = self._specs.get(str(tool))
            if spec is None:
                result = self._response(request_id, "error", error_code="unsupported_tool")
            else:
                try:
                    validate_tool_args(tool, args)
                    handler = getattr(self, spec.handler_name)
                    if spec.name == "ReadArchive":
                        result = handler(request_id, args)
                    else:
                        result = handler(request_id, args, request)
                except Exception as exc:
                    result = self._response(request_id, "error", {"message": str(exc)}, error_code="tool_exception")
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            with self._lock:
                self._request_cache[str(request_id)] = copy.deepcopy(result)
                while len(self._request_cache) > 512:
                    self._request_cache.pop(next(iter(self._request_cache)))
            return result

    def _search(self, request_id: str, args: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        view_id = str(args.get("view_id", "")).strip()
        crop = self._view(request, view_id)
        if not view_id or crop is None:
            return self._response(request_id, "error", error_code="view_not_available")
        if self.visual_index is None:
            return self._response(request_id, "not_available", error_code="visual_archive_disabled")
        matches, latency_ms = self.visual_index.search(crop, top_k=max(1, int(args.get("top_k", self.visual_config.get("top_k", 3)))))
        return self._response(
            request_id,
            "success",
            {"view_id": view_id, "matches": [dict(m, archive_id=m.get("archive_id", m.get("hull_number", ""))) for m in matches],
             "backend": self.visual_config.get("backend", type(self.visual_index).__name__),
             "matching_latency_ms": float(latency_ms)},
            [f"{request_id}:visual"],
        )

    def _read_archive(self, request_id: str, args: dict[str, Any]) -> dict[str, Any]:
        candidate_ids = args.get("candidate_ids", [])
        if not isinstance(candidate_ids, list) or not all(isinstance(item, str) for item in candidate_ids):
            return self._response(request_id, "error", error_code="invalid_candidate_ids")
        records = []
        requested_fields = list(dict.fromkeys(str(item).strip() for item in args.get("fields", []) if str(item).strip()))
        for candidate_id in dict.fromkeys(item.strip() for item in candidate_ids if item.strip()):
            description = self.database.lookup(candidate_id)
            if description is not None:
                getter = getattr(self.database, "get_ship_record", None)
                source_record = getter(candidate_id) if getter else None
                refs = [{"image_id": self._reference_id(r), "archive_id": candidate_id,
                         "prototype_id": r.get("prototype_id", "")}
                        for r in self._archive_records
                        if str(r.get("archive_id", r.get("hull_number", ""))) == candidate_id]
                limit = args.get("max_reference_views", 2)
                revision = hashlib.sha256(json.dumps({"record": source_record, "description": description, "refs": refs}, sort_keys=True, default=str).encode()).hexdigest()[:16]
                attributes = {}
                missing_fields = []
                if requested_fields:
                    for field in requested_fields:
                        if isinstance(source_record, dict) and field in source_record and str(source_record.get(field) or "").strip():
                            attributes[field] = source_record[field]
                        else:
                            missing_fields.append(field)
                records.append({
                    "archive_id": candidate_id,
                    "archived_hull_number": (source_record or {}).get("hull_number", ""),
                    "description": description,
                    "reference_views": refs[:limit],
                    "total_reference_views": len(refs),
                    "omitted_reference_views": max(0, len(refs) - limit),
                    "attributes": attributes,
                    "missing_fields": missing_fields,
                    "archive_ref": f"archive:{candidate_id}:{revision}",
                })
        return self._response(request_id, "success", {"records": records, "requested_fields": requested_fields}, [f"{request_id}:archive"])

    def _read_history(self, request_id: str, args: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        requested = args.get("view_ids")
        history = request.get("history", [])
        if requested is not None and (not isinstance(requested, list) or not all(isinstance(item, str) for item in requested)):
            return self._response(request_id, "error", error_code="invalid_view_ids")
        allowed = set(requested) if requested is not None else None
        selected = [item for item in history if item.get("tool") != "ReadHistory" and (
            allowed is None or allowed.intersection(item.get("view_ids", []) or [item.get("payload", {}).get("view_id")]))]
        view_records = [item for item in request.get("view_records", []) if allowed is None or item.get("view_id") in allowed]
        return self._response(request_id, "success", {"history": selected, "views": view_records,
                             "selected_view_ids": [v["view_id"] for v in view_records]}, [f"{request_id}:history"])

    def _verify(self, request_id: str, args: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
        view_id = str(args.get("view_id", "")).strip()
        crop = self._view(request, view_id)
        if not view_id or crop is None:
            return self._response(request_id, "error", error_code="view_not_available")
        fields = args.get("fields", [])
        if not isinstance(fields, list) or not all(isinstance(item, str) for item in fields):
            return self._response(request_id, "error", error_code="invalid_fields")
        question = str(args.get("question", "") or "").strip()
        transcription = args.get("mode") == "hull_transcription" or any(f in {"hull_number", "observed_hull_number"} for f in fields)
        if transcription:
            question = "Independently transcribe only visible hull-number characters. Use ? for uncertain characters. Do not guess missing digits. Report readability and alternatives."
            fields = ["observed_hull_number", "readability", "alternatives"]
        if self.verify_fn is None:
            return self._response(request_id, "not_available", error_code="verification_service_disabled")
        result = self.verify_fn(crop, question, list(dict.fromkeys(fields)))
        if not isinstance(result, dict):
            return self._response(request_id, "error", error_code="invalid_verification_result")
        payload = dict(result)
        payload["view_id"] = view_id
        payload["observation_mode"] = "blind_hull_transcription" if transcription else "targeted_observation"
        evidence = f"{request_id}:semantic"
        return self._response(request_id, "success", payload, [evidence])


def tool_result_to_agent_result(tool: str, result: dict[str, Any]) -> AgentResult | None:
    """Adapt real tool output to the existing TrackManager evidence path."""
    if result.get("status") != "success":
        return None
    payload = result.get("payload", {}) or {}
    if tool == "SearchArchive":
        matches = list(payload.get("matches", []) or [])
        candidate = str(matches[0].get("hull_number", "")) if matches else ""
        score = float(matches[0].get("score", 0.0)) if matches else 0.0
        second = float(matches[1].get("score", 0.0)) if len(matches) > 1 else 0.0
        return AgentResult(
            match_type="none",
            visual_match_ids=[str(item.get("hull_number", "")) for item in matches if item.get("hull_number")],
            visual_matches=matches,
            visual_candidate_id=candidate,
            visual_similarity_score=score,
            visual_margin=score - second,
            visual_matching_latency_ms=float(payload.get("matching_latency_ms", 0.0)),
            visual_embedding_latency_ms=float(payload.get("matching_latency_ms", 0.0)),
            visual_backend=str(payload.get("backend", "unknown")),
        )
    if tool == "VerifyEvidence":
        semantic_matches = list(payload.get("semantic_matches", []) or [])
        return AgentResult(
            hull_number=str(payload.get("hull_number", "") or ""),
            description=str(payload.get("description", "") or ""),
            identity_features=dict(payload.get("identity_features", {}) or {}),
            match_type="semantic" if semantic_matches else "none",
            semantic_matches=semantic_matches,
            semantic_match_ids=[str(item.get("hull_number", "")) for item in semantic_matches if item.get("hull_number")],
        )
    return None
