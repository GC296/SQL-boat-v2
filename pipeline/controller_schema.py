"""Strict wire schema for the central VLM controller."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


TOOL_NAMES = {"SearchArchive", "ReadArchive", "ReadHistory", "VerifyEvidence"}
FINISH_DECISIONS = {"known", "out_of_archive", "review"}


class ControllerProtocolError(ValueError):
    """Raised when a controller response cannot be executed safely."""


@dataclass(frozen=True)
class ControllerDecision:
    kind: str
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    decision: str = ""
    archive_id: str = ""
    evidence_refs: tuple[str, ...] = ()
    belief_patch: tuple[dict[str, Any], ...] = ()
    reason: str = ""
    gap_id: str = ""
    decision_target: str = ""
    assessment: dict[str, Any] = field(default_factory=dict)
    operation_purpose: str = ""


TOOL_ARGUMENTS = {
    "SearchArchive": {"view_id", "top_k"},
    "ReadArchive": {"candidate_ids", "fields", "max_reference_views"},
    "ReadHistory": {"view_ids"},
    "VerifyEvidence": {"view_id", "fields", "question", "mode"},
}


def validate_tool_args(tool: str, args: dict[str, Any]) -> None:
    if tool not in TOOL_ARGUMENTS or not isinstance(args, dict):
        raise ControllerProtocolError("invalid tool or args")
    if set(args) - TOOL_ARGUMENTS[tool]:
        raise ControllerProtocolError(f"unknown arguments for {tool}: {sorted(set(args) - TOOL_ARGUMENTS[tool])}")
    if tool in {"SearchArchive", "VerifyEvidence"}:
        if not isinstance(args.get("view_id"), str) or not args["view_id"].strip():
            raise ControllerProtocolError("view_id must be a non-empty string")
    for key in ("top_k", "max_reference_views"):
        if key in args and (type(args[key]) is not int or not 1 <= args[key] <= 50):
            raise ControllerProtocolError(f"{key} must be an integer from 1 to 50")
    for key in ("candidate_ids", "fields", "view_ids"):
        if key in args:
            _string_list(args[key], key)
            if args[key] is None or len(args[key]) > 50:
                raise ControllerProtocolError(f"{key} must be a list of at most 50 strings")
    if tool == "ReadArchive" and not args.get("candidate_ids"):
        raise ControllerProtocolError("ReadArchive requires candidate_ids")
    if "question" in args and (not isinstance(args["question"], str) or len(args["question"]) > 4000):
        raise ControllerProtocolError("question must be a string of at most 4000 characters")
    if "mode" in args and args["mode"] not in {"observe", "hull_transcription"}:
        raise ControllerProtocolError("VerifyEvidence mode must be observe or hull_transcription")


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        raise ControllerProtocolError("controller output must be a JSON object")
    text = value.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ControllerProtocolError("controller output is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ControllerProtocolError("controller output must be a JSON object")
    return parsed


def _string_list(value: Any, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ControllerProtocolError(f"{field_name} must be a list of non-empty strings")
    return tuple(dict.fromkeys(item.strip() for item in value))


def _patch_list(value: Any) -> tuple[dict[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ControllerProtocolError("belief_patch must be a list of objects")
    return tuple(dict(item) for item in value)


def parse_controller_output(value: Any) -> ControllerDecision:
    """Parse the mutually exclusive tool/continue/finish response union."""
    payload = _json_object(value)
    kind = str(payload.get("kind", "")).strip().lower()
    assessment = payload.get("assessment", {})
    if not isinstance(assessment, dict):
        raise ControllerProtocolError("assessment must be an object")
    if set(assessment) - {"leading_candidate", "readiness", "blocking_gap_ids", "basis_refs", "reason"}:
        raise ControllerProtocolError("unsupported assessment fields")
    for key in ("leading_candidate", "readiness", "reason"):
        if key in assessment and not isinstance(assessment[key], str):
            raise ControllerProtocolError(f"assessment.{key} must be a string")
    if "readiness" in assessment and assessment["readiness"] not in {"ready", "unresolved", "conflicted"}:
        raise ControllerProtocolError("invalid readiness")
    for key in ("blocking_gap_ids", "basis_refs"):
        if key in assessment:
            _string_list(assessment[key], key)
    if kind != "tool" and any(payload.get(k) for k in ("tool", "args")):
        raise ControllerProtocolError("tool fields require kind=tool")
    if kind != "finish" and any(payload.get(k) for k in ("decision", "archive_id")):
        raise ControllerProtocolError("terminal fields require kind=finish")
    common = {
        "evidence_refs": _string_list(payload.get("evidence_refs"), "evidence_refs"),
        "belief_patch": _patch_list(payload.get("belief_patch")),
        "reason": str(payload.get("reason", "") or "").strip(),
        "gap_id": str(payload.get("gap_id", "") or "").strip(),
        "decision_target": str(payload.get("decision_target", "") or "").strip(),
        "assessment": dict(assessment),
        "operation_purpose": str(payload.get("operation_purpose", "") or "").strip(),
    }
    if kind == "tool":
        tool = str(payload.get("tool", "")).strip()
        args = payload.get("args", {})
        if tool not in TOOL_NAMES:
            raise ControllerProtocolError(f"unsupported controller tool: {tool or '<missing>'}")
        if not isinstance(args, dict):
            raise ControllerProtocolError("tool args must be an object")
        validate_tool_args(tool, args)
        return ControllerDecision(kind="tool", tool=tool, args=dict(args), **common)
    if kind == "continue":
        return ControllerDecision(kind="continue", **common)
    if kind == "finish":
        decision = str(payload.get("decision", "")).strip().lower()
        if decision not in FINISH_DECISIONS:
            raise ControllerProtocolError(f"unsupported terminal decision: {decision or '<missing>'}")
        archive_id = str(payload.get("archive_id", "") or "").strip()
        if decision == "known" and not archive_id:
            raise ControllerProtocolError("known finish requires archive_id")
        if decision != "known" and archive_id:
            raise ControllerProtocolError("archive_id is only allowed for known finish")
        return ControllerDecision(kind="finish", decision=decision, archive_id=archive_id, **common)
    raise ControllerProtocolError("kind must be tool, continue, or finish")
