"""Reproducible, side-effect-free checks for the final Agent design audit."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np

from pipeline.central_controller import CentralVLMController
from pipeline.identity_tools import IdentityToolService


class AuditDatabase:
    _data = [{"hull_number": "A1", "description": "archive vessel"}]

    def lookup(self, archive_id: str) -> str | None:
        return "archive vessel" if archive_id == "A1" else None


class AuditVisualIndex:
    def __init__(self, score: float = 0.2):
        self.score = score
        self.calls = 0

    def search(self, _crop: Any, top_k: int = 3):
        self.calls += 1
        return ([{"hull_number": "A1", "score": self.score}][:top_k], 1.0)


def crop():
    return np.zeros((8, 8, 3), dtype=np.uint8)


def check_ooa_ignores_visual_score_in_multimodal_mode() -> dict[str, Any]:
    responses = iter([
        {"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "E1:f1"}},
        {"kind": "finish", "decision": "out_of_archive", "evidence_refs": ["__LATEST__"], "reason": "controller proposal"},
    ])
    def infer(context, _images):
        response = dict(next(responses))
        if response.get("evidence_refs") == ["__LATEST__"]:
            evidence = context["snapshot"]["evidence"]
            response["evidence_refs"] = evidence[-1]["evidence_refs"] if evidence else []
        return response
    controller = CentralVLMController(
        AuditDatabase(),
        visual_index=AuditVisualIndex(0.2),
        config={"query_limit": 3, "min_ooa_observations": 2, "min_ooa_score": 0.5, "max_control_steps": 8},
        controller_infer=infer,
    )
    controller.observe("E1", 1, 1, crop())
    snapshot = controller.observe("E1", 1, 2, crop())
    controller.shutdown()
    return {"status": snapshot["status"], "out_of_archive_ready": snapshot["out_of_archive_ready"]}


def check_rejected_finish_is_retried_by_controller() -> dict[str, Any]:
    responses = iter([
        {"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "E1:f1"}},
        {"kind": "finish", "decision": "known", "archive_id": "A1", "evidence_refs": ["__LATEST__"], "reason": "controller proposal"},
        {"kind": "finish", "decision": "review", "evidence_refs": ["__LATEST__"], "reason": "controller retry"},
    ])
    callback_calls: list[str] = []

    def reject(*args):
        callback_calls.append(str(args[3]))
        return {"status": "rejected", "reason": "simulated first callback failure"} if len(callback_calls) == 1 else {"status": "accepted"}

    def infer(context, _images):
        response = dict(next(responses))
        if response.get("evidence_refs") == ["__LATEST__"]:
            evidence = context["snapshot"]["evidence"]
            response["evidence_refs"] = evidence[-1]["evidence_refs"] if evidence else []
        return response

    controller = CentralVLMController(
        AuditDatabase(),
        visual_index=AuditVisualIndex(0.9),
        config={"query_limit": 1, "max_control_steps": 4},
        controller_infer=infer,
        on_finish=reject,
    )
    snapshot = controller.observe("E1", 1, 1, crop())
    controller.shutdown()
    return {"status": snapshot["status"], "callback_decisions": callback_calls}


def check_error_request_is_idempotent() -> dict[str, Any]:
    class BrokenVisualIndex:
        def __init__(self):
            self.calls = 0

        def search(self, _crop, top_k=3):
            self.calls += 1
            raise RuntimeError("simulated service failure")

    index = BrokenVisualIndex()
    service = IdentityToolService(AuditDatabase(), visual_index=index)
    request = {"views": {"v1": crop()}}
    first = service.execute("same-request", "SearchArchive", {"view_id": "v1"}, request)
    second = service.execute("same-request", "SearchArchive", {"view_id": "v1"}, request)
    return {"first": first["status"], "second": second["status"], "backend_calls": index.calls}


def check_belief_reaches_next_context() -> dict[str, Any]:
    responses = iter([
        {"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "E1:f1"}},
        {"kind": "continue", "reason": "wait"},
    ])
    contexts: list[str] = []

    def infer(context, _images):
        contexts.append(context["prompt"])
        return next(responses)

    controller = CentralVLMController(
        AuditDatabase(),
        visual_index=AuditVisualIndex(0.9),
        config={"query_limit": 1, "max_control_steps": 4},
        controller_infer=infer,
    )
    controller.observe("E1", 1, 1, crop())
    controller.shutdown()
    return {
        "controller_calls": len(contexts),
        "second_context_has_evidence": len(contexts) > 1 and "evidence" in contexts[1],
        "second_context_has_claims": len(contexts) > 1 and "claims" in contexts[1],
    }


def check_verify_arguments_reach_service() -> dict[str, Any]:
    received: dict[str, Any] = {}

    def verify(view, question, fields):
        received.update({"shape": list(view.shape), "question": question, "fields": fields})
        return {"observed": {"bridge_position": "left"}, "visibility": "visible"}

    service = IdentityToolService(AuditDatabase(), verify_fn=verify)
    result = service.execute(
        "verify-request", "VerifyEvidence",
        {"view_id": "v1", "question": "distinguish A1 and A2", "fields": ["bridge_position"]},
        {"views": {"v1": crop()}},
    )
    return {"status": result["status"], "received": received}


def main() -> None:
    results = {
        "ooa_multimodal": check_ooa_ignores_visual_score_in_multimodal_mode(),
        "rejected_finish_retry": check_rejected_finish_is_retried_by_controller(),
        "error_idempotency": check_error_request_is_idempotent(),
        "belief_context": check_belief_reaches_next_context(),
        "verify_arguments": check_verify_arguments_reach_service(),
    }
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
