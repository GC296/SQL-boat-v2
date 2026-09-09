import numpy as np
import pytest
import re

from pipeline.central_controller import CentralVLMController
from pipeline.agent_state import EntityEpisode
from pipeline.controller_schema import ControllerProtocolError, parse_controller_output
from pipeline.identity_tools import IdentityToolService


class FakeDatabase:
    def lookup(self, archive_id):
        return {"A1": "white vessel"}.get(archive_id)


class FakeVisualIndex:
    def __init__(self, matches):
        self.matches = matches
        self.calls = 0

    def search(self, crop, top_k=3):
        self.calls += 1
        return list(self.matches)[:top_k], 1.5


def crop():
    return np.zeros((16, 24, 3), dtype=np.uint8)


def controller(responses, visual=None, **config):
    calls = []
    finishes = []
    queue = list(responses)

    def infer(context, images):
        calls.append(context)
        response = dict(queue.pop(0) if queue else {"kind": "continue", "reason": "wait"})
        if "__LATEST__" in response.get("evidence_refs", []):
            evidence = context["snapshot"]["evidence"]
            response["evidence_refs"] = evidence[-1]["evidence_refs"] if evidence else []
        return response

    def finish(entity_id, track_id, frame_id, decision, archive_id, refs, reason):
        finishes.append((entity_id, decision, archive_id, reason))
        return {"status": "accepted"}

    instance = CentralVLMController(
        FakeDatabase(),
        visual_index=visual,
        config={"query_limit": 3, "max_control_steps": 8, **config},
        image_encoder=lambda image: "encoded",
        controller_infer=infer,
        on_finish=finish,
    )
    return instance, calls, finishes


def test_controller_schema_is_a_mutually_exclusive_union():
    assert parse_controller_output({"kind": "continue", "reason": "wait"}).kind == "continue"
    assert parse_controller_output({"kind": "tool", "tool": "ReadHistory", "args": {}}).tool == "ReadHistory"
    with pytest.raises(ControllerProtocolError):
        parse_controller_output({"kind": "finish", "decision": "known"})


def test_controller_executes_vlm_selected_tool_then_finishes():
    visual = FakeVisualIndex([{"hull_number": "A1", "score": 0.91}, {"hull_number": "A2", "score": 0.40}])
    instance, calls, finishes = controller([
        {"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "E1:f1", "top_k": 2}},
        {"kind": "finish", "decision": "known", "archive_id": "A1", "evidence_refs": ["__LATEST__"], "reason": "archive support"},
    ], visual=visual)

    snapshot = instance.observe("E1", 1, 1, crop())

    assert visual.calls == 1
    assert len(calls) == 2
    assert snapshot["resources"]["used"] == 1
    assert finishes == [("E1", "known", "A1", "archive support")]
    assert snapshot["status"] == "known"
    assert snapshot["evidence"][0]["tool"] == "SearchArchive"


def test_multimodal_terminal_known_is_not_overridden_by_low_visual_score():
    visual = FakeVisualIndex([{"hull_number": "A1", "score": 0.01}])
    instance, calls, finishes = controller([
        {"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "E1:f1"}},
        {"kind": "finish", "decision": "known", "archive_id": "A1", "evidence_refs": ["__LATEST__"], "reason": "semantic evidence"},
    ], visual=visual)

    snapshot = instance.observe("E1", 1, 1, crop())

    assert finishes[0][1] == "known"
    assert snapshot["status"] == "known"
    assert "out_of_archive_ready" not in calls[0]["prompt"]


def test_legacy_visual_only_keeps_ooa_guard():
    visual = FakeVisualIndex([{"hull_number": "A1", "score": 0.01}])
    instance, _, finishes = controller([
        {"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "E1:f1"}},
        {"kind": "finish", "decision": "out_of_archive", "evidence_refs": ["__LATEST__"]},
    ], visual=visual, terminal_policy="legacy_visual_only")

    snapshot = instance.observe("E1", 1, 1, crop())

    assert finishes == []
    assert snapshot["status"] == "active"


def test_entities_have_independent_memory_and_budget():
    instance, calls, _ = controller([
        {"kind": "continue", "reason": "wait"},
        {"kind": "continue", "reason": "wait"},
    ])
    first = instance.observe("E1", 1, 1, crop())
    second = instance.observe("E2", 2, 1, crop())

    assert first["entity_id"] == "E1"
    assert second["entity_id"] == "E2"
    assert first["tracklet_history"] == [1]
    assert second["tracklet_history"] == [2]
    assert first["resources"]["used"] == second["resources"]["used"] == 0
    assert len(calls) == 2


def test_continue_does_not_spin_without_a_new_observation():
    instance, calls, _ = controller([{"kind": "continue", "reason": "wait"}])
    first = instance.observe("E1", 1, 1, crop())
    second = instance.observe("E1", 1, 1, crop())

    assert first["observation_version"] == second["observation_version"] == 1
    assert len(calls) == 1


def test_budget_exhaustion_does_not_become_ooa_without_rejection_evidence():
    visual = FakeVisualIndex([{"hull_number": "A1", "score": 0.90}])
    instance, _, finishes = controller([
        {"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "E1:f1", "top_k": 1}},
        {"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "E1:f1", "top_k": 2}},
        {"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "E1:f1", "top_k": 3}},
        {"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "E1:f1", "top_k": 3}},
    ], visual=visual)

    snapshot = instance.observe("E1", 1, 1, crop())

    assert snapshot["resources"]["used"] == 3
    assert visual.calls == 3
    assert snapshot["status"] == "review"
    assert all(decision != "out_of_archive" for _, decision, _, _ in finishes)
    assert any(item.get("phase") == "protocol_error" for item in snapshot["actions"])


def test_terminal_rejection_is_fed_back_before_a_later_commit():
    visual = FakeVisualIndex([{"hull_number": "A1", "score": 0.01}])
    callback_count = 0

    def finish(*args):
        nonlocal callback_count
        callback_count += 1
        return {"status": "rejected", "reason": "first callback failure"} if callback_count == 1 else {"status": "accepted"}

    instance, calls, _ = controller([
        {"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "E1:f1"}},
        {"kind": "finish", "decision": "known", "archive_id": "A1", "evidence_refs": ["__LATEST__"]},
        {"kind": "finish", "decision": "known", "archive_id": "A1", "evidence_refs": ["__LATEST__"]},
    ], visual=visual)
    instance._on_finish = finish

    snapshot = instance.observe("E1", 1, 1, crop())

    assert callback_count == 2
    assert snapshot["status"] == "known"
    assert any("validation_rejected" in call["prompt"] for call in calls[1:])


def test_verify_evidence_callback_does_not_enter_legacy_identity_fusion():
    visual = FakeVisualIndex([{"hull_number": "A1", "score": 0.01}])
    semantic_results = []
    legacy_results = []

    def verify(_view, _question, _fields):
        return {"observed": {"bridge_position": "left"}, "visibility": "visible"}

    instance = CentralVLMController(
        FakeDatabase(),
        visual_index=visual,
        config={"query_limit": 2, "max_control_steps": 4},
        image_encoder=lambda image: "encoded",
        verify_fn=verify,
        controller_infer=lambda context, images: (
            {"kind": "tool", "tool": "VerifyEvidence", "args": {"view_id": "E1:f1", "question": "bridge?", "fields": ["bridge_position"]}}
            if len(semantic_results) == 0 else {"kind": "continue", "reason": "wait"}
        ),
        on_verify_result=lambda track_id, frame_id, result: semantic_results.append(result),
        on_agent_result=lambda track_id, frame_id, result: legacy_results.append(result),
    )

    instance.observe("E1", 1, 1, crop())
    snapshot = instance.store.get("E1").snapshot()
    instance.shutdown()

    assert len(semantic_results) == 1
    assert legacy_results == []
    assert snapshot["claims"][0]["attribute"] == "bridge_position"


def test_repeated_low_archive_support_allows_ooa_finish():
    visual = FakeVisualIndex([{"hull_number": "A1", "score": 0.20}])
    instance, _, finishes = controller([
        {"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "E1:f1"}},
        {"kind": "continue", "reason": "wait for another view"},
        {"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "E1:f2"}},
        {"kind": "finish", "decision": "out_of_archive", "evidence_refs": ["__LATEST__"], "reason": "repeated weak support"},
    ], visual=visual)

    instance.observe("E1", 1, 1, crop())
    instance.observe("E1", 1, 2, crop())
    snapshot = instance.store.get("E1").snapshot()

    assert finishes[0][1] == "out_of_archive"
    assert snapshot["status"] == "out_of_archive"


def test_invalid_future_view_is_rejected_and_entity_can_continue():
    instance, calls, _ = controller([
        {"kind": "tool", "tool": "VerifyEvidence", "args": {"view_id": "future", "fields": ["hull_color"]}},
        {"kind": "continue", "reason": "wait"},
    ])

    snapshot = instance.observe("E1", 1, 1, crop())

    assert snapshot["status"] == "active"
    assert len(calls) == 2
    assert any(item.get("phase") == "protocol_error" for item in snapshot["actions"])


def test_closed_entity_accepts_tracking_updates_without_new_controller_call():
    instance, calls, _ = controller([
        {"kind": "finish", "decision": "review", "reason": "send to operator"},
    ])
    first = instance.observe("E1", 1, 1, crop())
    second = instance.observe("E1", 2, 2, crop())

    assert first["status"] == second["status"] == "review"
    assert second["tracklet_history"] == [1, 2]
    assert len(calls) == 1


def test_tool_request_id_is_idempotent():
    visual = FakeVisualIndex([{"hull_number": "A1", "score": 0.90}])
    service = IdentityToolService(FakeDatabase(), visual_index=visual)
    request = {"views": {"v1": crop()}, "history": []}
    first = service.execute("r1", "SearchArchive", {"view_id": "v1"}, request)
    second = service.execute("r1", "SearchArchive", {"view_id": "v1"}, request)

    assert first == second
    assert visual.calls == 1


def test_successful_archive_search_is_not_repeated_for_same_view_and_arguments():
    visual = FakeVisualIndex([{"hull_number": "A1", "score": 0.90}])
    instance, _, _ = controller([
        {"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "E1:f1", "top_k": 2}},
        {"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "E1:f1", "top_k": 2}},
        {"kind": "finish", "decision": "review", "reason": "visual evidence is insufficient"},
    ], visual=visual)

    snapshot = instance.observe("E1", 1, 1, crop())

    assert visual.calls == 1
    assert snapshot["resources"]["used"] == 1
    assert snapshot["status"] == "review"
    assert any("already succeeded" in action.get("message", "") for action in snapshot["actions"])


def test_read_archive_honors_requested_fields():
    class StructuredDatabase(FakeDatabase):
        def get_ship_record(self, archive_id):
            return {"hull_number": archive_id, "bridge_position": "left", "hull_color": "white"}

    service = IdentityToolService(StructuredDatabase())
    result = service.execute(
        "read-1",
        "ReadArchive",
        {"candidate_ids": ["A1"], "fields": ["bridge_position", "missing_field"]},
        {},
    )

    record = result["payload"]["records"][0]
    assert record["attributes"] == {"bridge_position": "left"}
    assert record["missing_fields"] == ["missing_field"]


def test_retracted_evidence_reopens_gap_and_disputes_claim():
    episode = EntityEpisode("S", "E1", "E1:1")
    episode.add_observation("v1", 1, crop())
    episode.append_evidence({"evidence_id": "e1", "request_id": "r1", "status": "success", "evidence_refs": ["r1:semantic"]})
    episode.add_claims([{
        "claim_id": "c1", "evidence_id": "e1", "view_ids": ["v1"],
        "subject_entity_id": "E1", "attribute": "bridge_position", "observed_value": "left",
    }])
    rejected = episode.apply_belief_patch(({
        "op": "upsert_gap",
        "gap": {"gap_id": "g1", "question": "left or right", "candidate_ids": ["A1"], "basis_refs": ["c1"]},
    }, {
        "op": "resolve_gap", "gap_id": "g1", "resolution_refs": ["c1"],
    }), {"e1"}, {"c1"}, {"A1"}, {"v1"})
    assert rejected == []
    assert episode.gaps[0]["status"] == "resolved"

    rejected = episode.apply_belief_patch(({
        "op": "set_evidence_status", "evidence_id": "e1", "status": "retracted", "reason": "source correction",
    },), {"e1"}, {"c1"}, {"A1"}, {"v1"})
    assert rejected == []
    assert episode.claims[0]["validity_status"] == "retracted"
    assert episode.gaps[0]["status"] == "unresolved"


def test_invalidated_evidence_can_be_revised_but_not_used_as_active_support():
    episode = EntityEpisode("S", "E1", "E1:1")
    episode.append_evidence({"evidence_id": "e1", "request_id": "r1", "status": "success"})
    rejected = episode.apply_belief_patch(({
        "op": "set_evidence_status", "evidence_id": "e1", "status": "disputed", "reason": "semantic conflict",
    },), {"e1"}, set(), {"A1"}, set(), {"e1"})
    assert rejected == []
    rejected = episode.apply_belief_patch(({
        "op": "upsert_gap",
        "gap": {"gap_id": "g1", "question": "is the bridge on the left", "candidate_ids": ["A1"], "basis_refs": ["e1"]},
    },), set(), set(), {"A1"}, set(), {"e1"})
    assert rejected == ["gap has invalid basis references"]


def test_async_observation_consumes_latest_cached_view():
    visual = FakeVisualIndex([{"hull_number": "A1", "score": 0.91}])
    instance, _, finishes = controller([
        {"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "E1:f1"}},
        {"kind": "finish", "decision": "known", "archive_id": "A1", "evidence_refs": ["__LATEST__"], "reason": "cached episode"},
    ], visual=visual)
    instance.submit_observation("E1", 1, 1, crop())
    instance.submit_observation("E1", 1, 2, crop())
    instance.wait_for_idle()
    snapshot = instance.store.get("E1").snapshot()
    instance.shutdown()
    assert snapshot["status"] == "known"
    assert snapshot["observation_version"] == 2
    assert finishes[0][1] == "known"
