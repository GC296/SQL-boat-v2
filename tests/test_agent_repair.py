"""Regression cases for the central VLM evidence/decision repair.

Model replies are scripted: these validate plumbing, not model accuracy.
"""
import json
import threading

import numpy as np
import pytest

from tests.test_central_controller import controller, crop, FakeDatabase, FakeVisualIndex
from pipeline.agent_state import EntityEpisode
from pipeline.controller_schema import ControllerProtocolError, parse_controller_output
from pipeline.central_controller import CentralVLMController
from pipeline.identity_tools import IdentityToolService


def test_exhausted_paid_budget_feedback_allows_vlm_known():
    visual = FakeVisualIndex([{"hull_number": "A1", "score": .91}])
    agent, calls, _ = controller([
        {"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "E1:f1", "top_k": 1}},
        {"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "E1:f1", "top_k": 2}},
        {"kind": "finish", "decision": "known", "archive_id": "A1", "evidence_refs": ["__LATEST__"]},
    ], visual=visual, query_limit=1)
    result = agent.observe("E1", 1, 1, crop())
    assert visual.calls == 1
    assert result["status"] == "known"
    assert result["terminal_record"]["decision_source"] == "vlm"
    assert calls[-1]["snapshot"]["resources"]["remaining"] == 0
    assert calls[-1]["snapshot"]["last_validation_feedback"]
    assert calls[-1]["snapshot"]["successful_search_count"] == 1
    agent.shutdown()


def test_eof_wait_is_explicit_system_fallback():
    agent, _, _ = controller([], max_control_steps=2, settlement_steps=1)
    agent.observe("E1", 1, 1, crop())
    result = agent.finish_eof()[0]
    assert result["terminal_record"]["system_fallback"] is True
    assert result["terminal_record"]["decision_source"] == "system_fallback"
    agent.shutdown()


@pytest.mark.parametrize("value", [True, "three", 0, 51, [], 1.5])
def test_bad_top_k_is_protocol_error(value):
    with pytest.raises(ControllerProtocolError):
        parse_controller_output({"kind": "tool", "tool": "SearchArchive", "args": {"view_id": "v1", "top_k": value}})


def test_history_images_take_priority_and_no_recursive_payload():
    agent, _, _ = controller([], max_context_views=1)
    episode = EntityEpisode("s", "E1", "ep")
    episode.add_observation("old", 1, crop())
    episode.add_observation("new", 2, crop()+1)
    for _ in range(12):
        decision = parse_controller_output({"kind": "tool", "tool": "ReadHistory", "args": {"view_ids": ["old"]}})
        agent._dispatch(episode, 1, 2, decision)
    context, images = agent._context(episode, None, False)
    assert context["snapshot"]["context_view_ids"] == ["old"]
    assert len(images) == 1
    assert len(json.dumps(episode.snapshot())) < 60000
    assert all("history" not in e.get("payload", {}) for e in episode.evidence)
    agent.shutdown()


def test_failed_image_encoding_preserves_manifest_alignment():
    agent, _, _ = controller([])
    agent._tools.image_encoder = lambda im: "ok" if not im.any() else (_ for _ in ()).throw(ValueError("bad image"))
    episode = EntityEpisode("s", "E1", "ep")
    episode.add_observation("old", 1, crop())
    episode.add_observation("new", 2, crop()+1)
    context, images = agent._context(episode, None, False)
    assert images == ["ok"]
    assert context["snapshot"]["image_view_ids"] == ["old"]
    assert context["snapshot"]["image_manifest"][0]["image_index"] == 0
    assert context["snapshot"]["image_errors"][0]["image_id"] == "new"
    agent.shutdown()


def test_read_archive_delivers_real_reference_image_and_distinct_number(tmp_path):
    import cv2
    image_path = tmp_path / "reference.png"
    cv2.imwrite(str(image_path), crop()+23)
    visual = FakeVisualIndex([])
    visual.records = [{"hull_number": "A1", "prototype_id": "p1", "image_path": str(image_path)}]
    class DB(FakeDatabase):
        def get_ship_record(self, archive_id):
            return {"hull_number": "真实123", "description": "white vessel"}
    agent = CentralVLMController(DB(), visual_index=visual, image_encoder=lambda im: str(int(im.mean())))
    episode = EntityEpisode("s", "E1", "ep")
    episode.add_observation("v1", 1, crop())
    agent._dispatch(episode, 1, 1, parse_controller_output({"kind": "tool", "tool": "ReadArchive", "args": {"candidate_ids": ["A1"]}}))
    context, images = agent._context(episode, None, False)
    record = episode.last_result["payload"]["records"][0]
    assert record["archive_id"] == "A1"
    assert record["archived_hull_number"] == "真实123"
    assert "23" in images
    assert any(m["role"] == "archive" for m in context["snapshot"]["image_manifest"])
    agent.shutdown()


def test_closed_memory_does_not_accumulate_images_and_pins_survive():
    episode = EntityEpisode("s", "E1", "ep", max_views=3)
    episode.add_observation("v0", 0, crop())
    episode.pinned_views.add("v0")
    for frame in range(1, 100):
        episode.add_observation(f"v{frame}", frame, crop())
    assert "v0" in episode.views and len(episode.views) == 3
    episode.close("review")
    for frame in range(100, 200):
        episode.add_observation(f"v{frame}", frame, crop())
    assert len(episode.views) == 3


def test_pending_entities_and_latest_observation_are_not_lost():
    entered, release = threading.Event(), threading.Event()
    seen = []
    def infer(context, images):
        snapshot = context["snapshot"]
        seen.append((snapshot["entity_id"], snapshot["current_view_id"]))
        if len(seen) == 1:
            entered.set()
            assert release.wait(5)
        return {"kind": "continue"}
    agent = CentralVLMController(FakeDatabase(), config={"max_workers": 1}, image_encoder=lambda im: "ok", controller_infer=infer)
    try:
        agent.submit_observation("E1", 1, 1, crop())
        assert entered.wait(5)
        agent.submit_observation("E2", 2, 1, crop())
        agent.submit_observation("E1", 1, 2, crop())
        release.set()
        agent.wait_for_idle(timeout=5)
        assert ("E2", "E2:f1") in seen
        assert ("E1", "E1:f2") in seen
        assert len(seen) == 3
    finally:
        release.set()
        agent.shutdown()


def test_hull_reading_does_not_receive_expected_archive_number():
    received = []
    service = IdentityToolService(FakeDatabase(), verify_fn=lambda im, q, fields: received.append((q, fields)) or {"observed_hull_number": "?", "readability": "unreadable"})
    result = service.execute("r1", "VerifyEvidence", {"view_id": "v1", "fields": ["hull_number"], "question": "Please confirm archive number 123456"}, {"views": {"v1": crop()}})
    assert "123456" not in received[0][0]
    assert result["payload"]["observed_hull_number"] == "?"
    assert result["payload"]["observation_mode"] == "blind_hull_transcription"


def test_same_patch_claim_can_support_candidate_and_invalid_patch_rolls_back():
    episode = EntityEpisode("s", "E1", "ep")
    episode.add_observation("v1", 1, crop())
    episode.append_evidence({"evidence_id": "e1", "request_id": "r1", "status": "success", "view_ids": ["v1"]})
    claim = {"claim_id": "c1", "evidence_id": "e1", "subject_entity_id": "E1", "view_ids": ["v1"], "attribute": "color", "observed_value": "white", "visibility": "clear", "source_role": "forged_tool"}
    patch = ({"op": "upsert_claim", "claim": claim}, {"op": "upsert_candidate_assessment", "assessment": {"archive_id": "A1", "support_claim_ids": ["c1"]}})
    assert episode.apply_belief_patch(patch, {"e1"}, set(), {"A1"}, {"v1"}) == []
    assert episode.claims[0]["source_role"] == "controller_observation"
    assert episode.candidate_assessments[0]["support_claim_ids"] == ["c1"]
    before = episode.snapshot()
    invalid = ({"op": "upsert_claim", "claim": {**claim, "observed_value": "black"}}, {"op": "resolve_gap", "gap_id": "missing", "resolution_refs": ["e1"]})
    assert episode.apply_belief_patch(invalid, {"e1"}, {"c1"}, {"A1"}, {"v1"})
    assert episode.snapshot()["claims"] == before["claims"]


def test_completed_track_accepts_central_terminal_without_fabricated_confidence():
    from pipeline.tracker import TrackManager
    tracker = TrackManager(max_stale_frames=1)
    item = tracker.get_or_create(1, 1)
    initial_uncertainty = item.last_uncertainty_score
    tracker.cleanup_stale(5)
    assert tracker.bind_controller_identity(1, "A1", "white")
    item = tracker.get_any(1)
    assert item.recognized and not item.pending
    assert item.episode_status == "closed"
    assert item.last_uncertainty_score == initial_uncertainty
    assert tracker.request_review(1, 6, "central_vlm_review", ["new conflict"])
    assert item.verified_identity == "" and not item.db_matched


def test_episode_control_budget_is_shared_across_new_views():
    agent, calls, _ = controller([], max_episode_steps=2, settlement_steps=1)
    agent.observe("E1", 1, 1, crop())
    agent.observe("E1", 1, 2, crop())
    final = agent.observe("E1", 1, 3, crop())
    assert len(calls) == 3
    assert calls[-1]["snapshot"]["settlement_only"]
    assert final["terminal_record"]["system_fallback"]
    agent.shutdown()


def test_http_uses_explicit_config_and_labels_images(monkeypatch):
    import httpx
    import tools
    captured = {}
    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return httpx.Response(200, request=httpx.Request("POST", url), json={"choices": [{"message": {"content": '{"kind":"continue"}'}, "finish_reason": "stop"}], "usage": {"total_tokens": 42}})
    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(tools, "_get_llm_cfg", lambda: pytest.fail("must use pipeline config"))
    result = tools._chat_json("task", ["fake"], llm_config={"base_url": "http://test.invalid/v1", "model": "test-model", "max_retries": 0}, image_manifest=[{"image_id": "old", "role": "observation"}])
    assert captured["url"] == "http://test.invalid/v1/chat/completions"
    assert captured["json"]["model"] == "test-model"
    assert "old" in captured["json"]["messages"][0]["content"][1]["text"]
    assert result["_vlm_usage"]["total_tokens"] == 42
    assert result["_vlm_finish_reason"] == "stop"


def test_cli_can_explicitly_enable_local_dinov2():
    from pipeline.cli import build_parser, _merge_args_to_config
    args = build_parser().parse_args(["clip.mp4", "--visual-backend", "dinov2", "--dinov2-repo", "dinov2-main", "--dinov2-weights", "weights.pth"])
    config = _merge_args_to_config(args, {})
    assert config["experiment"]["visual_archive"] == {"enabled": True, "backend": "dinov2", "model_repo": "dinov2-main", "weights": "weights.pth"}


def test_diagnostic_filesystem_error_does_not_interrupt_agent(tmp_path):
    from pipeline.agent_diagnostics import AgentDiagnostics
    invalid_root = tmp_path / "file"
    invalid_root.write_text("occupied")
    diagnostics = AgentDiagnostics({"diagnostics_enabled": True, "diagnostics_dir": str(invalid_root)})
    diagnostics.write({"event": "test"})
    assert diagnostics.image("YWJj")["persisted"] is False
