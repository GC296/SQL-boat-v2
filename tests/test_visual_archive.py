import cv2
import sys
from types import SimpleNamespace

from agent import AgentResult
from pipeline.visual_archive import (
    aggregate_identity_matches,
    visual_agreement_supports_confirmation,
    visual_consistency_supports_confirmation,
    visual_confirmation_block_reason,
    visual_only_rejection_supports_terminal,
    visual_open_set_override,
)
from pipeline.pipeline import ShipPipeline
from pipeline.visual_reranker import (
    QwenVisualRerankerIndex,
    _parse_scores,
    _score_url,
)


def test_visual_matches_group_prototypes_by_hull_number():
    records = [
        {"hull_number": "012", "prototype_id": "012_01", "image_path": "012_01.jpg"},
        {"hull_number": "012", "prototype_id": "012_02", "image_path": "012_02.jpg"},
        {"hull_number": "003", "prototype_id": "003_01", "image_path": "003_01.jpg"},
    ]

    matches = aggregate_identity_matches([0.72, 0.91, 0.83], records, top_k=3)

    assert matches == [
        {"hull_number": "012", "score": 0.91, "prototype_id": "012_02", "image_path": "012_02.jpg"},
        {"hull_number": "003", "score": 0.83, "prototype_id": "003_01", "image_path": "003_01.jpg"},
    ]


def test_agent_result_exposes_visual_sidecar_fields():
    result = AgentResult(
        visual_match_ids=["012", "003"],
        visual_matches=[{"hull_number": "012", "score": 0.81}],
        visual_candidate_id="012",
        visual_similarity_score=0.81,
        visual_margin=0.22,
        visual_embedding_latency_ms=12.5,
        visual_matching_latency_ms=12.5,
        visual_backend="qwen_vl_reranker",
        visual_model_request_count=2,
        visual_pairs_scored=16,
        visual_gate_reason="visual_identity_conflict",
    )

    assert result.visual_match_ids == ["012", "003"]
    assert result.visual_candidate_id == "012"
    assert result.visual_similarity_score == 0.81
    assert result.visual_margin == 0.22
    assert result.visual_embedding_latency_ms == 12.5
    assert result.visual_matching_latency_ms == 12.5
    assert result.visual_backend == "qwen_vl_reranker"
    assert result.visual_model_request_count == 2
    assert result.visual_pairs_scored == 16
    assert result.visual_gate_reason == "visual_identity_conflict"


def test_visual_gate_blocks_weak_agreement():
    config = {"decision_enabled": True, "min_support_score": 0.45}

    assert visual_confirmation_block_reason("012", "012", 0.4379, 0.10, config) == "visual_evidence_too_weak"


def test_visual_gate_blocks_nearly_tied_candidate():
    config = {
        "decision_enabled": True,
        "min_support_score": 0.45,
        "min_support_margin": 0.10,
    }

    assert visual_confirmation_block_reason("012", "012", 0.478215, 0.000471, config) == "visual_candidate_ambiguous"


def test_visual_gate_blocks_reliable_identity_conflict():
    config = {
        "decision_enabled": True,
        "min_support_score": 0.45,
        "conflict_min_score": 0.50,
        "conflict_min_margin": 0.05,
    }

    assert visual_confirmation_block_reason("012", "003", 0.5414, 0.0738, config) == "visual_identity_conflict"


def test_visual_gate_keeps_supported_agreement_and_blocks_ambiguous_conflict():
    config = {
        "decision_enabled": True,
        "min_support_score": 0.45,
        "conflict_min_score": 0.50,
        "conflict_min_margin": 0.05,
    }

    assert visual_confirmation_block_reason("003", "003", 0.65, 0.26, config) == ""
    assert visual_confirmation_block_reason("003", "012", 0.49, 0.01, config) == "visual_candidate_ambiguous"


def test_visual_gate_is_disabled_for_text_baselines():
    assert visual_confirmation_block_reason("012", "003", 0.90, 0.50, {"decision_enabled": False}) == ""


def test_structure_visual_agreement_promotes_uncertain_candidate():
    config = {
        "decision_enabled": True,
        "agreement_confirmation_enabled": True,
        "agreement_min_text_score": 0.70,
        "min_support_score": 0.56,
        "min_support_margin": 0.09,
    }

    assert visual_agreement_supports_confirmation("012", "uncertain", 0.74, "012", 0.87, 0.14, config)


def test_structure_visual_agreement_rejects_weak_or_conflicting_evidence():
    config = {
        "decision_enabled": True,
        "agreement_min_text_score": 0.70,
        "min_support_score": 0.56,
        "min_support_margin": 0.09,
    }

    assert not visual_agreement_supports_confirmation("012", "uncertain", 0.69, "012", 0.87, 0.14, config)
    assert not visual_agreement_supports_confirmation("012", "uncertain", 0.74, "003", 0.87, 0.14, config)
    assert not visual_agreement_supports_confirmation("012", "uncertain", 0.74, "012", 0.55, 0.14, config)
    assert not visual_agreement_supports_confirmation("012", "uncertain", 0.74, "012", 0.87, 0.08, config)


def test_visual_open_set_requires_repeated_low_score_before_rejection():
    config = {"decision_enabled": True, "open_set_enabled": True, "out_of_archive_score": 0.50}

    assert visual_open_set_override("012", "012", 0.44, 0.30, config) == (
        "uncertain",
        "visual_low_score_requires_reobservation",
    )
    assert visual_open_set_override("012", "012", 0.44, 0.30, config, low_score_observations=2) == (
        "out_of_archive",
        "visual_out_of_archive_repeated_low_score",
    )


def test_visual_open_set_treats_low_identity_margin_as_ambiguous():
    config = {
        "decision_enabled": True,
        "open_set_enabled": True,
        "out_of_archive_score": 0.50,
        "out_of_archive_margin": 0.20,
    }

    assert visual_open_set_override("012", "012", 0.611869, 0.188915, config) == (
        "uncertain",
        "visual_candidate_ambiguous",
    )


def test_visual_open_set_keeps_strong_visual_only_candidate_provisional():
    config = {"decision_enabled": True, "open_set_enabled": True}

    assert visual_open_set_override("", "V081", 0.936464, 0.25, config) == (
        "uncertain",
        "visual_candidate_requires_consistency",
    )


def test_visual_only_open_set_requires_repeated_weak_score_and_margin():
    config = {
        "decision_enabled": True,
        "open_set_enabled": True,
        "identity_decision_mode": "visual_only",
        "visual_only_reject_score": 0.70,
        "visual_only_reject_margin": 0.08,
        "min_out_of_archive_observations": 2,
    }

    assert visual_open_set_override("", "V081", 0.65, 0.05, config) == (
        "uncertain",
        "visual_only_weak_evidence_requires_reobservation",
    )
    assert visual_open_set_override("", "V081", 0.65, 0.05, config, low_score_observations=2) == (
        "out_of_archive",
        "visual_only_repeated_weak_evidence",
    )
    assert visual_open_set_override("", "V081", 0.65, 0.12, config, low_score_observations=2) == (
        "uncertain",
        "visual_only_gray_zone",
    )
    assert visual_open_set_override("", "V081", 0.82, 0.05, config, low_score_observations=2) == (
        "uncertain",
        "visual_only_gray_zone",
    )


def test_repeated_strong_visual_candidate_can_rescue_missing_text_evidence():
    config = {
        "decision_enabled": True,
        "visual_only_confirmation_enabled": True,
        "visual_only_min_observations": 2,
        "visual_only_min_score": 0.90,
        "visual_only_min_margin": 0.08,
    }

    assert not visual_consistency_supports_confirmation("", "out_of_archive", "V081", 0.94, 0.18, 1, config)
    assert visual_consistency_supports_confirmation("", "out_of_archive", "V081", 0.94, 0.18, 2, config)
    assert not visual_consistency_supports_confirmation("", "out_of_archive", "V081", 0.89, 0.18, 2, config)
    assert visual_consistency_supports_confirmation("", "out_of_archive", "V081", 0.94, 0.07, 2, config)
    assert not visual_consistency_supports_confirmation("012", "uncertain", "V081", 0.94, 0.18, 2, config)


def test_score_gate_accepts_one_observation_without_margin():
    config = {
        "decision_enabled": True,
        "visual_only_confirmation_enabled": True,
        "visual_only_decision_rule": "score_gate",
        "visual_only_min_observations": 1,
        "visual_only_min_score": 0.80,
        "visual_only_strong_score": 0.99,
    }

    assert visual_consistency_supports_confirmation("", "unknown", "V081", 0.80, 0.001, 1, config)
    assert not visual_consistency_supports_confirmation("", "unknown", "V081", 0.79, 0.90, 1, config)


def test_score_gate_rejection_ignores_margin_after_required_observations():
    config = {
        "decision_enabled": True,
        "open_set_enabled": True,
        "identity_decision_mode": "visual_only",
        "visual_only_decision_rule": "score_gate",
        "visual_only_reject_score": 0.80,
        "min_out_of_archive_observations": 2,
    }

    assert not visual_only_rejection_supports_terminal("V081", 0.79, 0.90, 1, 1, config)
    assert visual_only_rejection_supports_terminal("V081", 0.79, 0.90, 2, 2, config)
    assert visual_open_set_override("", "V081", 0.79, 0.90, config, low_score_observations=2, observation_count=2) == (
        "out_of_archive",
        "visual_only_score_gate_rejected",
    )


def test_repeated_strong_visual_candidate_can_resolve_text_conflict():
    config = {
        "decision_enabled": True,
        "visual_only_confirmation_enabled": True,
        "visual_only_min_observations": 2,
        "visual_only_min_score": 0.90,
        "visual_only_min_margin": 0.05,
    }

    assert visual_consistency_supports_confirmation("V110", "conflicting", "012", 0.95, 0.06, 2, config)
    assert not visual_consistency_supports_confirmation("V110", "uncertain", "012", 0.95, 0.06, 2, config)


def test_visual_only_identity_decision_ignores_text_hull():
    from types import SimpleNamespace

    pipeline = ShipPipeline.__new__(ShipPipeline)
    pipeline._visual_cfg = {
        "decision_enabled": True,
        "open_set_enabled": True,
        "visual_only_confirmation_enabled": True,
        "visual_only_min_observations": 2,
        "visual_only_min_score": 0.90,
        "visual_only_min_margin": 0.05,
    }
    info = SimpleNamespace(visual_consistent_observations=1, visual_low_score_observations=0)
    result = AgentResult(
        hull_number="012",
        match_type="exact",
        visual_candidate_id="V081",
        visual_similarity_score=0.95,
        visual_margin=0.20,
        visual_matches=[{"hull_number": "V081", "score": 0.95}],
    )

    first = pipeline._decide_visual_only_identity(info, result)
    assert first.state == "uncertain"
    assert first.identity == "V081"
    assert first.evidence_source.startswith("visual_only:")

    info.visual_consistent_observations = 2
    second = pipeline._decide_visual_only_identity(info, result)
    assert second.state == "structure_verified"
    assert second.identity == "V081"
    assert second.evidence_source == "visual_consistency"


def test_visual_open_set_marks_reliable_modality_disagreement_conflicting():
    config = {"decision_enabled": True, "open_set_enabled": True}

    assert visual_open_set_override("012", "003", 0.80, 0.30, config) == (
        "conflicting",
        "visual_identity_conflict",
    )


def test_visual_open_set_keeps_supported_identity_and_disabled_baseline():
    enabled = {"decision_enabled": True, "open_set_enabled": True}

    assert visual_open_set_override("003", "003", 0.65, 0.22, enabled) == ("", "")
    assert visual_open_set_override("003", "003", 0.20, 0.01, {"open_set_enabled": False}) == ("", "")


def test_reranker_score_url_accepts_root_and_openai_base_urls():
    assert _score_url("http://localhost:7894") == "http://localhost:7894/score"
    assert _score_url("http://localhost:7894/v1") == "http://localhost:7894/score"


def test_reranker_scores_are_restored_by_response_index():
    payload = {"data": [{"index": 1, "score": 0.2}, {"index": 0, "score": 0.9}]}

    assert _parse_scores(payload, 2) == [0.9, 0.2]


def test_qwen_reranker_directly_scores_all_archive_images(tmp_path, monkeypatch):
    import numpy as np

    archive = tmp_path / "archive"
    for hull_number, value in (("003", 40), ("012", 220)):
        hull_dir = archive / hull_number
        hull_dir.mkdir(parents=True)
        image = np.full((20, 30, 3), value, dtype=np.uint8)
        assert cv2.imwrite(str(hull_dir / f"{hull_number}_01.jpg"), image)

    requests = []

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"index": 0, "score": 0.31}, {"index": 1, "score": 0.87}]}

    def fake_post(url, headers, json, timeout):
        requests.append((url, headers, json, timeout))
        return FakeResponse()

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        SimpleNamespace(
            HTTPError=Exception,
            Timeout=lambda **kwargs: kwargs,
            post=fake_post,
        ),
    )
    matcher = QwenVisualRerankerIndex(
        {
            "archive_root": str(archive),
            "model": "Qwen/Qwen3-VL-Reranker-2B",
            "base_url": "http://localhost:7894/v1",
            "request_batch_size": 8,
            "max_retries": 0,
        }
    )

    matches, latency_ms = matcher.search(np.full((24, 32, 3), 200, dtype=np.uint8), top_k=2)

    assert [item["hull_number"] for item in matches] == ["012", "003"]
    assert [item["score"] for item in matches] == [0.87, 0.31]
    assert latency_ms >= 0.0
    assert len(requests) == 1
    assert matcher.last_request_count == 1
    assert matcher.last_pair_count == 2
    assert requests[0][0] == "http://localhost:7894/score"
    assert len(requests[0][2]["documents"]) == 2
    assert requests[0][2]["queries"]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
