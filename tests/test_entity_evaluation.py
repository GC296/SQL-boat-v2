import json

from experiments.build_entity_annotations import build_entities
from experiments.evaluate_entities import evaluate_entities


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_build_entity_annotations_groups_tracklets_by_video_and_vessel():
    entities = build_entities([
        {"video_id": "V1", "track_id": 1, "vessel_id": "小蓝", "known_or_unknown": "unknown", "hull_number": "", "frame_start": 1, "frame_end": 10, "risk_label": "low"},
        {"video_id": "V1", "track_id": 2, "vessel_id": "小蓝", "known_or_unknown": "unknown", "hull_number": "", "frame_start": 20, "frame_end": 30, "risk_label": "low"},
    ])
    assert len(entities) == 1
    assert entities[0]["member_track_ids"] == [1, 2]
    assert entities[0]["known_or_unknown"] == "unknown"


def test_entity_evaluator_reports_association_and_identity(tmp_path):
    annotations = tmp_path / "entities_gt.jsonl"
    write_jsonl(annotations, [
        {"video_id": "V1", "entity_id": "GT1", "member_track_ids": [1, 2], "known_or_unknown": "known", "hull_number": "012"},
        {"video_id": "V1", "entity_id": "GT2", "member_track_ids": [3, 4], "known_or_unknown": "unknown", "hull_number": ""},
    ])
    write_jsonl(tmp_path / "entities.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "member_track_ids": [1, 2], "verified_identity": "012", "identity_state": "structure_verified"},
        {"video_id": "V1", "entity_id": "E2", "member_track_ids": [3, 4], "verified_identity": "", "identity_state": "out_of_archive"},
    ])
    write_jsonl(tmp_path / "edge_cloud.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "upload_bytes": 200, "request_count": 2},
        {"video_id": "V1", "entity_id": "E2", "upload_bytes": 200},
    ])
    metrics = evaluate_entities(tmp_path, annotations)
    assert metrics["entity_association_pair_f1"] == 1.0
    assert metrics["entity_archive_matching_precision"] == 1.0
    assert metrics["entity_archive_matching_success_rate"] == 1.0
    assert metrics["entity_unknown_rejection_recall"] == 1.0
    assert metrics["avg_vlm_calls_per_entity"] == 1.5


def test_entity_evaluator_treats_matching_singletons_as_perfect_association(tmp_path):
    annotations = tmp_path / "entities_gt.jsonl"
    write_jsonl(annotations, [{"video_id": "V1", "entity_id": "GT1", "member_track_ids": [1], "known_or_unknown": "unknown", "hull_number": ""}])
    write_jsonl(tmp_path / "entities.jsonl", [{"video_id": "V1", "entity_id": "E1", "member_track_ids": [1], "verified_identity": "", "identity_state": "out_of_archive"}])
    metrics = evaluate_entities(tmp_path, annotations)
    assert metrics["entity_association_pair_precision"] == 1.0
    assert metrics["entity_association_pair_recall"] == 1.0
    assert metrics["entity_association_pair_f1"] == 1.0


def test_fragmented_unknown_counts_any_false_accept_and_all_fragment_costs(tmp_path):
    annotations = tmp_path / "entities_gt.jsonl"
    write_jsonl(annotations, [
        {"video_id": "V1", "entity_id": "GT1", "member_track_ids": [1, 2], "known_or_unknown": "unknown", "hull_number": ""},
    ])
    write_jsonl(tmp_path / "entities.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "member_track_ids": [1], "verified_identity": "", "identity_state": "unknown"},
        {"video_id": "V1", "entity_id": "E2", "member_track_ids": [2], "verified_identity": "012", "identity_state": "structure_verified"},
        {"video_id": "V1", "entity_id": "JUNK", "member_track_ids": [99], "verified_identity": "012", "identity_state": "structure_verified"},
    ])
    write_jsonl(tmp_path / "edge_cloud.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "upload_bytes": 100},
        {"video_id": "V1", "entity_id": "E2", "upload_bytes": 200},
        {"video_id": "V1", "entity_id": "JUNK", "upload_bytes": 999},
    ])
    metrics = evaluate_entities(tmp_path, annotations)
    assert metrics["predicted_entities"] == 2
    assert metrics["entity_unknown_false_acceptance_rate"] == 1.0
    assert metrics["entity_unresolved_unknown_rate"] == 0.0
    assert metrics["avg_vlm_calls_per_entity"] == 2.0
    assert metrics["avg_upload_bytes_per_entity"] == 300.0
    assert metrics["predicted_fragments_per_truth_entity"] == 2.0
    assert metrics["entity_oversegmentation_rate"] == 1.0


def test_entity_evaluator_reports_agent_task_cost_and_review_metrics(tmp_path):
    annotations = tmp_path / "entities_gt.jsonl"
    write_jsonl(annotations, [
        {"video_id": "V1", "entity_id": "GT1", "member_track_ids": [1, 2], "known_or_unknown": "known", "hull_number": "012", "review_required": True},
        {"video_id": "V1", "entity_id": "GT2", "member_track_ids": [3], "known_or_unknown": "unknown", "hull_number": ""},
        {"video_id": "V1", "entity_id": "GT3", "member_track_ids": [4, 5], "known_or_unknown": "known", "hull_number": "003"},
    ])
    write_jsonl(tmp_path / "entities.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "member_track_ids": [1, 2], "verified_identity": "", "identity_state": "review_requested"},
        {"video_id": "V1", "entity_id": "E2", "member_track_ids": [3], "verified_identity": "", "identity_state": "out_of_archive"},
        {"video_id": "V1", "entity_id": "E3", "member_track_ids": [4, 5], "verified_identity": "003", "identity_state": "confirmed"},
    ])
    write_jsonl(tmp_path / "edge_cloud.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "request_count": 2},
        {"video_id": "V1", "entity_id": "E2", "request_count": 1},
        {"video_id": "V1", "entity_id": "E3", "request_count": 1},
    ])
    write_jsonl(tmp_path / "episodes.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "costly_action_count": 2, "evidence_gain_sum": 0.8, "unnecessary_query_count": 0},
        {"video_id": "V1", "entity_id": "E2", "costly_action_count": 1, "evidence_gain_sum": 0.2, "unnecessary_query_count": 1},
        {"video_id": "V1", "entity_id": "E3", "costly_action_count": 1, "evidence_gain_sum": 0.1, "unnecessary_query_count": 0},
    ])
    write_jsonl(tmp_path / "review_records.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "completeness": 0.875},
    ])
    write_jsonl(tmp_path / "review_outcomes.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "verified_identity": "012", "timestamp": 1.0},
    ])

    metrics = evaluate_entities(tmp_path, annotations)

    assert metrics["autonomous_task_success_rate"] == 0.666667
    assert metrics["safe_handling_success_rate"] == 1.0
    assert metrics["evidence_gain_per_action"] == 0.275
    assert metrics["unnecessary_query_rate"] == 0.25
    assert metrics["success_per_vlm_call"] == 0.5
    assert metrics["inherited_identity_accuracy"] == 0.5
    assert metrics["review_requested_rate"] == 0.333333
    assert metrics["escalation_precision"] == 1.0
    assert metrics["review_needed_recall"] == 1.0
    assert metrics["review_record_completeness"] == 0.875
