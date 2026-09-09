import json

from experiments.evaluate import evaluate


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_evaluator_reports_primary_archive_metrics(tmp_path):
    write_jsonl(tmp_path / "observations.jsonl", [
        {"video_id": "V1", "track_id": 1, "frame_id": 10, "observation_quality": 0.8},
        {"video_id": "V1", "track_id": 2, "frame_id": 10, "observation_quality": 0.7},
    ])
    write_jsonl(tmp_path / "recognition.jsonl", [
        {"video_id": "V1", "track_id": 1, "frame_id": 20, "fused_hull_number": "A12", "verified_identity": "A12", "identity_state": "confirmed"},
        {"video_id": "V1", "track_id": 2, "frame_id": 20, "fused_hull_number": "A12", "verified_identity": "A12", "identity_state": "confirmed"},
    ])
    write_jsonl(tmp_path / "edge_cloud.jsonl", [
        {"video_id": "V1", "track_id": 1, "total_latency_ms": 100, "upload_bytes": 20, "request_count": 2, "success": True},
        {"video_id": "V1", "track_id": 2, "total_latency_ms": 200, "upload_bytes": 20, "success": True},
    ])
    write_jsonl(tmp_path / "actions.jsonl", [])
    annotations = {
        ("V1", 1): {"known_or_unknown": "known", "hull_number": "A12", "hull_visible": True, "frame_start": 10},
        ("V1", 2): {"known_or_unknown": "unknown", "hull_number": "", "frame_start": 10},
    }
    metrics = evaluate(tmp_path, annotations)
    assert metrics["hull_recognition_success_rate"] == 1.0
    assert metrics["archive_matching_precision"] == 0.5
    assert metrics["archive_matching_success_rate"] == 1.0
    assert metrics["unknown_false_acceptance_rate"] == 1.0
    assert metrics["false_match_rate"] == 0.5
    assert metrics["vlm_calls"] == 3
    assert metrics["avg_vlm_calls_per_track"] == 1.5
    assert metrics["avg_recognition_latency_ms"] == 150.0
    assert metrics["avg_confirmation_frames"] == 10.0


def test_annotations_filter_excluded_run_records_from_all_track_metrics(tmp_path):
    write_jsonl(tmp_path / "observations.jsonl", [
        {"video_id": "V1", "track_id": 1, "frame_id": 1, "observation_quality": 0.8},
        {"video_id": "V2", "track_id": 2, "frame_id": 1, "observation_quality": 0.1},
    ])
    write_jsonl(tmp_path / "recognition.jsonl", [
        {"video_id": "V1", "track_id": 1, "frame_id": 2, "verified_identity": "003", "identity_state": "confirmed"},
        {"video_id": "V2", "track_id": 2, "frame_id": 2, "verified_identity": "", "identity_state": "out_of_archive"},
    ])
    write_jsonl(tmp_path / "edge_cloud.jsonl", [
        {"video_id": "V1", "track_id": 1, "total_latency_ms": 100, "upload_bytes": 20, "request_count": 1, "success": True},
        {"video_id": "V2", "track_id": 2, "total_latency_ms": 900, "upload_bytes": 200, "request_count": 4, "success": True},
    ])
    write_jsonl(tmp_path / "actions.jsonl", [
        {"video_id": "V2", "track_id": 2, "action": "escalate", "reasons": []},
    ])
    annotations = {
        ("V1", 1): {"known_or_unknown": "known", "hull_number": "003", "hull_visible": False},
    }

    metrics = evaluate(tmp_path, annotations)

    assert metrics["tracks_evaluated"] == 1
    assert metrics["tracks_with_recognition"] == 1
    assert metrics["confirmed_target_rate"] == 1.0
    assert metrics["out_of_archive_target_rate"] == 0.0
    assert metrics["vlm_calls"] == 1
    assert metrics["avg_recognition_latency_ms"] == 100.0
    assert metrics["mean_observation_quality"] == 0.8
    assert metrics["false_escalation_count"] == 0


def test_out_of_archive_is_resolved_not_unresolved(tmp_path):
    write_jsonl(tmp_path / "observations.jsonl", [
        {"video_id": "V1", "track_id": 1, "frame_id": 1, "observation_quality": 0.8},
        {"video_id": "V1", "track_id": 2, "frame_id": 1, "observation_quality": 0.8},
    ])
    write_jsonl(tmp_path / "recognition.jsonl", [
        {"video_id": "V1", "track_id": 1, "frame_id": 2, "verified_identity": "002", "identity_state": "structure_verified"},
        {"video_id": "V1", "track_id": 2, "frame_id": 2, "verified_identity": "", "identity_state": "out_of_archive"},
    ])
    write_jsonl(tmp_path / "edge_cloud.jsonl", [])
    write_jsonl(tmp_path / "actions.jsonl", [])

    metrics = evaluate(tmp_path)

    assert metrics["confirmed_target_rate"] == 0.5
    assert metrics["out_of_archive_target_rate"] == 0.5
    assert metrics["resolved_target_rate"] == 1.0
    assert metrics["unresolved_target_rate"] == 0.0


def test_uncertain_unknown_is_not_counted_as_successful_rejection(tmp_path):
    write_jsonl(tmp_path / "observations.jsonl", [
        {"video_id": "V1", "track_id": 1, "frame_id": 1, "observation_quality": 0.8},
        {"video_id": "V1", "track_id": 2, "frame_id": 1, "observation_quality": 0.8},
    ])
    write_jsonl(tmp_path / "recognition.jsonl", [
        {"video_id": "V1", "track_id": 1, "frame_id": 2, "verified_identity": "", "identity_state": "out_of_archive"},
        {"video_id": "V1", "track_id": 2, "frame_id": 2, "verified_identity": "", "identity_state": "uncertain"},
    ])
    write_jsonl(tmp_path / "edge_cloud.jsonl", [])
    write_jsonl(tmp_path / "actions.jsonl", [])
    annotations = {
        ("V1", 1): {"known_or_unknown": "unknown", "hull_number": ""},
        ("V1", 2): {"known_or_unknown": "unknown", "hull_number": ""},
    }

    metrics = evaluate(tmp_path, annotations)

    assert metrics["unknown_false_acceptance_rate"] == 0.0
    assert metrics["unresolved_unknown_rate"] == 0.5
    assert metrics["unknown_rejection_recall"] == 0.5
    assert metrics["unknown_rejection_precision"] == 1.0
    assert metrics["unknown_rejection_f1"] == 0.666667


def test_unresolved_known_is_not_a_false_unknown_rejection(tmp_path):
    write_jsonl(tmp_path / "observations.jsonl", [
        {"video_id": "V1", "track_id": 1, "frame_id": 1, "observation_quality": 0.8},
        {"video_id": "V1", "track_id": 2, "frame_id": 1, "observation_quality": 0.8},
    ])
    write_jsonl(tmp_path / "recognition.jsonl", [
        {"video_id": "V1", "track_id": 1, "frame_id": 2, "verified_identity": "", "identity_state": "uncertain"},
        {"video_id": "V1", "track_id": 2, "frame_id": 2, "verified_identity": "", "identity_state": "out_of_archive"},
    ])
    write_jsonl(tmp_path / "edge_cloud.jsonl", [])
    write_jsonl(tmp_path / "actions.jsonl", [])
    annotations = {
        ("V1", 1): {"known_or_unknown": "known", "hull_number": "002", "hull_visible": False},
        ("V1", 2): {"known_or_unknown": "unknown", "hull_number": ""},
    }

    metrics = evaluate(tmp_path, annotations)

    assert metrics["unrecognized_known_rate"] == 1.0
    assert metrics["known_false_rejection_rate"] == 0.0
    assert metrics["unknown_rejection_precision"] == 1.0
