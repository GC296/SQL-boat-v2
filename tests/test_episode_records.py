import json

from experiments.build_episode_records import build_episode_records, write_episode_records


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_episode_builder_joins_agent_streams_into_review_record(tmp_path):
    write_jsonl(tmp_path / "entities.jsonl", [{
        "video_id": "V1",
        "entity_id": "E1",
        "member_track_ids": [1, 2],
        "identity_state": "review_requested",
    }])
    write_jsonl(tmp_path / "observations.jsonl", [{
        "video_id": "V1", "entity_id": "E1", "track_id": 2, "frame_id": 10,
        "bbox": [1, 2, 30, 40], "observation_quality": 0.8,
    }])
    write_jsonl(tmp_path / "actions.jsonl", [
        {
            "video_id": "V1", "entity_id": "E1", "track_id": 2, "frame_id": 10,
            "action": "verify_archive_identity", "should_query": True,
            "skill_signals": {"identity_uncertainty": 0.9, "identity_state": "conflicting"},
        },
        {
            "video_id": "V1", "entity_id": "E1", "track_id": 2, "frame_id": 20,
            "action": "request_remote_verification", "should_query": False, "score": 0.7,
            "reasons": ["active_query_budget_exhausted"], "identity_state_before": "conflicting",
            "skill_signals": {"identity_uncertainty": 0.6, "query_budget_used": 3, "query_budget_max": 3},
        },
    ])
    write_jsonl(tmp_path / "recognition.jsonl", [{
        "video_id": "V1", "entity_id": "E1", "track_id": 2, "frame_id": 10,
        "raw_hull_number": "012", "fused_hull_number": "012", "identity_state": "conflicting",
        "uncertainty": 0.3, "observed_structure_description": "white patrol vessel",
        "semantic_matches": [{"hull_number": "012", "score": 0.82}],
        "visual_matches": [{"hull_number": "012", "score": 0.76, "image_path": "archive/012.jpg"}],
    }])
    write_jsonl(tmp_path / "edge_cloud.jsonl", [{
        "video_id": "V1", "entity_id": "E1", "track_id": 2, "frame_id": 10,
        "request_count": 1, "upload_bytes": 120, "total_latency_ms": 300,
        "target_image_path": "evidence/targets/V1_E1.jpg",
    }])
    write_jsonl(tmp_path / "visual.jsonl", [])

    episodes, reviews = build_episode_records(tmp_path)

    assert len(episodes) == len(reviews) == 1
    assert episodes[0]["identity_state"] == "review_requested"
    assert episodes[0]["evidence_gain_per_action"] == 0.6
    assert episodes[0]["query_count"] == 1
    assert reviews[0]["target_view"]["image_path"] == "evidence/targets/V1_E1.jpg"
    assert reviews[0]["archive_image_paths"] == ["archive/012.jpg"]
    assert reviews[0]["completeness"] == 1.0

    counts = write_episode_records(tmp_path)
    assert counts == {"episodes": 1, "review_records": 1}
    assert (tmp_path / "episodes.jsonl").exists()
    assert (tmp_path / "review_records.jsonl").exists()
