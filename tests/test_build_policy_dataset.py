import json

from experiments.build_policy_dataset import build_policy_dataset


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_build_policy_dataset_labels_actions_without_same_frame_leakage(tmp_path):
    write_jsonl(tmp_path / "entities.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "member_track_ids": [1]},
        {"video_id": "V2", "entity_id": "E2", "member_track_ids": [2]},
    ])
    write_jsonl(tmp_path / "actions.jsonl", [
        {
            "video_id": "V1", "entity_id": "E1", "track_id": 1, "frame_id": 5,
            "action": "wait_better_view", "should_query": False,
            "identity_state_before": "unknown", "identity_state_after": "unknown",
            "skill_signals": {"observation_quality": 0.1, "query_budget_used": 0, "query_budget_max": 3},
        },
        {
            "video_id": "V1", "entity_id": "E1", "track_id": 1, "frame_id": 10,
            "action": "cloud_vlm_read", "should_query": True,
            "identity_state_before": "unknown", "identity_state_after": "unknown",
            "skill_signals": {
                "observation_quality": 0.8,
                "query_budget_used": 0,
                "query_budget_max": 3,
                "visual_observation_count": 1,
                "visual_consistent_observations": 1,
                "visual_low_score_observations": 0,
            },
        },
        {
            "video_id": "V1", "entity_id": "E1", "track_id": 1, "frame_id": 20,
            "action": "continue_tracking", "should_query": False,
            "identity_state_before": "structure_verified", "identity_state_after": "structure_verified",
            "skill_signals": {"observation_quality": 0.7, "query_budget_used": 1, "query_budget_max": 3},
        },
        {
            "video_id": "V2", "entity_id": "E2", "track_id": 2, "frame_id": 20,
            "action": "monitor_unknown", "should_query": False,
            "identity_state_before": "out_of_archive", "identity_state_after": "out_of_archive",
            "skill_signals": {"observation_quality": 0.7, "query_budget_used": 1, "query_budget_max": 3},
        },
    ])
    write_jsonl(tmp_path / "recognition.jsonl", [
        {
            "video_id": "V1", "entity_id": "E1", "track_id": 1, "frame_id": 10,
            "verified_identity": "", "archive_candidate_id": "012", "identity_state": "uncertain",
            "archive_similarity_score": 0.9, "structure_candidate_scores": {"012": 0.88},
            "visual_candidate_id": "012", "visual_similarity_score": 0.88, "visual_margin": 0.32,
            "visual_consistent_observations": 2,
            "identity_decision_mode": "visual_only",
        },
        {
            "video_id": "V2", "entity_id": "E2", "track_id": 2, "frame_id": 10,
            "verified_identity": "", "archive_candidate_id": "", "identity_state": "out_of_archive",
            "archive_similarity_score": 0.2, "structure_candidate_scores": {},
            "visual_candidate_id": "012", "visual_similarity_score": 0.3, "visual_margin": 0.05,
        },
    ])
    write_jsonl(tmp_path / "visual.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "track_id": 1, "frame_id": 10, "visual_candidate_id": "012", "visual_similarity_score": 0.88, "visual_margin": 0.32},
        {"video_id": "V2", "entity_id": "E2", "track_id": 2, "frame_id": 10, "visual_candidate_id": "012", "visual_similarity_score": 0.3, "visual_margin": 0.05},
    ])
    write_jsonl(tmp_path / "annotations.jsonl", [
        {"video_id": "V1", "entity_id": "V1_GT_01", "member_track_ids": [1], "known_or_unknown": "known", "hull_number": "012", "vessel_id": "known boat"},
        {"video_id": "V2", "entity_id": "V2_GT_01", "member_track_ids": [2], "known_or_unknown": "unknown", "hull_number": "", "vessel_id": "unknown boat"},
    ])

    output = tmp_path / "policy.jsonl"
    summary = build_policy_dataset(tmp_path, tmp_path / "annotations.jsonl", output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert summary["samples"] == 4
    assert [row["oracle_action"] for row in rows] == ["defer", "query", "stop_known", "stop_out_of_archive"]
    assert rows[1]["features"]["structure_score"] == 0.0
    assert rows[1]["features"]["visual_observation_count"] == 1
    assert rows[1]["features"]["visual_consistent_observations"] == 1
    assert rows[2]["features"]["structure_score"] == 0.0
    assert rows[2]["features"]["visual_only_mode"] == 1
    assert rows[2]["features"]["text_visual_agree"] == 0
    assert (tmp_path / "policy.summary.json").exists()
