import json

from experiments.train_learned_policy import _split_by_video, train_policy
from pipeline.learned_policy import LearnedPolicyModel


def _sample(video, index, label, features):
    return {
        "sample_id": f"{video}:{index}",
        "video_id": video,
        "entity_id": f"E{index}",
        "track_id": index,
        "frame_id": index,
        "features": features,
        "oracle_action": label,
        "teacher_action": label,
    }


def test_train_learned_policy_writes_loadable_model(tmp_path):
    rows = []
    templates = {
        "defer": {"observation_quality": 0.05, "identity_uncertainty": 0.8, "query_budget_fraction": 0.0},
        "query": {"observation_quality": 0.8, "identity_uncertainty": 0.9, "query_budget_fraction": 0.0},
        "stop_known": {"observation_quality": 0.8, "identity_uncertainty": 0.1, "structure_score": 0.95, "text_visual_agree": 1},
        "stop_out_of_archive": {"observation_quality": 0.8, "identity_uncertainty": 0.2, "structure_score": 0.1, "visual_similarity_score": 0.1},
        "escalate_review": {"observation_quality": 0.8, "identity_uncertainty": 0.8, "query_budget_fraction": 1.0},
    }
    for action, features in templates.items():
        for index in range(20):
            rows.append(_sample(f"V{index % 5}", index, action, features))
    dataset = tmp_path / "policy.jsonl"
    with dataset.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    output = tmp_path / "learned_policy.json"
    summary = train_policy(dataset, output, validation_fraction=0.0, epochs=300, learning_rate=0.1)
    model = LearnedPolicyModel.load(output)
    action, score, probabilities = model.predict(templates["stop_known"])

    assert summary["train_metrics"]["accuracy"] >= 0.95
    assert action == "stop_known"
    assert score == probabilities["stop_known"]
    assert output.with_suffix(".summary.json").exists()


def test_grouped_split_balances_action_distribution_without_video_leakage():
    rows = []
    for video_index in range(20):
        video = f"V{video_index:03d}"
        stop_known_count = 80 if video_index in {0, 1, 2, 3} else 2
        for index in range(20):
            rows.append(_sample(video, index, "stop_out_of_archive", {}))
        for index in range(stop_known_count):
            rows.append(_sample(video, 100 + index, "stop_known", {}))

    train, validation, validation_videos = _split_by_video(
        rows, 0.2, 7, target="oracle_action"
    )

    assert set(row["video_id"] for row in train).isdisjoint(validation_videos)
    assert set(row["video_id"] for row in validation) == set(validation_videos)
    total_known_rate = sum(row["oracle_action"] == "stop_known" for row in rows) / len(rows)
    validation_known_rate = sum(row["oracle_action"] == "stop_known" for row in validation) / len(validation)
    assert abs(validation_known_rate - total_known_rate) < 0.15
