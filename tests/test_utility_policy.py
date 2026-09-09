import json

from experiments.build_policy_utility_dataset import build_utility_dataset
from experiments.train_utility_policy import train_utility_policy
from pipeline.learned_policy import LearnedPolicyModel
from pipeline.policy import LookoutPolicy
from pipeline.tracker import TrackInfo


def _sample(video_id, index, truth_status, truth_hull, features, evidence):
    return {
        "sample_id": f"{video_id}:{index}",
        "video_id": video_id,
        "entity_id": f"E{index}",
        "track_id": index,
        "frame_id": index,
        "features": features,
        "oracle_action": "query",
        "teacher_action": "query",
        "truth": {
            "known_or_unknown": truth_status,
            "hull_number": truth_hull,
        },
        "evidence": evidence,
    }


def _templates():
    return {
        "defer": _sample(
            "V1", 1, "unknown", "",
            {"observation_quality": 0.05, "identity_uncertainty": 0.9, "query_budget_used": 0, "query_budget_max": 3},
            {"identity_state_before": "unknown"},
        ),
        "query": _sample(
            "V2", 2, "known", "012",
            {"observation_quality": 0.8, "identity_uncertainty": 0.9, "evidence_opportunity_score": 0.9, "query_budget_used": 0, "query_budget_max": 3},
            {"identity_state_before": "unknown"},
        ),
        "stop_known": _sample(
            "V3", 3, "known", "012",
            {"observation_quality": 0.8, "identity_uncertainty": 0.1, "structure_score": 0.95, "visual_similarity_score": 0.9, "visual_margin": 0.3, "text_visual_agree": 1, "query_budget_used": 1, "query_budget_max": 3},
            {"recognition_identity_state": "structure_verified", "verified_identity": "012", "archive_candidate_id": "012", "visual_candidate_id": "012"},
        ),
        "stop_out_of_archive": _sample(
            "V4", 4, "unknown", "",
            {"observation_quality": 0.8, "identity_uncertainty": 0.2, "structure_score": 0.1, "visual_similarity_score": 0.1, "visual_margin": 0.03, "query_budget_used": 1, "query_budget_max": 3},
            {"recognition_identity_state": "out_of_archive"},
        ),
        "escalate_review": _sample(
            "V5", 5, "known", "012",
            {"observation_quality": 0.8, "identity_uncertainty": 0.9, "structure_score": 0.45, "visual_similarity_score": 0.45, "visual_margin": 0.02, "query_budget_used": 3, "query_budget_max": 3, "query_budget_fraction": 1.0},
            {"recognition_identity_state": "conflicting", "archive_candidate_id": "003", "visual_candidate_id": "012"},
        ),
    }


def test_build_policy_utility_dataset_covers_all_actions(tmp_path):
    dataset = tmp_path / "policy.jsonl"
    with dataset.open("w", encoding="utf-8") as handle:
        for sample in _templates().values():
            handle.write(json.dumps(sample) + "\n")
    output = tmp_path / "utility.jsonl"

    summary = build_utility_dataset(dataset, output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert {row["utility_action"] for row in rows} == set(_templates())
    assert summary["samples"] == 5
    assert all("utility_by_action" in row for row in rows)


def test_visual_only_utilities_stop_on_repeated_match_and_weak_unknown(tmp_path):
    known = _sample(
        "V6", 6, "known", "012",
        {
            "observation_quality": 0.8,
            "identity_uncertainty": 0.6,
            "evidence_opportunity_score": 0.7,
            "query_budget_used": 1,
            "query_budget_max": 3,
            "visual_similarity_score": 0.88,
            "visual_margin": 0.18,
            "visual_consistent_observations": 2,
            "visual_only_mode": 1,
        },
        {"recognition_identity_state": "uncertain", "visual_candidate_id": "012"},
    )
    unknown = _sample(
        "V7", 7, "unknown", "",
        {
            "observation_quality": 0.8,
            "identity_uncertainty": 0.7,
            "evidence_opportunity_score": 0.7,
            "query_budget_used": 1,
            "query_budget_max": 3,
            "structure_score": 0.95,
                "visual_similarity_score": 0.45,
                "visual_margin": 0.05,
                "visual_consistent_observations": 1,
                "visual_observation_count": 2,
                "visual_low_score_observations": 2,
                "visual_only_mode": 1,
        },
        {"recognition_identity_state": "uncertain", "visual_candidate_id": "003"},
    )

    dataset = tmp_path / "visual_only.jsonl"
    with dataset.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(known) + "\n")
        handle.write(json.dumps(unknown) + "\n")
    output = tmp_path / "visual_only_utility.jsonl"

    build_utility_dataset(dataset, output)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert rows[0]["utility_action"] == "stop_known"
    assert rows[1]["utility_action"] == "stop_out_of_archive"


def test_train_utility_policy_writes_runtime_compatible_model(tmp_path):
    base = list(_templates().values())
    utility_input = tmp_path / "utility.jsonl"
    raw_input = tmp_path / "raw.jsonl"
    with raw_input.open("w", encoding="utf-8") as handle:
        for repeat in range(12):
            for sample in base:
                row = {**sample, "video_id": f"{sample['video_id']}_{repeat % 4}", "sample_id": f"{sample['sample_id']}:{repeat}"}
                handle.write(json.dumps(row) + "\n")
    build_utility_dataset(raw_input, utility_input)

    output = tmp_path / "utility_policy.json"
    summary = train_utility_policy(utility_input, output, validation_fraction=0.0, l2=0.01)
    model = LearnedPolicyModel.load(output)
    action, _, _ = model.predict(_templates()["stop_known"]["features"])

    assert summary["train_metrics"]["accuracy"] >= 0.8
    assert action == "stop_known"
    assert model.payload["model_type"] == "linear_utility_regression"


def test_learned_utility_mode_uses_same_runtime_action_interface(tmp_path):
    actions = ["defer", "query", "stop_known", "stop_out_of_archive", "escalate_review"]
    bias = [0.0] * len(actions)
    bias[actions.index("query")] = 5.0
    model_path = tmp_path / "utility_policy.json"
    model_path.write_text(json.dumps({
        "actions": actions,
        "feature_names": ["observation_quality"],
        "feature_mean": [0.0],
        "feature_std": [1.0],
        "weights": [[0.0] * len(actions)],
        "bias": bias,
        "model_type": "linear_utility_regression",
    }), encoding="utf-8")
    track = TrackInfo(track_id=1, observation_quality=0.8, identity_state="unknown")

    decision = LookoutPolicy({
        "mode": "learned_utility",
        "learned_model_path": str(model_path),
    }).decide(track, 10)

    assert decision.should_query
    assert decision.action == "cloud_vlm_read"
    assert decision.skill_signals["learned_policy_family"] == "utility"


def test_runtime_loads_mlp_utility_policy_without_torch(tmp_path):
    actions = ["defer", "query", "stop_known", "stop_out_of_archive", "escalate_review"]
    model_path = tmp_path / "mlp_policy.json"
    model_path.write_text(json.dumps({
        "actions": actions,
        "feature_names": ["observation_quality"],
        "feature_mean": [0.0],
        "feature_std": [1.0],
        "model_type": "mlp_utility_regression",
        "layers": [
            {
                "weights": [[1.0, -1.0]],
                "bias": [0.0, 0.0],
                "activation": "relu",
            },
            {
                "weights": [
                    [0.0, 5.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                "bias": [0.0, 0.0, 0.0, 0.0, 0.0],
                "activation": "linear",
            },
        ],
    }), encoding="utf-8")

    model = LearnedPolicyModel.load(model_path)
    action, score, probabilities = model.predict({"observation_quality": 1.0})

    assert model.payload["model_type"] == "mlp_utility_regression"
    assert action == "query"
    assert score == probabilities["query"]
