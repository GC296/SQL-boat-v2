from config import load_config


def test_project_archive_thresholds_preserve_calibrated_debug_setting():
    config = load_config("config.yaml")
    archive = config["experiment"]["archive"]
    visual = config["experiment"]["visual_archive"]
    assert visual["identity_decision_mode"] == "fused"
    assert visual["visual_only_decision_rule"] == "score_gate"
    assert visual["visual_only_min_score"] == 0.80
    assert visual["visual_only_min_margin"] == 0.03
    assert visual["visual_only_strong_score"] == 0.90
    assert visual["visual_only_reject_score"] == 0.80
    assert visual["visual_only_reject_margin"] == 0.08
    assert archive["structure_in_archive_threshold"] == 0.735
    assert archive["structure_uncertain_threshold"] == 0.70
    assert archive["min_structure_observations"] == 1
    assert archive["strong_single_threshold"] == 0.88
    assert archive["min_consistent_observations"] == 2
    assert archive["min_consistency_ratio"] == 0.67
