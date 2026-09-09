import json

import yaml

from experiments.calibrate_visual_reranker import calibrate_thresholds, write_frozen_config


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_calibration_uses_entity_members_and_writes_frozen_config(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_jsonl(
        run_dir / "visual.jsonl",
        [
            {"video_id": "V1", "track_id": 1, "visual_candidate_id": "012", "visual_similarity_score": 0.82, "visual_margin": 0.22},
            {"video_id": "V2", "track_id": 2, "visual_candidate_id": "003", "visual_similarity_score": 0.76, "visual_margin": 0.18},
            {"video_id": "V3", "track_id": 3, "visual_candidate_id": "012", "visual_similarity_score": 0.58, "visual_margin": 0.07},
            {"video_id": "V4", "track_id": 4, "visual_candidate_id": "003", "visual_similarity_score": 0.44, "visual_margin": 0.02},
        ],
    )
    annotations = tmp_path / "annotations.jsonl"
    _write_jsonl(
        annotations,
        [
            {"video_id": "V1", "member_track_ids": [1], "known_or_unknown": "known", "hull_number": "012"},
            {"video_id": "V2", "member_track_ids": [2], "known_or_unknown": "known", "hull_number": "003"},
            {"video_id": "V3", "member_track_ids": [3], "known_or_unknown": "unknown", "hull_number": ""},
            {"video_id": "V4", "member_track_ids": [4], "known_or_unknown": "unknown", "hull_number": ""},
        ],
    )

    summary = calibrate_thresholds(
        run_dir,
        annotations,
        max_ufar=0.0,
        score_step=0.01,
        margin_step=0.01,
    )

    assert summary["matched_observations"] == 4
    assert summary["selected"]["KAcc"] == 1.0
    assert summary["selected"]["UFAR"] == 0.0
    assert summary["selected"]["ATS"] == 1.0
    assert summary["objective"] == "kacc"
    assert summary["selected"]["balanced_accuracy"] == 1.0
    assert summary["known_entities"] == 2
    assert summary["unknown_entities"] == 2
    assert summary["raw_known_top1_correct"] == 2
    assert summary["raw_known_top1_accuracy"] == 1.0

    base_config = tmp_path / "base.yaml"
    base_config.write_text("experiment:\n  visual_archive:\n    enabled: false\n", encoding="utf-8")
    output_config = tmp_path / "frozen.yaml"
    write_frozen_config(base_config, output_config, summary["selected"])
    frozen = yaml.safe_load(output_config.read_text(encoding="utf-8"))
    visual = frozen["experiment"]["visual_archive"]
    assert visual["backend"] == "qwen_vl_reranker"
    assert visual["min_support_score"] == summary["selected"]["score_threshold"]
    assert visual["out_of_archive_margin"] == summary["selected"]["margin_threshold"]
