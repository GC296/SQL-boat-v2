import json
from experiments.bootstrap_annotations import build_drafts


def test_build_track_annotation_drafts(tmp_path):
    records = [
        {"video_id": "V1", "track_id": 2, "frame_id": 10, "bbox": [1, 2, 30, 40], "confidence": 0.7, "observation_quality": 0.4},
        {"video_id": "V1", "track_id": 2, "frame_id": 20, "bbox": [2, 3, 32, 42], "confidence": 0.8, "observation_quality": 0.9},
    ]
    with (tmp_path / "observations.jsonl").open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item) + "\n")
    (tmp_path / "recognition.jsonl").write_text(json.dumps({"video_id": "V1", "track_id": 2, "frame_id": 20, "fused_hull_number": "A12", "identity_state": "probable", "risk_level": "low"}) + "\n", encoding="utf-8")
    drafts = build_drafts(tmp_path)
    assert len(drafts) == 1
    assert drafts[0]["frame_start"] == 10
    assert drafts[0]["frame_end"] == 20
    assert drafts[0]["best_frame_id"] == 20
    assert drafts[0]["predicted_hull_number"] == "A12"
    assert drafts[0]["annotation_status"] == "pending"
