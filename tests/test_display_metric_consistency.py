import json

from experiments.evaluate import evaluate
from pipeline.archive import decide_identity
from pipeline.demo import DemoRenderer
from pipeline.tracker import TrackInfo


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_cosine_score_082_requires_repeated_evidence_below_strong_single_threshold():
    decision = decide_identity(
        {"002": 0.82},
        in_archive_threshold=0.80,
        uncertain_threshold=0.70,
        min_margin=0.0,
        evidence_count=1,
        min_structure_observations=1,
    )
    assert decision.state == "uncertain"
    assert decision.identity == "002"


def test_video_and_metrics_share_verified_identity(tmp_path):
    track = TrackInfo(
        track_id=1,
        recognized=True,
        verified_identity="002",
        archive_candidate_id="002",
        archive_similarity_score=0.82,
        identity_state="structure_verified",
        identity_evidence_source="structure_semantic",
        db_matched=True,
    )
    lines = DemoRenderer._display_lines(track)
    assert "Verified identity: 002" in lines
    assert DemoRenderer._color(track) == (0, 200, 0)

    _write_jsonl(tmp_path / "observations.jsonl", [{"video_id": "V1", "track_id": 1, "frame_id": 1, "observation_quality": 0.8}])
    _write_jsonl(tmp_path / "recognition.jsonl", [{"video_id": "V1", "track_id": 1, "frame_id": 2, "fused_hull_number": "", "verified_identity": "002", "identity_state": "structure_verified"}])
    _write_jsonl(tmp_path / "edge_cloud.jsonl", [])
    _write_jsonl(tmp_path / "actions.jsonl", [])
    metrics = evaluate(tmp_path, {("V1", 1): {"known_or_unknown": "known", "hull_number": "002", "hull_visible": False, "frame_start": 1}})
    assert metrics["archive_matching_success_rate"] == 1.0
    assert metrics["identity_verification_accuracy"] == 1.0
    assert metrics["hull_recognition_success_rate"] == 0.0
