import json
import re
from pathlib import Path

import cv2
import numpy as np

from experiments.generate_case_studies import (
    _choose_recognition,
    _english_structure_description,
    generate_case_studies,
)


def _write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_generates_three_self_contained_case_reports(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    image_path = tmp_path / "evidence.jpg"
    cv2.imwrite(str(image_path), np.full((180, 320, 3), (42, 118, 178), dtype=np.uint8))
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [
        {"video_id": "V1", "video_path": str(tmp_path / "missing.mp4")},
        {"video_id": "V2", "video_path": str(tmp_path / "missing.mp4")},
        {"video_id": "V3", "video_path": str(tmp_path / "missing.mp4")},
    ])
    _write_jsonl(run_dir / "recognition.jsonl", [
        {
            "video_id": "V1", "entity_id": "E1", "track_id": 2, "member_track_ids": [1, 2], "frame_id": 100,
            "verified_identity": "012", "archive_candidate_id": "012", "archive_similarity_score": 0.94,
            "identity_state": "structure_verified", "raw_hull_number": "012", "target_image_path": str(image_path),
            "observed_structure_description": "黄色船体配黑色上层建筑，船首有白色浮筒，船尾装有推进器，船身中部有圆形标志。",
            "visual_similarity_score": 0.91, "visual_margin": 0.48,
            "visual_matches": [{"hull_number": "012", "score": 0.91, "image_path": str(image_path)}],
            "identity_evidence_source": "structure_visual",
        },
        {
            "video_id": "V2", "entity_id": "E2", "track_id": 3, "member_track_ids": [3], "frame_id": 120,
            "archive_candidate_id": "012", "archive_similarity_score": 0.62, "identity_state": "out_of_archive",
            "target_image_path": str(image_path), "observed_structure_description": "Blue compact workboat with an open canopy.",
            "visual_similarity_score": 0.41, "visual_margin": 0.03,
            "visual_matches": [{"hull_number": "012", "score": 0.41, "image_path": str(image_path)}],
            "identity_evidence_source": "visual_open_set",
        },
        {
            "video_id": "V3", "entity_id": "E3", "track_id": 4, "member_track_ids": [4, 5], "frame_id": 150,
            "archive_candidate_id": "003", "archive_similarity_score": 0.81, "identity_state": "conflicting",
            "target_image_path": str(image_path), "observed_structure_description": "Yellow hull but inconsistent number evidence.",
            "visual_similarity_score": 0.55, "visual_margin": 0.01,
            "visual_matches": [{"hull_number": "003", "score": 0.55, "image_path": str(image_path)}],
            "identity_evidence_source": "conflicting_sources",
        },
    ])
    _write_jsonl(run_dir / "entity_episodes.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "member_track_ids": [1, 2], "terminal_state": "confirmed", "verified_identity": "012", "uncertainty": 0.08},
        {"video_id": "V2", "entity_id": "E2", "member_track_ids": [3], "terminal_state": "out_of_archive", "uncertainty": 0.12},
        {"video_id": "V3", "entity_id": "E3", "member_track_ids": [4, 5], "state": "review_requested", "escalated": True, "uncertainty": 0.72},
    ])
    _write_jsonl(run_dir / "entity_actions.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "track_id": 2, "frame_id": 100, "action": "verify_identity", "status": "completed", "next_state": "confirmed", "rationale": "Cross-frame evidence converged."},
        {"video_id": "V2", "entity_id": "E2", "track_id": 3, "frame_id": 120, "action": "terminate", "status": "completed", "next_state": "out_of_archive", "rationale": "Archive support remained insufficient."},
        {"video_id": "V3", "entity_id": "E3", "track_id": 4, "frame_id": 150, "action": "escalate", "status": "completed", "previous_state": "conflicting", "next_state": "review_requested", "rationale": "Conflict persisted after budget exhaustion."},
    ])
    _write_jsonl(run_dir / "review_queue.jsonl", [
        {"video_id": "V3", "entity_id": "E3", "status": "delivered", "package": {"entity_id": "E3", "member_track_ids": [4, 5], "identity_state": "conflicting", "reasons": ["query_budget_exhausted"], "evidence_image": str(image_path)}},
    ])

    results = generate_case_studies(run_dir, manifest, render_pdfs=False)

    assert [result["case_type"] for result in results] == ["known", "unknown", "conflict"]
    for result in results:
        report = result["html"].read_text(encoding="utf-8")
        assert "data:image/jpeg;base64," in report
        assert str(image_path) not in report
        assert "Attributed historical evidence" in report
        assert "Hull and structural evidence" in report
        assert not re.search(r"[\u3400-\u9fff]", report)

    conflict_report = results[2]["html"].read_text(encoding="utf-8")
    assert "Leading archive candidate" in conflict_report
    assert "Candidate identity 003" in conflict_report

    conflict_only = generate_case_studies(
        run_dir,
        manifest,
        output_dir=tmp_path / "conflict-only",
        selectors={"conflict": "V3:E3"},
        render_pdfs=False,
        case_types=("conflict",),
    )
    assert len(conflict_only) == 1
    assert conflict_only[0]["case_type"] == "conflict"
    assert conflict_only[0]["html"].name.startswith("case_03_conflict_")


def test_supports_original_episode_streams_and_relative_review_images(tmp_path):
    run_dir = tmp_path / "run"
    evidence_dir = run_dir / "evidence" / "targets"
    evidence_dir.mkdir(parents=True)
    image_path = evidence_dir / "conflict.jpg"
    cv2.imwrite(str(image_path), np.full((120, 220, 3), (35, 92, 160), dtype=np.uint8))
    manifest = tmp_path / "manifest.jsonl"
    _write_jsonl(manifest, [
        {"video_id": video_id, "video_path": str(tmp_path / "missing.mp4")}
        for video_id in ("V1", "V2", "V3")
    ])
    _write_jsonl(run_dir / "recognition.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "track_id": 1, "frame_id": 10, "verified_identity": "012", "identity_state": "confirmed", "target_image_path": str(image_path)},
        {"video_id": "V2", "entity_id": "E2", "track_id": 2, "frame_id": 20, "identity_state": "out_of_archive", "target_image_path": str(image_path)},
        {"video_id": "V3", "entity_id": "E3", "track_id": 3, "frame_id": 30, "identity_state": "conflicting"},
    ])
    _write_jsonl(run_dir / "episodes.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "member_track_ids": [1, 4], "identity_state": "confirmed", "verified_identity": "012"},
        {"video_id": "V2", "entity_id": "E2", "member_track_ids": [2], "identity_state": "out_of_archive"},
        {"video_id": "V3", "entity_id": "E3", "member_track_ids": [3], "identity_state": "review_requested", "review_requested": True},
    ])
    _write_jsonl(run_dir / "actions.jsonl", [
        {"video_id": "V3", "entity_id": "E3", "track_id": 3, "frame_id": 30, "action": "request_remote_verification", "identity_state_before": "conflicting", "reasons": ["active_query_budget_exhausted"]},
    ])
    _write_jsonl(run_dir / "review_records.jsonl", [
        {
            "video_id": "V3", "entity_id": "E3", "identity_state": "conflicting", "reasons": ["active_query_budget_exhausted"],
            "target_view": {"track_id": 3, "frame_id": 30, "image_path": "evidence/targets/conflict.jpg"},
            "archive_image_paths": ["evidence/targets/conflict.jpg"],
        },
    ])

    results = generate_case_studies(run_dir, manifest, render_pdfs=False)

    conflict = next(result for result in results if result["case_type"] == "conflict")
    report = conflict["html"].read_text(encoding="utf-8")
    assert "data:image/jpeg;base64," in report
    assert "evidence/targets/conflict.jpg" not in report


def test_annotations_exclude_known_false_rejections_and_filter_archive_reference(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    image_path = tmp_path / "archive.jpg"
    cv2.imwrite(str(image_path), np.full((120, 220, 3), (60, 120, 180), dtype=np.uint8))
    manifest = tmp_path / "manifest.jsonl"
    annotations = tmp_path / "entities.jsonl"
    _write_jsonl(manifest, [
        {"video_id": video_id, "video_path": str(tmp_path / "missing.mp4")}
        for video_id in ("V1", "V2", "V3", "V4")
    ])
    _write_jsonl(annotations, [
        {"video_id": "V1", "entity_id": "GT1", "member_track_ids": [1], "known_or_unknown": "known", "hull_number": "012"},
        {"video_id": "V2", "entity_id": "GT2", "member_track_ids": [2], "known_or_unknown": "known", "hull_number": "003"},
        {"video_id": "V3", "entity_id": "GT3", "member_track_ids": [3], "known_or_unknown": "unknown", "hull_number": ""},
        {"video_id": "V4", "entity_id": "GT4", "member_track_ids": [4], "known_or_unknown": "unknown", "hull_number": ""},
    ])
    _write_jsonl(run_dir / "recognition.jsonl", [
        {
            "video_id": "V1", "entity_id": "E1", "track_id": 1, "member_track_ids": [1], "frame_id": 10,
            "verified_identity": "012", "archive_candidate_id": "012", "archive_similarity_score": 0.91,
            "identity_state": "confirmed", "target_image_path": str(image_path),
            "visual_matches": [
                {"hull_number": "003", "score": 0.95, "image_path": str(image_path)},
                {"hull_number": "012", "score": 0.88, "image_path": str(image_path)},
            ],
        },
        {"video_id": "V2", "entity_id": "E2", "track_id": 2, "member_track_ids": [2], "frame_id": 20, "identity_state": "out_of_archive", "archive_similarity_score": 0.99, "target_image_path": str(image_path)},
        {"video_id": "V3", "entity_id": "E3", "track_id": 3, "member_track_ids": [3], "frame_id": 30, "identity_state": "out_of_archive", "archive_candidate_id": "012", "archive_similarity_score": 0.82, "target_image_path": str(image_path), "visual_matches": [{"hull_number": "012", "score": 0.72, "image_path": str(image_path)}]},
        {"video_id": "V4", "entity_id": "E4", "track_id": 4, "member_track_ids": [4], "frame_id": 40, "identity_state": "conflicting", "archive_candidate_id": "003", "archive_similarity_score": 0.79, "target_image_path": str(image_path), "visual_matches": [{"hull_number": "003", "score": 0.70, "image_path": str(image_path)}]},
    ])
    _write_jsonl(run_dir / "visual.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "track_id": 1, "frame_id": 10, "visual_candidate_id": "012", "visual_similarity_score": 0.88, "visual_margin": 0.42, "visual_matches": [{"hull_number": "012", "score": 0.88, "image_path": str(image_path)}]},
    ])
    _write_jsonl(run_dir / "episodes.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "member_track_ids": [1], "identity_state": "confirmed", "verified_identity": "012"},
        {"video_id": "V2", "entity_id": "E2", "member_track_ids": [2], "identity_state": "out_of_archive"},
        {"video_id": "V3", "entity_id": "E3", "member_track_ids": [3], "identity_state": "out_of_archive"},
        {"video_id": "V4", "entity_id": "E4", "member_track_ids": [4], "identity_state": "review_requested", "review_requested": True},
    ])
    _write_jsonl(run_dir / "actions.jsonl", [
        {"video_id": "V4", "entity_id": "E4", "track_id": 4, "frame_id": 40, "action": "request_remote_verification", "identity_state_before": "conflicting"},
    ])

    results = generate_case_studies(run_dir, manifest, annotations=annotations, render_pdfs=False)

    by_type = {result["case_type"]: result for result in results}
    assert by_type["unknown"]["video_id"] == "V3"
    assert by_type["conflict"]["video_id"] == "V4"
    known_report = by_type["known"]["html"].read_text(encoding="utf-8")
    unknown_report = by_type["unknown"]["html"].read_text(encoding="utf-8")
    assert known_report.count('<div class="image-panel archive-panel">') == 1
    assert "Registered identity 003" not in known_report
    assert "0.880" in known_report
    assert "0.420" in known_report
    assert "Matched archive vessel" not in unknown_report
    conflict_report = by_type["conflict"]["html"].read_text(encoding="utf-8")
    assert "Conflicting Identity Evidence and Review" in conflict_report
    assert "REVIEW REQUESTED" in conflict_report


def test_uses_annotated_unresolved_unknown_when_no_conflict_exists(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    image_path = tmp_path / "evidence.jpg"
    cv2.imwrite(str(image_path), np.full((80, 140, 3), 120, dtype=np.uint8))
    manifest = tmp_path / "manifest.jsonl"
    annotations = tmp_path / "entities.jsonl"
    _write_jsonl(manifest, [
        {"video_id": video_id, "video_path": str(tmp_path / "missing.mp4")}
        for video_id in ("V1", "V2", "V3")
    ])
    _write_jsonl(annotations, [
        {"video_id": "V1", "member_track_ids": [1], "known_or_unknown": "known", "hull_number": "012"},
        {"video_id": "V2", "member_track_ids": [2], "known_or_unknown": "unknown", "hull_number": ""},
        {"video_id": "V3", "member_track_ids": [3], "known_or_unknown": "unknown", "hull_number": ""},
    ])
    _write_jsonl(run_dir / "recognition.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "track_id": 1, "member_track_ids": [1], "frame_id": 10, "identity_state": "confirmed", "verified_identity": "012", "target_image_path": str(image_path)},
        {"video_id": "V2", "entity_id": "E2", "track_id": 2, "member_track_ids": [2], "frame_id": 20, "identity_state": "out_of_archive", "archive_similarity_score": 0.65, "target_image_path": str(image_path)},
        {"video_id": "V3", "entity_id": "E3", "track_id": 3, "member_track_ids": [3], "frame_id": 30, "identity_state": "uncertain", "archive_candidate_id": "012", "archive_similarity_score": 0.84, "target_image_path": str(image_path)},
    ])
    _write_jsonl(run_dir / "episodes.jsonl", [
        {"video_id": "V1", "entity_id": "E1", "member_track_ids": [1], "identity_state": "confirmed", "verified_identity": "012"},
        {"video_id": "V2", "entity_id": "E2", "member_track_ids": [2], "identity_state": "out_of_archive"},
        {"video_id": "V3", "entity_id": "E3", "member_track_ids": [3], "identity_state": "uncertain", "uncertainty": 0.71},
    ])

    results = generate_case_studies(run_dir, manifest, annotations=annotations, render_pdfs=False)

    conflict = next(result for result in results if result["case_type"] == "conflict")
    assert conflict["video_id"] == "V3"
    report = conflict["html"].read_text(encoding="utf-8")
    assert "Persistent Uncertainty and Review" in report
    assert "UNRESOLVED / REVIEW" in report


def test_translates_retained_chinese_structure_text_to_english():
    translated = _english_structure_description(
        "黄色船体配黑色上层建筑，船首有白色浮筒，船尾装有推进器，船身中部有圆形标志。"
    )

    assert translated == (
        "A yellow hull with a black superstructure; a white fender at the bow; "
        "propulsion equipment at the stern; a circular marking amidships."
    )
    assert not re.search(r"[\u3400-\u9fff]", translated)


def test_review_request_without_incompatible_evidence_is_not_a_conflict():
    from experiments.generate_case_studies import _bundle_summary

    summary = _bundle_summary({
        "episodes": [{"identity_state": "review_requested", "review_requested": True}],
        "actions": [{"action": "request_remote_verification", "previous_state": "uncertain"}],
    })

    assert summary["review"] is True
    assert summary["conflict"] is False


def test_conflict_report_uses_the_recognition_that_triggered_conflict():
    selected = _choose_recognition({"recognition": [
        {
            "frame_id": 30,
            "identity_state": "conflicting",
            "structure_conflict_observations": 1,
            "archive_similarity_score": 0.81,
        },
        {
            "frame_id": 60,
            "identity_state": "uncertain",
            "structure_conflict_observations": 0,
            "archive_similarity_score": 0.84,
        },
    ]}, "conflict")

    assert selected["frame_id"] == 30
