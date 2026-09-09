from experiments.reconcile_test_annotations import reconcile_annotations


def _track(video_id, track_id, vessel_id, identity_state, hull_number=""):
    return {
        "video_id": video_id,
        "track_id": track_id,
        "vessel_id": vessel_id,
        "known_or_unknown": identity_state,
        "hull_number": hull_number,
        "frame_start": 1,
        "frame_end": 10,
        "hull_visible": False,
        "risk_label": "low",
        "annotation_status": "confirmed",
    }


def test_reconcile_filters_stale_rows_and_merges_missing_test_video():
    manifest = [
        {"video_id": "V1", "split": "test"},
        {"video_id": "V2", "split": "test"},
        {"video_id": "V3", "split": "policy_train"},
    ]
    base = [
        _track("V1", 1, "ship-a", "known", "003"),
        _track("V3", 1, "ship-stale", "unknown"),
    ]
    additional = [_track("V2", 2, "ship-b", "unknown")]

    tracks, entities, report = reconcile_annotations(manifest, base, additional)

    assert [(row["video_id"], row["track_id"]) for row in tracks] == [("V1", 1), ("V2", 2)]
    assert [row["video_id"] for row in entities] == ["V1", "V2"]
    assert report["removed_non_test_videos"] == ["V3"]
    assert report["missing_test_videos"] == []
    assert report["passed"] is True


def test_reconcile_rejects_missing_test_video():
    manifest = [{"video_id": "V1", "split": "test"}, {"video_id": "V2", "split": "test"}]
    base = [_track("V1", 1, "ship-a", "unknown")]

    tracks, entities, report = reconcile_annotations(manifest, base, [])

    assert len(tracks) == 1
    assert entities == []
    assert report["missing_test_videos"] == ["V2"]
    assert report["passed"] is False
