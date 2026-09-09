from experiments.prepare_final_dataset import audit_manifest, build_inventory


def record(video_id, path, split, voyage):
    return {
        "video_id": video_id,
        "video_path": str(path),
        "split": split,
        "voyage_id": voyage,
        "location": "port_a",
        "date": "2026-01-01",
        "dataset_subset": "public",
        "source_url": "https://example.com/video",
        "license": "research-use",
    }


def test_no_validation_manifest_passes_with_disjoint_voyages(tmp_path):
    archive_video = tmp_path / "archive.mp4"
    test_video = tmp_path / "test.mp4"
    archive_video.write_bytes(b"archive")
    test_video.write_bytes(b"test")
    rows = [
        record("V1", archive_video, "train", "voyage_1"),
        record("V2", test_video, "test", "voyage_2"),
    ]

    audit = audit_manifest(rows, {"train", "archive"}, "test")
    inventory = build_inventory(rows, {"train", "archive"}, "test", probe=False, hash_videos=False)

    assert audit["passed"] is True
    assert audit["split_counts"] == {"test": 1, "train": 1}
    assert [item["protocol_role"] for item in inventory] == ["archive_construction", "final_test"]


def test_validation_split_is_rejected(tmp_path):
    video = tmp_path / "val.mp4"
    video.write_bytes(b"val")
    audit = audit_manifest([record("V1", video, "val", "voyage_1")], {"train", "archive"}, "test")

    assert audit["passed"] is False
    assert any("val split" in error for error in audit["errors"])


def test_voyage_cannot_cross_archive_and_test_roles(tmp_path):
    archive_video = tmp_path / "archive.mp4"
    test_video = tmp_path / "test.mp4"
    archive_video.write_bytes(b"archive")
    test_video.write_bytes(b"test")
    rows = [
        record("V1", archive_video, "train", "voyage_1"),
        record("V2", test_video, "test", "voyage_1"),
    ]

    audit = audit_manifest(rows, {"train", "archive"}, "test")

    assert audit["passed"] is False
    assert any("crosses archive and test" in error for error in audit["errors"])


def test_missing_public_source_metadata_is_reported(tmp_path):
    video = tmp_path / "test.mp4"
    video.write_bytes(b"test")
    row = record("V1", video, "test", "voyage_1")
    row.pop("source_url")

    audit = audit_manifest([row], {"train", "archive"}, "test")

    assert audit["passed"] is True
    assert any("source_url" in warning for warning in audit["warnings"])
