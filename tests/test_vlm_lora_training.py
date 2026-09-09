import json

from training.bootstrap_vlm_annotations import build_annotation_drafts
from training.build_vlm_lora_dataset import build_samples, validate_group_splits
from training.common import identity_target
from training.metrics import evaluate_predictions
from training.promote_pseudo_labels import promote_predictions


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_dataset_builder_blanks_unreadable_hull_and_keeps_external_images(tmp_path):
    image = tmp_path / "external_ship.jpg"
    image.write_bytes(b"jpeg")
    annotations = tmp_path / "annotations.jsonl"
    write_jsonl(annotations, [{
        "sample_id": "external-1",
        "image": image.name,
        "split": "train",
        "vessel_id": "external-vessel-1",
        "hull_number": "A123",
        "hull_visible": False,
        "annotation_status": "confirmed",
        "identity_features": {"hull_color": "blue"},
    }])

    samples = build_samples([annotations], tmp_path, "train")

    assert len(samples) == 1
    assert samples[0]["target"]["hull_number"] == ""
    assert samples[0]["target"]["identity_features"]["hull_color"] == "blue"
    assert samples[0]["group_id"] == "external-vessel-1"


def test_dataset_builder_detects_entity_leakage(tmp_path):
    image = tmp_path / "ship.jpg"
    image.write_bytes(b"jpeg")
    annotations = tmp_path / "annotations.jsonl"
    write_jsonl(annotations, [
        {"image": image.name, "split": "train", "vessel_id": "same-vessel", "annotation_status": "confirmed"},
        {"image": image.name, "split": "test", "vessel_id": "same-vessel", "annotation_status": "confirmed"},
    ])

    leakage = validate_group_splits(build_samples([annotations], tmp_path, "train"))

    assert leakage == {"same-vessel": ["test", "train"]}


def test_dataset_builder_infers_split_from_annotation_filename(tmp_path):
    image = tmp_path / "ship.jpg"
    image.write_bytes(b"jpeg")
    annotations = tmp_path / "val_tracks.jsonl"
    write_jsonl(annotations, [{
        "image": image.name,
        "vessel_id": "validation-vessel",
        "annotation_status": "confirmed",
    }])

    samples = build_samples([annotations], tmp_path, "train")

    assert samples[0]["split"] == "val"


def test_lora_metrics_include_hull_hallucination_and_attributes():
    rows = [
        {
            "hull_visible": True,
            "json_valid": True,
            "target": {"hull_number": "A12", "identity_features": {"hull_color": "blue"}},
            "prediction": {"hull_number": "A12", "identity_features": {"hull_color": "blue"}},
        },
        {
            "hull_visible": False,
            "json_valid": True,
            "target": {"hull_number": "", "identity_features": {}},
            "prediction": {"hull_number": "B99", "identity_features": {}},
        },
    ]

    metrics = evaluate_predictions(rows)

    assert metrics["hull_exact_accuracy"] == 1.0
    assert metrics["hull_hallucination_rate"] == 1.0
    assert metrics["attribute_macro_exact_accuracy"] == 1.0


def test_description_is_derived_from_attributes():
    target = identity_target({
        "description": "stale model description",
        "identity_features": {"hull_color": "blue"},
    })

    assert target["description"] == "船体颜色：blue"


def test_annotation_bootstrap_reuses_track_and_recognition_fields(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_jsonl(run_dir / "recognition.jsonl", [{
        "video_id": "V1",
        "track_id": 2,
        "frame_id": 40,
        "observed_structure_description": "blue hull",
        "identity_features": {"hull_color": "blue"},
    }])
    drafts = build_annotation_drafts([{
        "video_id": "V1",
        "track_id": 2,
        "representative_crop": "crop.jpg",
        "hull_number": "A12",
        "hull_visible": True,
        "vessel_id": "ship-1",
    }], run_dir)

    assert drafts[0]["description"] == "blue hull"
    assert drafts[0]["identity_features"]["hull_color"] == "blue"
    assert drafts[0]["needs_manual_review"] is True


def test_teacher_pseudo_labels_keep_trusted_hull_and_attributes():
    promoted, rejected = promote_predictions([{
        "sample_id": "sample-1",
        "split": "train",
        "image": "ship.jpg",
        "hull_visible": True,
        "target": {"hull_number": "A12", "identity_features": {}},
        "prediction": {
            "hull_number": "A12",
            "identity_features": {"vessel_type": "workboat", "hull_color": "blue"},
        },
        "json_valid": True,
        "model": "teacher-vlm",
    }])

    assert rejected == {}
    assert promoted[0]["target"]["hull_number"] == "A12"
    assert promoted[0]["target"]["identity_features"]["hull_color"] == "blue"
    assert promoted[0]["label_source"] == "teacher_pseudo_label"


def test_teacher_pseudo_labels_reject_hull_conflicts():
    promoted, rejected = promote_predictions([{
        "hull_visible": True,
        "target": {"hull_number": "A12"},
        "prediction": {
            "hull_number": "B99",
            "identity_features": {"vessel_type": "workboat", "hull_color": "blue"},
        },
        "json_valid": True,
    }])

    assert promoted == []
    assert rejected == {"hull_conflict": 1}
