import csv
import json
from pathlib import Path

from experiments.apply_archive50_expansion import apply_expansion


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_apply_archive_expansion_updates_all_linked_files(tmp_path):
    crop_dir = tmp_path / "data" / "annotations" / "public_pool_representative_crops"
    crop_dir.mkdir(parents=True)
    selection = []
    manifest = []
    tracks = []
    entities = []
    for index in range(33):
        video_id = f"V{index + 100:03d}"
        entity_id = f"{video_id}_GT_01"
        crop = crop_dir / f"{video_id}_track_1_frame_1.jpg"
        crop.write_bytes(b"jpeg")
        selection.append({
            "archive_id": video_id,
            "video_id": video_id,
            "entity_id": entity_id,
            "description": f"description {video_id}",
            "secondary_description": f"secondary {video_id}",
        })
        manifest.append({"video_id": video_id, "video_path": f"/{video_id}.mp4", "split": "test"})
        tracks.append({
            "video_id": video_id,
            "track_id": 1,
            "known_or_unknown": "unknown",
            "hull_number": "",
            "representative_crop": str(crop),
            "best_observation_quality": 0.8,
        })
        entities.append({
            "video_id": video_id,
            "entity_id": entity_id,
            "member_track_ids": [1],
            "known_or_unknown": "unknown",
            "hull_number": "",
        })

    paths = {
        "selection": tmp_path / "selection.jsonl",
        "vlm_drafts": tmp_path / "vlm_drafts.jsonl",
        "manifest": tmp_path / "manifest.jsonl",
        "tracks": tmp_path / "tracks.jsonl",
        "entities": tmp_path / "entities.jsonl",
        "ships": tmp_path / "ships.csv",
        "archive": tmp_path / "archive",
        "report": tmp_path / "report.json",
    }
    vlm_drafts = [
        {
            "video_id": row["video_id"],
            "track_id": 1,
            "description": f"vlm description {row['video_id']}",
        }
        for row in manifest
    ]
    for key in ("selection", "vlm_drafts", "manifest", "tracks", "entities"):
        write_jsonl(paths[key], locals()[key])
    paths["ships"].write_text("prototype_id,hull_number,description\nOLD_01,OLD,old vessel\n", encoding="utf-8")

    report = apply_expansion(
        paths["selection"],
        paths["vlm_drafts"],
        paths["manifest"],
        paths["tracks"],
        paths["entities"],
        paths["ships"],
        paths["archive"],
        paths["report"],
    )

    assert report["selected_identities"] == 33
    assert report["distinct_known_identities"] == 33
    assert report["new_visual_prototypes"] == 33
    updated_entities = [json.loads(line) for line in paths["entities"].read_text(encoding="utf-8").splitlines()]
    assert all(row["known_or_unknown"] == "known" for row in updated_entities)
    assert all(row["hull_number"] == row["video_id"] for row in updated_entities)
    updated_manifest = [json.loads(line) for line in paths["manifest"].read_text(encoding="utf-8").splitlines()]
    assert all(row["archive_enrolled"] for row in updated_manifest)
    with paths["ships"].open(encoding="utf-8-sig", newline="") as handle:
        ship_rows = list(csv.DictReader(handle))
    assert len(ship_rows) == 100
    assert (paths["archive"] / "V100" / "V100_01.jpg").exists()
