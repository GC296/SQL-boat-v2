"""Apply the curated 50-identity archive expansion to final annotations."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def backup_once(path: Path, suffix: str = ".before_archive50") -> Path:
    backup = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    if not backup.exists():
        shutil.copy2(path, backup)
    return backup


def append_note(value: Any, note: str) -> str:
    existing = str(value or "").strip()
    if note in existing:
        return existing
    return f"{existing}; {note}".strip("; ")


def apply_expansion(
    selection_path: Path,
    vlm_drafts_path: Path,
    manifest_path: Path,
    tracks_path: Path,
    entities_path: Path,
    ships_path: Path,
    archive_root: Path,
    report_path: Path,
    max_prototypes: int = 3,
    min_prototype_quality: float = 0.60,
    max_text_prototypes: int = 4,
) -> dict[str, Any]:
    selection = read_jsonl(selection_path)
    vlm_drafts = read_jsonl(vlm_drafts_path) if vlm_drafts_path.exists() else []
    archive_ids = [str(row["archive_id"]).strip() for row in selection]
    entity_ids = [str(row["entity_id"]).strip() for row in selection]
    if len(selection) != 33:
        raise ValueError(f"Expected 33 selected identities, found {len(selection)}")
    if len(set(archive_ids)) != len(archive_ids) or len(set(entity_ids)) != len(entity_ids):
        raise ValueError("Selection contains duplicate archive_id or entity_id values")

    for path in (manifest_path, tracks_path, entities_path, ships_path):
        if not path.exists():
            raise FileNotFoundError(path)
        backup_once(path)

    manifest = read_jsonl(manifest_path)
    tracks = read_jsonl(tracks_path)
    entities = read_jsonl(entities_path)
    selection_by_entity = {str(row["entity_id"]): row for row in selection}
    entities_by_id = {str(row["entity_id"]): row for row in entities}
    tracks_by_key = {(str(row["video_id"]), int(row["track_id"])): row for row in tracks}
    draft_by_track = {
        (str(row.get("video_id", "")), int(row.get("track_id", 0))): row
        for row in vlm_drafts
    }

    missing_entities = sorted(set(selection_by_entity) - set(entities_by_id))
    if missing_entities:
        raise ValueError(f"Selected entities are missing: {missing_entities}")

    selected_track_keys: dict[tuple[str, int], dict[str, Any]] = {}
    for entity_id, selected in selection_by_entity.items():
        entity = entities_by_id[entity_id]
        if str(entity.get("video_id")) != str(selected["video_id"]):
            raise ValueError(f"Video mismatch for {entity_id}")
        if str(entity.get("known_or_unknown", "")).lower() not in {"unknown", "known"}:
            raise ValueError(f"Invalid status for {entity_id}: {entity.get('known_or_unknown')}")
        for track_id in entity.get("member_track_ids", []):
            key = (str(entity["video_id"]), int(track_id))
            if key not in tracks_by_key:
                raise ValueError(f"Missing member track {key} for {entity_id}")
            selected_track_keys[key] = selected

    note = "archive50_enrolled_from_public_representative_crop"
    for entity in entities:
        selected = selection_by_entity.get(str(entity.get("entity_id", "")))
        if not selected:
            continue
        archive_id = str(selected["archive_id"])
        entity.update({
            "vessel_id": archive_id,
            "known_or_unknown": "known",
            "hull_number": archive_id,
            "hull_visible": False,
            "annotation_status": "confirmed",
            "notes": append_note(entity.get("notes"), note),
        })

    for track in tracks:
        selected = selected_track_keys.get((str(track.get("video_id", "")), int(track.get("track_id", 0))))
        if not selected:
            continue
        archive_id = str(selected["archive_id"])
        track.update({
            "vessel_id": archive_id,
            "known_or_unknown": "known",
            "hull_number": archive_id,
            "hull_visible": False,
            "annotation_status": "confirmed",
            "notes": append_note(track.get("notes"), note),
        })

    selections_by_video: dict[str, list[str]] = {}
    for selected in selection:
        selections_by_video.setdefault(str(selected["video_id"]), []).append(str(selected["archive_id"]))
    manifest_videos = {str(row.get("video_id", "")) for row in manifest}
    if missing_videos := sorted(set(selections_by_video) - manifest_videos):
        raise ValueError(f"Selected videos are missing from manifest: {missing_videos}")
    for row in manifest:
        video_id = str(row.get("video_id", ""))
        if video_id in selections_by_video:
            row["archive_enrolled"] = True
            row["archive_identity_ids"] = sorted(selections_by_video[video_id])

    archive_root.mkdir(parents=True, exist_ok=True)
    copied_images: list[dict[str, Any]] = []
    for selected in selection:
        video_id = str(selected["video_id"])
        entity = entities_by_id[str(selected["entity_id"])]
        member_rows = [
            tracks_by_key[(video_id, int(track_id))]
            for track_id in entity.get("member_track_ids", [])
        ]
        member_rows.sort(key=lambda row: float(row.get("best_observation_quality", 0.0) or 0.0), reverse=True)
        sources = []
        for track in member_rows:
            quality = float(track.get("best_observation_quality", 0.0) or 0.0)
            if sources and quality < min_prototype_quality:
                continue
            source = Path(str(track.get("representative_crop", "")))
            if "public_pool_representative_crops" not in source.as_posix() or not source.exists():
                continue
            sources.append((source, track))
            if len(sources) >= max(1, max_prototypes):
                break
        if not sources:
            raise FileNotFoundError(f"No usable representative crop for {selected['entity_id']}")
        identity_dir = archive_root / str(selected["archive_id"])
        identity_dir.mkdir(parents=True, exist_ok=True)
        for old_prototype in identity_dir.glob(f"{selected['archive_id']}_*"):
            if old_prototype.is_file():
                old_prototype.unlink()
        for index, (source, track) in enumerate(sources, start=1):
            target = identity_dir / f"{selected['archive_id']}_{index:02d}{source.suffix.lower()}"
            shutil.copy2(source, target)
            copied_images.append({
                "archive_id": selected["archive_id"],
                "entity_id": selected["entity_id"],
                "track_id": int(track["track_id"]),
                "quality": float(track.get("best_observation_quality", 0.0) or 0.0),
                "source": source.as_posix(),
                "target": target.as_posix(),
            })

    with ships_path.open(encoding="utf-8-sig", newline="") as handle:
        ship_rows = list(csv.DictReader(handle))
    existing_ids = {str(row.get("hull_number", "")).strip() for row in ship_rows}
    conflicting_ids = sorted(set(archive_ids) & existing_ids)
    if conflicting_ids:
        ship_rows = [row for row in ship_rows if str(row.get("hull_number", "")).strip() not in set(conflicting_ids)]
    text_prototypes: list[dict[str, Any]] = []
    for selected in selection:
        archive_id = str(selected["archive_id"])
        entity = entities_by_id[str(selected["entity_id"])]
        descriptions = []
        member_rows = [
            tracks_by_key[(str(selected["video_id"]), int(track_id))]
            for track_id in entity.get("member_track_ids", [])
        ]
        member_rows.sort(key=lambda row: float(row.get("best_observation_quality", 0.0) or 0.0), reverse=True)
        for track in member_rows:
            draft = draft_by_track.get((str(selected["video_id"]), int(track["track_id"])), {})
            description = str(draft.get("description", "") or "").strip()
            if description and description not in descriptions:
                descriptions.append(description)
            if len(descriptions) >= max(1, max_text_prototypes - 2):
                break
        for key in ("description", "secondary_description"):
            description = str(selected.get(key, "") or "").strip()
            if description and description not in descriptions:
                descriptions.append(description)
        descriptions = descriptions[:max(1, max_text_prototypes)]
        for index, description in enumerate(descriptions, start=1):
            prototype = {
                "prototype_id": f"{archive_id}_{index:02d}",
                "hull_number": archive_id,
                "description": description,
            }
            ship_rows.append(prototype)
            text_prototypes.append(prototype)
    with ships_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prototype_id", "hull_number", "description"])
        writer.writeheader()
        writer.writerows(ship_rows)

    write_jsonl(manifest_path, manifest)
    write_jsonl(tracks_path, tracks)
    write_jsonl(entities_path, entities)

    known_entities = [row for row in entities if str(row.get("known_or_unknown", "")).lower() == "known"]
    unknown_entities = [row for row in entities if str(row.get("known_or_unknown", "")).lower() == "unknown"]
    known_identity_counts = Counter(str(row.get("hull_number", "")) for row in known_entities)
    report = {
        "selection": str(selection_path),
        "selected_identities": len(selection),
        "selected_archive_ids": archive_ids,
        "manifest_rows": len(manifest),
        "track_rows": len(tracks),
        "entity_rows": len(entities),
        "known_entities": len(known_entities),
        "unknown_entities": len(unknown_entities),
        "distinct_known_identities": len(known_identity_counts),
        "known_identity_counts": dict(sorted(known_identity_counts.items())),
        "ship_rows": len(ship_rows),
        "ship_identities": len({str(row.get("hull_number", "")).strip() for row in ship_rows}),
        "new_text_prototypes": len(text_prototypes),
        "vlm_drafts": str(vlm_drafts_path),
        "new_visual_prototypes": len(copied_images),
        "archive_root": str(archive_root),
        "copied_images": copied_images,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=Path("data/annotations/archive50_selection.jsonl"))
    parser.add_argument("--vlm-drafts", type=Path, default=Path("data/annotations/public_pool_vlm_drafts.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/annotations/manifest.jsonl"))
    parser.add_argument("--tracks", type=Path, default=Path("data/annotations/test_tracks.final_public.jsonl"))
    parser.add_argument("--entities", type=Path, default=Path("data/annotations/test_entities.final_public.jsonl"))
    parser.add_argument("--ships", type=Path, default=Path("data/ships.csv"))
    parser.add_argument("--archive-root", type=Path, default=Path("data/archive/visual_prototypes"))
    parser.add_argument("--report", type=Path, default=Path("data/annotations/archive50_expansion_report.json"))
    parser.add_argument("--max-prototypes", type=int, default=3)
    parser.add_argument("--min-prototype-quality", type=float, default=0.60)
    parser.add_argument("--max-text-prototypes", type=int, default=4)
    args = parser.parse_args()
    report = apply_expansion(
        args.selection,
        args.vlm_drafts,
        args.manifest,
        args.tracks,
        args.entities,
        args.ships,
        args.archive_root,
        args.report,
        max_prototypes=args.max_prototypes,
        min_prototype_quality=args.min_prototype_quality,
        max_text_prototypes=args.max_text_prototypes,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
