"""Reconcile confirmed track annotations with the frozen manifest test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.build_entity_annotations import build_entities


VALID_IDENTITY_STATES = {"known", "unknown"}
VALID_RISK_LABELS = {"low", "medium", "high"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _track_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("video_id", "")), int(row.get("track_id", -1))


def reconcile_annotations(
    manifest_rows: list[dict[str, Any]],
    base_tracks: list[dict[str, Any]],
    additional_tracks: list[dict[str, Any]],
    split: str = "test",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    test_video_ids = {
        str(row.get("video_id", ""))
        for row in manifest_rows
        if str(row.get("split", "")) == split
    }
    if not test_video_ids:
        raise ValueError(f"Manifest contains no videos for split {split!r}")

    merged: dict[tuple[str, int], dict[str, Any]] = {}
    duplicate_additional: list[tuple[str, int]] = []
    for row in base_tracks:
        if str(row.get("video_id", "")) in test_video_ids:
            merged[_track_key(row)] = dict(row)
    seen_additional: set[tuple[str, int]] = set()
    for row in additional_tracks:
        key = _track_key(row)
        if key in seen_additional:
            duplicate_additional.append(key)
        seen_additional.add(key)
        if key[0] in test_video_ids:
            merged[key] = dict(row)

    rows = [merged[key] for key in sorted(merged)]
    annotated_video_ids = {str(row.get("video_id", "")) for row in rows}
    missing_videos = sorted(test_video_ids - annotated_video_ids)
    removed_non_test_videos = sorted(
        {str(row.get("video_id", "")) for row in base_tracks} - test_video_ids
    )
    errors: list[str] = []
    if duplicate_additional:
        errors.append(f"duplicate additional track keys: {duplicate_additional}")
    if missing_videos:
        errors.append(f"test videos without confirmed track annotations: {missing_videos}")

    for row in rows:
        video_id, track_id = _track_key(row)
        prefix = f"video={video_id}, track={track_id}"
        identity_state = str(row.get("known_or_unknown", ""))
        hull_number = str(row.get("hull_number", "") or "").strip()
        if not str(row.get("vessel_id", "")).strip():
            errors.append(f"{prefix}: vessel_id is required")
        if identity_state not in VALID_IDENTITY_STATES:
            errors.append(f"{prefix}: known_or_unknown must be known or unknown")
        if identity_state == "known" and not hull_number:
            errors.append(f"{prefix}: known target requires hull_number")
        if identity_state == "unknown" and hull_number:
            errors.append(f"{prefix}: unknown target must not have hull_number")
        if str(row.get("risk_label", "")) not in VALID_RISK_LABELS:
            errors.append(f"{prefix}: risk_label must be low, medium, or high")
        if str(row.get("annotation_status", "")) != "confirmed":
            errors.append(f"{prefix}: annotation_status must be confirmed")
        if int(row.get("frame_end", -1)) < int(row.get("frame_start", 0)):
            errors.append(f"{prefix}: frame_end is earlier than frame_start")

    entities: list[dict[str, Any]] = []
    if not errors:
        entities = build_entities(rows)
    report = {
        "split": split,
        "manifest_test_videos": len(test_video_ids),
        "base_track_rows": len(base_tracks),
        "additional_track_rows": len(additional_tracks),
        "output_track_rows": len(rows),
        "output_entities": len(entities),
        "removed_non_test_videos": removed_non_test_videos,
        "missing_test_videos": missing_videos,
        "errors": errors,
        "passed": not errors,
    }
    return rows, entities, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter stale annotations by the final manifest and merge newly confirmed test tracks."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--base-tracks", required=True, type=Path)
    parser.add_argument("--additional-tracks", type=Path)
    parser.add_argument("--output-tracks", required=True, type=Path)
    parser.add_argument("--output-entities", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    additional = read_jsonl(args.additional_tracks) if args.additional_tracks else []
    tracks, entities, report = reconcile_annotations(
        read_jsonl(args.manifest),
        read_jsonl(args.base_tracks),
        additional,
        split=args.split,
    )
    report_path = args.report or args.output_tracks.with_suffix(".audit.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(f"annotation reconciliation failed with {len(report['errors'])} error(s)")
    write_jsonl(args.output_tracks, tracks)
    write_jsonl(args.output_entities, entities)


if __name__ == "__main__":
    main()
