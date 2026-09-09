"""Audit and freeze experiment inputs for a no-validation final protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any


REQUIRED_MANIFEST_FIELDS = (
    "video_id",
    "video_path",
    "split",
    "voyage_id",
    "location",
    "date",
)
SOURCE_METADATA_FIELDS = ("dataset_subset", "source_url", "license")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {"video_readable": False}
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {
            "video_readable": True,
            "fps": round(fps, 6),
            "frame_count": frame_count,
            "duration_s": round(frame_count / fps, 6) if fps > 0 else 0.0,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
    finally:
        capture.release()


def audit_manifest(
    records: list[dict[str, Any]],
    archive_splits: set[str],
    test_split: str,
    check_paths: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    video_ids: dict[str, int] = {}
    video_paths: dict[str, str] = {}
    voyage_splits: dict[str, set[str]] = defaultdict(set)

    for line_number, record in enumerate(records, 1):
        missing = [field for field in REQUIRED_MANIFEST_FIELDS if not str(record.get(field, "")).strip()]
        if missing:
            errors.append(f"line {line_number}: missing required fields: {', '.join(missing)}")
            continue

        video_id = str(record["video_id"]).strip()
        video_path = str(record["video_path"]).strip()
        split = str(record["split"]).strip()
        voyage_id = str(record["voyage_id"]).strip()

        if video_id in video_ids:
            errors.append(f"line {line_number}: duplicate video_id {video_id!r}; first seen on line {video_ids[video_id]}")
        else:
            video_ids[video_id] = line_number

        normalized_path = str(Path(video_path)).casefold()
        if normalized_path in video_paths:
            errors.append(f"line {line_number}: video_path is reused by {video_paths[normalized_path]} and {video_id}")
        else:
            video_paths[normalized_path] = video_id

        try:
            date.fromisoformat(str(record["date"]))
        except ValueError:
            errors.append(f"line {line_number}, video={video_id}: date must use YYYY-MM-DD")

        if split == "val":
            errors.append(f"line {line_number}, video={video_id}: val split is not allowed by the no-validation protocol")
        elif split not in archive_splits | {test_split}:
            warnings.append(f"line {line_number}, video={video_id}: unrecognized split {split!r}")

        voyage_splits[voyage_id].add(split)
        if check_paths and not Path(video_path).is_file():
            errors.append(f"line {line_number}, video={video_id}: video file does not exist: {video_path}")

        missing_source = [field for field in SOURCE_METADATA_FIELDS if not str(record.get(field, "")).strip()]
        if missing_source:
            warnings.append(
                f"line {line_number}, video={video_id}: source metadata still needs {', '.join(missing_source)}"
            )

    for voyage_id, splits in sorted(voyage_splits.items()):
        if test_split in splits and splits & archive_splits:
            errors.append(
                f"voyage {voyage_id!r} crosses archive and test roles: {sorted(splits)}; keep all clips from one voyage together"
            )

    split_counts = Counter(str(record.get("split", "")) for record in records)
    if not split_counts.get(test_split):
        warnings.append(f"no records use final test split {test_split!r}")
    if not sum(split_counts.get(split, 0) for split in archive_splits):
        warnings.append(f"no records use archive-construction splits {sorted(archive_splits)}")

    return {
        "protocol": "no_validation_frozen_config",
        "records": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "voyages": len(voyage_splits),
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }


def build_inventory(
    records: list[dict[str, Any]],
    archive_splits: set[str],
    test_split: str,
    probe: bool,
    hash_videos: bool,
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        split = str(record.get("split", ""))
        row["protocol_role"] = (
            "archive_construction" if split in archive_splits else "final_test" if split == test_split else "unassigned"
        )
        path = Path(str(record.get("video_path", "")))
        row["path_exists"] = path.is_file()
        row["file_size_bytes"] = path.stat().st_size if path.is_file() else 0
        if probe and path.is_file():
            row.update(probe_video(path))
        if hash_videos and path.is_file():
            row["sha256"] = sha256_file(path)
        inventory.append(row)
    return inventory


def build_review_rows(inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = (
        "video_id",
        "video_path",
        "split",
        "protocol_role",
        "dataset_subset",
        "voyage_id",
        "location",
        "date",
        "source_url",
        "license",
        "expected_entity_count",
        "independent_archive_reference",
        "archive_reference_note",
        "include_in_final_report",
        "review_status",
        "notes",
    )
    rows = []
    for item in inventory:
        row = {field: item.get(field, "") for field in fields}
        row["review_status"] = row["review_status"] or "pending"
        rows.append(row)
    return rows


def build_input_lock(
    manifest: Path,
    config: Path | None,
    archive_csv: Path | None,
    visual_archive_root: Path | None,
) -> dict[str, Any]:
    locked: dict[str, Any] = {
        "manifest": {"path": str(manifest), "sha256": sha256_file(manifest)},
    }
    for key, path in (("config", config), ("archive_csv", archive_csv)):
        if path is not None:
            if not path.is_file():
                raise FileNotFoundError(f"{key} does not exist: {path}")
            locked[key] = {"path": str(path), "sha256": sha256_file(path)}
    if visual_archive_root is not None:
        if not visual_archive_root.is_dir():
            raise FileNotFoundError(f"visual archive does not exist: {visual_archive_root}")
        files = sorted(
            path for path in visual_archive_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        locked["visual_archive"] = {
            "path": str(visual_archive_root),
            "images": [
                {
                    "path": path.relative_to(visual_archive_root).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in files
            ],
        }
    return locked


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    seen: set[str] = set()
    for record in records:
        for field in record:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and freeze a final dataset without a validation split")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/annotations/final_protocol"))
    parser.add_argument("--archive-splits", default="train,archive")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--archive-csv", type=Path)
    parser.add_argument("--visual-archive-root", type=Path)
    parser.add_argument("--probe-videos", action="store_true")
    parser.add_argument("--hash-videos", action="store_true")
    parser.add_argument("--skip-path-check", action="store_true")
    args = parser.parse_args()

    archive_splits = {value.strip() for value in args.archive_splits.split(",") if value.strip()}
    records = read_jsonl(args.manifest)
    audit = audit_manifest(records, archive_splits, args.test_split, check_paths=not args.skip_path_check)
    inventory = build_inventory(records, archive_splits, args.test_split, args.probe_videos, args.hash_videos)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "manifest_audit.json", audit)
    write_jsonl(args.output_dir / "video_inventory.jsonl", inventory)
    write_csv(args.output_dir / "video_inventory.csv", inventory)
    write_csv(args.output_dir / "manual_review.csv", build_review_rows(inventory))
    write_json(
        args.output_dir / "protocol_lock.json",
        build_input_lock(args.manifest, args.config, args.archive_csv, args.visual_archive_root),
    )

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if audit["errors"]:
        raise SystemExit(f"final dataset audit failed with {len(audit['errors'])} error(s)")


if __name__ == "__main__":
    main()
