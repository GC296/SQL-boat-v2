from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from identity_schema import IDENTITY_FIELDS, normalize_identity_features
from training.common import as_bool, read_jsonl, write_jsonl


def _recognition_drafts(run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = run_dir / "recognition.jsonl"
    if not path.exists():
        return {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(path):
        key = (str(row.get("video_id", "")), str(row.get("track_id", "")))
        grouped[key].append(row)
    return {
        key: sorted(rows, key=lambda row: int(row.get("frame_id", 0) or 0))[-1]
        for key, rows in grouped.items()
    }


def build_annotation_drafts(tracks: list[dict[str, Any]], run_dir: Path | None = None) -> list[dict[str, Any]]:
    recognitions = _recognition_drafts(run_dir) if run_dir else {}
    drafts: list[dict[str, Any]] = []
    for index, track in enumerate(tracks, start=1):
        video_id = str(track.get("video_id", "") or "")
        track_id = str(track.get("track_id", "") or "")
        recognition = recognitions.get((video_id, track_id), {})
        raw_features = recognition.get("identity_features")
        if not isinstance(raw_features, dict):
            raw_features = {}
        features = normalize_identity_features(raw_features)
        drafts.append({
            "sample_id": str(track.get("sample_id", "") or f"{video_id}:{track_id}:{index}"),
            "image": str(track.get("representative_crop", track.get("image", "")) or ""),
            "split": str(track.get("split", "") or ""),
            "vessel_id": str(track.get("vessel_id", "") or f"{video_id}:track:{track_id}"),
            "video_id": video_id,
            "track_id": track_id,
            "hull_number": str(track.get("hull_number", "") or ""),
            "hull_visible": as_bool(track.get("hull_visible"), default=bool(track.get("hull_number"))),
            "known_or_unknown": str(track.get("known_or_unknown", "") or ""),
            "description": str(
                recognition.get("observed_structure_description", recognition.get("description", "")) or ""
            ).strip(),
            "identity_features": features,
            "annotation_status": "draft",
            "needs_manual_review": True,
            "annotation_source": {
                "hull": "track_annotation",
                "description": "recognition_log_draft" if recognition else "empty",
                "identity_features": "recognition_log_draft" if any(features.values()) else "empty",
            },
        })
    return drafts


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reviewable VLM evidence annotation drafts")
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    drafts = build_annotation_drafts(read_jsonl(args.tracks), args.run_dir)
    write_jsonl(args.output, drafts)
    summary = {
        "drafts": len(drafts),
        "with_images": sum(bool(row["image"]) for row in drafts),
        "with_draft_attributes": sum(any(row["identity_features"].values()) for row in drafts),
        "fields_to_review": list(IDENTITY_FIELDS),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
