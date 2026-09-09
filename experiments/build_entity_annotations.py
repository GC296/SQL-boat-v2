"""Build entity-level ground truth by grouping manually verified tracklets."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_entities(track_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in track_rows:
        vessel_id = str(row.get("vessel_id", "")).strip()
        if not vessel_id:
            raise ValueError(f"missing vessel_id: video={row.get('video_id')} track={row.get('track_id')}")
        groups[(str(row["video_id"]), vessel_id)].append(row)
    entities: list[dict[str, Any]] = []
    per_video_index: dict[str, int] = defaultdict(int)
    for (video_id, vessel_id), rows in sorted(groups.items()):
        labels = {(str(row.get("known_or_unknown", "")), str(row.get("hull_number", "") or "")) for row in rows}
        if len(labels) != 1:
            raise ValueError(f"inconsistent identity labels for video={video_id}, vessel={vessel_id}: {sorted(labels)}")
        per_video_index[video_id] += 1
        known_or_unknown, hull_number = next(iter(labels))
        entities.append({
            "video_id": video_id,
            "entity_id": f"{video_id}_GT_{per_video_index[video_id]:02d}",
            "vessel_id": vessel_id,
            "member_track_ids": sorted(int(row["track_id"]) for row in rows),
            "frame_start": min(int(row.get("frame_start", 0)) for row in rows),
            "frame_end": max(int(row.get("frame_end", 0)) for row in rows),
            "known_or_unknown": known_or_unknown,
            "hull_number": hull_number,
            "hull_visible": any(str(row.get("hull_visible", "false")).lower() in {"true", "1", "yes"} for row in rows),
            "risk_label": max((str(row.get("risk_label", "low")) for row in rows), key={"low": 0, "medium": 1, "high": 2}.get),
            "annotation_status": "confirmed",
        })
    return entities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    entities = build_entities(read_jsonl(args.tracks))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for entity in entities:
            handle.write(json.dumps(entity, ensure_ascii=False) + "\n")
    print(json.dumps({"tracklets": sum(len(item["member_track_ids"]) for item in entities), "entities": len(entities), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
