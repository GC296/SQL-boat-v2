from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from identity_schema import IDENTITY_FIELDS, canonical_identity_description, normalize_identity_features


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "visible"}


def resolve_image_path(raw_path: str, source: Path, image_root: Path) -> Path:
    image_path = Path(raw_path).expanduser()
    candidates = [image_path] if image_path.is_absolute() else [
        image_root / image_path,
        source.parent / image_path,
        source.parent.parent / image_path,
        source.parent.parent.parent / image_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def identity_target(record: Mapping[str, Any]) -> dict[str, Any]:
    raw_features = record.get("identity_features")
    if not isinstance(raw_features, Mapping):
        raw_features = {field: record.get(field, "") for field in IDENTITY_FIELDS}
    features = normalize_identity_features(raw_features)
    annotated_hull = str(record.get("hull_number", "") or "").strip()
    hull_visible = as_bool(record.get("hull_visible"), default=bool(annotated_hull))
    hull_number = annotated_hull if hull_visible else ""
    canonical_description = canonical_identity_description(features)
    description = canonical_description or str(record.get("description", "") or "").strip()
    return {
        "hull_number": hull_number,
        "description": description,
        "identity_features": features,
    }


def compact_target_json(target: Mapping[str, Any]) -> str:
    return json.dumps(dict(target), ensure_ascii=False, separators=(",", ":"))


def sample_group_id(record: Mapping[str, Any]) -> str:
    vessel_id = str(record.get("vessel_id", "") or "").strip()
    if vessel_id:
        return vessel_id
    entity_id = str(record.get("entity_id", "") or "").strip()
    video_id = str(record.get("video_id", "") or "").strip()
    if entity_id:
        return f"{video_id}:{entity_id}" if video_id else entity_id
    track_id = str(record.get("track_id", "") or "").strip()
    return f"{video_id}:track:{track_id}" if video_id or track_id else "ungrouped"


def normalized_edit_similarity(left: str, right: str) -> float:
    left, right = str(left or ""), str(right or "")
    if not left and not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        for column_index, right_char in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[column_index] + 1,
                previous[column_index - 1] + int(left_char != right_char),
            ))
        previous = current
    distance = previous[-1]
    return 1.0 - distance / max(1, len(left), len(right))
