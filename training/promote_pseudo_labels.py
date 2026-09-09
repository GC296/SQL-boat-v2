from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from identity_schema import normalize_identity_features
from training.common import compact_target_json, identity_target, read_jsonl, write_jsonl


def promote_predictions(
    rows: list[dict[str, Any]],
    min_attributes: int = 2,
    require_hull_agreement: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    promoted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for row in rows:
        if not bool(row.get("json_valid", False)):
            rejected["invalid_json"] += 1
            continue
        prediction = row.get("prediction", {}) or {}
        expected = row.get("target", {}) or {}
        hull_visible = bool(row.get("hull_visible", False))
        expected_hull = str(expected.get("hull_number", "") or "").strip()
        predicted_hull = str(prediction.get("hull_number", "") or "").strip()
        if hull_visible and expected_hull and require_hull_agreement and predicted_hull != expected_hull:
            rejected["hull_conflict"] += 1
            continue
        features = normalize_identity_features(prediction.get("identity_features", {}))
        nonempty_attributes = sum(bool(value) for value in features.values())
        if nonempty_attributes < min_attributes:
            rejected["insufficient_attributes"] += 1
            continue
        target = identity_target({
            "hull_number": expected_hull,
            "hull_visible": hull_visible,
            "identity_features": features,
        })
        promoted.append({
            key: value
            for key, value in row.items()
            if key not in {"prediction", "raw_response", "json_valid", "target", "target_text"}
        } | {
            "target": target,
            "target_text": compact_target_json(target),
            "label_source": "teacher_pseudo_label",
            "teacher_model": str(row.get("model", "") or ""),
            "pseudo_label_reviewed": False,
            "pseudo_label_attribute_count": nonempty_attributes,
        })
    return promoted, dict(rejected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter teacher predictions into a LoRA pseudo-label dataset")
    parser.add_argument("--predictions", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-attributes", type=int, default=2)
    parser.add_argument("--allow-hull-conflict", action="store_true")
    args = parser.parse_args()

    rows = [row for path in args.predictions for row in read_jsonl(path)]
    promoted, rejected = promote_predictions(
        rows,
        min_attributes=max(0, args.min_attributes),
        require_hull_agreement=not args.allow_hull_conflict,
    )
    write_jsonl(args.output, promoted)
    summary = {
        "teacher_predictions": len(rows),
        "accepted_pseudo_labels": len(promoted),
        "rejected": rejected,
        "splits": dict(Counter(str(row.get("split", "")) for row in promoted)),
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
