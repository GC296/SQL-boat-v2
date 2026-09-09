from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any

from identity_schema import IDENTITY_FIELDS, normalize_identity_features
from training.common import normalized_edit_similarity


def evaluate_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    visible_exact: list[float] = []
    visible_edit: list[float] = []
    unreadable_hallucinations: list[float] = []
    attribute_scores: dict[str, list[float]] = defaultdict(list)
    empty_attribute_fills: list[float] = []
    valid_json: list[float] = []
    for row in rows:
        target = row.get("target", {}) or {}
        prediction = row.get("prediction", {}) or {}
        valid_json.append(float(bool(row.get("json_valid", False))))
        expected_hull = str(target.get("hull_number", "") or "").strip()
        predicted_hull = str(prediction.get("hull_number", "") or "").strip()
        if bool(row.get("hull_visible", bool(expected_hull))):
            visible_exact.append(float(predicted_hull == expected_hull))
            visible_edit.append(normalized_edit_similarity(predicted_hull, expected_hull))
        else:
            unreadable_hallucinations.append(float(bool(predicted_hull)))
        expected_features = normalize_identity_features(target.get("identity_features", {}))
        predicted_features = normalize_identity_features(prediction.get("identity_features", {}))
        for field in IDENTITY_FIELDS:
            if expected_features[field]:
                attribute_scores[field].append(float(predicted_features[field] == expected_features[field]))
            elif predicted_features[field]:
                empty_attribute_fills.append(1.0)
            else:
                empty_attribute_fills.append(0.0)
    per_field = {field: round(mean(values), 6) for field, values in attribute_scores.items() if values}
    return {
        "samples": len(rows),
        "json_valid_rate": round(mean(valid_json), 6) if valid_json else 0.0,
        "hull_readable_samples": len(visible_exact),
        "hull_exact_accuracy": round(mean(visible_exact), 6) if visible_exact else 0.0,
        "hull_normalized_edit_similarity": round(mean(visible_edit), 6) if visible_edit else 0.0,
        "hull_unreadable_samples": len(unreadable_hallucinations),
        "hull_hallucination_rate": round(mean(unreadable_hallucinations), 6) if unreadable_hallucinations else 0.0,
        "attribute_macro_exact_accuracy": round(mean(per_field.values()), 6) if per_field else 0.0,
        "attribute_false_fill_rate": round(mean(empty_attribute_fills), 6) if empty_attribute_fills else 0.0,
        "attribute_exact_accuracy_by_field": per_field,
    }

