"""Evaluate identity, policy, edge-cloud, and risk outputs."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_annotations(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    output = {}
    for item in read_jsonl(path):
        output[(str(item["video_id"]), int(item["track_id"]))] = item
    return output


def normalized_edit_similarity(left: str, right: str) -> float:
    left, right = left or "", right or ""
    if not left and not right:
        return 1.0
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return 1.0 - previous[-1] / max(1, len(left), len(right))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def evaluate(run_dir: Path, annotations: dict[tuple[str, int], dict[str, Any]] | None = None) -> dict[str, Any]:
    recognitions = read_jsonl(run_dir / "recognition.jsonl")
    actions = read_jsonl(run_dir / "actions.jsonl")
    cloud = read_jsonl(run_dir / "edge_cloud.jsonl")
    observations = read_jsonl(run_dir / "observations.jsonl")

    if annotations is not None:
        allowed_tracks = set(annotations)

        def belongs_to_evaluation(row: dict[str, Any]) -> bool:
            try:
                key = (str(row.get("video_id", "")), int(row.get("track_id", -1)))
            except (TypeError, ValueError):
                return False
            return key in allowed_tracks

        recognitions = [row for row in recognitions if belongs_to_evaluation(row)]
        actions = [row for row in actions if belongs_to_evaluation(row)]
        cloud = [row for row in cloud if belongs_to_evaluation(row)]
        observations = [row for row in observations if belongs_to_evaluation(row)]

    latest: dict[tuple[str, int], dict[str, Any]] = {}
    recognition_history: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    first_observation: dict[tuple[str, int], int] = {}
    observed_tracks: set[tuple[str, int]] = set()
    for item in observations:
        key = (str(item.get("video_id", "")), int(item["track_id"]))
        observed_tracks.add(key)
        frame_id = int(item.get("frame_id", 0))
        first_observation[key] = min(frame_id, first_observation.get(key, frame_id))
    for item in recognitions:
        key = (str(item.get("video_id", "")), int(item["track_id"]))
        recognition_history[key].append(item)
        latest[key] = item
    for items in recognition_history.values():
        items.sort(key=lambda item: int(item.get("frame_id", 0)))

    evaluation_tracks = set(annotations) if annotations else (observed_tracks | set(latest))
    calls_per_track: dict[tuple[str, int], int] = defaultdict(int)
    for item in cloud:
        key = (str(item.get("video_id", "")), int(item["track_id"]))
        calls_per_track[key] += max(1, int(item.get("request_count", 1)))
    latencies = [float(item.get("total_latency_ms", 0.0)) for item in cloud if item.get("success", True)]
    total_vlm_calls = sum(max(1, int(item.get("request_count", 1))) for item in cloud)

    identity_states = [str(item.get("identity_state", "unknown")) for item in latest.values()]
    accepted_states = {"confirmed", "structure_verified"}
    unresolved_states = {"unknown", "uncertain", "conflicting"}
    resolved_states = accepted_states | {"out_of_archive"}

    metrics: dict[str, Any] = {
        "tracks_evaluated": len(evaluation_tracks),
        "tracks_with_recognition": len(latest),
        "confirmed_target_rate": mean([state in accepted_states for state in identity_states]) if identity_states else 0.0,
        "out_of_archive_target_rate": mean([state == "out_of_archive" for state in identity_states]) if identity_states else 0.0,
        "resolved_target_rate": mean([state in resolved_states for state in identity_states]) if identity_states else 0.0,
        "unresolved_target_rate": mean([state in unresolved_states for state in identity_states]) if identity_states else 0.0,
        "vlm_calls": total_vlm_calls,
        "avg_vlm_calls_per_track": sum(calls_per_track.values()) / max(1, len(evaluation_tracks)),
        "vlm_calls_per_target": total_vlm_calls / max(1, len(evaluation_tracks)),
        "avg_recognition_latency_ms": mean(latencies) if latencies else 0.0,
        "mean_observation_quality": mean([float(item["observation_quality"]) for item in observations]) if observations else 0.0,
        "upload_bytes_per_target": sum(int(item.get("upload_bytes", 0)) for item in cloud) / max(1, len(evaluation_tracks)),
        "end_to_end_p50_ms": percentile(latencies, 0.50),
        "end_to_end_p95_ms": percentile(latencies, 0.95),
        "false_escalation_count": sum(1 for item in actions if item.get("action") in {"remote_report", "escalate"} and "high_risk" not in item.get("reasons", [])),
    }

    if annotations:
        known_total = unknown_total = readable_known = 0
        exact_hull = correct_matches = accepted_matches = 0
        unresolved_known = false_matches = unknown_false_accepts = 0
        known_false_rejections = 0
        unknown_rejections = unresolved_unknown = 0
        edit_scores: list[float] = []
        confirmation_frames: list[int] = []

        for key, truth in annotations.items():
            pred = latest.get(key, {})
            target = str(truth.get("hull_number", "") or "")
            hull_guess = str(pred.get("fused_hull_number", "") or "")
            identity_guess = str(pred.get("verified_identity", "") or "")
            state = str(pred.get("identity_state", "unknown"))
            is_unknown = truth.get("known_or_unknown") == "unknown"
            is_accepted = bool(identity_guess) and state in {"confirmed", "structure_verified"}
            is_correct = bool(target) and identity_guess == target and is_accepted
            hull_visible = as_bool(truth.get("hull_visible", True))

            if is_accepted:
                accepted_matches += 1
            if is_unknown:
                unknown_total += 1
                if is_accepted:
                    unknown_false_accepts += 1
                    false_matches += 1
                elif state == "out_of_archive":
                    unknown_rejections += 1
                else:
                    unresolved_unknown += 1
                continue

            known_total += 1
            if hull_visible:
                readable_known += 1
                exact_hull += int(hull_guess == target)
                edit_scores.append(normalized_edit_similarity(hull_guess, target))
            if is_accepted:
                if is_correct:
                    correct_matches += 1
                else:
                    false_matches += 1
            elif state == "out_of_archive":
                known_false_rejections += 1
                false_matches += 1
            else:
                unresolved_known += 1

            if target:
                for item in recognition_history.get(key, []):
                    if item.get("verified_identity") == target and item.get("identity_state") in {"confirmed", "structure_verified"}:
                        start_frame = first_observation.get(key, int(truth.get("frame_start", 0)))
                        confirmation_frames.append(max(0, int(item.get("frame_id", 0)) - start_frame))
                        break

        unknown_precision_denominator = unknown_rejections + known_false_rejections
        unknown_precision = unknown_rejections / max(1, unknown_precision_denominator)
        unknown_recall = unknown_rejections / max(1, unknown_total)
        metrics.update({
            "known_tracks": known_total,
            "unknown_tracks": unknown_total,
            "hull_readable_known_tracks": readable_known,
            "hull_recognition_success_rate": exact_hull / max(1, readable_known),
            "exact_hull_accuracy": exact_hull / max(1, readable_known),
            "normalized_edit_similarity": mean(edit_scores) if edit_scores else 0.0,
            "archive_matching_precision": correct_matches / max(1, accepted_matches),
            "archive_matching_success_rate": correct_matches / max(1, known_total),
            "identity_verification_accuracy": correct_matches / max(1, known_total),
            "unrecognized_known_rate": unresolved_known / max(1, known_total),
            "known_false_rejection_rate": known_false_rejections / max(1, known_total),
            "false_match_rate": false_matches / max(1, len(annotations)),
            "unknown_false_acceptance_rate": unknown_false_accepts / max(1, unknown_total),
            "unresolved_unknown_rate": unresolved_unknown / max(1, unknown_total),
            "unknown_rejection_precision": unknown_precision,
            "unknown_rejection_recall": unknown_recall,
            "unknown_rejection_f1": 2 * unknown_precision * unknown_recall / max(1e-9, unknown_precision + unknown_recall),
            "avg_confirmation_frames": mean(confirmation_frames) if confirmation_frames else 0.0,
            "confirmed_known_tracks": len(confirmation_frames),
        })
    return {key: round(value, 6) if isinstance(value, float) else value for key, value in metrics.items()}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    annotations = load_annotations(args.annotations) if args.annotations else None
    metrics = evaluate(args.run_dir, annotations)
    output = args.output or args.run_dir / "metrics.json"
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
