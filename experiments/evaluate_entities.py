"""Evaluate tracklet reconciliation and archive identity at vessel-entity level."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _members(row: dict[str, Any]) -> set[int]:
    return {int(value) for value in row.get("member_track_ids", [])}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _pairs(rows: list[dict[str, Any]]) -> set[tuple[str, int, int]]:
    output: set[tuple[str, int, int]] = set()
    for row in rows:
        video_id = str(row.get("video_id", ""))
        members = sorted(_members(row))
        output.update((video_id, left, right) for left, right in itertools.combinations(members, 2))
    return output


def _filter_predictions_to_annotations(
    predictions: list[dict[str, Any]],
    truth: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    annotated: dict[str, set[int]] = defaultdict(set)
    for row in truth:
        annotated[str(row.get("video_id", ""))].update(_members(row))
    filtered: list[dict[str, Any]] = []
    for row in predictions:
        video_id = str(row.get("video_id", ""))
        members = sorted(_members(row) & annotated.get(video_id, set()))
        if not members:
            continue
        filtered.append({**row, "member_track_ids": members})
    return filtered


def evaluate_entities(run_dir: Path, annotations_path: Path) -> dict[str, Any]:
    truth = read_jsonl(annotations_path)
    raw_predictions = list({
        (str(row.get("video_id", "")), str(row.get("entity_id", ""))): row
        for row in read_jsonl(run_dir / "entities.jsonl")
    }.values())
    predictions = _filter_predictions_to_annotations(raw_predictions, truth)
    edge = read_jsonl(run_dir / "edge_cloud.jsonl")
    episodes = read_jsonl(run_dir / "episodes.jsonl")
    review_records = read_jsonl(run_dir / "review_records.jsonl")
    review_outcomes = read_jsonl(run_dir / "review_outcomes.jsonl")
    episode_by_key = {(str(row.get("video_id", "")), str(row.get("entity_id", ""))): row for row in episodes}
    review_by_key = {(str(row.get("video_id", "")), str(row.get("entity_id", ""))): row for row in review_records}
    outcomes_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in review_outcomes:
        outcomes_by_key[(str(row.get("video_id", "")), str(row.get("entity_id", "")))].append(row)

    truth_pairs, predicted_pairs = _pairs(truth), _pairs(predictions)
    pair_intersection = len(truth_pairs & predicted_pairs)
    pair_precision = 1.0 if not predicted_pairs and not truth_pairs else pair_intersection / max(1, len(predicted_pairs))
    pair_recall = 1.0 if not predicted_pairs and not truth_pairs else pair_intersection / max(1, len(truth_pairs))
    pair_f1 = 2 * pair_precision * pair_recall / max(1e-9, pair_precision + pair_recall)

    calls: dict[tuple[str, str], int] = defaultdict(int)
    upload_bytes: dict[tuple[str, str], int] = defaultdict(int)
    for row in edge:
        key = (str(row.get("video_id", "")), str(row.get("entity_id", "")))
        calls[key] += max(1, int(row.get("request_count", 1)))
        upload_bytes[key] += int(row.get("upload_bytes", 0))

    known = unknown = correct_known = false_known_accept = false_known_reject = unresolved_known = 0
    unknown_accept = unknown_reject = unresolved_unknown = 0
    matched_tracklets = 0
    entity_calls: list[int] = []
    entity_uploads: list[int] = []
    fragments_per_truth: list[int] = []
    complete_reconciliations = 0
    autonomous_correct = review_correct = 0
    review_required_entities = escalated_entities = correctly_escalated = 0
    inherited_opportunities = inherited_correct = 0
    evaluated_episode_keys: set[tuple[str, str]] = set()
    review_completeness: list[float] = []

    truth_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in truth:
        truth_by_video[str(row.get("video_id", ""))].append(row)
    undersegmented_predictions = 0
    for pred in predictions:
        video_id = str(pred.get("video_id", ""))
        overlapping_truths = sum(bool(_members(pred) & _members(row)) for row in truth_by_video.get(video_id, []))
        undersegmented_predictions += int(overlapping_truths > 1)

    for truth_row in truth:
        video_id = str(truth_row.get("video_id", ""))
        truth_members = _members(truth_row)
        fragments = [
            pred for pred in predictions
            if str(pred.get("video_id", "")) == video_id and bool(_members(pred) & truth_members)
        ]
        fragments_per_truth.append(len(fragments))
        covered = set().union(*(_members(pred) & truth_members for pred in fragments)) if fragments else set()
        matched_tracklets += len(covered)
        complete_reconciliations += int(len(fragments) == 1 and _members(fragments[0]) == truth_members)

        fragment_keys = {(video_id, str(pred.get("entity_id", ""))) for pred in fragments}
        evaluated_episode_keys.update(fragment_keys)
        entity_calls.append(sum(calls.get(key, 0) for key in fragment_keys))
        entity_uploads.append(sum(upload_bytes.get(key, 0) for key in fragment_keys))
        escalated = any(key in review_by_key for key in fragment_keys)
        requires_review = _as_bool(truth_row.get("review_required", truth_row.get("requires_review", False)))
        review_required_entities += int(requires_review)
        escalated_entities += int(escalated)
        correctly_escalated += int(escalated and requires_review)
        review_completeness.extend(float(review_by_key[key].get("completeness", 0.0)) for key in fragment_keys if key in review_by_key)
        outcomes = [outcome for key in fragment_keys for outcome in outcomes_by_key.get(key, [])]
        latest_outcome = max(outcomes, key=lambda row: float(row.get("timestamp", 0.0)), default={})

        accepted = [
            str(pred.get("verified_identity", "") or "")
            for pred in fragments
            if str(pred.get("verified_identity", "") or "")
            and str(pred.get("identity_state", "unknown")) in {"confirmed", "structure_verified"}
        ]
        states = [str(pred.get("identity_state", "unknown")) for pred in fragments]

        autonomously_correct = False
        resolved_by_review = False
        if truth_row.get("known_or_unknown") == "unknown":
            unknown += 1
            if accepted:
                unknown_accept += 1
            elif states and all(state == "out_of_archive" for state in states):
                unknown_reject += 1
                autonomously_correct = True
            else:
                unresolved_unknown += 1
            outcome_state = str(latest_outcome.get("identity_state", latest_outcome.get("decision", ""))).lower()
            resolved_by_review = bool(latest_outcome) and outcome_state in {"out_of_archive", "unknown", "reject", "rejected"}
        else:
            known += 1
            target = str(truth_row.get("hull_number", "") or "")
            if any(identity != target for identity in accepted):
                false_known_accept += 1
            elif any(identity == target for identity in accepted):
                correct_known += 1
                autonomously_correct = True
            elif states and all(state == "out_of_archive" for state in states):
                false_known_reject += 1
            else:
                unresolved_known += 1
            outcome_identity = str(latest_outcome.get("verified_identity", latest_outcome.get("identity", "")) or "")
            resolved_by_review = bool(latest_outcome) and outcome_identity == target
            opportunities = max(0, len(truth_members) - 1)
            inherited_opportunities += opportunities
            if opportunities and autonomously_correct and len(fragments) == 1 and _members(fragments[0]) == truth_members:
                inherited_correct += opportunities

        autonomous_correct += int(autonomously_correct)
        review_correct += int(not autonomously_correct and resolved_by_review)

    truth_tracklets = sum(len(_members(row)) for row in truth)
    predicted_tracklet_memberships = sum(len(_members(row)) for row in predictions)
    evaluated_episodes = [episode_by_key[key] for key in evaluated_episode_keys if key in episode_by_key]
    costly_actions = sum(int(row.get("costly_action_count", 0)) for row in evaluated_episodes)
    evidence_gain_sum = sum(float(row.get("evidence_gain_sum", 0.0)) for row in evaluated_episodes)
    unnecessary_queries = sum(int(row.get("unnecessary_query_count", 0)) for row in evaluated_episodes)
    entity_total = known + unknown
    total_calls = sum(entity_calls)
    metrics = {
        "entity_annotations": len(truth),
        "predicted_entities": len(predictions),
        "entity_association_pair_precision": pair_precision,
        "entity_association_pair_recall": pair_recall,
        "entity_association_pair_f1": pair_f1,
        "entity_tracklet_coverage": matched_tracklets / max(1, truth_tracklets),
        "tracklets_per_predicted_entity": predicted_tracklet_memberships / max(1, len(predictions)),
        "predicted_fragments_per_truth_entity": mean(fragments_per_truth) if fragments_per_truth else 0.0,
        "complete_entity_reconciliation_rate": complete_reconciliations / max(1, len(truth)),
        "entity_oversegmentation_rate": sum(count > 1 for count in fragments_per_truth) / max(1, len(truth)),
        "entity_undersegmentation_rate": undersegmented_predictions / max(1, len(predictions)),
        "known_entities": known,
        "unknown_entities": unknown,
        "entity_archive_matching_precision": correct_known / max(1, correct_known + false_known_accept + unknown_accept),
        "entity_archive_matching_success_rate": correct_known / max(1, known),
        "entity_identity_verification_accuracy": correct_known / max(1, known),
        "entity_known_false_acceptance_rate": false_known_accept / max(1, known),
        "entity_known_false_rejection_rate": false_known_reject / max(1, known),
        "entity_unresolved_known_rate": unresolved_known / max(1, known),
        "entity_unknown_false_acceptance_rate": unknown_accept / max(1, unknown),
        "entity_unknown_rejection_precision": unknown_reject / max(1, unknown_reject + false_known_reject),
        "entity_unknown_rejection_recall": unknown_reject / max(1, unknown),
        "entity_unknown_rejection_f1": 2 * unknown_reject / max(1, 2 * unknown_reject + false_known_reject + unresolved_unknown + unknown_accept),
        "entity_unresolved_unknown_rate": unresolved_unknown / max(1, unknown),
        "avg_vlm_calls_per_entity": mean(entity_calls) if entity_calls else 0.0,
        "avg_upload_bytes_per_entity": mean(entity_uploads) if entity_uploads else 0.0,
        "known_identity_accuracy": correct_known / max(1, known),
        "unknown_false_acceptance_rate": unknown_accept / max(1, unknown),
        "unknown_rejection_recall": unknown_reject / max(1, unknown),
        "autonomous_task_success_rate": autonomous_correct / max(1, entity_total),
        "safe_handling_success_rate": (autonomous_correct + review_correct) / max(1, entity_total),
        "review_resolved_entities": review_correct,
        "evidence_gain_per_action": evidence_gain_sum / max(1, costly_actions),
        "unnecessary_query_rate": unnecessary_queries / max(1, costly_actions),
        "success_per_vlm_call": autonomous_correct / max(1, total_calls),
        "inherited_identity_accuracy": inherited_correct / max(1, inherited_opportunities),
        "inherited_identity_opportunities": inherited_opportunities,
        "review_requested_rate": escalated_entities / max(1, entity_total),
        "escalation_precision": correctly_escalated / max(1, escalated_entities),
        "review_needed_recall": correctly_escalated / max(1, review_required_entities),
        "review_record_completeness": mean(review_completeness) if review_completeness else 0.0,
    }
    return {key: round(value, 6) if isinstance(value, float) else value for key, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = evaluate_entities(args.run_dir, args.annotations)
    output = args.output or args.run_dir / "entity_metrics.json"
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
