"""Audit known-vessel identity failures with evidence details.

The entity evaluator reports aggregate known false rejection, unresolved known
and wrong identity counts.  This script expands those counts back into concrete
truth entities, predicted fragments, recognition records and visual/archive
evidence so that failures can be inspected manually.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


KNOWN_ACCEPT_STATES = {"confirmed", "structure_verified"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _members(row: dict[str, Any]) -> set[int]:
    return {int(value) for value in row.get("member_track_ids", []) if str(value).strip()}


def _frame(row: dict[str, Any]) -> int:
    return int(row.get("frame_id", row.get("last_frame", row.get("frame_end", 0))) or 0)


def _track_id(row: dict[str, Any]) -> int:
    return int(row.get("track_id", 0) or 0)


def _norm_identity(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    base = text.split("_", 1)[0]
    return base.zfill(3) if base.isdigit() else base


def _unique(values: list[Any]) -> list[Any]:
    output: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if value in {"", None}:
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        key = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else str(value)
        if key not in seen:
            seen.add(key)
            output.append(value)
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
        if members:
            filtered.append({**row, "member_track_ids": members})
    return filtered


def _track_entity_index(entities: list[dict[str, Any]], stream_rows: list[dict[str, Any]]) -> dict[tuple[str, int], str]:
    index: dict[tuple[str, int], str] = {}
    for row in entities:
        video_id = str(row.get("video_id", ""))
        entity_id = str(row.get("entity_id", ""))
        for track_id in _members(row):
            if video_id and entity_id:
                index[(video_id, track_id)] = entity_id
    for row in stream_rows:
        video_id = str(row.get("video_id", ""))
        entity_id = str(row.get("entity_id", ""))
        track_id = _track_id(row)
        if video_id and entity_id and track_id:
            index[(video_id, track_id)] = entity_id
    return index


def _rows_for_fragment(
    rows: list[dict[str, Any]],
    video_id: str,
    fragment_ids: set[str],
    track_ids: set[int],
    track_entity: dict[tuple[str, int], str],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("video_id", "")) != video_id:
            continue
        entity_id = str(row.get("entity_id", ""))
        track_id = _track_id(row)
        inferred_entity = entity_id or track_entity.get((video_id, track_id), "")
        if inferred_entity in fragment_ids or track_id in track_ids:
            selected.append(row)
    return sorted(selected, key=lambda item: (_frame(item), _track_id(item)))


def _top_visual(row: dict[str, Any]) -> dict[str, Any]:
    matches = row.get("visual_matches", []) or []
    if matches:
        top = matches[0]
        return {
            "identity": _norm_identity(top.get("hull_number", top.get("identity", ""))),
            "score": float(top.get("score", 0.0) or 0.0),
            "image_path": str(top.get("image_path", "") or ""),
        }
    return {
        "identity": _norm_identity(row.get("visual_candidate_id", "")),
        "score": float(row.get("visual_similarity_score", 0.0) or 0.0),
        "image_path": "",
    }


def _recognition_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        visual = _top_visual(row)
        output.append({
            "frame_id": _frame(row),
            "track_id": _track_id(row),
            "entity_id": str(row.get("entity_id", "") or ""),
            "identity_state": str(row.get("identity_state", "") or ""),
            "verified_identity": _norm_identity(row.get("verified_identity", "")),
            "archive_candidate_id": _norm_identity(row.get("archive_candidate_id", "")),
            "archive_similarity_score": float(row.get("archive_similarity_score", 0.0) or 0.0),
            "raw_hull_number": str(row.get("raw_hull_number", "") or ""),
            "fused_hull_number": str(row.get("fused_hull_number", "") or ""),
            "visual_identity": visual["identity"],
            "visual_score": visual["score"],
            "visual_margin": float(row.get("visual_margin", 0.0) or 0.0),
            "visual_observation_count": int(row.get("visual_observation_count", 0) or 0),
            "visual_consistent_observations": int(row.get("visual_consistent_observations", 0) or 0),
            "visual_low_score_observations": int(row.get("visual_low_score_observations", 0) or 0),
            "visual_image_path": visual["image_path"],
            "description": str(row.get("observed_structure_description", "") or "")[:240],
            "visual_gate_reason": str(row.get("visual_gate_reason", "") or ""),
            "identity_evidence_source": str(row.get("identity_evidence_source", "") or ""),
            "identity_decision_mode": str(row.get("identity_decision_mode", "fused") or "fused"),
        })
    return output


def _latest_by_frame(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (_frame(row), _track_id(row)), reverse=True)[:limit]


def _classify_known_truth(truth_row: dict[str, Any], fragments: list[dict[str, Any]]) -> tuple[str, list[str], list[str]]:
    target = _norm_identity(truth_row.get("hull_number", ""))
    accepted = [
        _norm_identity(pred.get("verified_identity", ""))
        for pred in fragments
        if _norm_identity(pred.get("verified_identity", ""))
        and str(pred.get("identity_state", "unknown")) in KNOWN_ACCEPT_STATES
    ]
    states = [str(pred.get("identity_state", "unknown")) for pred in fragments]
    if any(identity and identity != target for identity in accepted):
        return "known_wrong_identity", accepted, states
    if any(identity == target for identity in accepted):
        return "known_correct", accepted, states
    if states and all(state == "out_of_archive" for state in states):
        return "known_false_rejection", accepted, states
    if any(state == "out_of_archive" for state in states):
        return "known_partial_rejection", accepted, states
    return "known_unresolved", accepted, states


def audit_identity_errors(
    run_dir: Path,
    entity_annotations: Path,
    *,
    include_correct: bool = False,
    evidence_limit: int = 6,
) -> list[dict[str, Any]]:
    truth = read_jsonl(entity_annotations)
    raw_entities = list({
        (str(row.get("video_id", "")), str(row.get("entity_id", ""))): row
        for row in read_jsonl(run_dir / "entities.jsonl")
    }.values())
    predictions = _filter_predictions_to_annotations(raw_entities, truth)
    recognitions = read_jsonl(run_dir / "recognition.jsonl")
    visual_rows = read_jsonl(run_dir / "visual.jsonl")
    actions = read_jsonl(run_dir / "actions.jsonl")
    edge_rows = read_jsonl(run_dir / "edge_cloud.jsonl")
    track_entity = _track_entity_index(predictions, recognitions + visual_rows + actions + edge_rows)

    rows: list[dict[str, Any]] = []
    for truth_row in truth:
        if str(truth_row.get("known_or_unknown", "")).lower() != "known":
            continue
        video_id = str(truth_row.get("video_id", ""))
        truth_members = _members(truth_row)
        fragments = [
            pred for pred in predictions
            if str(pred.get("video_id", "")) == video_id and bool(_members(pred) & truth_members)
        ]
        category, accepted, states = _classify_known_truth(truth_row, fragments)
        if category == "known_correct" and not include_correct:
            continue

        fragment_ids = {str(pred.get("entity_id", "")) for pred in fragments if str(pred.get("entity_id", ""))}
        fragment_tracks = set().union(*(_members(pred) for pred in fragments)) if fragments else set()
        related_tracks = truth_members | fragment_tracks
        related_recognition = _rows_for_fragment(recognitions, video_id, fragment_ids, related_tracks, track_entity)
        related_visual = _rows_for_fragment(visual_rows, video_id, fragment_ids, related_tracks, track_entity)
        related_actions = _rows_for_fragment(actions, video_id, fragment_ids, related_tracks, track_entity)
        related_edge = _rows_for_fragment(edge_rows, video_id, fragment_ids, related_tracks, track_entity)
        latest_recognition = _recognition_summary(_latest_by_frame(related_recognition, evidence_limit))
        latest_visual = _latest_by_frame(related_visual, evidence_limit)
        learned_actions = Counter(
            str((row.get("skill_signals", {}) or {}).get("learned_policy_action", "") or "")
            for row in related_actions
            if (row.get("skill_signals", {}) or {}).get("learned_policy_action")
        )
        review_actions = []
        blocked_review_actions = []
        for action_row in related_actions:
            if str(action_row.get("action", "")) not in {"request_remote_verification", "remote_report"}:
                continue
            signals = action_row.get("skill_signals", {}) or {}
            review_record = {
                "frame_id": _frame(action_row),
                "track_id": _track_id(action_row),
                "entity_id": str(action_row.get("entity_id", "") or ""),
                "action": str(action_row.get("action", "") or ""),
                "identity_state_before": str(action_row.get("identity_state_before", "") or ""),
                "identity_state_after": str(action_row.get("identity_state_after", "") or ""),
                "learned_policy_action": str(signals.get("learned_policy_action", "") or ""),
                "learned_policy_confidence": float(signals.get("learned_policy_confidence", 0.0) or 0.0),
                "query_budget_used": int(signals.get("query_budget_used", 0) or 0),
                "query_budget_max": int(signals.get("query_budget_max", 0) or 0),
                "reasons": list(action_row.get("reasons", []) or []),
            }
            if str(action_row.get("identity_state_after", "")) == "review_requested":
                review_actions.append(review_record)
            elif action_row.get("review_request_applied") is False:
                blocked_review_actions.append(review_record)
        query_count = sum(max(1, int(row.get("request_count", 1) or 1)) for row in related_edge)
        rows.append({
            "category": category,
            "video_id": video_id,
            "truth_entity_id": str(truth_row.get("entity_id", "") or ""),
            "truth_vessel_id": str(truth_row.get("vessel_id", "") or ""),
            "truth_hull_number": _norm_identity(truth_row.get("hull_number", "")),
            "truth_member_track_ids": sorted(truth_members),
            "frame_start": int(truth_row.get("frame_start", 0) or 0),
            "frame_end": int(truth_row.get("frame_end", 0) or 0),
            "fragment_count": len(fragments),
            "predicted_entity_ids": sorted(fragment_ids),
            "predicted_member_track_ids": sorted(fragment_tracks),
            "predicted_states": states,
            "accepted_identities": accepted,
            "verified_identities": _unique([_norm_identity(pred.get("verified_identity", "")) for pred in fragments]),
            "archive_candidate_ids": _unique([_norm_identity(pred.get("archive_candidate_id", "")) for pred in fragments]),
            "recognition_count": len(related_recognition),
            "visual_count": len(related_visual),
            "query_count": query_count,
            "latest_recognition": latest_recognition,
            "latest_visual_candidates": [
                {
                    "frame_id": _frame(row),
                    "track_id": _track_id(row),
                    "entity_id": str(row.get("entity_id", "") or ""),
                    "visual_candidate_id": _norm_identity(row.get("visual_candidate_id", "")),
                    "visual_similarity_score": float(row.get("visual_similarity_score", 0.0) or 0.0),
                    "visual_margin": float(row.get("visual_margin", 0.0) or 0.0),
                    "visual_observation_count": int(row.get("visual_observation_count", 0) or 0),
                    "visual_consistent_observations": int(row.get("visual_consistent_observations", 0) or 0),
                }
                for row in latest_visual
            ],
            "target_image_paths": _unique([str(row.get("target_image_path", "") or "") for row in related_edge])[:evidence_limit],
            "learned_policy_actions": dict(learned_actions),
            "review_actions": review_actions,
            "blocked_review_actions": blocked_review_actions,
        })
    order = {
        "known_false_rejection": 0,
        "known_partial_rejection": 1,
        "known_wrong_identity": 2,
        "known_unresolved": 3,
        "known_correct": 4,
    }
    return sorted(rows, key=lambda row: (order.get(row["category"], 9), row["video_id"], row["truth_entity_id"]))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "category",
        "video_id",
        "truth_entity_id",
        "truth_vessel_id",
        "truth_hull_number",
        "truth_member_track_ids",
        "predicted_entity_ids",
        "predicted_states",
        "accepted_identities",
        "verified_identities",
        "archive_candidate_ids",
        "fragment_count",
        "recognition_count",
        "visual_count",
        "query_count",
        "frame_start",
        "frame_end",
        "target_image_paths",
        "learned_policy_actions",
        "review_actions",
        "blocked_review_actions",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: json.dumps(row.get(field, ""), ensure_ascii=False) if isinstance(row.get(field), (list, dict)) else row.get(field, "")
                for field in fields
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--entity-annotations", "--annotations", required=True, type=Path)
    parser.add_argument("--category", default="problem", choices=[
        "problem",
        "known_false_rejection",
        "known_partial_rejection",
        "known_wrong_identity",
        "known_unresolved",
        "known_correct",
        "all",
    ])
    parser.add_argument("--include-correct", action="store_true")
    parser.add_argument("--evidence-limit", type=int, default=6)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--print-limit", type=int, default=80)
    args = parser.parse_args()

    rows = audit_identity_errors(
        args.run_dir,
        args.entity_annotations,
        include_correct=args.include_correct or args.category in {"all", "known_correct"},
        evidence_limit=args.evidence_limit,
    )
    if args.category == "problem":
        rows = [row for row in rows if row["category"] != "known_correct"]
    elif args.category != "all":
        rows = [row for row in rows if row["category"] == args.category]

    output = args.output or args.run_dir / "identity_error_audit.jsonl"
    csv_output = args.csv_output or args.run_dir / "identity_error_audit.csv"
    _write_jsonl(output, rows)
    _write_csv(csv_output, rows)

    counts = Counter(row["category"] for row in rows)
    print(json.dumps({
        "run_dir": str(args.run_dir),
        "annotations": str(args.entity_annotations),
        "rows": len(rows),
        "counts": dict(counts),
        "output": str(output),
        "csv_output": str(csv_output),
    }, ensure_ascii=False, indent=2, sort_keys=True))
    for row in rows[:max(0, args.print_limit)]:
        latest = row["latest_recognition"][0] if row["latest_recognition"] else {}
        review = row["review_actions"][-1] if row["review_actions"] else {}
        print(
            f"{row['category']} video={row['video_id']} truth={row['truth_entity_id']} "
            f"vessel={row['truth_vessel_id']} hull={row['truth_hull_number']} "
            f"tracks={row['truth_member_track_ids']} pred_entities={row['predicted_entity_ids']} "
            f"states={row['predicted_states']} accepted={row['accepted_identities']} "
            f"last_frame={latest.get('frame_id', 'N/A')} last_state={latest.get('identity_state', 'N/A')} "
            f"last_candidate={latest.get('archive_candidate_id', 'N/A')} visual={latest.get('visual_identity', 'N/A')} "
            f"visual_score={latest.get('visual_score', 'N/A')} visual_margin={latest.get('visual_margin', 'N/A')} "
            f"visual_count={latest.get('visual_observation_count', 'N/A')} "
            f"visual_consistent={latest.get('visual_consistent_observations', 'N/A')} "
            f"visual_low={latest.get('visual_low_score_observations', 'N/A')} "
            f"review_frame={review.get('frame_id', 'N/A')} review_before={review.get('identity_state_before', 'N/A')} "
            f"learned_action={review.get('learned_policy_action', 'N/A')} "
            f"budget={review.get('query_budget_used', 'N/A')}/{review.get('query_budget_max', 'N/A')}"
        )


if __name__ == "__main__":
    main()
