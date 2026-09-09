"""Build offline training samples for the learned cognitive policy.

The builder converts a policy-collection run into supervised samples.  It does
not retrain the policy and it does not modify the experiment outputs; it only
joins action, recognition, visual, entity, and ground-truth annotation streams
into a compact JSONL dataset.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


QUERY_ACTIONS = {"cloud_vlm_read", "verify_archive_identity", "reobserve_conflicting_identity"}
REVIEW_ACTIONS = {"request_remote_verification", "remote_report", "escalate_for_review"}
TERMINAL_KNOWN_STATES = {"confirmed", "structure_verified"}
TERMINAL_UNKNOWN_STATES = {"out_of_archive"}
ACTION_ORDER = ("defer", "query", "stop_known", "stop_out_of_archive", "escalate_review")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


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


def _members(row: dict[str, Any]) -> set[int]:
    return {int(value) for value in row.get("member_track_ids", []) if str(value).strip()}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _latest_before(rows: list[dict[str, Any]], frame_id: int) -> dict[str, Any]:
    eligible = [row for row in rows if _frame(row) < frame_id]
    if not eligible:
        return {}
    return max(eligible, key=lambda row: (_frame(row), _float(row.get("timestamp", 0.0))))


def _entity_index(entities: list[dict[str, Any]], stream_rows: list[dict[str, Any]]) -> dict[tuple[str, int], str]:
    index: dict[tuple[str, int], str] = {}
    for row in entities:
        video_id = str(row.get("video_id", ""))
        entity_id = str(row.get("entity_id", ""))
        for track_id in _members(row):
            index[(video_id, track_id)] = entity_id
    for row in stream_rows:
        video_id = str(row.get("video_id", ""))
        entity_id = str(row.get("entity_id", ""))
        track_id = _track_id(row)
        if video_id and entity_id and track_id:
            index[(video_id, track_id)] = entity_id
    return index


def _truth_index(annotations: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    truth_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in annotations:
        truth_by_video[str(row.get("video_id", ""))].append(row)

    index: dict[tuple[str, str], dict[str, Any]] = {}
    for video_id, rows in truth_by_video.items():
        for row in rows:
            entity_id = str(row.get("entity_id", ""))
            if entity_id:
                index[(video_id, entity_id)] = row
            for track_id in _members(row):
                index[(video_id, f"T{track_id}")] = row
    return index


def _best_truth_for(
    video_id: str,
    entity_id: str,
    track_id: int,
    entity_members: set[int],
    annotations: list[dict[str, Any]],
    truth_by_key: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any] | None:
    keyed = truth_by_key.get((video_id, entity_id)) or truth_by_key.get((video_id, f"T{track_id}"))
    if keyed:
        return keyed
    candidates = [row for row in annotations if str(row.get("video_id", "")) == video_id]
    if not candidates:
        return None
    members = set(entity_members)
    if track_id:
        members.add(track_id)
    best_row: dict[str, Any] | None = None
    best_overlap = 0
    for row in candidates:
        overlap = len(members & _members(row))
        if overlap > best_overlap:
            best_overlap = overlap
            best_row = row
    return best_row if best_overlap > 0 else None


def _teacher_action(action: dict[str, Any]) -> str:
    name = str(action.get("action", "") or "")
    if name in REVIEW_ACTIONS:
        return "escalate_review"
    if _bool(action.get("should_query")) or name in QUERY_ACTIONS:
        return "query"
    if name in {"wait_better_view", "defer_reasoning"}:
        return "defer"
    if name == "monitor_unknown":
        return "stop_out_of_archive"
    state_after = str(action.get("identity_state_after", "") or "")
    if state_after in TERMINAL_KNOWN_STATES:
        return "stop_known"
    if state_after in TERMINAL_UNKNOWN_STATES:
        return "stop_out_of_archive"
    return "defer"


def _is_truth_known(truth: dict[str, Any]) -> bool:
    return str(truth.get("known_or_unknown", "") or "").strip().lower() == "known"


def _is_truth_unknown(truth: dict[str, Any]) -> bool:
    return str(truth.get("known_or_unknown", "") or "").strip().lower() == "unknown"


def _matches_truth_identity(row: dict[str, Any], truth: dict[str, Any]) -> bool:
    truth_hull = _norm_identity(truth.get("hull_number", ""))
    if not truth_hull:
        return False
    candidates = [
        row.get("verified_identity", ""),
        row.get("archive_candidate_id", ""),
        row.get("fused_hull_number", ""),
        row.get("visual_candidate_id", ""),
    ]
    return any(_norm_identity(value) == truth_hull for value in candidates)


def _oracle_action(
    truth: dict[str, Any],
    action: dict[str, Any],
    recognition: dict[str, Any],
    features: dict[str, Any],
    *,
    min_quality: float,
    known_archive_score: float,
    unknown_archive_score: float,
    known_visual_score: float,
    known_visual_margin: float,
    known_visual_strong_score: float,
    unknown_visual_score: float,
    unknown_visual_margin: float,
    min_visual_rejection_observations: int,
    default_query_budget: int,
) -> str:
    quality = _float(features.get("observation_quality", 0.0))
    if quality < min_quality:
        return "defer"

    query_budget_used = _int(features.get("query_budget_used", 0))
    query_budget_max = _int(features.get("query_budget_max", 0), default_query_budget) or default_query_budget
    budget_exhausted = query_budget_used >= query_budget_max

    identity_state = str(recognition.get("identity_state", action.get("identity_state_before", "")) or "")
    archive_score = _float(recognition.get("archive_similarity_score", 0.0))
    structure_score = _float(features.get("structure_score", 0.0))
    visual_score = _float(features.get("visual_similarity_score", 0.0))
    visual_margin = _float(features.get("visual_margin", 0.0))
    visual_consistent = _int(features.get("visual_consistent_observations", 0))
    visual_observations = _int(features.get("visual_observation_count", 0))
    visual_low_score_observations = _int(features.get("visual_low_score_observations", 0))
    visual_only_mode = bool(_int(features.get("visual_only_mode", 0)))
    support = max(archive_score, structure_score, visual_score)

    if _is_truth_known(truth):
        if visual_only_mode:
            truth_hull = _norm_identity(truth.get("hull_number", ""))
            visual_candidate = _norm_identity(recognition.get("visual_candidate_id", ""))
            repeated_correct_visual = bool(
                truth_hull
                and visual_candidate == truth_hull
                and visual_consistent >= 2
                and (
                    visual_score >= known_visual_strong_score
                    or (
                        visual_score >= known_visual_score
                        and visual_margin >= known_visual_margin
                    )
                )
            )
            if repeated_correct_visual or (
                identity_state in TERMINAL_KNOWN_STATES and visual_candidate == truth_hull
            ):
                return "stop_known"
            return "escalate_review" if budget_exhausted else "query"
        if _matches_truth_identity(recognition, truth) and (
            identity_state in TERMINAL_KNOWN_STATES or support >= known_archive_score
        ):
            return "stop_known"
        return "escalate_review" if budget_exhausted else "query"

    if _is_truth_unknown(truth):
        if visual_only_mode:
            visual_candidate = _norm_identity(recognition.get("visual_candidate_id", ""))
            weak_visual = bool(visual_candidate) and (
                visual_score < unknown_visual_score and visual_margin < unknown_visual_margin
            )
            repeated_weak_visual = (
                weak_visual
                and visual_observations >= min_visual_rejection_observations
                and visual_low_score_observations >= min_visual_rejection_observations
            )
            if identity_state in TERMINAL_UNKNOWN_STATES or repeated_weak_visual:
                return "stop_out_of_archive"
            return "escalate_review" if budget_exhausted else "query"
        weak_archive = max(archive_score, structure_score) < unknown_archive_score
        weak_visual = visual_score < unknown_visual_score or visual_margin < unknown_visual_margin
        if identity_state in TERMINAL_UNKNOWN_STATES or (query_budget_used > 0 and weak_archive and weak_visual):
            return "stop_out_of_archive"
        return "escalate_review" if budget_exhausted else "query"

    return "escalate_review" if budget_exhausted else "query"


def _features(
    action: dict[str, Any],
    recognition: dict[str, Any],
    visual: dict[str, Any],
    entity: dict[str, Any],
    default_query_budget: int,
) -> dict[str, Any]:
    signals = dict(action.get("skill_signals", {}) or {})
    quality = _float(signals.get("observation_quality", 0.0))
    if not quality:
        quality = _float(recognition.get("observation_quality", 0.0))
    if not quality:
        quality = _float(action.get("observation_quality", 0.0))

    query_budget_max = _int(signals.get("query_budget_max", 0), default_query_budget) or default_query_budget
    query_budget_used = _int(signals.get("query_budget_used", 0))
    structure_scores = recognition.get("structure_candidate_scores", {}) or {}
    structure_score = max((_float(value) for value in structure_scores.values()), default=0.0)
    semantic_matches = recognition.get("semantic_matches", []) or []
    text_candidate = _norm_identity(
        recognition.get("archive_candidate_id", "")
        or recognition.get("verified_identity", "")
        or (semantic_matches[0].get("hull_number", "") if semantic_matches else "")
    )
    visual_candidate = _norm_identity(
        visual.get("visual_candidate_id", "") or recognition.get("visual_candidate_id", "")
    )
    identity_decision_mode = str(
        signals.get("identity_decision_mode", recognition.get("identity_decision_mode", "fused")) or "fused"
    ).lower()
    if identity_decision_mode == "visual_only":
        text_candidate = ""
        structure_score = 0.0

    return {
        "observation_quality": round(quality, 6),
        "identity_uncertainty": round(_float(signals.get("identity_uncertainty", recognition.get("uncertainty", 1.0)), 1.0), 6),
        "evidence_opportunity_score": round(_float(signals.get("evidence_opportunity_score", action.get("score", 0.0))), 6),
        "query_budget_used": query_budget_used,
        "query_budget_max": query_budget_max,
        "query_budget_fraction": round(min(1.0, query_budget_used / max(1, query_budget_max)), 6),
        "temporal_gap_frames": _int(signals.get("temporal_gap_frames", 0)),
        "quality_improvement": round(_float(signals.get("quality_improvement", 0.0)), 6),
        "novel_tracklet_view": int(_bool(signals.get("novel_tracklet_view", False))),
        "structure_score": round(structure_score, 6),
        "structure_evidence_count": 0 if identity_decision_mode == "visual_only" else _int(recognition.get("structure_evidence_count", entity.get("structure_evidence_count", 0))),
        "structure_consistent_observations": 0 if identity_decision_mode == "visual_only" else _int(recognition.get("structure_consistent_observations", 0)),
        "structure_conflict_observations": 0 if identity_decision_mode == "visual_only" else _int(recognition.get("structure_conflict_observations", 0)),
        "visual_similarity_score": round(_float(visual.get("visual_similarity_score", recognition.get("visual_similarity_score", 0.0))), 6),
        "visual_margin": round(_float(visual.get("visual_margin", recognition.get("visual_margin", 0.0))), 6),
        "visual_observation_count": _int(signals.get("visual_observation_count", recognition.get("visual_observation_count", 0))),
        "visual_consistent_observations": _int(signals.get("visual_consistent_observations", recognition.get("visual_consistent_observations", 0))),
        "visual_low_score_observations": _int(signals.get("visual_low_score_observations", recognition.get("visual_low_score_observations", 0))),
        "text_visual_agree": int(bool(text_candidate and visual_candidate and text_candidate == visual_candidate)),
        "visual_only_mode": int(identity_decision_mode == "visual_only"),
        "recognized_before": int(str(action.get("identity_state_before", "unknown")) not in {"unknown", ""}),
        "risk_score": round(_float(signals.get("risk_score", recognition.get("risk_score", 0.0))), 6),
    }


def build_policy_dataset(
    run_dir: Path,
    annotations_path: Path,
    output_path: Path,
    *,
    default_query_budget: int = 3,
    min_quality: float = 0.25,
    known_archive_score: float = 0.80,
    unknown_archive_score: float = 0.70,
    known_visual_score: float = 0.70,
    known_visual_margin: float = 0.03,
    known_visual_strong_score: float = 0.90,
    unknown_visual_score: float = 0.70,
    unknown_visual_margin: float = 0.08,
    min_visual_rejection_observations: int = 2,
) -> dict[str, Any]:
    actions = read_jsonl(run_dir / "actions.jsonl")
    recognitions = read_jsonl(run_dir / "recognition.jsonl")
    visual_rows = read_jsonl(run_dir / "visual.jsonl")
    entities = read_jsonl(run_dir / "entities.jsonl")
    annotations = read_jsonl(annotations_path)

    all_stream_rows = actions + recognitions + visual_rows
    entity_by_track = _entity_index(entities, all_stream_rows)
    entity_by_key = {(str(row.get("video_id", "")), str(row.get("entity_id", ""))): row for row in entities}
    truth_by_key = _truth_index(annotations)
    recognitions_by_entity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    visual_by_entity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in recognitions:
        video_id = str(row.get("video_id", ""))
        track_id = _track_id(row)
        entity_id = str(row.get("entity_id", "")) or entity_by_track.get((video_id, track_id), f"T{track_id}")
        recognitions_by_entity[(video_id, entity_id)].append(row)
    for row in visual_rows:
        video_id = str(row.get("video_id", ""))
        track_id = _track_id(row)
        entity_id = str(row.get("entity_id", "")) or entity_by_track.get((video_id, track_id), f"T{track_id}")
        visual_by_entity[(video_id, entity_id)].append(row)

    for rows in recognitions_by_entity.values():
        rows.sort(key=lambda row: (_frame(row), _track_id(row)))
    for rows in visual_by_entity.values():
        rows.sort(key=lambda row: (_frame(row), _track_id(row)))

    rows: list[dict[str, Any]] = []
    skipped_no_truth = 0
    for action in sorted(actions, key=lambda row: (str(row.get("video_id", "")), _frame(row), _track_id(row))):
        video_id = str(action.get("video_id", ""))
        track_id = _track_id(action)
        entity_id = str(action.get("entity_id", "")) or entity_by_track.get((video_id, track_id), f"T{track_id}")
        entity = entity_by_key.get((video_id, entity_id), {})
        entity_members = _members(entity)
        truth = _best_truth_for(video_id, entity_id, track_id, entity_members, annotations, truth_by_key)
        if not truth:
            skipped_no_truth += 1
            continue
        frame_id = _frame(action)
        recognition = _latest_before(recognitions_by_entity.get((video_id, entity_id), []), frame_id)
        visual = _latest_before(visual_by_entity.get((video_id, entity_id), []), frame_id)
        feature_values = _features(action, recognition, visual, entity, default_query_budget)
        teacher = _teacher_action(action)
        oracle = _oracle_action(
            truth,
            action,
            recognition,
            feature_values,
            min_quality=min_quality,
            known_archive_score=known_archive_score,
            unknown_archive_score=unknown_archive_score,
            known_visual_score=known_visual_score,
            known_visual_margin=known_visual_margin,
            known_visual_strong_score=known_visual_strong_score,
            unknown_visual_score=unknown_visual_score,
            unknown_visual_margin=unknown_visual_margin,
            min_visual_rejection_observations=min_visual_rejection_observations,
            default_query_budget=default_query_budget,
        )
        rows.append({
            "sample_id": f"{video_id}:{entity_id}:T{track_id}:F{frame_id}",
            "video_id": video_id,
            "entity_id": entity_id,
            "track_id": track_id,
            "frame_id": frame_id,
            "features": feature_values,
            "oracle_action": oracle,
            "teacher_action": teacher,
            "teacher_should_query": _bool(action.get("should_query")),
            "truth": {
                "entity_id": str(truth.get("entity_id", "")),
                "known_or_unknown": str(truth.get("known_or_unknown", "")),
                "hull_number": _norm_identity(truth.get("hull_number", "")),
                "vessel_id": str(truth.get("vessel_id", "")),
            },
            "evidence": {
                "identity_state_before": str(action.get("identity_state_before", "")),
                "identity_state_after": str(action.get("identity_state_after", "")),
                "recognition_identity_state": str(recognition.get("identity_state", "")),
                "verified_identity": _norm_identity(recognition.get("verified_identity", "")),
                "archive_candidate_id": _norm_identity(recognition.get("archive_candidate_id", "")),
                "visual_candidate_id": _norm_identity(visual.get("visual_candidate_id", recognition.get("visual_candidate_id", ""))),
                "action": str(action.get("action", "")),
                "reasons": list(action.get("reasons", []) or []),
            },
        })

    write_jsonl(output_path, rows)
    action_counts = Counter(row["oracle_action"] for row in rows)
    teacher_counts = Counter(row["teacher_action"] for row in rows)
    summary = {
        "run_dir": str(run_dir),
        "annotations": str(annotations_path),
        "output": str(output_path),
        "samples": len(rows),
        "skipped_no_truth": skipped_no_truth,
        "oracle_action_counts": {name: action_counts.get(name, 0) for name in ACTION_ORDER},
        "teacher_action_counts": {name: teacher_counts.get(name, 0) for name in ACTION_ORDER},
        "oracle_query_rate": round(action_counts.get("query", 0) / max(1, len(rows)), 6),
        "teacher_query_rate": round(sum(1 for row in rows if row["teacher_should_query"]) / max(1, len(rows)), 6),
        "truth_known_samples": sum(1 for row in rows if row["truth"]["known_or_unknown"] == "known"),
        "truth_unknown_samples": sum(1 for row in rows if row["truth"]["known_or_unknown"] == "unknown"),
        "thresholds": {
            "default_query_budget": default_query_budget,
            "min_quality": min_quality,
            "known_archive_score": known_archive_score,
            "unknown_archive_score": unknown_archive_score,
            "known_visual_score": known_visual_score,
            "known_visual_margin": known_visual_margin,
            "known_visual_strong_score": known_visual_strong_score,
            "unknown_visual_score": unknown_visual_score,
            "unknown_visual_margin": unknown_visual_margin,
            "min_visual_rejection_observations": min_visual_rejection_observations,
        },
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build supervised samples for learned cognitive-policy training.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Experiment run directory containing actions/recognition/visual/entities JSONL logs.")
    parser.add_argument("--annotations", required=True, type=Path, help="Entity-level ground truth annotations for the same split.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path for policy samples.")
    parser.add_argument("--default-query-budget", type=int, default=3)
    parser.add_argument("--min-quality", type=float, default=0.25)
    parser.add_argument("--known-archive-score", type=float, default=0.80)
    parser.add_argument("--unknown-archive-score", type=float, default=0.70)
    parser.add_argument("--known-visual-score", type=float, default=0.70)
    parser.add_argument("--known-visual-margin", type=float, default=0.03)
    parser.add_argument("--known-visual-strong-score", type=float, default=0.90)
    parser.add_argument("--unknown-visual-score", type=float, default=0.70)
    parser.add_argument("--unknown-visual-margin", type=float, default=0.08)
    parser.add_argument("--min-visual-rejection-observations", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_policy_dataset(
        args.run_dir,
        args.annotations,
        args.output,
        default_query_budget=args.default_query_budget,
        min_quality=args.min_quality,
        known_archive_score=args.known_archive_score,
        unknown_archive_score=args.unknown_archive_score,
        known_visual_score=args.known_visual_score,
        known_visual_margin=args.known_visual_margin,
        known_visual_strong_score=args.known_visual_strong_score,
        unknown_visual_score=args.unknown_visual_score,
        unknown_visual_margin=args.unknown_visual_margin,
        min_visual_rejection_observations=args.min_visual_rejection_observations,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
