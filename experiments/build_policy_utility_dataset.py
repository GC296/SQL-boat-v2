"""Add counterfactual action utilities to policy samples.

This script keeps the existing policy samples intact and adds a utility vector
for every cognitive action.  The utilities are automatically derived from
ground-truth identity status, the evidence state available before the action,
and configurable task costs.  No extra manual labels are required.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ACTIONS = ["defer", "query", "stop_known", "stop_out_of_archive", "escalate_review"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _norm_identity(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    base = text.split("_", 1)[0]
    return base.zfill(3) if base.isdigit() else base


def _clip(value: float, lower: float = -1.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))


def _truth_status(sample: dict[str, Any]) -> str:
    return str((sample.get("truth", {}) or {}).get("known_or_unknown", "") or "").strip().lower()


def _truth_hull(sample: dict[str, Any]) -> str:
    return _norm_identity((sample.get("truth", {}) or {}).get("hull_number", ""))


def _candidate_ids(sample: dict[str, Any]) -> set[str]:
    evidence = sample.get("evidence", {}) or {}
    return {
        _norm_identity(evidence.get("verified_identity", "")),
        _norm_identity(evidence.get("archive_candidate_id", "")),
        _norm_identity(evidence.get("visual_candidate_id", "")),
    } - {""}


def _matches_known_truth(sample: dict[str, Any]) -> bool:
    truth_hull = _truth_hull(sample)
    return bool(truth_hull and truth_hull in _candidate_ids(sample))


def _best_candidate_support(sample: dict[str, Any]) -> float:
    features = sample.get("features", {}) or {}
    return max(
        _float(features.get("structure_score", 0.0)),
        _float(features.get("visual_similarity_score", 0.0)),
        0.85 if _float(features.get("text_visual_agree", 0.0)) > 0.0 else 0.0,
    )


def _state(sample: dict[str, Any]) -> str:
    evidence = sample.get("evidence", {}) or {}
    return str(
        evidence.get("recognition_identity_state", "")
        or evidence.get("identity_state_before", "")
        or evidence.get("identity_state_after", "")
        or ""
    )


def _utility_by_action(
    sample: dict[str, Any],
    *,
    query_cost: float,
    latency_cost: float,
    defer_cost: float,
    escalation_cost: float,
    false_accept_penalty: float,
    false_reject_penalty: float,
    wrong_known_penalty: float,
    uncertainty_bonus: float,
    known_visual_score: float,
    known_visual_margin: float,
    known_visual_strong_score: float,
    unknown_visual_score: float,
    unknown_visual_margin: float,
    min_visual_rejection_observations: int,
) -> dict[str, float]:
    features = sample.get("features", {}) or {}
    status = _truth_status(sample)
    known_truth = status == "known"
    unknown_truth = status == "unknown"
    quality = _float(features.get("observation_quality", 0.0))
    uncertainty = _float(features.get("identity_uncertainty", 1.0), 1.0)
    opportunity = _float(features.get("evidence_opportunity_score", uncertainty), uncertainty)
    budget_fraction = _float(features.get("query_budget_fraction", 0.0))
    budget_used = _float(features.get("query_budget_used", 0.0))
    budget_max = max(1.0, _float(features.get("query_budget_max", 3.0), 3.0))
    budget_exhausted = budget_used >= budget_max or budget_fraction >= 0.999
    support = _best_candidate_support(sample)
    visual_score = _float(features.get("visual_similarity_score", 0.0))
    visual_margin = _float(features.get("visual_margin", 0.0))
    visual_consistent = int(_float(features.get("visual_consistent_observations", 0.0)))
    visual_observations = int(_float(features.get("visual_observation_count", 0.0)))
    visual_low_score_observations = int(_float(features.get("visual_low_score_observations", 0.0)))
    visual_only_mode = bool(int(_float(features.get("visual_only_mode", 0.0))))
    visual_candidate = _norm_identity((sample.get("evidence", {}) or {}).get("visual_candidate_id", ""))
    weak_visual = not visual_candidate or (
        visual_score < unknown_visual_score and visual_margin < unknown_visual_margin
    )
    repeated_weak_visual = (
        bool(visual_candidate)
        and weak_visual
        and visual_observations >= min_visual_rejection_observations
        and visual_low_score_observations >= min_visual_rejection_observations
    )
    weak_archive = repeated_weak_visual if visual_only_mode else (
        support < 0.55 and visual_score < 0.50 and visual_margin < 0.20
    )
    terminal_state = _state(sample)
    matched_truth = _matches_known_truth(sample)
    low_quality = quality < 0.25
    repeated_visual_match = (
        visual_only_mode
        and matched_truth
        and visual_consistent >= 2
        and (
            visual_score >= known_visual_strong_score
            or (
                visual_score >= known_visual_score
                and visual_margin >= known_visual_margin
            )
        )
    )
    visual_known_ready = repeated_visual_match or (
        matched_truth and terminal_state in {"confirmed", "structure_verified"}
    )
    visual_unknown_ready = visual_only_mode and unknown_truth and repeated_weak_visual

    query = opportunity + uncertainty_bonus * uncertainty - query_cost - latency_cost - 0.20 * budget_fraction
    if low_quality:
        query -= 0.60
    if budget_exhausted:
        query -= 1.00
    if terminal_state in {"confirmed", "structure_verified", "out_of_archive"}:
        query -= 0.55
    if known_truth and not matched_truth:
        query += 0.25
    if unknown_truth and not weak_archive and terminal_state != "out_of_archive":
        query += 0.20
    if visual_known_ready or visual_unknown_ready:
        query -= 0.45

    defer = -defer_cost
    if low_quality:
        defer += 0.85
    else:
        defer -= 0.20 + 0.35 * opportunity
    if budget_exhausted:
        defer -= 0.25

    stop_known = -0.20
    if known_truth and visual_known_ready:
        stop_known += 1.10 + 0.40 * min(1.0, support)
    elif known_truth and matched_truth:
        stop_known -= 0.45
    elif known_truth:
        stop_known -= wrong_known_penalty
    elif unknown_truth:
        stop_known -= false_accept_penalty
    if low_quality:
        stop_known -= 0.20
    if terminal_state in {"confirmed", "structure_verified"}:
        stop_known += 0.30
    if terminal_state in {"conflicting", "uncertain", "review_requested"} and not visual_known_ready:
        stop_known -= 0.55

    stop_unknown = -0.15
    if unknown_truth and (weak_archive or terminal_state == "out_of_archive"):
        stop_unknown += 0.90
    elif unknown_truth:
        stop_unknown += 0.25
    elif known_truth:
        stop_unknown -= false_reject_penalty
    if low_quality:
        stop_unknown -= 0.10
    if terminal_state == "out_of_archive":
        stop_unknown += 0.30
    if terminal_state in {"conflicting", "uncertain", "review_requested"} and not visual_unknown_ready:
        stop_unknown -= 0.30

    escalate = -escalation_cost
    conflict_or_exhausted = terminal_state in {"conflicting", "uncertain", "review_requested"} or budget_exhausted
    if conflict_or_exhausted:
        escalate += 0.65 + 0.25 * uncertainty
    if known_truth and not matched_truth and budget_exhausted:
        escalate += 0.30
    if unknown_truth and not weak_archive and budget_exhausted:
        escalate += 0.25
    if not conflict_or_exhausted and not budget_exhausted:
        escalate -= 0.25

    return {
        "defer": round(_clip(defer), 6),
        "query": round(_clip(query), 6),
        "stop_known": round(_clip(stop_known), 6),
        "stop_out_of_archive": round(_clip(stop_unknown), 6),
        "escalate_review": round(_clip(escalate), 6),
    }


def build_utility_dataset(
    dataset_path: Path,
    output_path: Path,
    *,
    query_cost: float = 0.18,
    latency_cost: float = 0.04,
    defer_cost: float = 0.08,
    escalation_cost: float = 0.35,
    false_accept_penalty: float = 1.00,
    false_reject_penalty: float = 0.90,
    wrong_known_penalty: float = 0.95,
    uncertainty_bonus: float = 0.20,
    known_visual_score: float = 0.70,
    known_visual_margin: float = 0.03,
    known_visual_strong_score: float = 0.90,
    unknown_visual_score: float = 0.70,
    unknown_visual_margin: float = 0.08,
    min_visual_rejection_observations: int = 2,
) -> dict[str, Any]:
    rows = read_jsonl(dataset_path)
    output_rows: list[dict[str, Any]] = []
    best_counts: Counter[str] = Counter()
    oracle_agreement = 0
    teacher_agreement = 0
    for row in rows:
        utilities = _utility_by_action(
            row,
            query_cost=query_cost,
            latency_cost=latency_cost,
            defer_cost=defer_cost,
            escalation_cost=escalation_cost,
            false_accept_penalty=false_accept_penalty,
            false_reject_penalty=false_reject_penalty,
            wrong_known_penalty=wrong_known_penalty,
            uncertainty_bonus=uncertainty_bonus,
            known_visual_score=known_visual_score,
            known_visual_margin=known_visual_margin,
            known_visual_strong_score=known_visual_strong_score,
            unknown_visual_score=unknown_visual_score,
            unknown_visual_margin=unknown_visual_margin,
            min_visual_rejection_observations=min_visual_rejection_observations,
        )
        best_action = max(ACTIONS, key=lambda action: (utilities[action], -ACTIONS.index(action)))
        best_counts[best_action] += 1
        oracle_agreement += int(str(row.get("oracle_action", "")) == best_action)
        teacher_agreement += int(str(row.get("teacher_action", "")) == best_action)
        output_rows.append({
            **row,
            "utility_by_action": utilities,
            "utility_action": best_action,
            "utility_margin": round(
                sorted(utilities.values(), reverse=True)[0] - sorted(utilities.values(), reverse=True)[1],
                6,
            ),
        })

    write_jsonl(output_path, output_rows)
    summary = {
        "input": str(dataset_path),
        "output": str(output_path),
        "samples": len(output_rows),
        "utility_action_counts": {action: best_counts.get(action, 0) for action in ACTIONS},
        "oracle_action_agreement": round(oracle_agreement / max(1, len(output_rows)), 6),
        "teacher_action_agreement": round(teacher_agreement / max(1, len(output_rows)), 6),
        "costs": {
            "query_cost": query_cost,
            "latency_cost": latency_cost,
            "defer_cost": defer_cost,
            "escalation_cost": escalation_cost,
            "false_accept_penalty": false_accept_penalty,
            "false_reject_penalty": false_reject_penalty,
            "wrong_known_penalty": wrong_known_penalty,
            "uncertainty_bonus": uncertainty_bonus,
        },
        "visual_thresholds": {
            "known_visual_score": known_visual_score,
            "known_visual_margin": known_visual_margin,
            "known_visual_strong_score": known_visual_strong_score,
            "unknown_visual_score": unknown_visual_score,
            "unknown_visual_margin": unknown_visual_margin,
            "min_visual_rejection_observations": min_visual_rejection_observations,
        },
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build counterfactual utility labels for cognitive policy training.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--query-cost", type=float, default=0.18)
    parser.add_argument("--latency-cost", type=float, default=0.04)
    parser.add_argument("--defer-cost", type=float, default=0.08)
    parser.add_argument("--escalation-cost", type=float, default=0.35)
    parser.add_argument("--false-accept-penalty", type=float, default=1.00)
    parser.add_argument("--false-reject-penalty", type=float, default=0.90)
    parser.add_argument("--wrong-known-penalty", type=float, default=0.95)
    parser.add_argument("--uncertainty-bonus", type=float, default=0.20)
    parser.add_argument("--known-visual-score", type=float, default=0.70)
    parser.add_argument("--known-visual-margin", type=float, default=0.03)
    parser.add_argument("--known-visual-strong-score", type=float, default=0.90)
    parser.add_argument("--unknown-visual-score", type=float, default=0.70)
    parser.add_argument("--unknown-visual-margin", type=float, default=0.08)
    parser.add_argument("--min-visual-rejection-observations", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_utility_dataset(
        args.dataset,
        args.output,
        query_cost=args.query_cost,
        latency_cost=args.latency_cost,
        defer_cost=args.defer_cost,
        escalation_cost=args.escalation_cost,
        false_accept_penalty=args.false_accept_penalty,
        false_reject_penalty=args.false_reject_penalty,
        wrong_known_penalty=args.wrong_known_penalty,
        uncertainty_bonus=args.uncertainty_bonus,
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
