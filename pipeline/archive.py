"""Open-set identity decisions from temporal and archive evidence."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IdentityDecision:
    identity: str
    state: str
    confidence: float
    candidates: list[tuple[str, float]]
    evidence_source: str = "none"


def decide_identity(
    candidate_scores: dict[str, float],
    exact_match: str = "",
    in_archive_threshold: float = 0.78,
    uncertain_threshold: float = 0.55,
    min_margin: float = 0.08,
    evidence_count: int = 0,
    min_structure_observations: int = 2,
    strong_single_threshold: float = 0.88,
    consistent_in_archive_threshold: float | None = None,
    min_rejection_observations: int = 2,
    consistent_observations: int = 0,
    conflict_observations: int = 0,
    best_single_score: float = 0.0,
    min_consistent_score: float = 0.0,
    min_consistent_margin: float = 0.0,
    min_consistent_observations: int = 2,
    min_consistency_ratio: float = 0.67,
    exact_threshold: float | None = None,
    probable_threshold: float | None = None,
    unknown_threshold: float | None = None,
) -> IdentityDecision:
    if exact_threshold is not None or probable_threshold is not None or unknown_threshold is not None:
        total = sum(max(0.0, value) for value in candidate_scores.values()) or 1.0
        legacy = {key: max(0.0, value) / total for key, value in candidate_scores.items()}
        ranked_legacy = sorted(legacy.items(), key=lambda item: (-item[1], item[0]))
        if not ranked_legacy:
            return IdentityDecision("", "unknown", 0.0, [], "none")
        best_id, confidence = ranked_legacy[0]
        if unknown_threshold is not None and confidence < unknown_threshold:
            return IdentityDecision("", "unknown", round(confidence, 4), ranked_legacy, "legacy")
        if exact_threshold is not None and confidence >= exact_threshold:
            return IdentityDecision(best_id, "confirmed", round(confidence, 4), ranked_legacy, "legacy")

    if exact_match:
        return IdentityDecision(exact_match, "confirmed", 1.0, [(exact_match, 1.0)], "exact_hull")
    if not candidate_scores:
        return IdentityDecision("", "unknown", 0.0, [], "none")
    ranked = sorted(
        ((key, min(1.0, max(0.0, value))) for key, value in candidate_scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
    best_id, confidence = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = confidence - second
    consistent_threshold = in_archive_threshold if consistent_in_archive_threshold is None else consistent_in_archive_threshold
    required_consistent_observations = max(2, min_consistent_observations, min_structure_observations)
    strong_single = (
        evidence_count == 1
        and best_single_score >= strong_single_threshold
        and min_consistent_margin >= min_margin
    )
    consistency_ratio = consistent_observations / max(1, consistent_observations + conflict_observations)
    consistent_multi = (
        confidence >= consistent_threshold
        and margin >= min_margin
        and evidence_count >= required_consistent_observations
        and consistent_observations >= required_consistent_observations
        and consistency_ratio >= min_consistency_ratio
        and min_consistent_score >= uncertain_threshold
        and min_consistent_margin >= min_margin
    )
    legacy_confirmation = (
        consistent_observations == 0
        and evidence_count >= required_consistent_observations
        and confidence >= consistent_threshold
        and margin >= min_margin
    )
    if strong_single or consistent_multi or legacy_confirmation:
        state = "structure_verified"
    elif conflict_observations > 0 and consistent_observations > 0 and consistency_ratio < min_consistency_ratio:
        state = "conflicting"
    elif confidence >= uncertain_threshold or evidence_count < max(1, min_rejection_observations):
        state = "uncertain"
    else:
        state = "out_of_archive"
        best_id = ""
    return IdentityDecision(
        best_id,
        state,
        round(confidence, 4),
        [(key, round(score, 4)) for key, score in ranked],
        "structure_semantic",
    )
