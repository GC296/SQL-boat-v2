from pipeline.archive import decide_identity


def test_exact_hull_has_highest_priority():
    decision = decide_identity({"B34": 0.99}, exact_match="A12", evidence_count=3)
    assert decision.identity == "A12"
    assert decision.state == "confirmed"
    assert decision.evidence_source == "exact_hull"


def test_structure_score_defines_archive_state():
    verified = decide_identity({"A12": 0.84, "B34": 0.61}, evidence_count=3)
    uncertain = decide_identity({"A12": 0.70, "B34": 0.65}, evidence_count=3)
    rejected = decide_identity({"A12": 0.42}, evidence_count=3)
    assert verified.state == "structure_verified" and verified.identity == "A12"
    assert uncertain.state == "uncertain"
    assert rejected.state == "out_of_archive" and rejected.identity == ""


def test_structure_requires_multiple_observations():
    decision = decide_identity({"A12": 0.90}, evidence_count=1, min_structure_observations=2)
    assert decision.state == "uncertain"


def test_project_default_allows_one_strong_structure_observation():
    decision = decide_identity(
        {"A12": 0.90},
        evidence_count=1,
        min_structure_observations=1,
        consistent_observations=1,
        best_single_score=0.90,
        min_consistent_score=0.90,
        min_consistent_margin=0.90,
    )
    assert decision.state == "structure_verified"
    assert decision.identity == "A12"


def test_medium_single_observation_requires_temporal_confirmation():
    first = decide_identity(
        {"A12": 0.80, "B34": 0.60},
        evidence_count=1,
        min_structure_observations=1,
        in_archive_threshold=0.735,
        strong_single_threshold=0.88,
        consistent_observations=1,
        best_single_score=0.80,
        min_consistent_score=0.80,
        min_consistent_margin=0.20,
        min_consistent_observations=2,
    )
    repeated = decide_identity(
        {"A12": 0.81, "B34": 0.59},
        evidence_count=2,
        in_archive_threshold=0.735,
        strong_single_threshold=0.88,
        consistent_observations=2,
        best_single_score=0.82,
        min_consistent_score=0.80,
        min_consistent_margin=0.20,
        min_consistent_observations=2,
    )
    assert first.state == "uncertain"
    assert repeated.state == "structure_verified"


def test_temporal_consistency_tolerates_one_conflict_when_majority_is_stable():
    decision = decide_identity(
        {"A12": 0.79, "B34": 0.68},
        evidence_count=3,
        in_archive_threshold=0.735,
        uncertain_threshold=0.70,
        min_margin=0.03,
        consistent_observations=2,
        conflict_observations=1,
        best_single_score=0.82,
        min_consistent_score=0.75,
        min_consistent_margin=0.05,
        min_consistent_observations=2,
        min_consistency_ratio=0.66,
    )
    assert decision.state == "structure_verified"


def test_conflicting_temporal_candidates_remain_conflicting():
    decision = decide_identity(
        {"A12": 0.80, "B34": 0.78},
        evidence_count=2,
        in_archive_threshold=0.735,
        consistent_observations=1,
        conflict_observations=1,
        best_single_score=0.82,
        min_consistent_score=0.82,
        min_consistent_margin=0.04,
        min_consistent_observations=2,
        min_consistency_ratio=0.67,
    )
    assert decision.state == "conflicting"


def test_low_structure_score_requires_repeated_negative_evidence_before_rejection():
    first = decide_identity(
        {"A12": 0.68},
        evidence_count=1,
        uncertain_threshold=0.70,
        min_rejection_observations=2,
        consistent_observations=1,
        best_single_score=0.68,
        min_consistent_score=0.68,
    )
    repeated = decide_identity(
        {"A12": 0.68},
        evidence_count=2,
        uncertain_threshold=0.70,
        min_rejection_observations=2,
        consistent_observations=2,
        best_single_score=0.68,
        min_consistent_score=0.68,
    )
    assert first.state == "uncertain"
    assert repeated.state == "out_of_archive"


def test_consistent_medium_evidence_can_confirm_below_strong_single_threshold():
    decision = decide_identity(
        {"A12": 0.73, "B34": 0.60},
        evidence_count=3,
        in_archive_threshold=0.735,
        consistent_in_archive_threshold=0.73,
        strong_single_threshold=0.88,
        consistent_observations=3,
        best_single_score=0.74,
        min_consistent_score=0.71,
        min_consistent_margin=0.10,
        min_consistent_observations=2,
    )
    assert decision.state == "structure_verified"
    assert decision.identity == "A12"
