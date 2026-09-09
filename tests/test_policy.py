from pipeline.policy import LookoutPolicy
from pipeline.pipeline import ShipPipeline
from pipeline.policy import PolicyDecision
from pipeline.tracker import TrackInfo, TrackManager
import json


def write_learned_model(path, preferred_action):
    actions = ["defer", "query", "stop_known", "stop_out_of_archive", "escalate_review"]
    bias = [0.0] * len(actions)
    bias[actions.index(preferred_action)] = 10.0
    path.write_text(json.dumps({
        "actions": actions,
        "feature_names": ["observation_quality"],
        "feature_mean": [0.0],
        "feature_std": [1.0],
        "weights": [[0.0] * len(actions)],
        "bias": bias,
    }), encoding="utf-8")

def test_policy_waits_for_better_view():
    track = TrackInfo(track_id=1, recognized=False, observation_quality=0.1)
    decision = LookoutPolicy({"mode": "full", "min_query_quality": 0.25}).decide(track, 10)
    assert not decision.should_query and decision.action == "wait_better_view"

def test_uncertainty_policy_requeries_after_gap():
    track = TrackInfo(track_id=1, recognized=True, observation_quality=0.8, last_uncertainty_score=0.9, first_seen_frame=1, last_recognized_frame=10)
    decision = LookoutPolicy({"mode": "uncertainty", "min_gap_frames": 20, "uncertainty_threshold": 0.65}).decide(track, 40)
    assert decision.should_query and decision.action == "cloud_vlm_read"


def test_policy_reports_high_risk_unknown_with_skill_provenance():
    track = TrackInfo(track_id=1, recognized=True, observation_quality=0.8, identity_state="out_of_archive", risk_level="high", risk_score=0.9)
    decision = LookoutPolicy({"mode": "full"}).decide(track, 40)
    assert not decision.should_query
    assert decision.action == "remote_report"
    assert "risk_skill_high" in decision.reasons
    assert decision.skill_signals["identity_state"] == "out_of_archive"


def test_gray_zone_requests_better_follow_up_observation():
    track = TrackInfo(
        track_id=1,
        recognized=True,
        identity_state="uncertain",
        observation_quality=0.84,
        last_recognition_quality=0.75,
        last_recognized_frame=10,
        recognition_attempts=1,
        structure_evidence_count=1,
    )
    decision = LookoutPolicy({"mode": "full", "min_gap_frames": 20, "min_quality_improvement": 0.03}).decide(track, 30)
    assert decision.should_query
    assert decision.action == "verify_archive_identity"


def test_gray_zone_waits_for_quality_improvement():
    track = TrackInfo(
        track_id=1,
        recognized=True,
        identity_state="uncertain",
        observation_quality=0.76,
        last_recognition_quality=0.75,
        last_recognized_frame=10,
        recognition_attempts=1,
    )
    decision = LookoutPolicy({"mode": "full", "min_gap_frames": 20, "min_quality_improvement": 0.03}).decide(track, 30)
    assert not decision.should_query
    assert decision.action == "wait_better_view"


def test_gray_zone_stops_at_query_budget():
    track = TrackInfo(
        track_id=1,
        recognized=True,
        identity_state="uncertain",
        observation_quality=0.90,
        last_recognition_quality=0.70,
        recognition_attempts=2,
    )
    decision = LookoutPolicy({"mode": "full", "max_queries_per_track": 2}).decide(track, 100)
    assert not decision.should_query
    assert decision.action == "request_remote_verification"


def test_unknown_state_also_stops_at_entity_query_budget():
    track = TrackInfo(
        track_id=1,
        recognized=True,
        identity_state="unknown",
        observation_quality=0.9,
        recognition_attempts=3,
        last_uncertainty_score=1.0,
    )
    decision = LookoutPolicy({"mode": "full", "max_queries_per_entity": 3}).decide(track, 100)

    assert decision.should_query is False
    assert decision.action == "request_remote_verification"
    assert "identity_skill_unknown" in decision.reasons


def test_fixed_interval_baseline_ignores_active_quality_gate():
    track = TrackInfo(
        track_id=1,
        recognized=True,
        identity_state="uncertain",
        observation_quality=0.75,
        last_recognition_quality=0.75,
        last_recognized_frame=10,
        recognition_attempts=1,
    )
    decision = LookoutPolicy({"mode": "fixed_interval", "fixed_interval_frames": 20}).decide(track, 30)
    assert decision.should_query
    assert decision.action == "cloud_vlm_read"


def test_gray_zone_requeries_on_reassociated_tracklet_without_quality_gain():
    track = TrackInfo(
        track_id=2,
        entity_id="E1",
        member_track_ids=[1, 2],
        reassociation_count=1,
        recognized=True,
        identity_state="uncertain",
        observation_quality=0.75,
        last_recognition_quality=0.75,
        last_recognition_track_id=1,
        last_recognized_frame=10,
        recognition_attempts=1,
    )
    decision = LookoutPolicy({"mode": "full", "min_gap_frames": 20, "min_quality_improvement": 0.03}).decide(track, 30)
    assert decision.should_query
    assert decision.action == "verify_archive_identity"
    assert "active_novel_tracklet_view" in decision.reasons


def test_gray_zone_allows_bounded_temporal_confirmation_retry():
    track = TrackInfo(
        track_id=1,
        recognized=True,
        identity_state="uncertain",
        observation_quality=0.75,
        last_recognition_quality=0.75,
        last_recognition_track_id=1,
        last_recognized_frame=10,
        recognition_attempts=1,
        structure_evidence_count=1,
    )
    policy = LookoutPolicy({
        "mode": "full",
        "min_gap_frames": 20,
        "confirmation_retry_gap_frames": 100,
        "min_confirmation_observations": 2,
        "min_quality_improvement": 0.03,
    })
    early = policy.decide(track, 30)
    retry = policy.decide(track, 110)
    assert not early.should_query
    assert retry.should_query
    assert "active_temporal_confirmation_retry" in retry.reasons


def test_review_requested_entity_is_terminal_for_identity_queries():
    track = TrackInfo(
        track_id=1,
        identity_state="review_requested",
        episode_status="review_requested",
        observation_quality=0.9,
        recognition_attempts=3,
    )
    decision = LookoutPolicy({"mode": "full", "max_queries_per_entity": 3}).decide(track, 200)

    assert decision.should_query is False
    assert decision.action == "continue_tracking"
    assert decision.reasons == ["identity_skill_review_requested"]
    assert decision.skill_signals["query_budget_fraction"] == 1.0


def test_policy_exposes_evidence_opportunity_inputs():
    track = TrackInfo(
        track_id=2,
        identity_state="conflicting",
        observation_quality=0.8,
        last_recognition_quality=0.6,
        last_recognized_frame=10,
        recognition_attempts=1,
        risk_level="medium",
    )
    decision = LookoutPolicy({"mode": "full", "max_queries_per_entity": 4}).decide(track, 100)

    assert decision.skill_signals["quality_improvement"] == 0.2
    assert decision.skill_signals["temporal_gap_frames"] == 90
    assert decision.skill_signals["query_budget_used"] == 1
    assert decision.skill_signals["query_budget_max"] == 4
    assert 0.0 <= decision.skill_signals["evidence_opportunity_score"] <= 1.0


def test_policy_exposes_visual_consistency_inputs():
    track = TrackInfo(
        track_id=3,
        identity_state="uncertain",
        observation_quality=0.8,
        visual_candidate_id="V081",
        visual_similarity_score=0.94,
        visual_margin=0.18,
        visual_observation_count=2,
        visual_consistent_observations=2,
        visual_low_score_observations=0,
    )

    decision = LookoutPolicy({"mode": "full"}).decide(track, 100)

    assert decision.skill_signals["visual_observation_count"] == 2
    assert decision.skill_signals["visual_consistent_observations"] == 2
    assert decision.skill_signals["visual_low_score_observations"] == 0


def test_policy_marks_visual_only_mode():
    track = TrackInfo(
        track_id=4,
        identity_decision_mode="visual_only",
        observation_quality=0.8,
        structure_candidate_scores={"012": 0.95},
        structure_evidence_count=2,
    )
    decision = LookoutPolicy({"mode": "full"}).decide(track, 10)
    assert decision.skill_signals["visual_only_mode"] == 1
    assert decision.skill_signals["text_visual_agree"] == 0
    assert decision.skill_signals["structure_score"] == 0.0
    assert decision.skill_signals["structure_evidence_count"] == 0


def test_learned_policy_queries_with_model_scores(tmp_path):
    model_path = tmp_path / "policy.json"
    write_learned_model(model_path, "query")
    track = TrackInfo(track_id=1, recognized=False, identity_state="unknown", observation_quality=0.8)

    decision = LookoutPolicy({"mode": "learned", "learned_model_path": str(model_path)}).decide(track, 10)

    assert decision.should_query
    assert decision.action == "cloud_vlm_read"
    assert decision.skill_signals["learned_policy_action"] == "query"


def test_learned_policy_stops_known_only_when_eligible(tmp_path):
    model_path = tmp_path / "policy.json"
    write_learned_model(model_path, "stop_known")
    eligible = TrackInfo(
        track_id=1,
        recognized=True,
        identity_state="structure_verified",
        verified_identity="012",
        observation_quality=0.8,
    )
    unknown = TrackInfo(track_id=2, recognized=False, identity_state="unknown", observation_quality=0.8)
    policy = LookoutPolicy({"mode": "learned", "learned_model_path": str(model_path)})

    stop = policy.decide(eligible, 10)
    fallback = policy.decide(unknown, 10)

    assert not stop.should_query
    assert stop.action == "continue_tracking"
    assert fallback.should_query
    assert fallback.action == "cloud_vlm_read"


def test_learned_visual_only_policy_confirms_repeated_candidate(tmp_path):
    model_path = tmp_path / "policy.json"
    write_learned_model(model_path, "stop_known")
    track = TrackInfo(
        track_id=1,
        recognized=True,
        identity_state="uncertain",
        identity_decision_mode="visual_only",
        observation_quality=0.8,
        visual_candidate_id="012",
        visual_similarity_score=0.88,
        visual_margin=0.18,
        visual_observation_count=3,
        visual_consistent_observations=3,
        recognition_attempts=2,
    )

    decision = LookoutPolicy({
        "mode": "learned_utility",
        "learned_model_path": str(model_path),
        "min_confirmation_observations": 2,
    }).decide(track, 100)

    assert decision.should_query is False
    assert decision.action == "confirm_visual_identity"
    assert "repeated_visual_candidate_guard" in decision.reasons
    assert decision.skill_signals["visual_confirmation_guard_ready"] == 1

    track.visual_similarity_score = 0.65
    blocked = LookoutPolicy({
        "mode": "learned_utility",
        "learned_model_path": str(model_path),
        "min_confirmation_observations": 2,
    }).decide(track, 120)

    assert blocked.should_query is True
    assert blocked.action == "verify_archive_identity"
    assert "visual_confirmation_guard_blocked" in blocked.reasons


def test_learned_visual_only_policy_guards_stop_out_of_archive(tmp_path):
    model_path = tmp_path / "policy.json"
    write_learned_model(model_path, "stop_out_of_archive")
    policy = LookoutPolicy({
        "mode": "learned_utility",
        "learned_model_path": str(model_path),
        "max_queries_per_track": 3,
        "min_visual_rejection_observations": 2,
    })
    strong_candidate = TrackInfo(
        track_id=1,
        recognized=True,
        identity_state="uncertain",
        identity_decision_mode="visual_only",
        observation_quality=0.8,
        visual_candidate_id="012",
        visual_similarity_score=0.73,
        visual_margin=0.14,
        visual_observation_count=1,
        visual_consistent_observations=1,
        visual_low_score_observations=0,
        recognition_attempts=1,
    )

    query = policy.decide(strong_candidate, 100)
    strong_candidate.recognition_attempts = 3
    review = policy.decide(strong_candidate, 150)
    strong_candidate.recognition_attempts = 1
    strong_candidate.visual_low_score_observations = 2
    reject = policy.decide(strong_candidate, 200)

    assert query.should_query is True
    assert query.action == "verify_archive_identity"
    assert "visual_rejection_guard_blocked" in query.reasons
    assert review.should_query is False
    assert review.action == "request_remote_verification"
    assert reject.should_query is True
    assert reject.action == "verify_archive_identity"

    strong_candidate.visual_observation_count = 2
    protected = policy.decide(strong_candidate, 250)

    assert protected.should_query is True
    assert protected.action == "verify_archive_identity"
    assert protected.skill_signals["visual_rejection_evidence_weak"] == 0
    assert "visual_evidence_not_weak" in protected.reasons

    strong_candidate.visual_similarity_score = 0.65
    strong_candidate.visual_margin = 0.05
    repeated_reject = policy.decide(strong_candidate, 300)

    assert repeated_reject.should_query is False
    assert repeated_reject.action == "monitor_unknown"
    assert repeated_reject.skill_signals["visual_rejection_guard_ready"] == 1
    assert repeated_reject.skill_signals["visual_rejection_evidence_weak"] == 1


def test_score_gate_policy_rejects_at_exhausted_budget(tmp_path):
    model_path = tmp_path / "policy.json"
    write_learned_model(model_path, "query")
    track = TrackInfo(
        track_id=1,
        recognized=True,
        identity_state="uncertain",
        identity_decision_mode="visual_only",
        observation_quality=0.8,
        visual_candidate_id="012",
        visual_similarity_score=0.79,
        visual_margin=0.90,
        visual_observation_count=2,
        visual_consistent_observations=1,
        visual_low_score_observations=2,
        recognition_attempts=3,
    )

    decision = LookoutPolicy({
        "mode": "learned_utility",
        "learned_model_path": str(model_path),
        "visual_only_decision_rule": "score_gate",
        "visual_confirmation_min_score": 0.80,
        "visual_rejection_max_score": 0.80,
        "min_visual_rejection_observations": 2,
        "max_queries_per_track": 3,
    }).decide(track, 100)

    assert decision.should_query is False
    assert decision.action == "monitor_unknown"
    assert "visual_score_gate_budget_exhausted" in decision.reasons


def test_learned_escalation_cannot_override_verified_identity(tmp_path):
    model_path = tmp_path / "policy.json"
    write_learned_model(model_path, "escalate_review")
    track = TrackInfo(
        track_id=1,
        recognized=True,
        identity_state="structure_verified",
        verified_identity="012",
        observation_quality=0.8,
        recognition_attempts=3,
    )

    decision = LookoutPolicy({
        "mode": "learned",
        "learned_model_path": str(model_path),
        "max_queries_per_track": 3,
    }).decide(track, 100)

    assert decision.should_query is False
    assert decision.action == "continue_tracking"
    assert "learned_terminal_known_mask" in decision.reasons


def test_learned_escalation_requeries_before_review_is_eligible(tmp_path):
    model_path = tmp_path / "policy.json"
    write_learned_model(model_path, "escalate_review")
    track = TrackInfo(
        track_id=1,
        recognized=True,
        identity_state="uncertain",
        observation_quality=0.8,
        recognition_attempts=1,
    )

    decision = LookoutPolicy({
        "mode": "learned",
        "learned_model_path": str(model_path),
        "max_queries_per_track": 3,
    }).decide(track, 100)

    assert decision.should_query is True
    assert decision.action == "verify_archive_identity"
    assert "learned_review_not_yet_eligible" in decision.reasons


def test_learned_escalation_requests_review_after_budget_exhaustion(tmp_path):
    model_path = tmp_path / "policy.json"
    write_learned_model(model_path, "escalate_review")
    track = TrackInfo(
        track_id=1,
        recognized=True,
        identity_state="conflicting",
        observation_quality=0.8,
        recognition_attempts=3,
    )

    decision = LookoutPolicy({
        "mode": "learned",
        "learned_model_path": str(model_path),
        "max_queries_per_track": 3,
    }).decide(track, 100)

    assert decision.should_query is False
    assert decision.action == "request_remote_verification"


def test_pipeline_commits_learned_stop_out_of_archive():
    class StopUnknownPolicy:
        mode = "learned_utility"

        @staticmethod
        def decide(track, frame_id, experience_hint):
            return PolicyDecision(
                False,
                "monitor_unknown",
                0.91,
                ["learned_policy", "learned_action_stop_out_of_archive", "learned_stop_out_of_archive"],
                {"learned_policy_action": "stop_out_of_archive"},
            )

    class CapturingLogger:
        def __init__(self):
            self.rows = []

        def log(self, channel, payload):
            self.rows.append((channel, payload))

    pipeline = ShipPipeline.__new__(ShipPipeline)
    pipeline._tracker = TrackManager(memory_mode="full")
    pipeline._tracker.get_or_create(1, 1)
    pipeline._tracker.apply_identity_decision(1, "V081", "uncertain", 0.75)
    pipeline._policy = StopUnknownPolicy()
    pipeline._experience_hint = lambda info: None
    pipeline._memory_mode = "full"
    pipeline._identity_decision_mode = "fused"
    pipeline._experiment_logger = CapturingLogger()

    assert pipeline._should_query_track(1, 30) is False
    info = pipeline._tracker.get(1)
    assert info.identity_state == "out_of_archive"
    assert info.identity_evidence_source == "learned_policy_stop_out_of_archive"
    assert pipeline._experiment_logger.rows[0][1]["identity_state_after"] == "out_of_archive"


def test_pipeline_commits_learned_visual_identity_confirmation():
    class ConfirmVisualPolicy:
        mode = "learned_utility"

        @staticmethod
        def decide(track, frame_id, experience_hint):
            return PolicyDecision(
                False,
                "confirm_visual_identity",
                0.91,
                ["learned_policy", "learned_action_stop_known", "learned_stop_known"],
                {"learned_policy_action": "stop_known"},
            )

    class FakeDatabase:
        @staticmethod
        def lookup(identity):
            return "known vessel" if identity == "012" else None

    class CapturingLogger:
        def __init__(self):
            self.rows = []

        def log(self, channel, payload):
            self.rows.append((channel, payload))

    pipeline = ShipPipeline.__new__(ShipPipeline)
    pipeline._tracker = TrackManager(memory_mode="full")
    pipeline._tracker.get_or_create(1, 1)
    pipeline._tracker.update_visual_match(1, "012", 0.76, 0.18)
    pipeline._tracker.update_visual_match(1, "012", 0.76, 0.18)
    pipeline._tracker.apply_identity_decision(1, "012", "uncertain", 0.76)
    pipeline._policy = ConfirmVisualPolicy()
    pipeline._experience_hint = lambda info: None
    pipeline._memory_mode = "full"
    pipeline._identity_decision_mode = "visual_only"
    pipeline._db = FakeDatabase()
    pipeline._experiment_logger = CapturingLogger()

    assert pipeline._should_query_track(1, 30) is False
    info = pipeline._tracker.get(1)
    assert info.identity_state == "structure_verified"
    assert info.verified_identity == "012"
    assert info.identity_evidence_source == "learned_visual_policy"
    assert pipeline._experiment_logger.rows[0][1]["identity_state_after"] == "structure_verified"


def test_pipeline_does_not_relog_pending_review_state():
    class CapturingPolicy:
        mode = "learned_utility"

        def decide(self, track, frame_id, experience_hint):
            raise AssertionError("review-pending tracks must not invoke the policy")

    class CapturingLogger:
        def log(self, channel, payload):
            raise AssertionError("review-pending tracks must not emit repeated actions")

    pipeline = ShipPipeline.__new__(ShipPipeline)
    pipeline._tracker = TrackManager(memory_mode="full")
    pipeline._tracker.get_or_create(1, 1)
    pipeline._tracker.request_review(1, 20, "request_remote_verification", ["budget_exhausted"])
    pipeline._policy = CapturingPolicy()
    pipeline._identity_decision_mode = "visual_only"
    pipeline._experiment_logger = CapturingLogger()

    assert pipeline._should_query_track(1, 30) is False
