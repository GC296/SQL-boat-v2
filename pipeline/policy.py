"""Configurable skill-driven active watchkeeping policy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PolicyDecision:
    should_query: bool
    action: str
    score: float
    reasons: list[str]
    skill_signals: dict[str, Any] = field(default_factory=dict)


class LookoutPolicy:
    def __init__(self, config: dict[str, Any] | None = None):
        cfg = config or {}
        self.mode = str(cfg.get("mode", "full")).lower()
        self.threshold = float(cfg.get("uncertainty_threshold", 0.65))
        self.min_gap_frames = int(cfg.get("min_gap_frames", 45))
        self.fixed_interval_frames = int(cfg.get("fixed_interval_frames", 150))
        self.min_query_quality = float(cfg.get("min_query_quality", 0.25))
        self.risk_bonus = float(cfg.get("risk_bonus", 0.20))
        self.archive_bonus = float(cfg.get("archive_bonus", 0.10))
        self.experience_bonus = float(cfg.get("experience_bonus", 0.10))
        self.min_structure_observations = int(cfg.get("min_structure_observations", 2))
        self.min_confirmation_observations = int(cfg.get("min_confirmation_observations", 2))
        self.visual_confirmation_min_score = float(cfg.get("visual_confirmation_min_score", 0.70))
        self.visual_confirmation_min_margin = float(cfg.get("visual_confirmation_min_margin", 0.03))
        self.visual_confirmation_strong_score = float(cfg.get("visual_confirmation_strong_score", 0.90))
        self.visual_only_decision_rule = str(cfg.get("visual_only_decision_rule", "legacy")).strip().lower()
        self.min_visual_rejection_observations = max(1, int(cfg.get("min_visual_rejection_observations", 2)))
        self.visual_rejection_max_score = float(cfg.get("visual_rejection_max_score", 0.70))
        self.visual_rejection_max_margin = float(cfg.get("visual_rejection_max_margin", 0.08))
        self.confirmation_retry_gap_frames = int(cfg.get("confirmation_retry_gap_frames", max(self.min_gap_frames * 2, 90)))
        self.gray_zone_reobserve = bool(cfg.get("gray_zone_reobserve", True))
        self.max_queries_per_track = max(1, int(cfg.get("max_queries_per_entity", cfg.get("max_queries_per_track", 3))))
        self.require_quality_improvement = bool(cfg.get("require_quality_improvement", True))
        self.min_quality_improvement = max(0.0, float(cfg.get("min_quality_improvement", 0.03)))
        self.quality_gain_weight = max(0.0, float(cfg.get("quality_gain_weight", 0.25)))
        self.new_tracklet_weight = max(0.0, float(cfg.get("new_tracklet_weight", 0.15)))
        self.conflict_weight = max(0.0, float(cfg.get("conflict_weight", 0.15)))
        self.budget_penalty = max(0.0, float(cfg.get("budget_penalty", 0.20)))
        self.learned_model_path = str(cfg.get("learned_model_path", "models/policy/learned_policy_v1.json"))
        self._learned_model = None
        if self.mode in {"learned", "learned_utility"}:
            from pipeline.learned_policy import LearnedPolicyModel
            self._learned_model = LearnedPolicyModel.load(self.learned_model_path)

    @staticmethod
    def _signals(track: Any, quality: float, uncertainty: float) -> dict[str, Any]:
        verified_identity = str(getattr(track, "verified_identity", ""))
        archive_candidate_id = str(getattr(track, "archive_candidate_id", ""))
        visual_candidate_id = str(getattr(track, "visual_candidate_id", ""))
        identity_decision_mode = str(getattr(track, "identity_decision_mode", "fused"))
        text_candidate = "" if identity_decision_mode == "visual_only" else (verified_identity or archive_candidate_id)
        structure_scores = getattr(track, "structure_candidate_scores", {}) or {"": 0.0}
        structure_score = 0.0 if identity_decision_mode == "visual_only" else max(structure_scores.values())
        structure_evidence_count = 0 if identity_decision_mode == "visual_only" else int(getattr(track, "structure_evidence_count", 0))
        structure_consistent_observations = 0 if identity_decision_mode == "visual_only" else int(getattr(track, "structure_consistent_observations", 0))
        structure_conflict_observations = 0 if identity_decision_mode == "visual_only" else int(getattr(track, "structure_conflict_observations", 0))
        return {
            "observation_quality": round(quality, 4),
            "identity_state": str(getattr(track, "identity_state", "unknown")),
            "identity_evidence_source": str(getattr(track, "identity_evidence_source", "none")),
            "identity_uncertainty": round(uncertainty, 4),
            "structure_score": round(structure_score, 4),
            "structure_evidence_count": structure_evidence_count,
            "structure_consistent_observations": structure_consistent_observations,
            "structure_conflict_observations": structure_conflict_observations,
            "structure_best_single_score": round(float(getattr(track, "structure_best_single_score", 0.0)), 4),
            "visual_similarity_score": round(float(getattr(track, "visual_similarity_score", 0.0)), 4),
            "visual_margin": round(float(getattr(track, "visual_margin", 0.0)), 4),
            "visual_observation_count": int(getattr(track, "visual_observation_count", 0)),
            "visual_consistent_observations": int(getattr(track, "visual_consistent_observations", 0)),
            "visual_low_score_observations": int(getattr(track, "visual_low_score_observations", 0)),
            "text_visual_agree": int(bool(text_candidate and visual_candidate_id and text_candidate == visual_candidate_id)),
            "visual_only_mode": int(identity_decision_mode == "visual_only"),
            "recognized_before": int(bool(getattr(track, "recognized", False))),
            "recognition_attempts": int(getattr(track, "recognition_attempts", 0)),
            "risk_level": str(getattr(track, "risk_level", "low")),
            "risk_score": round(float(getattr(track, "risk_score", 0.0)), 4),
        }

    def _learned_query_action(self, identity_state: str) -> str:
        if identity_state == "uncertain":
            return "verify_archive_identity"
        if identity_state == "conflicting":
            return "reobserve_conflicting_identity"
        return "cloud_vlm_read"

    def _decide_learned(
        self,
        track: Any,
        identity_state: str,
        recognition_attempts: int,
        quality: float,
        signals: dict[str, Any],
    ) -> PolicyDecision:
        if self._learned_model is None:
            raise RuntimeError("learned policy mode requires a loaded policy model")
        learned_action, confidence, probabilities = self._learned_model.predict(signals)
        signals.update({
            "learned_policy_action": learned_action,
            "learned_policy_confidence": round(float(confidence), 6),
            "learned_policy_probabilities": probabilities,
        })
        policy_family = "utility" if self.mode == "learned_utility" else "classification"
        signals["learned_policy_family"] = policy_family
        reasons = ["learned_policy", f"learned_policy_{policy_family}", f"learned_action_{learned_action}"]
        budget_exhausted = recognition_attempts >= self.max_queries_per_track
        terminal_known = identity_state in {"confirmed", "structure_verified"} or bool(getattr(track, "verified_identity", ""))
        repeated_failure = int(getattr(track, "failed_recognition_count", 0)) >= 2
        visual_score = float(getattr(track, "visual_similarity_score", 0.0))
        visual_margin = float(getattr(track, "visual_margin", 0.0))
        visual_candidate = str(getattr(track, "visual_candidate_id", ""))
        visual_only_mode = str(getattr(track, "identity_decision_mode", "fused")) == "visual_only"
        if self.visual_only_decision_rule == "score_gate":
            visual_only_confirmation_ready = (
                visual_only_mode
                and bool(visual_candidate)
                and int(getattr(track, "visual_observation_count", 0)) >= self.min_confirmation_observations
                and visual_score >= self.visual_confirmation_min_score
            )
        else:
            visual_only_confirmation_ready = (
                visual_only_mode
                and bool(visual_candidate)
                and int(getattr(track, "visual_consistent_observations", 0)) >= self.min_confirmation_observations
                and (
                    visual_score >= self.visual_confirmation_strong_score
                    or (
                        visual_score >= self.visual_confirmation_min_score
                        and visual_margin >= self.visual_confirmation_min_margin
                    )
                )
            )
        visual_rejection_observation_count = int(getattr(track, "visual_observation_count", 0))
        if self.visual_only_decision_rule == "score_gate":
            visual_rejection_weak = not visual_candidate or visual_score < self.visual_rejection_max_score
        else:
            visual_rejection_weak = (
                not visual_candidate
                or (
                    visual_score < self.visual_rejection_max_score
                    and visual_margin < self.visual_rejection_max_margin
                )
            )
        if visual_only_mode:
            from pipeline.visual_archive import visual_only_rejection_supports_terminal

            visual_rejection_ready = visual_only_rejection_supports_terminal(
                visual_candidate,
                visual_score,
                visual_margin,
                visual_rejection_observation_count,
                int(getattr(track, "visual_low_score_observations", 0)),
                {
                    "min_out_of_archive_observations": self.min_visual_rejection_observations,
                    "visual_only_reject_score": self.visual_rejection_max_score,
                    "visual_only_reject_margin": self.visual_rejection_max_margin,
                    "visual_only_decision_rule": self.visual_only_decision_rule,
                },
            )
        else:
            visual_rejection_ready = True
        signals["visual_confirmation_guard_ready"] = int(visual_only_confirmation_ready)
        signals["visual_confirmation_min_score"] = self.visual_confirmation_min_score
        signals["visual_confirmation_min_margin"] = self.visual_confirmation_min_margin
        signals["visual_confirmation_strong_score"] = self.visual_confirmation_strong_score
        signals["visual_rejection_guard_ready"] = int(visual_rejection_ready)
        signals["visual_rejection_evidence_weak"] = int(visual_rejection_weak)
        signals["visual_rejection_observation_count"] = visual_rejection_observation_count
        signals["visual_rejection_min_observations"] = self.min_visual_rejection_observations
        signals["visual_rejection_max_score"] = self.visual_rejection_max_score
        signals["visual_rejection_max_margin"] = self.visual_rejection_max_margin

        if terminal_known:
            return PolicyDecision(
                False,
                "continue_tracking",
                float(confidence),
                reasons + ["learned_terminal_known_mask"],
                signals,
            )
        if identity_state == "out_of_archive":
            return PolicyDecision(
                False,
                "monitor_unknown",
                float(confidence),
                reasons + ["learned_terminal_out_of_archive_mask"],
                signals,
            )
        if (
            visual_only_mode
            and self.visual_only_decision_rule == "score_gate"
            and budget_exhausted
            and visual_rejection_ready
        ):
            return PolicyDecision(
                False,
                "monitor_unknown",
                float(confidence),
                reasons + ["visual_score_gate_budget_exhausted"],
                signals,
            )

        if learned_action == "defer" or (quality < self.min_query_quality and learned_action == "query"):
            return PolicyDecision(False, "wait_better_view", float(confidence), reasons + ["learned_defer_or_low_quality"], signals)
        if learned_action == "query":
            if budget_exhausted:
                return PolicyDecision(False, "request_remote_verification", float(confidence), reasons + ["learned_query_budget_exhausted"], signals)
            return PolicyDecision(True, self._learned_query_action(identity_state), float(confidence), reasons, signals)
        if learned_action == "escalate_review":
            if budget_exhausted or repeated_failure:
                return PolicyDecision(False, "request_remote_verification", float(confidence), reasons, signals)
            if quality >= self.min_query_quality:
                return PolicyDecision(
                    True,
                    self._learned_query_action(identity_state),
                    float(confidence),
                    reasons + ["learned_review_not_yet_eligible"],
                    signals,
                )
            return PolicyDecision(
                False,
                "wait_better_view",
                float(confidence),
                reasons + ["learned_review_not_yet_eligible", "quality_skill_low_observation"],
                signals,
            )
        if learned_action == "stop_known":
            if terminal_known:
                return PolicyDecision(False, "continue_tracking", float(confidence), reasons + ["learned_stop_known"], signals)
            if visual_only_confirmation_ready:
                return PolicyDecision(
                    False,
                    "confirm_visual_identity",
                    float(confidence),
                    reasons + ["learned_stop_known", "repeated_visual_candidate_guard"],
                    signals,
                )
            if budget_exhausted:
                return PolicyDecision(False, "request_remote_verification", float(confidence), reasons + ["learned_stop_known_not_eligible", "visual_confirmation_guard_blocked"], signals)
            return PolicyDecision(True, self._learned_query_action(identity_state), float(confidence), reasons + ["learned_stop_known_not_eligible", "visual_confirmation_guard_blocked"], signals)
        if learned_action == "stop_out_of_archive":
            if identity_state == "out_of_archive" or (
                recognition_attempts > 0 and not terminal_known and visual_rejection_ready
            ):
                return PolicyDecision(False, "monitor_unknown", float(confidence), reasons + ["learned_stop_out_of_archive"], signals)
            rejection_block_reason = (
                "visual_rejection_observation_guard_blocked"
                if visual_rejection_weak
                else "visual_evidence_not_weak"
            )
            if budget_exhausted:
                return PolicyDecision(
                    False,
                    "request_remote_verification",
                    float(confidence),
                    reasons + ["learned_stop_unknown_not_eligible", "visual_rejection_guard_blocked", rejection_block_reason],
                    signals,
                )
            return PolicyDecision(
                True,
                self._learned_query_action(identity_state),
                float(confidence),
                reasons + ["learned_stop_unknown_not_eligible", "visual_rejection_guard_blocked", rejection_block_reason],
                signals,
            )
        return PolicyDecision(False, "continue_tracking", float(confidence), reasons + ["learned_policy_unknown_action"], signals)

    def decide(self, track: Any, frame_id: int, experience_hint: dict[str, Any] | None = None) -> PolicyDecision:
        if track is None or getattr(track, "pending", False):
            return PolicyDecision(False, "continue_tracking", 0.0, ["missing_or_pending_track"], {})
        if str(getattr(track, "identity_state", "unknown")) == "review_requested":
            signals = self._signals(track, float(getattr(track, "observation_quality", 0.0)), float(getattr(track, "last_uncertainty_score", 1.0)))
            signals.update({
                "query_budget_used": int(getattr(track, "recognition_attempts", 0)),
                "query_budget_max": self.max_queries_per_track,
                "query_budget_fraction": round(min(1.0, int(getattr(track, "recognition_attempts", 0)) / self.max_queries_per_track), 4),
            })
            return PolicyDecision(False, "continue_tracking", 0.0, ["identity_skill_review_requested"], signals)
        quality = float(getattr(track, "observation_quality", 0.0))
        last_frame = int(getattr(track, "last_recognized_frame", 0) or getattr(track, "first_seen_frame", 0))
        gap = max(0, frame_id - last_frame)
        recognized = bool(getattr(track, "recognized", False))
        uncertainty = float(getattr(track, "last_uncertainty_score", 1.0))
        identity_state = str(getattr(track, "identity_state", "unknown"))
        recognition_attempts = int(getattr(track, "recognition_attempts", 0))
        last_recognition_quality = float(getattr(track, "last_recognition_quality", 0.0))
        last_recognition_track_id = int(getattr(track, "last_recognition_track_id", 0))
        novel_tracklet_view = last_recognition_track_id > 0 and int(getattr(track, "track_id", 0)) != last_recognition_track_id
        quality_improved = quality >= last_recognition_quality + self.min_quality_improvement
        structure_evidence_count = int(getattr(track, "structure_evidence_count", 0))
        confirmation_retry_due = structure_evidence_count < self.min_confirmation_observations and gap >= self.confirmation_retry_gap_frames
        risk_level = str(getattr(track, "risk_level", "low"))
        signals = self._signals(track, quality, uncertainty)
        budget_fraction = min(1.0, recognition_attempts / self.max_queries_per_track)
        quality_gain = max(0.0, quality - last_recognition_quality)
        evidence_opportunity = min(1.0, max(0.0,
            uncertainty
            + self.quality_gain_weight * quality_gain
            + self.new_tracklet_weight * float(novel_tracklet_view)
            + self.conflict_weight * float(identity_state == "conflicting")
            + self.risk_bonus * float(risk_level == "high")
            - self.budget_penalty * budget_fraction
        ))
        signals.update({
            "quality_improvement": round(quality_gain, 4),
            "novel_tracklet_view": novel_tracklet_view,
            "temporal_gap_frames": gap,
            "query_budget_used": recognition_attempts,
            "query_budget_max": self.max_queries_per_track,
            "query_budget_fraction": round(budget_fraction, 4),
            "evidence_opportunity_score": round(evidence_opportunity, 4),
        })

        if self.mode in {"learned", "learned_utility"}:
            return self._decide_learned(track, identity_state, recognition_attempts, quality, signals)

        if quality < self.min_query_quality:
            return PolicyDecision(False, "wait_better_view", uncertainty, ["quality_skill_low_observation"], signals)
        if risk_level == "high" and identity_state in {"unknown", "uncertain", "conflicting", "out_of_archive"}:
            return PolicyDecision(False, "remote_report", round(evidence_opportunity, 4), ["risk_skill_high", f"identity_skill_{identity_state}"], signals)
        active_modes = {"uncertainty", "uncertainty_archive", "experience", "full"}
        if self.mode in active_modes and identity_state in {"unknown", "uncertain", "conflicting"} and recognition_attempts >= self.max_queries_per_track:
            return PolicyDecision(False, "request_remote_verification", round(evidence_opportunity, 4), ["active_query_budget_exhausted", f"identity_skill_{identity_state}"], signals)
        if identity_state in {"uncertain", "conflicting"} and self.gray_zone_reobserve and self.mode in active_modes:
            if gap < self.min_gap_frames:
                return PolicyDecision(False, "continue_tracking", uncertainty, ["waiting_structure_observation_gap"], signals)
            if self.require_quality_improvement and not quality_improved and not novel_tracklet_view and not confirmation_retry_due:
                return PolicyDecision(False, "wait_better_view", uncertainty, ["active_observation_quality_not_improved"], signals)
            action = "verify_archive_identity" if identity_state == "uncertain" else "reobserve_conflicting_identity"
            if novel_tracklet_view:
                observation_reason = "active_novel_tracklet_view"
            elif confirmation_retry_due:
                observation_reason = "active_temporal_confirmation_retry"
            else:
                observation_reason = "active_observation_quality_improved"
            return PolicyDecision(True, action, round(evidence_opportunity, 4), ["identity_skill_gray_zone", observation_reason], signals)
        if identity_state == "out_of_archive":
            return PolicyDecision(False, "monitor_unknown", uncertainty, ["identity_skill_out_of_archive"], signals)
        if identity_state in {"confirmed", "structure_verified"} and self.mode not in {"fixed_interval"}:
            return PolicyDecision(False, "continue_tracking", uncertainty, [f"identity_skill_{identity_state}"], signals)

        reasons: list[str] = []
        if self.mode in {"recognize_once", "single"}:
            query = not recognized
        elif self.mode == "fixed_interval":
            query = not recognized or gap >= self.fixed_interval_frames
        else:
            score = uncertainty
            if risk_level == "high":
                score += self.risk_bonus
                reasons.append("risk_skill_bonus")
            if identity_state in {"unknown", "uncertain", "conflicting"} or getattr(track, "semantic_match_ids", []):
                score += self.archive_bonus
                reasons.append(f"identity_skill_{identity_state}")
            if self.mode in {"experience", "full"} and experience_hint:
                score += self.experience_bonus * float(experience_hint.get("success_rate", 0.0))
                reasons.append("experience_memory_hint")
            query = not recognized or (gap >= self.min_gap_frames and score >= self.threshold)
            uncertainty = min(1.0, score)
        if query:
            reasons.extend(getattr(track, "uncertainty_reasons", []) or ["identity_evidence_required"])
            return PolicyDecision(True, "cloud_vlm_read", round(uncertainty, 4), list(dict.fromkeys(reasons)), signals)
        return PolicyDecision(False, "continue_tracking", round(uncertainty, 4), reasons or ["skills_evidence_sufficient"], signals)
