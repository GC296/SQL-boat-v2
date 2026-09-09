"""Visible-video watchkeeping risk estimation from image-plane evidence."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RiskAssessment:
    level: str
    score: float
    reasons: list[str]


def assess_track_risk(track_info: Any, config: dict[str, Any] | None = None, nearby_tracks: int = 0) -> tuple[str, list[str], float]:
    cfg = config or {}
    if track_info is None:
        return "low", ["missing_track"], 0.0
    trajectory = getattr(track_info, "trajectory", []) or []
    score = 0.0
    reasons: list[str] = []
    if len(trajectory) >= 4:
        early = trajectory[max(0, len(trajectory) - 8)]
        latest = trajectory[-1]
        scale_growth = latest["scale"] - early["scale"]
        image_motion = ((latest["cx"] - early["cx"]) ** 2 + (latest["cy"] - early["cy"]) ** 2) ** 0.5
        if scale_growth >= float(cfg.get("scale_growth_threshold", 0.012)):
            score += 0.35
            reasons.append("persistent_scale_growth")
        if image_motion >= float(cfg.get("image_motion_threshold", 0.10)):
            score += 0.15
            reasons.append("substantial_image_plane_motion")
        if len(trajectory) >= int(cfg.get("persistence_frames", 15)):
            score += 0.10
            reasons.append("persistent_track")
    uncertainty = float(getattr(track_info, "last_uncertainty_score", 0.0))
    if uncertainty >= float(cfg.get("identity_uncertainty_threshold", 0.70)):
        score += 0.15
        reasons.append("high_identity_uncertainty")
    if getattr(track_info, "identity_state", "unknown") == "conflicting":
        score += 0.15
        reasons.append("conflicting_identity_evidence")
    if int(getattr(track_info, "failed_recognition_count", 0)) >= 2:
        score += 0.10
        reasons.append("repeated_recognition_failures")
    if nearby_tracks >= int(cfg.get("multi_target_threshold", 3)):
        score += 0.15
        reasons.append("dense_multi_target_context")
    high_threshold = float(cfg.get("high_threshold", 0.65))
    medium_threshold = float(cfg.get("medium_threshold", 0.30))
    level = "high" if score >= high_threshold else ("medium" if score >= medium_threshold else "low")
    return level, reasons or ["stable_visible_track"], round(min(1.0, score), 4)


def explain_encounter(track_info: Any) -> str:
    if track_info is None:
        return "No target state is available."
    reasons = getattr(track_info, "risk_reasons", []) or []
    identity = getattr(track_info, "db_match_id", "") or getattr(track_info, "hull_number", "") or "unknown"
    level = getattr(track_info, "risk_level", "low")
    reason_text = ", ".join(reasons) if reasons else "stable visible evidence"
    return f"Visible-video watchkeeping assessment for {identity}: {level} risk based on {reason_text}. The estimate does not imply metric range, CPA, or TCPA."
