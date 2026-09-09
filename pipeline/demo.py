"""OpenCV visualization for vessel tracking and archive verification."""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class DemoRenderer:
    def __init__(self, show_fps: bool = True, show_track_id: bool = True, font_scale: float = 0.5):
        self._show_fps = show_fps
        self._show_track_id = show_track_id
        self._font_scale = font_scale

    def render(self, frame: np.ndarray, detections: list[Any], tracks: dict[int, Any], fps_info: dict[str, float] | None = None, frame_id: int = 0, queue_depth: int = 0, max_queue: int = 0) -> np.ndarray:
        canvas = frame.copy()
        for detection in detections:
            self._render_detection(canvas, detection, tracks.get(detection.track_id))
        if self._show_fps and fps_info:
            self._render_hud(canvas, fps_info, frame_id, queue_depth, max_queue)
        return canvas

    @staticmethod
    def _color(track: Any) -> tuple[int, int, int]:
        if track and getattr(track, "risk_level", "low") == "high":
            return (0, 0, 255)
        state = str(getattr(track, "identity_state", "unknown")) if track else "unknown"
        if state in {"confirmed", "structure_verified"}:
            return (0, 200, 0)
        if state in {"uncertain", "conflicting"}:
            return (0, 200, 255)
        if state == "out_of_archive":
            return (0, 0, 255)
        if track and getattr(track, "pending", False):
            return (255, 255, 0)
        return (180, 180, 180)

    def _render_detection(self, canvas: np.ndarray, detection: Any, track: Any) -> None:
        x1, y1, x2, y2 = detection.bbox
        color = self._color(track)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        if self._show_track_id:
            cv2.putText(canvas, f"Track {detection.track_id}", (x1, max(16, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, self._font_scale, color, 1)
        if track:
            self._render_panel(canvas, self._display_lines(track), x1, y2, color)

    @staticmethod
    def _display_lines(track: Any) -> list[str]:
        if not getattr(track, "recognized", False):
            return ["recognizing..."] if getattr(track, "pending", False) else ["tracking"]
        hull = str(getattr(track, "hull_number", "") or "-")
        candidate = str(getattr(track, "archive_candidate_id", "") or "-")
        verified = str(getattr(track, "verified_identity", "") or "-")
        score = float(getattr(track, "archive_similarity_score", 0.0) or 0.0)
        state = str(getattr(track, "identity_state", "unknown"))
        source = str(getattr(track, "identity_evidence_source", "none"))
        lines = [
            f"Entity: {str(getattr(track, 'entity_id', '') or '-')}",
            f"Hull read: {hull}",
            f"Archive candidate: {candidate}  sim={score:.3f}",
            f"Verified identity: {verified}",
            f"State: {state}",
            f"Evidence: {source}",
        ]
        risk = str(getattr(track, "risk_level", "low"))
        if risk != "low":
            lines.append(f"Risk: {risk}")
        return lines

    @staticmethod
    def _render_panel(canvas: np.ndarray, lines: list[str], x: int, y: int, color: tuple[int, int, int]) -> None:
        line_height = 18
        width = max(220, max((len(line) for line in lines), default=0) * 8 + 12)
        height = len(lines) * line_height + 8
        frame_height, frame_width = canvas.shape[:2]
        panel_x = max(0, min(x, frame_width - width - 1))
        panel_y = y + 4 if y + height + 4 < frame_height else max(0, y - height - 4)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + width, panel_y + height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + width, panel_y + height), color, 1)
        for index, line in enumerate(lines):
            cv2.putText(canvas, line, (panel_x + 5, panel_y + 16 + index * line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _render_hud(canvas: np.ndarray, fps_info: dict[str, float], frame_id: int, queue_depth: int, max_queue: int) -> None:
        lines = [f"Frame: {frame_id}"] + [f"{name}: {fps:.1f} FPS" for name, fps in fps_info.items()]
        if max_queue > 0:
            lines.append(f"Queue: {queue_depth}/{max_queue}")
        for index, line in enumerate(lines):
            cv2.putText(canvas, line, (10, 18 + index * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
