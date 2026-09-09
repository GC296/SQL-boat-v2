"""
ShipPipeline — 主流水线编排

三步链路（硬编码模式，无 LangChain Agent 依赖）：
  Step1: VLM 识别 → hull_number + description
  Step2: db.lookup(hull_number) → 精确匹配
  Step3: db.semantic_search_prototypes(description) → 多原型结构语义检索

级联模式（concurrent_mode=false）：
  YOLO 检测 → 三步链路 → 绑定结果 → 绘制输出

并发模式（concurrent_mode=true）：
  YOLO 检测（主循环同步）→ crop 送入队列 → VLM worker 线程并发推理
  → 结果按帧时间戳严格顺序出队 → 绑定到对应帧绘制输出
"""

from __future__ import annotations

import base64
import logging
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from agent import AgentResult
from pipeline.detector import Detection
from pipeline.demo import DemoRenderer
from pipeline.output import ScreenshotSaver
from pipeline.fps import FPSMeter, LatencyMeter
from pipeline.archive import IdentityDecision, decide_identity
from pipeline.edge_cloud import EdgeCloudExecutor
from pipeline.entity import EntityReconciler, aggregate_entity_identity
from pipeline.experiment import ExperimentLogger
from pipeline.grounding import create_grounding_provider
from pipeline.policy import LookoutPolicy
from pipeline.quality import compute_observation_quality
from pipeline.risk import assess_track_risk, explain_encounter
from pipeline.tracker import TrackManager
from pipeline.video_input import InputSource
from pipeline.central_controller import CentralVLMController

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


class ShipPipeline:
    """
    船弦号识别视频处理流水线。

    整合 YOLO 检测、三步链路识别、跟踪管理，支持级联/并发双模式。
    """

    def __init__(self, config: dict[str, Any] | None = None):
        if config is None:
            from config import load_config
            config = load_config()

        self._config = config
        pipe_cfg = config.get("pipeline", {})

        self._concurrent_mode: bool = bool(pipe_cfg.get("concurrent_mode", False))
        self._max_concurrent: int = pipe_cfg.get("max_concurrent") or 4
        self._max_queued_frames: int = pipe_cfg.get("max_queued_frames") or 30
        self._target_fps: float = float(pipe_cfg.get("target_fps", 0))  # 0=不限制
        self._frame_skip_interval: int = 1  # 帧跳过间隔，>1 时跳帧减少计算量
        self._process_every_n: int = max(1, pipe_cfg.get("process_every_n_frames") or 1)
        self._detect_every_n: int = max(1, pipe_cfg.get("detect_every_n_frames") or 1)
        self._demo_enabled: bool = bool(pipe_cfg.get("demo", False))
        self._save_screenshots: bool = bool(pipe_cfg.get("save_screenshots", True))
        self._save_output_video: bool = bool(pipe_cfg.get("save_output_video", False))
        self._enable_refresh: bool = bool(pipe_cfg.get("enable_refresh", False))
        self._skip_refresh_matched: bool = bool(pipe_cfg.get("skip_refresh_matched", False))
        self._gap_num: int = pipe_cfg.get("gap_num") or 150
        active_cfg = pipe_cfg.get("active_perception", {}) or {}
        self._active_perception_enabled: bool = bool(active_cfg.get("enabled", True))
        self._uncertainty_threshold: float = float(active_cfg.get("uncertainty_threshold", 0.65))
        self._active_min_gap_frames: int = int(active_cfg.get("min_gap_frames", 45))
        self._prompt_mode: str = pipe_cfg.get("prompt_mode") or "detailed"
        self._output_size: tuple[int, int] | None = None
        _os = pipe_cfg.get("output_size")
        if _os and len(_os) == 2:
            self._output_size = (int(_os[0]), int(_os[1]))
        self._stop_file: Path | None = Path(pipe_cfg["stop_file"]) if pipe_cfg.get("stop_file") else None

        from database import ShipDatabase
        self._db = ShipDatabase(config=config)

        self._grounder = create_grounding_provider(pipe_cfg)

        experiment_cfg = config.get("experiment", {}) or {}
        self._controller_mode = str(experiment_cfg.get("controller_mode", "legacy") or "legacy").strip().lower()
        if self._controller_mode not in {"legacy", "central_vlm", "mock"}:
            raise ValueError("experiment.controller_mode must be 'legacy', 'central_vlm', or 'mock'")
        ledger_cfg = experiment_cfg.get("temporal_ledger", {}) or {}
        self._memory_mode = str(experiment_cfg.get("memory_mode", "full")).lower()
        self._quality_cfg = experiment_cfg.get("quality", {}) or {}
        self._risk_cfg = experiment_cfg.get("risk", {}) or {}
        self._archive_cfg = experiment_cfg.get("archive", {}) or {}
        self._experience_cfg = experiment_cfg.get("experience", {}) or {}
        self._entity_cfg = experiment_cfg.get("entity_reconciliation", {}) or {}
        self._visual_cfg = experiment_cfg.get("visual_archive", {}) or {}
        self._identity_decision_mode = str(self._visual_cfg.get("identity_decision_mode", "fused")).strip().lower()
        if self._identity_decision_mode not in {"fused", "visual_only"}:
            raise ValueError("experiment.visual_archive.identity_decision_mode must be 'fused' or 'visual_only'")
        self._visual_backend = str(self._visual_cfg.get("backend", "dinov2")).strip().lower()
        self._visual_index = None
        if bool(self._visual_cfg.get("enabled", False)):
            try:
                if self._visual_backend in {"qwen_vl_reranker", "qwen3_vl_reranker", "reranker"}:
                    from pipeline.visual_reranker import get_qwen_visual_reranker_index

                    reranker_cfg = {
                        **dict(config.get("visual_reranker", {}) or {}),
                        **self._visual_cfg,
                    }
                    self._visual_index = get_qwen_visual_reranker_index(reranker_cfg)
                    self._visual_backend = "qwen_vl_reranker"
                elif self._visual_backend == "dinov2":
                    from pipeline.visual_archive import get_visual_prototype_index

                    self._visual_index = get_visual_prototype_index(self._visual_cfg)
                else:
                    raise ValueError(f"Unsupported visual archive backend: {self._visual_backend}")
                logger.info(
                    "Visual archive enabled: backend=%s prototypes=%d device=%s",
                    self._visual_backend,
                    len(self._visual_index.records),
                    self._visual_index.device,
                )
            except Exception:
                if not bool(self._visual_cfg.get("fail_open", True)):
                    raise
                logger.exception("Visual archive initialization failed; continuing without visual evidence")
        memory_write_cfg = experiment_cfg.get("memory_write", {}) or {}
        self._profile_memory_write = bool(memory_write_cfg.get("profile", False))
        self._experience_memory_write = bool(memory_write_cfg.get("experience", False))
        if not bool((experiment_cfg.get("temporal_ledger", {}) or {}).get("enabled", True)):
            self._memory_mode = "none"
        elif not bool(self._archive_cfg.get("enabled", True)) and self._memory_mode in {"tel_archive", "full"}:
            self._memory_mode = "tel"
        policy_cfg = dict(experiment_cfg.get("policy", {}) or {})
        policy_cfg.setdefault("mode", experiment_cfg.get("policy_mode", "full"))
        policy_cfg.setdefault("min_query_quality", self._quality_cfg.get("min_query_quality", 0.25))
        policy_cfg.setdefault("min_structure_observations", self._archive_cfg.get("min_structure_observations", 2))
        policy_cfg.setdefault(
            "min_confirmation_observations",
            self._visual_cfg.get("visual_only_min_observations", self._archive_cfg.get("min_consistent_observations", 2)),
        )
        policy_cfg.setdefault("visual_confirmation_min_score", self._visual_cfg.get("visual_only_min_score", 0.70))
        policy_cfg.setdefault("visual_confirmation_min_margin", self._visual_cfg.get("visual_only_min_margin", 0.03))
        policy_cfg.setdefault("visual_confirmation_strong_score", self._visual_cfg.get("visual_only_strong_score", 0.90))
        policy_cfg.setdefault("visual_only_decision_rule", self._visual_cfg.get("visual_only_decision_rule", "legacy"))
        policy_cfg.setdefault(
            "min_visual_rejection_observations",
            self._visual_cfg.get("min_out_of_archive_observations", 2),
        )
        policy_cfg.setdefault("visual_rejection_max_score", self._visual_cfg.get("visual_only_reject_score", 0.70))
        policy_cfg.setdefault("visual_rejection_max_margin", self._visual_cfg.get("visual_only_reject_margin", 0.08))
        self._policy = LookoutPolicy(policy_cfg)
        self._experiment_logger = ExperimentLogger(experiment_cfg)
        self._edge_cloud = EdgeCloudExecutor(experiment_cfg.get("network", {}) or {}, seed=int(experiment_cfg.get("seed", 42)))
        self._entity_reconciler = EntityReconciler(self._entity_cfg)
        self._tracker = TrackManager(max_stale_frames=pipe_cfg.get("max_stale_frames", 300), memory_mode=self._memory_mode, ledger_decay=ledger_cfg.get("decay", 0.97), ledger_min_support=ledger_cfg.get("min_support", 0.30))
        self._fps = FPSMeter(window_seconds=10.0)
        self._latency = LatencyMeter(window_seconds=10.0)
        self._renderer = DemoRenderer(show_fps=True, show_track_id=True)

        output_dir = pipe_cfg.get("output_dir", "./output")
        self._saver = ScreenshotSaver(output_dir=output_dir)

        # 并发模式（VLM worker 线程池）
        self._task_queue: queue.Queue = queue.Queue(maxsize=self._max_queued_frames)
        self._result_queue: queue.Queue = queue.Queue(maxsize=self._max_queued_frames)
        self._workers: list[threading.Thread] = []
        self._stop_event = threading.Event()

        # 背压控制：跟踪管道积压
        self._frames_submitted: int = 0   # 主循环已提交到 raw_writer 的帧数
        self._frames_encoded_ref: list[int] | None = pipe_cfg.get("_frames_encoded_ref")  # 共享引用 [frames_fed]
        self._max_pipe_lag: int = pipe_cfg.get("max_pipe_lag", 45)  # 最大允许积压帧数

        # Agent 运行链路日志
        self._agent_trace: list[dict[str, Any]] = []
        self._trace_lock = threading.Lock()
        self._max_trace_entries = 500

        # 渲染帧缓存（非处理帧复用，避免重复 PIL 渲染）
        self._cached_display_frame: np.ndarray | None = None

        self._central_controller: CentralVLMController | None = None
        self._central_terminal_policy = "multimodal_evidence"
        if self._controller_mode in {"central_vlm", "mock"}:
            controller_cfg = dict(experiment_cfg.get("central_controller", {}) or {})
            self._central_terminal_policy = str(controller_cfg.get("terminal_policy", "multimodal_evidence")).strip().lower()
            controller_cfg.setdefault("query_limit", (experiment_cfg.get("policy", {}) or {}).get("max_queries_per_entity", 3))
            controller_cfg.setdefault("min_ooa_observations", self._visual_cfg.get("min_out_of_archive_observations", 2))
            controller_cfg.setdefault("min_ooa_score", self._visual_cfg.get("out_of_archive_score", 0.50))
            def controller_infer(context, images):
                from tools import _controller_infer
                return _controller_infer(context, images, llm_config=self._config.get("llm", {}))
            if self._controller_mode == "mock":
                mock_responses = list(controller_cfg.get("mock_responses", []) or [])
                if not mock_responses:
                    mock_responses = [{"kind": "continue", "reason": "mock controller waiting"}]

                def controller_infer(_context: dict[str, Any], _images: list[str]) -> dict[str, Any]:
                    response = mock_responses.pop(0) if mock_responses else {"kind": "continue", "reason": "mock controller waiting"}
                    return dict(response)

            self._central_controller = CentralVLMController(
                self._db,
                visual_index=self._visual_index,
                visual_config=self._visual_cfg,
                config=controller_cfg,
                image_encoder=self._encode_image,
                controller_infer=controller_infer,
                verify_fn=self._central_verify_evidence,
                on_agent_result=self._handle_agent_result,
                on_verify_result=self._handle_central_verify_result,
                on_visual_result=self._handle_central_visual_result,
                on_finish=self._handle_central_finish,
                on_log=self._log_central_event,
            )

        logger.info(
            "ShipPipeline 初始化: mode=%s, workers=%d, process_every=%d, refresh=%s(gap=%d, skip_matched=%s)",
            "concurrent" if self._concurrent_mode else "cascade",
            self._max_concurrent,
            self._process_every_n,
            "on" if self._enable_refresh else "off",
            self._gap_num,
            "on" if self._skip_refresh_matched else "off",
        )

    # ── 链路日志 ──────────────────────────────

    def _log_agent_trace(self, event_type: str, track_id: int, frame_id: int, content: str = "", **extra: Any) -> None:
        entry = {"type": event_type, "track_id": track_id, "frame_id": frame_id, "content": content, "timestamp": time.time(), **extra}
        with self._trace_lock:
            self._agent_trace.append(entry)
            if len(self._agent_trace) > self._max_trace_entries:
                self._agent_trace = self._agent_trace[-(self._max_trace_entries // 2):]

    def _log_track_summary(self, track_id: int) -> None:
        with self._trace_lock:
            entries = [e for e in self._agent_trace if e["track_id"] == track_id]
        if not entries:
            return
        latest_frame = max(e["frame_id"] for e in entries)
        entries = [e for e in entries if e["frame_id"] == latest_frame]
        types = {e["type"]: e["content"] for e in entries}
        step1 = types.get("step1_vlm") or "—"
        step2 = types.get("step2_lookup") or "—"
        step3 = types.get("step3_result") or types.get("step3_fallback") or "—"
        logger.info("[Track %d] frame=%d | Step1(VLM): %s | Step2(Lookup): %s | Step3(Result): %s", track_id, latest_frame, step1, step2, step3)

    # ── 工具方法 ──────────────────────────────

    @staticmethod
    def _encode_image(image: np.ndarray) -> str:
        # quality=85 平衡清晰度与速度（弦号文字需要足够清晰度）
        success, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise RuntimeError("图像编码失败")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    def _central_verify_evidence(self, crop: np.ndarray, question: str, fields: list[str]) -> dict[str, Any]:
        from tools import _vlm_verify

        return _vlm_verify(crop, question=question, fields=fields, llm_config=self._config.get("llm", {}))

    def _log_central_event(self, event_type: str, payload: dict[str, Any]) -> None:
        track_id = int(payload.get("track_id", 0) or 0)
        frame_id = int(payload.get("frame_id", 0) or 0)
        trace_payload = {key: value for key, value in payload.items() if key not in {"track_id", "frame_id"}}
        self._log_agent_trace(f"central_{event_type}", track_id, frame_id, content=str(payload.get("reason", payload.get("status", ""))), **trace_payload)
        stream = "controller_calls" if event_type == "controller_call" else "tool_requests" if event_type in {"tool_request", "tool_result"} else "agent_events"
        self._experiment_logger.log(stream, payload)
        self._print_central_trace(event_type, payload)

    @staticmethod
    def _short_trace_value(value: Any, limit: int = 180) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 3)] + "..."

    def _print_central_trace(self, event_type: str, payload: dict[str, Any]) -> None:
        """Print the controller's observable state/action trace.

        This exposes the returned decision and state transitions, not hidden
        model reasoning. It is intentionally compact because the full event
        payload is already written to the experiment log.
        """
        entity_id = str(payload.get("entity_id", "-") or "-")
        frame_id = int(payload.get("frame_id", 0) or 0)
        prefix = f"[Agent {entity_id} | frame {frame_id}]"

        if event_type == "controller_call":
            decision = payload.get("decision", {}) or {}
            kind = str(decision.get("kind", "protocol_error"))
            if kind == "tool":
                action = f"tool={decision.get('tool', '?')} args={decision.get('args', {})}"
            elif kind == "finish":
                action = f"finish={decision.get('decision', '?')} archive={decision.get('archive_id', '') or '-'}"
            else:
                action = "continue"
            reason = self._short_trace_value(decision.get("reason", ""), 220) or "(no reason returned)"
            refs = list(decision.get("evidence_refs", []) or [])
            patch_count = len(decision.get("belief_patch", []) or [])
            logger.info(
                "%s step=%s state=v%s | VLM action: %s | reason: %s | refs=%s patches=%d",
                prefix,
                payload.get("control_step", "?"),
                payload.get("state_version", "?"),
                action,
                reason,
                refs or "-",
                patch_count,
            )
            return

        if event_type == "controller_infer_start":
            logger.info(
                "%s | VLM deciding: view=%s observations=%s evidence=%s claims=%s assessment_candidates=%s visual_candidate=%s visual_score=%.3f visual_searches=%s gaps=%s | budget=%s/%s (reserved=%s)",
                prefix,
                payload.get("current_view_id", "-") or "-",
                payload.get("observation_count", 0),
                payload.get("evidence_count", 0),
                payload.get("claim_count", 0),
                payload.get("candidate_assessment_count", 0),
                payload.get("visual_candidate_id", "-") or "-",
                payload.get("visual_score", 0.0),
                payload.get("visual_search_count", 0),
                payload.get("gap_count", 0),
                payload.get("budget_used", 0),
                payload.get("query_limit", 0),
                payload.get("budget_reserved", 0),
            )
            return

        if event_type == "tool_request":
            logger.info(
                "%s | execute %s args=%s | state=v%s",
                prefix,
                payload.get("tool", "?"),
                payload.get("args", {}),
                payload.get("state_version", "?"),
            )
            return

        if event_type == "tool_result":
            changes = payload.get("state_changes", {}) or {}
            detail = payload.get("payload_summary", {}) or {}
            logger.info(
                "%s | result %s status=%s evidence=%s | budget=%s/%s | update=%s | detail=%s",
                prefix,
                payload.get("tool", "?"),
                payload.get("status", "?"),
                payload.get("evidence_refs", []) or "-",
                payload.get("used", "?"),
                payload.get("query_limit", "?"),
                changes or "none",
                detail or "none",
            )
            return

        if event_type == "belief_update":
            logger.info(
                "%s | belief update: %s",
                prefix,
                payload.get("changes", {}) or "no observable change",
            )
            return

        if event_type == "finish_rejected":
            logger.warning(
                "%s | terminal proposal rejected: decision=%s reason=%s",
                prefix,
                payload.get("decision", "?"),
                payload.get("reason", "unknown"),
            )
            return

        if event_type == "protocol_error":
            logger.error(
                "%s | controller protocol error: %s",
                prefix,
                payload.get("error", "unknown"),
            )
            return

        if event_type == "controller_invalid_response":
            logger.error(
                "%s | invalid VLM JSON: %s | returned=%s",
                prefix,
                payload.get("error", "unknown"),
                self._short_trace_value(payload.get("raw_response", ""), 420) or "-",
            )
            return

        if event_type == "episode_closed":
            logger.info(
                "%s | episode closed: decision=%s archive=%s reason=%s",
                prefix,
                payload.get("decision", "?"),
                payload.get("archive_id", "") or "-",
                self._short_trace_value(payload.get("reason", ""), 220),
            )
            return

        if event_type == "continue_wait":
            logger.info(
                "%s | waiting for next observation (observation_version=%s)",
                prefix,
                payload.get("observation_version", "?"),
            )

    def _handle_central_finish(self, entity_id: str, track_id: int, frame_id: int, decision: str,
                               archive_id: str, evidence_refs: list[str], reason: str) -> dict[str, Any]:
        """Project an entity decision to its tracks; never select an identity here."""
        info = self._tracker.get_any(track_id)
        if info is not None and info.entity_id != entity_id:
            return {"status": "rejected", "reason": "entity_track_mismatch"}
        if decision == "known" and (not archive_id or self._db.lookup(archive_id) is None):
            return {"status": "rejected", "reason": "archive_identity_not_found"}
        if self._central_terminal_policy == "legacy_visual_only":
            if info is None:
                return {"status": "rejected", "reason": "track_not_found"}
            if decision == "known" and not (
                archive_id == info.visual_candidate_id
                and info.visual_similarity_score >= float(self._visual_cfg.get("visual_only_min_score", .8))
                and info.visual_observation_count >= int(self._visual_cfg.get("visual_only_min_observations", 1))
            ):
                return {"status": "rejected", "reason": "archive_identity_guard_blocked"}
            if decision == "out_of_archive" and not info.visual_low_score_observations:
                return {"status": "rejected", "reason": "open_set_guard_blocked"}
        tracks = self._tracker.tracks_for_entity(entity_id)
        if not tracks:
            owner = self._central_controller.store.get(entity_id) if self._central_controller else None
            if owner is None:
                return {"status": "rejected", "reason": "entity_not_found"}
            return {"status": "accepted", "reason": "entity_record_only", "projection_pending": True}
        for target in tracks:
            if decision in {"known", "out_of_archive"}:
                applied = self._tracker.bind_controller_identity(target.track_id, archive_id if decision == "known" else "",
                    self._db.lookup(archive_id) or "" if decision == "known" else "",
                    state="confirmed" if decision == "known" else "out_of_archive")
            elif decision == "review":
                applied = target.episode_status == "review_requested" or self._tracker.request_review(target.track_id, frame_id, "central_vlm_review", [reason],
                    {"entity_id": entity_id, "evidence_refs": evidence_refs})
            else:
                return {"status": "rejected", "reason": "invalid_terminal_decision"}
            if not applied:
                return {"status": "rejected", "reason": "track_projection_failed"}
        return {"status": "accepted", "reason": "terminal_recorded"}

    def _handle_central_visual_result(self, track_id: int, frame_id: int, result: AgentResult) -> None:
        info = self._tracker.get_any(track_id)
        if info is None or (getattr(result, "entity_id", info.entity_id) != info.entity_id):
            return
        if getattr(result, "episode_id", ""):
            owner = self._central_controller.store.get(info.entity_id)
            if owner is None or owner.episode_id != result.episode_id or owner.session_id != result.session_id:
                return
        self._tracker.set_identity_decision_mode(track_id, self._identity_decision_mode)
        self._tracker.update_visual_match(
            track_id,
            result.visual_candidate_id,
            result.visual_similarity_score,
            result.visual_margin,
            low_score_threshold=float(self._visual_cfg.get("out_of_archive_score", 0.50)),
        )
        self._experiment_logger.log("visual", {
            "track_id": track_id,
            "entity_id": info.entity_id,
            "frame_id": frame_id,
            "source": "central_vlm_tool",
            "visual_backend": result.visual_backend,
            "visual_candidate_id": result.visual_candidate_id,
            "visual_similarity_score": result.visual_similarity_score,
            "visual_margin": result.visual_margin,
            "visual_matches": result.visual_matches,
            "visual_matching_latency_ms": result.visual_matching_latency_ms,
        })

    def _handle_central_verify_result(self, track_id: int, frame_id: int, result: AgentResult) -> None:
        """Expose semantic tool output without invoking the legacy identity path."""
        info = self._tracker.get_any(track_id)
        if info is None or (getattr(result, "entity_id", info.entity_id) != info.entity_id):
            return
        if getattr(result, "episode_id", ""):
            owner = self._central_controller.store.get(info.entity_id)
            if owner is None or owner.episode_id != result.episode_id or owner.session_id != result.session_id:
                return
        self._tracker.bind_semantic_matches(track_id, result.semantic_match_ids)
        self._experiment_logger.log("semantic_evidence", {
            "track_id": track_id,
            "entity_id": info.entity_id,
            "frame_id": frame_id,
            "source": "central_vlm_tool",
            "semantic_match_ids": list(result.semantic_match_ids),
            "semantic_matches": list(result.semantic_matches),
            "identity_features": dict(result.identity_features),
            "description": result.description,
        })

    def _central_process(self, detections: list[Detection], frame_id: int) -> None:
        if self._central_controller is None:
            return
        for det in detections:
            if det.crop is None or det.crop.size == 0:
                continue
            info = self._tracker.get(det.track_id)
            if info is None:
                continue
            episode = self._central_controller.store.get(info.entity_id)
            if episode is not None and episode.status != "active":
                terminal = episode.terminal_record
                if terminal.get("commit_status") == "committed" and not info.recognized and info.identity_state != "review_requested":
                    self._handle_central_finish(info.entity_id, det.track_id, frame_id, terminal["decision"], terminal.get("archive_id", ""), terminal.get("evidence_refs", []), terminal.get("reason", ""))
                continue
            source_frame = getattr(det, "source_frame_id", frame_id)
            self._central_controller.submit_observation(info.entity_id, det.track_id, source_frame, det.crop.copy(), info=info)

    # ── 三步链路核心 ──────────────────────────

    def _run_three_step_chain(self, crop: np.ndarray, track_id: int = 0, frame_id: int = 0) -> AgentResult:
        """
        三步链路：VLM识别 → 精确查找 → 语义检索。

        Step1: _vlm_infer(crop) → 弦号 + 描述
        Step2: db.lookup(hull_number) → 有弦号时精确查找
        Step3: db.semantic_search_prototypes(description) → 舷号未匹配或无舷号时多原型结构检索
        """
        from tools import _vlm_infer

        # Step1: VLM 识别
        crop_b64 = self._encode_image(crop)
        vlm_result = _vlm_infer(crop_b64, prompt_mode=self._prompt_mode)
        hull_number = vlm_result.get("hull_number", "")
        description = vlm_result.get("description", "")
        identity_features = dict(vlm_result.get("identity_features", {}) or {})

        self._log_agent_trace(
            "step1_vlm", track_id=track_id, frame_id=frame_id,
            content=f"弦号={hull_number or '(无)'} 描述={description[:40] if description else '(无)'}",
        )

        if not hull_number and not description:
            return AgentResult(answer="VLM 未返回结果")

        if self._identity_decision_mode == "visual_only":
            self._log_agent_trace(
                "identity_lookup_skipped",
                track_id=track_id,
                frame_id=frame_id,
                content="visual_only mode: text retained for reporting, archive identity lookup skipped",
            )
            result = AgentResult(
                hull_number=hull_number,
                description=description,
                identity_features=identity_features,
                match_type="none",
            )
        else:
            result = self._local_lookup_retrieve(
                hull_number,
                description,
                identity_features=identity_features,
                track_id=track_id,
                frame_id=frame_id,
            )
        result.vlm_http_attempts = max(1, int(vlm_result.get("_vlm_http_attempts", 1)))
        result.vlm_model = str(vlm_result.get("_vlm_model", "") or "")
        result.vlm_base_url = str(vlm_result.get("_vlm_base_url", "") or "")
        result.vlm_fallback_used = bool(vlm_result.get("_vlm_fallback_used", False))
        return result

    def _local_lookup_retrieve(self, hull_number: str, description: str, identity_features: dict[str, str] | None = None, track_id: int = 0, frame_id: int = 0) -> AgentResult:
        """Use exact hull evidence and attribute-aware structural archive matching."""
        exact_matched = False
        if hull_number:
            archive_description = self._db.lookup(hull_number)
            exact_matched = archive_description is not None
            if exact_matched:
                description = description or archive_description
        prototype_search = getattr(self._db, "semantic_search_prototypes", self._db.semantic_search)
        raw_matches = prototype_search(description) if description else []
        semantic_matches = [{**match, "embedding_score": float(match.get("score", 0.0))} for match in raw_matches]
        semantic_ids = [result["hull_number"] for result in semantic_matches if result.get("hull_number")]
        match_type = "exact" if exact_matched else ("semantic" if semantic_ids else "none")
        if track_id:
            self._log_agent_trace(
                "identity_lookup", track_id=track_id, frame_id=frame_id,
                content=f"hull={hull_number or '(none)'} match={match_type} structure_candidates={semantic_matches}",
            )
        return AgentResult(
            hull_number=hull_number,
            description=description,
            identity_features=identity_features,
            match_type=match_type,
            semantic_match_ids=semantic_ids,
            semantic_matches=semantic_matches,
        )

    def _run_recognition(self, crop: np.ndarray, track_id: int = 0, frame_id: int = 0) -> AgentResult:
        """Run cloud VLM reasoning and record communication and latency costs."""
        track_info = self._tracker.get(track_id)
        target_image_path = self._experiment_logger.save_image(
            "targets",
            crop,
            entity_id=track_info.entity_id if track_info else "",
            track_id=track_id,
            frame_id=frame_id,
        )
        encoded, encoded_buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        upload_bytes = int(encoded_buffer.nbytes) if encoded else int(crop.nbytes)
        with self._latency.measure("vlm"):
            result, metrics = self._edge_cloud.run("cloud_vlm_read", upload_bytes, lambda: self._run_three_step_chain(crop, track_id=track_id, frame_id=frame_id), outcome_fn=lambda item: item.match_type)
        if self._visual_index is not None:
            try:
                visual_matches, visual_latency_ms = self._visual_index.search(crop, top_k=int(self._visual_cfg.get("top_k", 3)))
                result.visual_matches = visual_matches
                result.visual_match_ids = [str(item.get("hull_number", "")) for item in visual_matches if item.get("hull_number")]
                if visual_matches:
                    result.visual_candidate_id = str(visual_matches[0].get("hull_number", ""))
                    result.visual_similarity_score = float(visual_matches[0].get("score", 0.0))
                    second_score = float(visual_matches[1].get("score", 0.0)) if len(visual_matches) > 1 else 0.0
                    result.visual_margin = result.visual_similarity_score - second_score
                result.visual_embedding_latency_ms = float(visual_latency_ms)
                result.visual_matching_latency_ms = float(visual_latency_ms)
                result.visual_backend = self._visual_backend
                result.visual_model_request_count = int(getattr(self._visual_index, "last_request_count", 0))
                result.visual_pairs_scored = int(getattr(self._visual_index, "last_pair_count", 0))
                info = self._tracker.get(track_id)
                self._experiment_logger.log("visual", {"track_id": track_id, "entity_id": info.entity_id if info else "", "frame_id": frame_id, "visual_backend": result.visual_backend, "visual_candidate_id": result.visual_candidate_id, "visual_similarity_score": result.visual_similarity_score, "visual_margin": result.visual_margin, "visual_embedding_latency_ms": result.visual_embedding_latency_ms, "visual_matching_latency_ms": result.visual_matching_latency_ms, "visual_model_request_count": result.visual_model_request_count, "visual_pairs_scored": result.visual_pairs_scored, "visual_matches": result.visual_matches})
            except Exception as exc:
                if not bool(self._visual_cfg.get("fail_open", True)):
                    raise
                logger.warning("Visual prototype search failed (track=%d, frame=%d): %s", track_id, frame_id, exc)
        request_count = max(1, int(getattr(result, "vlm_http_attempts", 1)))
        metrics["request_count"] = request_count
        metrics["upload_bytes"] = int(metrics.get("upload_bytes", upload_bytes)) * request_count
        if target_image_path:
            metrics["target_image_path"] = target_image_path
        info = self._tracker.get(track_id)
        self._experiment_logger.log("edge_cloud", {"track_id": track_id, "entity_id": info.entity_id if info else "", "frame_id": frame_id, **metrics})
        return result

    def _experience_hint(self, info: Any) -> dict[str, Any] | None:
        if not self._experience_cfg.get("enabled", True) or info is None:
            return None
        return self._db.experience_action_hint(scene_type="visible", uncertainty=info.last_uncertainty_score, failure_reason="repeated_failure" if info.failed_recognition_count >= 2 else "", top_k=int(self._experience_cfg.get("top_k", 5)), min_similarity=float(self._experience_cfg.get("min_similarity", 0.60)))

    def _should_query_track(self, track_id: int, frame_id: int) -> bool:
        info = self._tracker.get(track_id)
        self._tracker.set_identity_decision_mode(track_id, self._identity_decision_mode)
        info = self._tracker.get(track_id)
        if info and info.identity_state == "review_requested":
            return False
        decision = self._policy.decide(info, frame_id, self._experience_hint(info))
        identity_state_before = info.identity_state if info else "unknown"
        metadata = {"policy_score": decision.score, "policy_mode": self._policy.mode, "skill_signals": decision.skill_signals}
        if decision.action == "confirm_visual_identity" and info and info.visual_candidate_id:
            candidate_id = str(info.visual_candidate_id)
            visual_score = float(info.visual_similarity_score)
            applied = self._tracker.apply_identity_decision(
                track_id,
                candidate_id,
                "structure_verified",
                visual_score,
                evidence_source="learned_visual_policy",
                preserve_verified_identity=True,
            )
            if applied:
                self._tracker.bind_db_match(
                    track_id,
                    candidate_id,
                    self._db.lookup(candidate_id) or "",
                    evidence_source="learned_visual_policy",
                    confidence=visual_score,
                )
            metadata["policy_identity_applied"] = applied
            metadata["policy_identity_candidate"] = candidate_id
        if (
            decision.action == "monitor_unknown"
            and (
                "learned_stop_out_of_archive" in decision.reasons
                or "visual_score_gate_budget_exhausted" in decision.reasons
            )
            and identity_state_before not in {"confirmed", "structure_verified", "out_of_archive"}
        ):
            metadata["policy_identity_applied"] = self._tracker.apply_identity_decision(
                track_id,
                "",
                "out_of_archive",
                decision.score,
                evidence_source="learned_policy_stop_out_of_archive",
                preserve_verified_identity=True,
            )
        if decision.action in {"request_remote_verification", "remote_report"}:
            metadata["review_request_applied"] = self._tracker.request_review(
                track_id,
                frame_id,
                decision.action,
                decision.reasons,
                metadata,
            )
        updated_info = self._tracker.get(track_id)
        identity_state_after = updated_info.identity_state if updated_info else identity_state_before
        metadata.update({"identity_state_before": identity_state_before, "identity_state_after": identity_state_after})
        self._tracker.record_action(track_id, frame_id, decision.action, decision.reasons, metadata)
        self._experiment_logger.log("actions", {
            "track_id": track_id,
            "entity_id": updated_info.entity_id if updated_info else "",
            "frame_id": frame_id,
            "action": decision.action,
            "should_query": decision.should_query,
            "score": decision.score,
            "reasons": decision.reasons,
            "skill_signals": decision.skill_signals,
            "identity_state_before": identity_state_before,
            "identity_state_after": identity_state_after,
            "review_request_applied": metadata.get("review_request_applied"),
            "episode_status": updated_info.episode_status if updated_info else "active",
            "memory_mode": self._memory_mode,
            "policy_mode": self._policy.mode,
        })
        return decision.should_query

    # 将标准化描述拆分为稳定档案属性，供记忆侧写使用。

    def _extract_profile_attributes(self, description: str) -> list[str]:
        if not description:
            return []
        tokens = []
        for raw in description.replace('，', ' ').replace(',', ' ').replace('；', ' ').replace(';', ' ').replace('：', ' ').split():
            token = raw.strip()
            if len(token) >= 2:
                tokens.append(token[:24])
        return list(dict.fromkeys(tokens[:8]))

    def _record_memory_updates(self, track_id: int, frame_id: int, agent_result: AgentResult, action_taken: str) -> None:
        info = self._tracker.get(track_id)
        if info is None:
            return
        matched = agent_result.match_type == 'exact'
        failure_reason = '' if matched else ('semantic_only' if agent_result.match_type == 'semantic' else 'unmatched')
        if self._profile_memory_write and agent_result.hull_number:
            self._db.upsert_profile_memory(
                agent_result.hull_number,
                db_match_id=info.db_match_id or agent_result.hull_number,
                visual_summary=agent_result.description,
                attributes=self._extract_profile_attributes(agent_result.description),
                matched=matched,
                misread='' if matched else (agent_result.semantic_match_ids[0] if agent_result.semantic_match_ids else ''),
                frame_id=frame_id,
            )
        if self._experience_memory_write:
            self._db.add_recognition_experience(
                track_id=track_id, frame_id=frame_id, scene_type=info.modality,
                uncertainty=info.last_uncertainty_score, hull_number_pred=agent_result.hull_number,
                db_match_id=info.db_match_id, match_type=agent_result.match_type, final_correct=matched,
                failure_reason=failure_reason, action_taken=action_taken, action_success=matched or bool(agent_result.semantic_match_ids),
                semantic_candidates=agent_result.semantic_match_ids, notes=agent_result.description[:200],
            )

    def _decide_visual_only_identity(self, info: Any, agent_result: AgentResult) -> IdentityDecision:
        from pipeline.visual_archive import (
            visual_consistency_supports_confirmation,
            visual_open_set_override,
        )

        candidates = [
            (str(item.get("hull_number", "")), float(item.get("score", 0.0) or 0.0))
            for item in agent_result.visual_matches
            if item.get("hull_number")
        ]
        visual_candidate = str(agent_result.visual_candidate_id or "")
        visual_score = float(agent_result.visual_similarity_score or 0.0)
        if not bool(self._visual_cfg.get("decision_enabled", False)):
            return IdentityDecision("", "unknown", 0.0, candidates, "visual_only_disabled")
        if visual_consistency_supports_confirmation(
            "",
            "unknown",
            visual_candidate,
            visual_score,
            agent_result.visual_margin,
            info.visual_consistent_observations,
            self._visual_cfg,
        ):
            agent_result.visual_gate_reason = "repeated_visual_identity_support"
            return IdentityDecision(
                visual_candidate,
                "structure_verified",
                visual_score,
                candidates,
                "visual_consistency",
            )
        state, reason = visual_open_set_override(
            "",
            visual_candidate,
            visual_score,
            agent_result.visual_margin,
            self._visual_cfg,
            low_score_observations=info.visual_low_score_observations,
            observation_count=int(getattr(info, "visual_observation_count", info.visual_low_score_observations)),
        )
        if state:
            agent_result.visual_gate_reason = reason
            identity = visual_candidate if state == "uncertain" else ""
            return IdentityDecision(identity, state, visual_score, candidates, f"visual_only:{reason}")
        if visual_candidate:
            agent_result.visual_gate_reason = "visual_candidate_requires_consistency"
            return IdentityDecision(
                visual_candidate,
                "uncertain",
                visual_score,
                candidates,
                "visual_only:visual_candidate_requires_consistency",
            )
        return IdentityDecision("", "unknown", 0.0, candidates, "visual_only:no_visual_candidate")

    def _commit_identity_decision(self, track_id: int, decision: IdentityDecision) -> None:
        candidate_identity = "" if decision.state == "out_of_archive" else (
            decision.identity or (decision.candidates[0][0] if decision.candidates else "")
        )
        applied = self._tracker.apply_identity_decision(
            track_id,
            candidate_identity,
            decision.state,
            decision.confidence,
            evidence_source=decision.evidence_source,
            preserve_verified_identity=bool(self._archive_cfg.get("preserve_verified_identity", True)),
            conflict_observations_to_downgrade=int(self._archive_cfg.get("conflict_observations_to_downgrade", 2)),
        )
        if applied and decision.state == "structure_verified" and decision.identity:
            self._tracker.bind_db_match(
                track_id,
                decision.identity,
                self._db.lookup(decision.identity) or "",
                evidence_source=decision.evidence_source,
                confidence=decision.confidence,
            )

    def _handle_agent_result(self, track_id: int, frame_id: int, agent_result: AgentResult, update_visual: bool = True) -> None:
        self._log_track_summary(track_id)
        self._tracker.bind_result(
            track_id,
            agent_result.hull_number,
            agent_result.description,
            frame_id=frame_id,
            identity_features=agent_result.identity_features,
            match_type=agent_result.match_type,
            semantic_match_ids=agent_result.semantic_match_ids,
            semantic_matches=agent_result.semantic_matches,
        )
        self._tracker.set_identity_decision_mode(track_id, self._identity_decision_mode)
        if update_visual:
            self._tracker.update_visual_match(
                track_id,
                agent_result.visual_candidate_id,
                agent_result.visual_similarity_score,
                agent_result.visual_margin,
                low_score_threshold=float(
                    self._visual_cfg.get("visual_only_reject_score", 0.70)
                    if self._identity_decision_mode == "visual_only"
                    else self._visual_cfg.get("out_of_archive_score", 0.50)
                ),
            )
        info = self._tracker.get(track_id)
        if self._identity_decision_mode == "visual_only" and info:
            decision = self._decide_visual_only_identity(info, agent_result)
            self._commit_identity_decision(track_id, decision)
        elif agent_result.match_type == "exact":
            archive_description = self._db.lookup(agent_result.hull_number) or agent_result.description
            self._tracker.bind_db_match(
                track_id,
                agent_result.hull_number,
                archive_description,
                evidence_source="exact_hull",
                confidence=1.0,
            )
        elif info and bool(self._archive_cfg.get("enabled", True)):
            decision = decide_identity(
                info.structure_candidate_scores,
                in_archive_threshold=float(self._archive_cfg.get("structure_in_archive_threshold", 0.78)),
                uncertain_threshold=float(self._archive_cfg.get("structure_uncertain_threshold", 0.55)),
                min_margin=float(self._archive_cfg.get("structure_min_margin", 0.08)),
                evidence_count=info.structure_evidence_count,
                min_structure_observations=int(self._archive_cfg.get("min_structure_observations", 2)),
                strong_single_threshold=float(self._archive_cfg.get("strong_single_threshold", 0.88)),
                consistent_in_archive_threshold=float(self._archive_cfg.get("consistent_in_archive_threshold", self._archive_cfg.get("structure_in_archive_threshold", 0.78))),
                min_rejection_observations=int(self._archive_cfg.get("min_rejection_observations", 2)),
                consistent_observations=info.structure_consistent_observations,
                conflict_observations=info.structure_conflict_observations,
                best_single_score=info.structure_best_single_score,
                min_consistent_score=info.structure_min_consistent_score,
                min_consistent_margin=info.structure_min_consistent_margin,
                min_consistent_observations=int(self._archive_cfg.get("min_consistent_observations", 2)),
                min_consistency_ratio=float(self._archive_cfg.get("min_consistency_ratio", 0.67)),
            )
            text_candidate_id = decision.identity or (decision.candidates[0][0] if decision.candidates else "")
            from pipeline.visual_archive import (
                visual_agreement_supports_confirmation,
                visual_consistency_supports_confirmation,
                visual_confirmation_block_reason,
                visual_open_set_override,
            )
            visual_only_confirmed = visual_consistency_supports_confirmation(
                text_candidate_id,
                decision.state,
                agent_result.visual_candidate_id,
                agent_result.visual_similarity_score,
                agent_result.visual_margin,
                info.visual_consistent_observations,
                self._visual_cfg,
            )
            override_state = ""
            override_reason = ""
            if visual_only_confirmed:
                agent_result.visual_gate_reason = "repeated_visual_identity_support"
                decision = IdentityDecision(
                    agent_result.visual_candidate_id,
                    "structure_verified",
                    agent_result.visual_similarity_score,
                    decision.candidates,
                    "visual_consistency",
                )
            else:
                override_state, override_reason = visual_open_set_override(
                    text_candidate_id,
                    agent_result.visual_candidate_id,
                    agent_result.visual_similarity_score,
                    agent_result.visual_margin,
                    self._visual_cfg,
                    low_score_observations=info.visual_low_score_observations,
                    observation_count=int(getattr(info, "visual_observation_count", info.visual_low_score_observations)),
                )
            if visual_only_confirmed:
                pass
            elif override_state:
                agent_result.visual_gate_reason = override_reason
                override_identity = text_candidate_id
                if override_state == "uncertain" and not override_identity:
                    override_identity = agent_result.visual_candidate_id
                decision = IdentityDecision(
                    "" if override_state == "out_of_archive" else override_identity,
                    override_state,
                    decision.confidence,
                    decision.candidates,
                    f"visual_open_set:{override_reason}",
                )
            elif visual_agreement_supports_confirmation(
                text_candidate_id,
                decision.state,
                decision.confidence,
                agent_result.visual_candidate_id,
                agent_result.visual_similarity_score,
                agent_result.visual_margin,
                self._visual_cfg,
            ):
                agent_result.visual_gate_reason = "structure_visual_identity_agreement"
                decision = IdentityDecision(
                    text_candidate_id,
                    "structure_verified",
                    decision.confidence,
                    decision.candidates,
                    "structure_visual_agreement",
                )
            elif decision.state == "structure_verified":
                gate_reason = visual_confirmation_block_reason(
                    decision.identity,
                    agent_result.visual_candidate_id,
                    agent_result.visual_similarity_score,
                    agent_result.visual_margin,
                    self._visual_cfg,
                )
                if gate_reason:
                    agent_result.visual_gate_reason = gate_reason
                    decision = IdentityDecision(
                        decision.identity,
                        "uncertain",
                        decision.confidence,
                        decision.candidates,
                        f"structure_visual_gate:{gate_reason}",
                    )
            self._commit_identity_decision(track_id, decision)
        self._record_memory_updates(track_id, frame_id, agent_result, action_taken="recognize")
        info = self._tracker.get(track_id)
        nearby_tracks = max(0, len(self._tracker.active_tracks) - 1)
        if bool(self._risk_cfg.get("enabled", True)):
            risk_level, risk_reasons, risk_score = assess_track_risk(info, self._risk_cfg, nearby_tracks=nearby_tracks)
        else:
            risk_level, risk_reasons, risk_score = "low", ["risk_module_disabled"], 0.0
        self._tracker.update_risk(track_id, frame_id, risk_level, risk_reasons, "", score=risk_score)
        info = self._tracker.get(track_id)
        explanation = explain_encounter(info)
        self._tracker.update_risk(track_id, frame_id, risk_level, risk_reasons, explanation, score=risk_score)
        self._experiment_logger.log("recognition", {
            "track_id": track_id,
            "entity_id": info.entity_id if info else "",
            "member_track_ids": info.member_track_ids if info else [track_id],
            "frame_id": frame_id,
            "raw_hull_number": agent_result.hull_number,
            "fused_hull_number": info.hull_number if info else "",
            "verified_identity": info.verified_identity if info else "",
            "archive_candidate_id": info.archive_candidate_id if info else "",
            "archive_similarity_score": info.archive_similarity_score if info else 0.0,
            "identity_state": info.identity_state if info else "unknown",
            "match_type": agent_result.match_type,
            "observed_structure_description": agent_result.description,
            "identity_features": agent_result.identity_features,
            "vlm_model": agent_result.vlm_model,
            "vlm_base_url": agent_result.vlm_base_url,
            "vlm_fallback_used": agent_result.vlm_fallback_used,
            "semantic_matches": agent_result.semantic_matches,
            "visual_backend": agent_result.visual_backend or self._visual_backend,
            "visual_candidate_id": agent_result.visual_candidate_id,
            "visual_similarity_score": agent_result.visual_similarity_score,
            "visual_margin": agent_result.visual_margin,
            "visual_observation_count": info.visual_observation_count if info else 0,
            "visual_consistent_observations": info.visual_consistent_observations if info else 0,
            "visual_low_score_observations": info.visual_low_score_observations if info else 0,
            "visual_embedding_latency_ms": agent_result.visual_embedding_latency_ms,
            "visual_matching_latency_ms": agent_result.visual_matching_latency_ms,
            "visual_model_request_count": agent_result.visual_model_request_count,
            "visual_pairs_scored": agent_result.visual_pairs_scored,
            "visual_gate_reason": agent_result.visual_gate_reason,
            "identity_decision_mode": self._identity_decision_mode,
            "visual_matches": agent_result.visual_matches,
            "structure_candidate_scores": info.structure_candidate_scores if info else {},
            "structure_evidence_count": info.structure_evidence_count if info else 0,
            "structure_consistent_observations": info.structure_consistent_observations if info else 0,
            "structure_conflict_observations": info.structure_conflict_observations if info else 0,
            "structure_best_single_score": info.structure_best_single_score if info else 0.0,
            "structure_min_consistent_score": info.structure_min_consistent_score if info else 0.0,
            "structure_min_consistent_margin": info.structure_min_consistent_margin if info else 0.0,
            "identity_evidence_source": info.identity_evidence_source if info else "none",
            "uncertainty": info.last_uncertainty_score if info else 1.0,
            "risk_level": risk_level,
            "risk_score": risk_score,
        })

    def _handle_agent_error(self, track_id: int, frame_id: int, error: str) -> None:
        logger.warning("识别出错 (track=%d, frame=%d): %s", track_id, frame_id, error)
        self._tracker.bind_result(track_id, hull_number="", description="", frame_id=frame_id)
        self._log_track_summary(track_id)

    # ── 级联模式 ──────────────────────────────

    def _cascade_process(self, detections: list[Detection], frame_id: int) -> None:
        if self._controller_mode in {"central_vlm", "mock"}:
            self._central_process(detections, frame_id)
            return
        for det in detections:
            if det.crop is None or det.crop.size == 0:
                continue
            if not self._should_query_track(det.track_id, frame_id):
                continue
            self._tracker.mark_pending(det.track_id)
            try:
                agent_result = self._run_recognition(det.crop, track_id=det.track_id, frame_id=frame_id)
                self._handle_agent_result(det.track_id, frame_id, agent_result)
            except Exception as e:
                self._handle_agent_error(det.track_id, frame_id, str(e))

    # ── 并发模式 ──────────────────────────────

    def _concurrent_process(self, detections: list[Detection], frame_id: int) -> None:
        if self._controller_mode in {"central_vlm", "mock"}:
            self._central_process(detections, frame_id)
            return
        if self._task_queue.qsize() > self._max_queued_frames // 2:
            return
        for det in detections:
            if det.crop is None or det.crop.size == 0:
                continue
            if not self._should_query_track(det.track_id, frame_id):
                continue
            self._tracker.mark_pending(det.track_id)
            try:
                self._task_queue.put_nowait({"frame_id": frame_id, "timestamp": time.time(), "track_id": det.track_id, "crop": det.crop.copy()})
            except queue.Full:
                self._tracker.cancel_pending(det.track_id)

    def _worker_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    task = self._task_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                track_id, frame_id, crop = task["track_id"], task["frame_id"], task["crop"]
                try:
                    agent_result = self._run_recognition(crop, track_id=track_id, frame_id=frame_id)
                except Exception as e:
                    agent_result = AgentResult(answer=str(e))
                try:
                    self._result_queue.put_nowait({"frame_id": frame_id, "track_id": track_id, "agent_result": agent_result})
                except queue.Full:
                    self._tracker.bind_result(track_id, hull_number="", description="", frame_id=frame_id)
        except Exception:
            logger.exception("Worker 线程意外退出")

    def _drain_results(self) -> int:
        count = 0
        while True:
            try:
                pending = self._result_queue.get_nowait()
                track_id, frame_id, agent_result = pending["track_id"], pending["frame_id"], pending["agent_result"]
                if agent_result.hull_number or agent_result.semantic_match_ids or agent_result.match_type in ("exact", "semantic"):
                    self._handle_agent_result(track_id, frame_id, agent_result)
                else:
                    self._handle_agent_error(track_id, frame_id, agent_result.answer or "无结果")
                count += 1
            except queue.Empty:
                break
        return count

    def _start_workers(self) -> None:
        self._stop_event.clear()
        self._workers.clear()
        for i in range(self._max_concurrent):
            w = threading.Thread(target=self._worker_loop, name=f"worker-{i}", daemon=True)
            w.start()
            self._workers.append(w)
        logger.info("启动 %d 个 Worker 线程", self._max_concurrent)

    def _stop_workers(self) -> None:
        self._stop_event.set()
        for w in self._workers:
            w.join(timeout=10.0)
        self._workers.clear()
        while True:
            try:
                self._task_queue.get_nowait()
            except queue.Empty:
                break
        remaining = self._drain_results()
        if remaining:
            logger.info("处理 %d 个残留结果", remaining)

    # ── MJPEG 帧写入器 ────────────────────────

    class _FrameWriter:
        """后台线程：异步编码 JPEG 并写入磁盘，不阻塞主检测循环。"""

        def __init__(self, stream_dir: Path, quality: int = 50):
            self._path = stream_dir / "latest.jpg"
            self._quality = quality
            self._queue: list = []  # 最多保留 1 帧
            self._lock = threading.Lock()
            self._stop = threading.Event()
            self._frame_count = 0
            self._drop_count = 0
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        def write(self, frame: np.ndarray) -> None:
            with self._lock:
                if self._queue:
                    self._drop_count += 1
                self._queue = [frame]

        def stop(self) -> None:
            self._stop.set()
            self._thread.join(timeout=2)
            if self._drop_count:
                logger.info("MJPEG 帧写入: 共 %d 帧, 丢弃 %d 帧", self._frame_count, self._drop_count)

        def _run(self) -> None:
            while not self._stop.is_set():
                frame = None
                with self._lock:
                    if self._queue:
                        frame = self._queue.pop(0)
                if frame is not None:
                    try:
                        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
                        tmp = self._path.with_suffix(".tmp")
                        with open(tmp, "wb") as f:
                            f.write(buf.tobytes())
                        tmp.rename(self._path)
                        self._frame_count += 1
                    except Exception:
                        pass
                else:
                    self._stop.wait(0.01)

    # ── Raw stdout 帧写入器 ────────────────────

    class _RawStdoutWriter:
        """后台线程：将原始 BGR 帧写入 stdout（供 ffmpeg H.264 编码）。自带背压。"""

        def __init__(self, max_pending: int = 2, pipe_output_size: tuple[int, int] | None = None):
            self._queue: list = []
            self._lock = threading.Lock()
            self._stop = threading.Event()
            self._frame_count = 0
            self._drop_count = 0
            self._max_pending = max_pending  # 最大待写帧数，超过时阻塞主循环
            self._pipe_output_size = pipe_output_size  # pipe 输出缩放尺寸，None 表示不缩放
            self._can_write = threading.Event()
            self._can_write.set()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

        def write(self, frame: np.ndarray) -> None:
            """提交帧到写入队列。队列满时阻塞（背压），不丢帧。"""
            # pipe 输出缩放（INTER_AREA 下采样，无编码开销）
            if self._pipe_output_size:
                ow, oh = self._pipe_output_size
                fh, fw = frame.shape[:2]
                if fw != ow or fh != oh:
                    frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
            # 等待队列有空间
            while not self._stop.is_set():
                with self._lock:
                    if len(self._queue) < self._max_pending:
                        self._queue.append(frame)
                        return
                # 队列满，等待写入线程消费
                self._can_write.wait(timeout=0.5)

        def stop(self) -> None:
            self._stop.set()
            self._can_write.set()  # 唤醒等待
            self._thread.join(timeout=2)
            if self._drop_count:
                logger.info("Raw stdout 写入: 共 %d 帧, 丢弃 %d 帧", self._frame_count, self._drop_count)

        def _run(self) -> None:
            import sys
            stdout = sys.stdout.buffer  # 二进制写入
            while not self._stop.is_set():
                frame = None
                with self._lock:
                    if self._queue:
                        frame = self._queue.pop(0)
                if frame is not None:
                    try:
                        # 直接写 raw BGR 数据（ffmpeg -f rawvideo -pix_fmt bgr24）
                        stdout.write(frame.tobytes())
                        stdout.flush()
                        self._frame_count += 1
                    except (BrokenPipeError, OSError):
                        break
                    # 通知主循环队列有空间了
                    self._can_write.set()
                else:
                    self._can_write.clear()
                    self._stop.wait(0.005)

    # ── H265/H264 转码 ────────────────────────

    _FFMPEG: str | None = None
    _FFPROBE: str | None = None

    @staticmethod
    def _find_binary(name: str) -> str | None:
        """查找二进制文件。"""
        import shutil
        found = shutil.which(name)
        if found:
            return found
        for path in [f"/usr/bin/{name}", f"/usr/local/bin/{name}"]:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
        return None

    @classmethod
    def _ensure_ffmpeg(cls):
        """延迟查找 ffmpeg/ffprobe。"""
        if cls._FFMPEG is None:
            cls._FFMPEG = cls._find_binary("ffmpeg") or ""
        if cls._FFPROBE is None:
            cls._FFPROBE = cls._find_binary("ffprobe") or ""

    @classmethod
    def _probe_video_codec(cls, video_path: str) -> str | None:
        """用 ffprobe 检测视频编码格式。"""
        cls._ensure_ffmpeg()
        if not cls._FFPROBE:
            return None
        try:
            ret = subprocess.run(
                [cls._FFPROBE, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name",
                 "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                capture_output=True, text=True, timeout=30,
            )
            if ret.returncode == 0:
                codec = ret.stdout.strip().lower()
                if codec:
                    return codec
        except Exception as e:
            logger.warning("ffprobe 检测失败: %s", e)
        return None

    @classmethod
    def _is_browser_compatible_codec(cls, codec: str | None) -> bool:
        """判断编码是否被主流浏览器原生支持。"""
        if codec is None:
            return False
        compatible = {"h264", "vp8", "vp9", "av1", "mpeg4part10"}
        return codec in compatible

    @classmethod
    def _transcode_to_h264(cls, source_path: str, target_path: str) -> bool:
        """将视频转码为 H264（浏览器兼容）。"""
        cls._ensure_ffmpeg()
        if not cls._FFMPEG:
            logger.error("ffmpeg 不可用，无法转码 H264")
            return False
        try:
            ret = subprocess.run(
                [cls._FFMPEG, "-y", "-i", source_path,
                 "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                 "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "128k",
                 "-movflags", "+faststart",
                 target_path],
                capture_output=True, timeout=600,
            )
            if ret.returncode == 0 and Path(target_path).exists() and Path(target_path).stat().st_size > 0:
                logger.info("H264 转码成功: %s", target_path)
                return True
            logger.warning("H264 转码失败: %s", ret.stderr.decode()[-300:] if ret.stderr else "")
        except Exception as e:
            logger.warning("H264 转码异常: %s", e)
        return False

    @classmethod
    def _transcode_to_h265(cls, source_path: str, target_path: str) -> bool:
        """将视频转码为 H265。"""
        cls._ensure_ffmpeg()
        if not cls._FFMPEG:
            logger.error("ffmpeg 不可用，无法转码 H265")
            return False
        try:
            ret = subprocess.run(
                [cls._FFMPEG, "-y", "-i", source_path,
                 "-c:v", "libx265", "-preset", "fast", "-crf", "28",
                 "-tag:v", "hvc1",
                 "-pix_fmt", "yuv420p",
                 "-c:a", "aac", "-b:a", "128k",
                 "-movflags", "+faststart",
                 target_path],
                capture_output=True, timeout=600,
            )
            if ret.returncode == 0 and Path(target_path).exists() and Path(target_path).stat().st_size > 0:
                logger.info("H265 转码成功: %s", target_path)
                return True
            logger.warning("H265 转码失败: %s", ret.stderr.decode()[-300:] if ret.stderr else "")
        except Exception as e:
            logger.warning("H265 转码异常: %s", e)
        return False

    @classmethod
    def _transcode_video(cls, source_path: str, target_path: str) -> bool:
        """
        转码视频，优先 H264（浏览器兼容），其次 H265。
        """
        if cls._transcode_to_h264(source_path, target_path):
            return True
        logger.warning("H264 转码失败，尝试 H265")
        return cls._transcode_to_h265(source_path, target_path)

    # ── 主流程 ────────────────────────────────

    def process(
        self,
        source: str | int | object,
        output_path: str | None = None,
        display: bool = False,
        max_frames: int = 0,
        frame_callback: Callable[[np.ndarray, int], None] | None = None,
        stream_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """
        运行完整的视频处理流水线。

        Args:
            source: 视频输入源（文件路径/相机号/RTSP URL/VirtualCamera 对象）。
            output_path: 输出视频路径（可选）。
            display: 是否实时显示窗口。
            max_frames: 最大处理帧数，0 表示不限制。
            frame_callback: 每帧处理完成后的回调函数。
            stream_dir: MJPEG 帧输出目录（将标注帧写入 latest.jpg 供流读取）。
        """
        # 如果配置了保存视频且未指定输出路径，自动生成输出路径
        if self._save_output_video and not output_path:
            demo_cfg = self._config.get("demo_video", {})
            output_dir = demo_cfg.get("output_dir", "./demo_output")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            if isinstance(source, str) and Path(source).is_file():
                stem = Path(source).stem
                output_path = str(Path(output_dir) / f"{stem}_{timestamp}.mp4")
            else:
                output_path = str(Path(output_dir) / f"output_{timestamp}.mp4")
            logger.info("自动保存推理视频: %s", output_path)

        input_src = InputSource(source)
        if self._central_controller is not None:
            source_name = Path(source).stem if isinstance(source, (str, Path)) else "stream"
            self._central_controller.reset_session(f"{source_name}-{int(time.time() * 1000)}")
        video_writer = None
        last_detections: list[Detection] = []
        frame_id = 0
        processed_frame_id = 0  # 仅统计实际处理的帧（跳帧后独立计数）
        total_detections = 0
        start_time = time.time()
        process_cpu_start = time.process_time()

        # 自动设置 target_fps：未指定时使用源视频 FPS（防止异步模式下跑太快）
        if self._target_fps <= 0:
            src_fps = input_src.source_fps
            if src_fps > 0:
                self._target_fps = src_fps
                logger.info("自动设置 target_fps = 源视频 FPS (%.1f)", src_fps)

        # 帧跳过计算：target_fps < source_fps 时跳帧减少计算量
        src_fps = input_src.source_fps
        if self._target_fps > 0 and src_fps > self._target_fps:
            self._frame_skip_interval = max(1, round(src_fps / self._target_fps))
            logger.info("帧跳过: 源 %.1f fps → 目标 %.1f fps, 每 %d 帧取 1 帧",
                        src_fps, self._target_fps, self._frame_skip_interval)

        # 帧输出：MJPEG 磁盘写入 + 可选 raw stdout
        frame_writer: ShipPipeline._FrameWriter | None = None
        raw_writer: ShipPipeline._RawStdoutWriter | None = None
        raw_stdout = self._config.get("pipeline", {}).get("raw_stdout", False)

        if raw_stdout:
            _pos = self._config.get("pipeline", {}).get("pipe_output_size")
            pipe_output_size = tuple(_pos) if _pos else None
            raw_writer = ShipPipeline._RawStdoutWriter(pipe_output_size=pipe_output_size)
            logger.info("Raw stdout 帧输出已启用（供 H.264 编码% s）", f", 缩放至 {pipe_output_size[0]}x{pipe_output_size[1]}" if pipe_output_size else "")
        elif stream_dir:
            stream_path = Path(stream_dir)
            stream_path.mkdir(parents=True, exist_ok=True)
            frame_writer = ShipPipeline._FrameWriter(stream_path, quality=70)
            logger.info("MJPEG 帧输出: %s", stream_path / "latest.jpg")

        no_output = self._config.get("pipeline", {}).get("no_output", False)

        # 如果配置了保存视频，优先使用 save_output_video 设置
        if self._save_output_video:
            no_output = False

        try:
            if output_path and not no_output:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(output_path, fourcc, input_src.source_fps, (input_src.width, input_src.height))
                if not video_writer.isOpened():
                    logger.error("无法创建输出视频: %s", output_path)
                    video_writer = None

            if self._concurrent_mode:
                self._start_workers()

            logger.info("开始处理: source=%s, mode=%s, workers=%d, refresh=%s(gap=%d, skip_matched=%s), detect_every=%d, process_every=%d",
                        source, "concurrent" if self._concurrent_mode else "cascade",
                        self._max_concurrent,
                        "on" if self._enable_refresh else "off", self._gap_num,
                        "on" if self._skip_refresh_matched else "off",
                        self._detect_every_n, self._process_every_n)

            # 停止信号文件路径（外部可以通过创建此文件来请求停止）
            stop_file = None
            if stream_dir:
                stop_file = Path(stream_dir) / "__STOP__"
            elif self._stop_file:
                stop_file = self._stop_file

            while True:
                # 检查停止信号文件
                if stop_file and stop_file.exists():
                    logger.info("检测到停止信号文件，优雅退出")
                    break

                ret, frame = input_src.read()
                if not ret:
                    break
                frame_id += 1
                frame_start_time = time.time()
                if max_frames > 0 and frame_id > max_frames:
                    break

                # 帧跳过：减少计算量，跳过的帧不执行检测/渲染/输出
                if self._frame_skip_interval > 1 and (frame_id % self._frame_skip_interval != 1):
                    continue
                processed_frame_id += 1

                self._fps.tick("stream")

                # YOLO 检测（主循环同步）— 用 processed_frame_id 避免跳帧后取模不匹配
                should_detect = (processed_frame_id % self._detect_every_n == 0)
                if should_detect:
                    try:
                        with self._latency.measure("yolo"):
                            detections = self._grounder.detect(frame, frame_id)
                    except Exception as e:
                        logger.error("YOLO 检测异常 (frame=%d): %s", frame_id, e)
                        detections = []
                    for detection in detections:
                        detection.source_frame_id = frame_id
                    last_detections = detections
                else:
                    detections = last_detections

                total_detections += len(detections)

                visible_track_ids = {int(det.track_id) for det in detections}
                for det in detections:
                    if bool(self._quality_cfg.get("enabled", True)):
                        quality = compute_observation_quality(det.crop, det.bbox, frame.shape, det.confidence, self._quality_cfg)
                    else:
                        from pipeline.quality import ObservationQuality
                        quality = ObservationQuality(1.0, {"quality_disabled": 1.0})
                    with self._latency.measure("entity_reconciliation"):
                        binding = self._entity_reconciler.observe(
                            det.track_id,
                            frame_id,
                            det.bbox,
                            det.crop,
                            frame.shape,
                            visible_track_ids=visible_track_ids,
                        )
                    if binding.reassociated and binding.source_track_id is not None:
                        self._tracker.inherit_entity_state(det.track_id, binding.source_track_id, binding.entity_id, binding.member_track_ids, frame_id)
                        self._experiment_logger.log("entity_links", {
                            "entity_id": binding.entity_id,
                            "source_track_id": binding.source_track_id,
                            "track_id": det.track_id,
                            "frame_id": frame_id,
                            "association_score": binding.association_score,
                            "member_track_ids": binding.member_track_ids,
                        })
                    info = self._tracker.record_observation(
                        det.track_id, frame_id, det.bbox, det.confidence, quality.score, quality.components,
                        frame_shape=frame.shape, entity_id=binding.entity_id, member_track_ids=binding.member_track_ids,
                    )
                    self._experiment_logger.log("observations", {
                        "track_id": det.track_id, "entity_id": binding.entity_id, "frame_id": frame_id,
                        "bbox": det.bbox, "confidence": det.confidence, "observation_quality": quality.score,
                        "quality_components": quality.components, "member_track_count": len(info.member_track_ids),
                    })

                # 推理：按间隔提交到 VLM worker 池（不阻塞主循环）
                should_process = (processed_frame_id % self._process_every_n == 0)
                if should_process:
                    if self._concurrent_mode:
                        self._concurrent_process(detections, frame_id)
                    else:
                        self._cascade_process(detections, frame_id)

                # 每帧都 drain VLM 结果（非阻塞，有就取）
                if self._concurrent_mode:
                    self._drain_results()

                if frame_id % 30 == 0:
                    self._tracker.cleanup_stale(frame_id)

                # 渲染：每帧都执行（用 tracker 最新状态，检测框 + 识别标签实时更新）
                if self._demo_enabled or output_path or display or frame_writer:
                    with self._latency.measure("demo"):
                        display_frame = self._renderer.render(
                            frame, last_detections, self._tracker.active_tracks,
                            self._fps.get_all_fps(), frame_id,
                            self._task_queue.qsize(), self._max_queued_frames,
                        )
                else:
                    display_frame = frame

                if self._save_screenshots and should_process:
                    active = self._tracker.active_tracks
                    if any(t.recognized for t in active.values()):
                        self._saver.save(display_frame, frame_id)

                if video_writer:
                    video_writer.write(display_frame)

                # 帧输出：raw stdout（H.264 编码用）或 MJPEG 磁盘写入
                if raw_writer:
                    out_frame = display_frame
                    if self._output_size:
                        ow, oh = self._output_size
                        fh, fw = display_frame.shape[:2]
                        if fw != ow or fh != oh:
                            out_frame = cv2.resize(display_frame, (ow, oh), interpolation=cv2.INTER_LINEAR)
                    raw_writer.write(out_frame)
                    self._frames_submitted += 1
                elif frame_writer:
                    frame_writer.write(display_frame)

                if frame_callback:
                    frame_callback(display_frame, frame_id)

                if display:
                    cv2.imshow("Ship Pipeline", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                # 帧率控制 + 背压：防止主循环跑太快淹没 ffmpeg
                if self._target_fps > 0:
                    frame_interval = 1.0 / self._target_fps
                    elapsed_since_frame = time.time() - frame_start_time
                    sleep_time = frame_interval - elapsed_since_frame

                    # 背压：管道积压太多时额外减速
                    if self._frames_encoded_ref is not None:
                        frames_encoded = self._frames_encoded_ref[0]
                        pipe_lag = self._frames_submitted - frames_encoded
                        if pipe_lag > self._max_pipe_lag:
                            extra_wait = min(2.0, (pipe_lag - self._max_pipe_lag) * 0.05)
                            sleep_time += extra_wait
                            if frame_id % 30 == 0:
                                logger.warning("管道积压 %d 帧 (max=%d)，额外等待 %.1fs", pipe_lag, self._max_pipe_lag, extra_wait)

                    if sleep_time > 0:
                        time.sleep(sleep_time)

                self._fps.tick("process")

                if self._fps.should_print("stream"):
                    elapsed = time.time() - start_time
                    stream_fps = self._fps.get_fps("stream")
                    process_fps = self._fps.get_fps("process")
                    latency_parts = []
                    for stage in ("yolo", "vlm", "demo"):
                        s = self._latency.get_stats(stage)
                        if s and s["count"] > 0:
                            latency_parts.append(f"{stage}: avg={s['avg']:.1f}ms p95={s['p95']:.1f}ms")
                    latency_str = f" | Latency: {' | '.join(latency_parts)}" if latency_parts else ""
                    logger.info("FPS: stream=%.1f process=%.1f | frames=%d elapsed=%ds tracks=%d%s", stream_fps, process_fps, frame_id, int(elapsed), len(self._tracker), latency_str)

            # ── 处理完成 ──

            if self._concurrent_mode:
                self._drain_results()

            if self._central_controller is not None:
                self._central_controller.wait_for_idle()
                self._central_controller.finish_eof()

            if (
                self._controller_mode == "legacy"
                and self._identity_decision_mode == "visual_only"
                and str(self._visual_cfg.get("visual_only_decision_rule", "legacy")).strip().lower() == "score_gate"
            ):
                finalized_unknowns = self._tracker.finalize_visual_score_gate(
                    float(self._visual_cfg.get("visual_only_reject_score", 0.80))
                )
                if finalized_unknowns:
                    logger.info("Visual score gate finalized %d unresolved track(s) as out-of-archive", finalized_unknowns)

            elapsed = time.time() - start_time
            process_cpu_seconds = time.process_time() - process_cpu_start
            peak_rss_mb = 0.0
            try:
                import resource
                peak_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                peak_rss_mb = peak_rss / 1024.0
            except (ImportError, AttributeError):
                peak_rss_mb = 0.0
            tracks = self._tracker.all_tracks
            total_recognized = sum(1 for t in tracks.values() if t.recognized)

            event_log = self._tracker.export_event_log()
            memory_summary = self._tracker.export_memory()
            entity_summary = self._entity_reconciler.export()
            for entity in entity_summary:
                entity.pop("appearance", None)
                appearance_bank = entity.pop("appearance_bank", [])
                entity["appearance_prototype_count"] = len(appearance_bank)
                member_tracks = [self._tracker.get_any(int(track_id)) for track_id in entity.get("member_track_ids", [])]
                identity = aggregate_entity_identity(member_tracks)
                if self._central_controller:
                    episode = self._central_controller.store.get(entity["entity_id"])
                    if episode and episode.terminal_record.get("commit_status") == "committed":
                        terminal = episode.terminal_record
                        state = {"known": "confirmed", "review": "review_requested"}.get(terminal["decision"], terminal["decision"])
                        identity.update({"identity_state": state, "verified_identity": terminal.get("archive_id", ""),
                                         "identity_evidence_source": terminal["decision_source"], "terminal_record": terminal,
                                         "control_calls": episode.control_calls, "tool_counts": dict(episode.tool_counts)})
                available_tracks = [track for track in member_tracks if track is not None]
                risk_track = max(available_tracks, key=lambda track: float(getattr(track, "risk_score", 0.0)), default=None)
                self._experiment_logger.log("entities", {
                    **entity,
                    **identity,
                    "risk_level": risk_track.risk_level if risk_track else "low",
                })
            exact_matches = sum(1 for t in tracks.values() if t.db_matched)
            semantic_only = sum(1 for t in tracks.values() if (not t.db_matched) and t.semantic_match_ids)
            high_risk_tracks = sum(1 for t in tracks.values() if getattr(t, "risk_level", "low") == "high")
            medium_risk_tracks = sum(1 for t in tracks.values() if getattr(t, "risk_level", "low") == "medium")
            avg_uncertainty = round(sum(getattr(t, "last_uncertainty_score", 0.0) for t in tracks.values()) / max(1, len(tracks)), 4)
            stats = {
                "total_frames": frame_id,
                "total_detections": total_detections,
                "total_tracks": len(tracks),
                "recognized_tracks": total_recognized,
                "total_entities": len(entity_summary),
                "exact_matches": exact_matches,
                "semantic_only_tracks": semantic_only,
                "high_risk_tracks": high_risk_tracks,
                "medium_risk_tracks": medium_risk_tracks,
                "avg_uncertainty": avg_uncertainty,
                "elapsed_seconds": round(elapsed, 1),
                "avg_fps": round(frame_id / elapsed, 1) if elapsed > 0 else 0,
                "source_fps": round(float(src_fps), 3),
                "realtime_factor": round((frame_id / elapsed) / src_fps, 4) if elapsed > 0 and src_fps > 0 else 0.0,
                "process_cpu_seconds": round(process_cpu_seconds, 3),
                "process_cpu_percent": round(100.0 * process_cpu_seconds / elapsed, 2) if elapsed > 0 else 0.0,
                "peak_rss_mb": round(peak_rss_mb, 3),
                "mode": "concurrent" if self._concurrent_mode else "cascade",
                "controller_mode": self._controller_mode,
                "screenshots_saved": self._saver.saved_count,
                "latency": self._latency.get_all_time_stats() if hasattr(self._latency, "get_all_time_stats") else self._latency.get_all_stats(),
                "rolling_latency": self._latency.get_all_stats(),
                "central_episodes": self._central_controller.snapshots() if self._central_controller else [],
                "track_memory": memory_summary,
                "event_log_tail": event_log[-100:],
                "agent_trace": self.agent_trace[-100:],
                "paper_summary": {
                    "grounding_provider": (self._config.get("pipeline", {}).get("grounding_provider") or "yolo"),
                    "evidence_tracks": len(memory_summary),
                    "evidence_entities": len(entity_summary),
                    "entity_reconciliation_enabled": self._entity_reconciler.enabled,
                    "active_perception_enabled": self._active_perception_enabled,
                    "profile_memory_enabled": True,
                    "experience_memory_enabled": bool(self._experience_cfg.get("enabled", True)),
                    "profile_memory_write": self._profile_memory_write,
                    "experience_memory_write": self._experience_memory_write,
                    "risk_explanation_enabled": bool(self._risk_cfg.get("enabled", True)),
                    "controller_mode": self._controller_mode,
                    "central_episode_count": len(self._central_controller.snapshots()) if self._central_controller else 0,
                },
            }

            self._experiment_logger.write_summary(stats)
            logger.info("=" * 50)
            logger.info("处理完成: 帧=%d 检测=%d 跟踪=%d 识别=%d 耗时=%.1fs FPS=%.1f",
                        stats["total_frames"], stats["total_detections"], stats["total_tracks"], stats["recognized_tracks"], stats["elapsed_seconds"], stats["avg_fps"])
            logger.info("=" * 50)

            # H265/H264 转码（输出视频存在时）
            if output_path and Path(output_path).exists() and Path(output_path).stat().st_size > 0:
                h265_path = Path(output_path).with_suffix(".h265.mp4")
                if self._transcode_video(output_path, str(h265_path)):
                    Path(output_path).unlink()
                    h265_path.rename(output_path)
                    logger.info("已替换为 H265/H264 编码: %s", output_path)
                else:
                    logger.warning("转码失败，保留原始 mp4v 文件: %s", output_path)

            return stats

        except KeyboardInterrupt:
            logger.info("用户中断")
            return {"total_frames": frame_id, "interrupted": True}

        finally:
            if self._central_controller is not None:
                self._central_controller.shutdown(wait_for_tasks=True)
            if self._concurrent_mode:
                self._stop_workers()
            if raw_writer:
                raw_writer.stop()
            if frame_writer:
                frame_writer.stop()
            input_src.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()
            self._grounder.cleanup()

    @property
    def agent_trace(self) -> list[dict[str, Any]]:
        with self._trace_lock:
            return list(self._agent_trace)

    def set_demo(self, enabled: bool) -> None:
        self._demo_enabled = enabled

    def set_prompt_mode(self, mode: str) -> None:
        if mode not in ("detailed", "brief"):
            raise ValueError(f"不支持的提示词模式: {mode}")
        self._prompt_mode = mode
        logger.info("提示词模式切换为: %s", mode)

    def switch_to_concurrent(self, enabled: bool) -> None:
        self._concurrent_mode = enabled
        logger.info("切换为 %s 模式", "并发" if enabled else "级联")
