"""配置读取 — 唯一配置源 config.yaml，返回字典"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

_DEFAULTS: dict[str, Any] = {
    "llm": {
        "model": "Qwen/Qwen3-VL-4B-AWQ",
        "use_lora": False,
        "lora_model": "vessel-evidence-lora",
        "lora_base_url": "http://localhost:7892/v1",
        "fallback_model": "Qwen/Qwen3-VL-4B-AWQ",
        "fallback_base_url": "http://localhost:7890/v1",
        "fallback_to_base": True,
        "output_schema": "simple",
        "fallback_output_schema": "structured",
        "fallback_output_schema_enabled": True,
        "enable_thinking": False,
        "structured_max_tokens": 768,
        "simple_max_tokens": 256,
        "api_key": "YOUR_API_KEY",
        "base_url": "http://localhost:7890/v1",
        "temperature": 0.0,
    },
    "embed": {
        "model": "Qwen3-Embedding-0.6B",
        "api_key": "YOUR_API_KEY",
        "base_url": "http://localhost:7891/v1",
    },
    "visual_reranker": {
        "model": "Qwen/Qwen3-VL-Reranker-2B",
        "api_key": "YOUR_API_KEY",
        "base_url": "http://localhost:7894",
        "request_timeout_seconds": 120,
        "max_retries": 1,
        "request_batch_size": 8,
        "jpeg_quality": 90,
        "instruction": "Determine whether the query image and document image depict the same physical vessel. Use stable vessel structure and markings while ignoring viewpoint, scale, illumination, water, and background differences.",
    },
    "retrieval": {"top_k": 3, "score_threshold": 0.5},
    "vector_store": {"persist_path": "./vector_store", "auto_rebuild": False},
    "database": {"backend": "csv", "csv_path": "./data/ships.csv", "sqlite_path": "./data/ships.db", "memory_sqlite_path": "./data/ship_memory.db"},
    "pipeline": {
        "concurrent_mode": False,
        "max_concurrent": 4,
        "max_queued_frames": 30,
        "process_every_n_frames": 30,
        "output_dir": "./output",
        "save_screenshots": True,
        "prompt_mode": "detailed",
        "enable_refresh": False,
        "skip_refresh_matched": False,
        "gap_num": 150,
        "demo": False,
        "yolo_model": "yolov8n.pt",
        "device": "",
        "conf_threshold": 0.25,
        "detect_every_n_frames": 1,
        "tracker": "bytetrack",
        "tracker_params": {
            "track_high_thresh": 0.5,
            "track_low_thresh": 0.05,
            "new_track_thresh": 0.6,
            "track_buffer": 90,
            "match_thresh": 0.5,
        },
        "detect_classes": [8],
        "max_stale_frames": 300,
    },
    "experiment": {
        "enabled": False, "run_name": "default", "output_dir": "./experiment_outputs", "controller_mode": "central_vlm",
        "save_jsonl": True, "save_crops": True, "crop_jpeg_quality": 90, "seed": 42, "split": "test",
        "memory_mode": "full", "policy_mode": "full",
        "quality": {"enabled": True, "min_query_quality": 0.25, "brightness_weight": 0.12, "blur_weight": 0.24, "contrast_weight": 0.12, "clipping_weight": 0.07, "scale_weight": 0.30, "confidence_weight": 0.15, "blur_reference": 300.0, "contrast_reference": 64.0, "target_area_reference": 0.08},
        "temporal_ledger": {"enabled": True, "decay": 0.97, "min_support": 0.30},
        "entity_reconciliation": {"enabled": True, "max_gap_frames": 180, "min_gap_frames": 1, "min_appearance_similarity": 0.78, "max_center_distance": 0.35, "max_log_area_ratio": 1.20, "min_association_score": 0.72, "appearance_weight": 0.70, "position_weight": 0.20, "scale_weight": 0.10, "appearance_ema": 0.80, "max_appearance_prototypes": 6, "appearance_novelty_threshold": 0.90},
        "archive": {"enabled": True, "structure_in_archive_threshold": 0.735, "structure_uncertain_threshold": 0.70, "structure_min_margin": 0.00, "min_structure_observations": 1, "strong_single_threshold": 0.88, "consistent_in_archive_threshold": 0.73, "min_consistent_observations": 2, "min_rejection_observations": 2, "min_consistency_ratio": 0.67, "preserve_verified_identity": True, "conflict_observations_to_downgrade": 2, "hull_structure_conflict_threshold": 0.85, "hull_structure_conflict_margin": 0.08, "hull_structure_min_consistency": 0.65},
        "visual_archive": {"enabled": False, "identity_decision_mode": "fused", "backend": "qwen_vl_reranker", "decision_enabled": False, "open_set_enabled": False, "fail_open": True, "archive_root": "data/archive/visual_prototypes", "model_repo": "models/dinov2_torch_home/hub/dinov2-main", "weights": "models/dinov2_torch_home/hub/checkpoints/dinov2_vits14_pretrain.pth", "device": "auto", "input_size": 224, "batch_size": 16, "top_k": 3, "min_support_score": 0.45, "min_support_margin": 0.10, "agreement_confirmation_enabled": True, "agreement_min_text_score": 0.70, "visual_only_confirmation_enabled": True, "visual_only_min_observations": 2, "visual_only_min_score": 0.70, "visual_only_min_margin": 0.03, "visual_only_strong_score": 0.90, "visual_only_reject_score": 0.70, "visual_only_reject_margin": 0.08, "conflict_min_score": 0.50, "conflict_min_margin": 0.05, "out_of_archive_score": 0.50, "out_of_archive_margin": 0.20, "min_out_of_archive_observations": 2},
        "experience": {"enabled": True, "top_k": 5, "min_similarity": 0.60},
        "memory_write": {"profile": True, "experience": True},
        "policy": {"uncertainty_threshold": 0.65, "min_gap_frames": 45, "fixed_interval_frames": 150, "gray_zone_reobserve": True, "max_queries_per_track": 3, "max_queries_per_entity": 3, "require_quality_improvement": True, "min_quality_improvement": 0.03, "min_confirmation_observations": 2, "confirmation_retry_gap_frames": 120, "risk_bonus": 0.20, "archive_bonus": 0.10, "experience_bonus": 0.10, "quality_gain_weight": 0.25, "new_tracklet_weight": 0.15, "conflict_weight": 0.15, "budget_penalty": 0.20, "learned_model_path": "models/policy/learned_policy_v1.json"},
        "risk": {"enabled": True, "scale_growth_threshold": 0.012, "image_motion_threshold": 0.10, "persistence_frames": 15, "identity_uncertainty_threshold": 0.70, "multi_target_threshold": 3, "medium_threshold": 0.30, "high_threshold": 0.65},
        "network": {"profile": "lan", "simulate": False, "latency_ms": 0, "jitter_ms": 0, "bandwidth_mbps": 100, "failure_probability": 0.0},
        "central_controller": {
            "terminal_policy": "multimodal_evidence", "query_limit": 3,
            "min_ooa_observations": 2, "min_ooa_score": 0.50,
            "max_control_steps": 12, "max_episode_steps": 48, "settlement_steps": 3,
            "max_memory_views": 64, "max_context_chars": 32000,
            "max_context_views": 4, "max_context_evidence": 20, "max_tokens": 1536,
            "charge_failed_queries": True, "max_workers": 4,
            "diagnostics_enabled": False, "diagnostics_dir": "output/agent_diagnostics",
            "diagnostics_max_bytes": 10000000, "diagnostics_image_limit": 256,
        },
    },
    "app": {"log_level": "INFO", "ship_db_path": "./data/ships.csv"},
    "web": {"host": "0.0.0.0", "port": 8000},
    "demo_video": {
        "dir": "./demovid",
        "output_dir": "./demo_output",
        "allowed_extensions": [".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm"],
        "max_file_size_mb": 500,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件 {path} 格式错误，期望顶层为字典")
    return data


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    user_dict: dict[str, Any] = {}
    if config_path:
        p = Path(config_path)
        if p.exists():
            user_dict = _load_yaml(p)
        else:
            logger.debug("配置文件不存在: %s，使用默认值", p)
    else:
        candidates = [Path.cwd() / "config.yaml", _DEFAULT_CONFIG_PATH]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                try:
                    user_dict = _load_yaml(candidate)
                except yaml.YAMLError as e:
                    raise SystemExit(f"配置文件解析失败: {e}")
                break
    return _deep_merge(_DEFAULTS, user_dict)
