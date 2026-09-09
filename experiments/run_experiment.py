"""Run one configured experiment over a split manifest."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from config import load_config
from experiments.build_episode_records import write_episode_records
from pipeline.pipeline import ShipPipeline

NETWORK_PROFILES = {
    "real": {"simulate": False, "latency_ms": 0, "jitter_ms": 0, "bandwidth_mbps": 1000, "failure_probability": 0.0},
    "lan": {"simulate": True, "latency_ms": 10, "jitter_ms": 3, "bandwidth_mbps": 100, "failure_probability": 0.0},
    "stable_mobile": {"simulate": True, "latency_ms": 60, "jitter_ms": 20, "bandwidth_mbps": 20, "failure_probability": 0.0},
    "limited": {"simulate": True, "latency_ms": 180, "jitter_ms": 60, "bandwidth_mbps": 3, "failure_probability": 0.01},
    "interruption": {"simulate": True, "latency_ms": 120, "jitter_ms": 80, "bandwidth_mbps": 8, "failure_probability": 0.15},
}


def load_manifest(path: Path, split: str) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if item.get("split", split) == split:
                    records.append(item)
    return records


def aggregate_deployment_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    total_frames = sum(int(item.get("total_frames", 0)) for item in summaries)
    total_elapsed = sum(float(item.get("elapsed_seconds", 0.0)) for item in summaries)
    total_cpu_seconds = sum(float(item.get("process_cpu_seconds", 0.0)) for item in summaries)
    weighted_source_fps = sum(float(item.get("source_fps", 0.0)) * int(item.get("total_frames", 0)) for item in summaries) / max(1, total_frames)
    latency_channels = sorted({channel for item in summaries for channel in (item.get("latency", {}) or {})})
    latency: dict[str, dict[str, float]] = {}
    for channel in latency_channels:
        rows = [(item.get("latency", {}) or {}).get(channel, {}) for item in summaries]
        rows = [row for row in rows if int(row.get("count", 0)) > 0]
        count = sum(int(row.get("count", 0)) for row in rows)
        latency[channel] = {
            "weighted_avg_ms": round(sum(float(row.get("avg", 0.0)) * int(row.get("count", 0)) for row in rows) / max(1, count), 3),
            "mean_video_p95_ms": round(sum(float(row.get("p95", 0.0)) for row in rows) / max(1, len(rows)), 3),
            "max_video_p95_ms": round(max((float(row.get("p95", 0.0)) for row in rows), default=0.0), 3),
            "count": count,
        }
    throughput_fps = total_frames / total_elapsed if total_elapsed > 0 else 0.0
    return {
        "videos": len(summaries),
        "total_frames": total_frames,
        "total_elapsed_seconds": round(total_elapsed, 3),
        "throughput_fps": round(throughput_fps, 3),
        "weighted_source_fps": round(weighted_source_fps, 3),
        "realtime_factor": round(throughput_fps / weighted_source_fps, 4) if weighted_source_fps > 0 else 0.0,
        "process_cpu_seconds": round(total_cpu_seconds, 3),
        "process_cpu_percent": round(100.0 * total_cpu_seconds / total_elapsed, 2) if total_elapsed > 0 else 0.0,
        "peak_rss_mb": round(max((float(item.get("peak_rss_mb", 0.0)) for item in summaries), default=0.0), 3),
        "latency": latency,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--memory-mode", default="full")
    parser.add_argument("--policy-mode", default="full")
    parser.add_argument("--policy-model", default=None, help="Path to learned policy JSON when --policy-mode learned")
    parser.add_argument("--network-profile", choices=NETWORK_PROFILES, default="lan")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--allow-memory-write", action="store_true", help="Allow profile and experience updates; use only for archive construction")
    parser.add_argument("--quality", choices=["on", "off"], default=None)
    parser.add_argument("--ledger", choices=["on", "off"], default=None)
    parser.add_argument("--archive", choices=["on", "off"], default=None)
    parser.add_argument("--visual-archive", choices=["on", "off"], default=None)
    parser.add_argument("--visual-decision", choices=["on", "off"], default=None)
    parser.add_argument("--visual-open-set", choices=["on", "off"], default=None)
    parser.add_argument("--experience", choices=["on", "off"], default=None)
    parser.add_argument("--risk", choices=["on", "off"], default=None)
    parser.add_argument("--entity-reconciliation", choices=["on", "off"], default=None)
    parser.add_argument("--keep-existing-output", action="store_true", help="Append to an existing run directory instead of replacing it")
    parser.add_argument("--display", action="store_true", help="Show the rendered output in a real-time OpenCV window; press q to stop")
    parser.add_argument("--save-output-video", action="store_true", help="Save the rendered result video for each input video")
    args = parser.parse_args()

    config = copy.deepcopy(load_config(args.config))
    exp = config.setdefault("experiment", {})
    exp.update({"enabled": True, "run_name": args.run_name, "split": args.split, "memory_mode": args.memory_mode, "policy_mode": args.policy_mode})
    if args.policy_model:
        exp.setdefault("policy", {})["learned_model_path"] = args.policy_model
    exp["memory_write"] = {"profile": bool(args.allow_memory_write), "experience": bool(args.allow_memory_write)}
    for key, value in (("quality", args.quality), ("temporal_ledger", args.ledger), ("archive", args.archive), ("experience", args.experience), ("risk", args.risk)):
        if value is not None:
            exp.setdefault(key, {})["enabled"] = value == "on"
    if args.entity_reconciliation is not None:
        exp.setdefault("entity_reconciliation", {})["enabled"] = args.entity_reconciliation == "on"
    if args.visual_archive is not None:
        exp.setdefault("visual_archive", {})["enabled"] = args.visual_archive == "on"
    if args.visual_decision is not None:
        exp.setdefault("visual_archive", {})["decision_enabled"] = args.visual_decision == "on"
    if args.visual_open_set is not None:
        visual_cfg = exp.setdefault("visual_archive", {})
        visual_cfg["open_set_enabled"] = args.visual_open_set == "on"
        if args.visual_open_set == "on":
            visual_cfg["enabled"] = True
            visual_cfg["decision_enabled"] = True
    exp.setdefault("network", {}).update(NETWORK_PROFILES[args.network_profile])
    exp["network"]["profile"] = args.network_profile
    config.setdefault("pipeline", {})["demo"] = bool(args.display or args.save_output_video)
    if args.save_output_video:
        config["pipeline"]["save_output_video"] = True
    output_root = Path(exp.get("output_dir", "./experiment_outputs")).resolve()
    run_dir = (output_root / args.run_name).resolve()
    if Path(os.path.commonpath([str(output_root), str(run_dir)])) != output_root or run_dir == output_root:
        raise ValueError("run-name must resolve to a child directory of experiment output_dir")
    if run_dir.exists() and not args.keep_existing_output:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

    videos = load_manifest(args.manifest, args.split)
    summaries = []
    for index, item in enumerate(videos):
        video_path = Path(item["video_path"])
        config["experiment"]["run_name"] = args.run_name
        config["experiment"]["video_id"] = item.get("video_id", video_path.stem)
        pipeline = ShipPipeline(config=config)
        stats = pipeline.process(str(video_path), display=args.display, max_frames=args.max_frames)
        summaries.append({"video_id": item.get("video_id", video_path.stem), "video_path": str(video_path), **stats})
        (run_dir / f"video_{index:04d}_summary.json").write_text(json.dumps(summaries[-1], ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (run_dir / "dataset_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    deployment_summary = aggregate_deployment_summaries(summaries)
    (run_dir / "deployment_summary.json").write_text(json.dumps(deployment_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    record_counts = write_episode_records(run_dir)
    (run_dir / "record_summary.json").write_text(json.dumps(record_counts, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
