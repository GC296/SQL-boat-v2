"""Build entity episodes and attributed review records from experiment streams."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ESCALATION_ACTIONS = {"request_remote_verification", "remote_report"}
AUTONOMOUS_TERMINAL_STATES = {"confirmed", "structure_verified", "out_of_archive"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _frame(row: dict[str, Any]) -> int:
    return int(row.get("frame_id", row.get("last_frame", 0)) or 0)


def _ordered(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (_frame(row), float(row.get("timestamp", 0.0))))


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[index])


def _unique_strings(values: list[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def _track_entity_index(streams: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, int], str]:
    index: dict[tuple[str, int], str] = {}
    for row in streams["entities"]:
        video_id = str(row.get("video_id", ""))
        entity_id = str(row.get("entity_id", ""))
        for track_id in row.get("member_track_ids", []):
            index[(video_id, int(track_id))] = entity_id
    for name in ("observations", "actions", "recognition", "edge_cloud", "visual"):
        for row in streams[name]:
            video_id = str(row.get("video_id", ""))
            track_id = int(row.get("track_id", 0) or 0)
            entity_id = str(row.get("entity_id", ""))
            if track_id and entity_id:
                index[(video_id, track_id)] = entity_id
    return index


def _group_streams(run_dir: Path) -> dict[tuple[str, str], dict[str, list[dict[str, Any]]]]:
    names = ("entities", "observations", "actions", "recognition", "edge_cloud", "visual")
    streams = {name: read_jsonl(run_dir / f"{name}.jsonl") for name in names}
    track_entities = _track_entity_index(streams)
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for name, rows in streams.items():
        for row in rows:
            video_id = str(row.get("video_id", ""))
            entity_id = str(row.get("entity_id", ""))
            if not entity_id:
                track_id = int(row.get("track_id", 0) or 0)
                entity_id = track_entities.get((video_id, track_id), "")
            if entity_id:
                grouped[(video_id, entity_id)][name].append(row)
    return grouped


def _candidate_evidence(recognitions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    candidates: list[dict[str, Any]] = []
    archive_images: list[str] = []
    seen: set[tuple[str, str, int]] = set()
    for row in recognitions:
        frame_id = _frame(row)
        for source, matches in (("semantic", row.get("semantic_matches", [])), ("visual", row.get("visual_matches", []))):
            for match in matches or []:
                identity = str(match.get("hull_number", match.get("identity", "")) or "")
                key = (source, identity, frame_id)
                if not identity or key in seen:
                    continue
                seen.add(key)
                image_path = str(match.get("image_path", "") or "")
                candidates.append({
                    "source": source,
                    "frame_id": frame_id,
                    "identity": identity,
                    "score": float(match.get("score", 0.0) or 0.0),
                    "image_path": image_path,
                })
                if image_path:
                    archive_images.append(image_path)
        for identity, score in (row.get("structure_candidate_scores", {}) or {}).items():
            key = ("temporal_structure", str(identity), frame_id)
            if key not in seen:
                seen.add(key)
                candidates.append({
                    "source": "temporal_structure",
                    "frame_id": frame_id,
                    "identity": str(identity),
                    "score": float(score or 0.0),
                    "image_path": "",
                })
    candidates.sort(key=lambda item: (-item["score"], item["source"], item["identity"]))
    return candidates, _unique_strings(archive_images)


def _evidence_gain(actions: list[dict[str, Any]], recognitions: list[dict[str, Any]]) -> tuple[list[float], int]:
    costly_actions = [row for row in actions if bool(row.get("should_query", False))]
    terminal_states = AUTONOMOUS_TERMINAL_STATES | {"review_requested"}
    unnecessary = sum(
        str((row.get("skill_signals", {}) or {}).get("identity_state", row.get("identity_state_before", "unknown"))) in terminal_states
        for row in costly_actions
    )
    gains: list[float] = []
    ordered_recognitions = _ordered(recognitions)
    for action in costly_actions:
        signals = action.get("skill_signals", {}) or {}
        before = float(signals.get("identity_uncertainty", 1.0) or 0.0)
        track_id = int(action.get("track_id", 0) or 0)
        frame_id = _frame(action)
        result = next((
            row for row in ordered_recognitions
            if int(row.get("track_id", 0) or 0) == track_id and _frame(row) == frame_id
        ), None)
        after = float(result.get("uncertainty", before) or 0.0) if result else before
        gains.append(round(max(0.0, before - after), 6))
    return gains, unnecessary


def _review_record(
    key: tuple[str, str],
    group: dict[str, list[dict[str, Any]]],
    episode: dict[str, Any],
) -> dict[str, Any] | None:
    escalation_actions = [row for row in _ordered(group["actions"]) if row.get("action") in ESCALATION_ACTIONS]
    if not escalation_actions:
        return None
    escalation = escalation_actions[-1]
    frame_id = _frame(escalation)
    recognitions = [row for row in _ordered(group["recognition"]) if _frame(row) <= frame_id]
    edge_rows = [row for row in _ordered(group["edge_cloud"]) if _frame(row) <= frame_id]
    observations = [row for row in _ordered(group["observations"]) if _frame(row) <= frame_id]
    selected_edge = edge_rows[-1] if edge_rows else {}
    target_frame = _frame(selected_edge) or frame_id
    selected_observation = next((row for row in reversed(observations) if _frame(row) <= target_frame), {})
    candidates, archive_images = _candidate_evidence(recognitions)
    hull_evidence = [{
        "track_id": int(row.get("track_id", 0) or 0),
        "frame_id": _frame(row),
        "raw_hull_number": str(row.get("raw_hull_number", "") or ""),
        "fused_hull_number": str(row.get("fused_hull_number", "") or ""),
        "description": str(row.get("observed_structure_description", "") or ""),
        "identity_state": str(row.get("identity_state", "unknown")),
        "uncertainty": float(row.get("uncertainty", 1.0) or 0.0),
    } for row in recognitions]
    signals = escalation.get("skill_signals", {}) or {}
    record = {
        "record_id": f"{key[0]}:{key[1]}:{frame_id}",
        "video_id": key[0],
        "entity_id": key[1],
        "member_track_ids": episode["member_track_ids"],
        "frame_id": frame_id,
        "action": str(escalation.get("action", "")),
        "reasons": list(escalation.get("reasons", [])),
        "identity_state": str(escalation.get("identity_state_before", signals.get("identity_state", "unknown"))),
        "uncertainty": float(signals.get("identity_uncertainty", 1.0) or 0.0),
        "policy_score": float(escalation.get("score", 0.0) or 0.0),
        "query_budget": {
            "used": int(signals.get("query_budget_used", episode["query_count"]) or 0),
            "maximum": int(signals.get("query_budget_max", 0) or 0),
        },
        "target_view": {
            "track_id": int(selected_edge.get("track_id", selected_observation.get("track_id", 0)) or 0),
            "frame_id": target_frame,
            "image_path": str(selected_edge.get("target_image_path", "") or ""),
            "bbox": selected_observation.get("bbox"),
            "observation_quality": float(selected_observation.get("observation_quality", 0.0) or 0.0),
        },
        "hull_and_structure_evidence": hull_evidence,
        "archive_candidates": candidates,
        "archive_image_paths": archive_images,
        "conflict_evidence": [row for row in hull_evidence if row["identity_state"] == "conflicting"],
    }
    required = {
        "entity": bool(record["entity_id"]),
        "frame": record["frame_id"] > 0,
        "action_reason": bool(record["action"] and record["reasons"]),
        "identity_state": bool(record["identity_state"]),
        "uncertainty": "uncertainty" in record,
        "target_image": bool(record["target_view"]["image_path"]),
        "archive_candidates": bool(record["archive_candidates"]),
        "archive_images": bool(record["archive_image_paths"]),
    }
    record["required_fields"] = required
    record["completeness"] = round(sum(required.values()) / len(required), 6)
    return record


def build_episode_records(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    review_records: list[dict[str, Any]] = []
    for key, group in sorted(_group_streams(run_dir).items()):
        for name in ("entities", "observations", "actions", "recognition", "edge_cloud", "visual"):
            group.setdefault(name, [])
        entities = _ordered(group["entities"])
        observations = _ordered(group["observations"])
        actions = _ordered(group["actions"])
        recognitions = _ordered(group["recognition"])
        edge_rows = _ordered(group["edge_cloud"])
        latest_entity = entities[-1] if entities else {}
        latest_recognition = recognitions[-1] if recognitions else {}
        escalation_actions = [row for row in actions if row.get("action") in ESCALATION_ACTIONS]
        identity_state = "review_requested" if escalation_actions else str(
            latest_entity.get("identity_state", latest_recognition.get("identity_state", "unknown"))
        )
        query_count = sum(max(1, int(row.get("request_count", 1) or 1)) for row in edge_rows)
        latencies = [float(row.get("total_latency_ms", 0.0) or 0.0) for row in edge_rows]
        gains, unnecessary_queries = _evidence_gain(actions, recognitions)
        member_tracks = set(int(value) for value in latest_entity.get("member_track_ids", []))
        member_tracks.update(int(row.get("track_id", 0) or 0) for row in observations + actions + recognitions if row.get("track_id"))
        all_frames = [_frame(row) for row in observations + actions + recognitions if _frame(row) > 0]
        episode = {
            "video_id": key[0],
            "entity_id": key[1],
            "member_track_ids": sorted(member_tracks),
            "first_frame": min(all_frames) if all_frames else int(latest_entity.get("first_frame", 0) or 0),
            "last_frame": max(all_frames) if all_frames else int(latest_entity.get("last_frame", 0) or 0),
            "identity_state": identity_state,
            "identity_state_before_review": str(latest_entity.get("identity_state_before_review", "")),
            "verified_identity": str(latest_entity.get("verified_identity", latest_recognition.get("verified_identity", "")) or ""),
            "archive_candidate_id": str(latest_entity.get("archive_candidate_id", latest_recognition.get("archive_candidate_id", "")) or ""),
            "uncertainty": float(latest_recognition.get("uncertainty", 1.0) or 0.0),
            "review_requested": bool(escalation_actions),
            "query_count": query_count,
            "upload_bytes": sum(int(row.get("upload_bytes", 0) or 0) for row in edge_rows),
            "latency_p50_ms": round(_percentile(latencies, 0.50), 6),
            "latency_p95_ms": round(_percentile(latencies, 0.95), 6),
            "costly_action_count": len(gains),
            "evidence_gain_sum": round(sum(gains), 6),
            "evidence_gain_per_action": round(mean(gains), 6) if gains else 0.0,
            "unnecessary_query_count": unnecessary_queries,
            "unnecessary_query_rate": round(unnecessary_queries / max(1, len(gains)), 6),
            "autonomously_resolved": identity_state in AUTONOMOUS_TERMINAL_STATES,
            "actions": actions,
            "recognition_evidence": recognitions,
        }
        episodes.append(episode)
        review = _review_record(key, group, episode)
        if review:
            review_records.append(review)
    return episodes, review_records


def write_episode_records(run_dir: Path) -> dict[str, int]:
    episodes, review_records = build_episode_records(run_dir)
    _write_jsonl(run_dir / "episodes.jsonl", episodes)
    _write_jsonl(run_dir / "review_records.jsonl", review_records)
    return {"episodes": len(episodes), "review_records": len(review_records)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(write_episode_records(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
