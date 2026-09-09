"""Audit which archive identities and truth vessels are judged in archive."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACCEPT_STATES = {"confirmed", "structure_verified"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _members(row: dict[str, Any]) -> set[int]:
    return {int(value) for value in row.get("member_track_ids", []) if str(value).strip()}


def _norm_identity(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    base = text.split("_", 1)[0]
    return base.zfill(3) if base.isdigit() else base


def _filter_predictions(predictions: list[dict[str, Any]], truth: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: dict[str, set[int]] = defaultdict(set)
    for row in truth:
        annotated[str(row.get("video_id", ""))].update(_members(row))
    output: list[dict[str, Any]] = []
    for row in predictions:
        video_id = str(row.get("video_id", ""))
        members = sorted(_members(row) & annotated.get(video_id, set()))
        if members:
            output.append({**row, "member_track_ids": members})
    return output


def _accepted_identity(row: dict[str, Any]) -> str:
    if str(row.get("identity_state", "")) not in ACCEPT_STATES:
        return ""
    return _norm_identity(row.get("verified_identity", "")) or _norm_identity(row.get("archive_candidate_id", ""))


def _best_truth(pred: dict[str, Any], truth_by_video: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    video_id = str(pred.get("video_id", ""))
    members = _members(pred)
    matches = [
        row for row in truth_by_video.get(video_id, [])
        if bool(_members(row) & members)
    ]
    return max(matches, key=lambda row: len(_members(row) & members), default={})


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def audit_archive_tendency(run_dir: Path, annotations: Path) -> dict[str, list[dict[str, Any]]]:
    truth = read_jsonl(annotations)
    raw_predictions = list({
        (str(row.get("video_id", "")), str(row.get("entity_id", ""))): row
        for row in read_jsonl(run_dir / "entities.jsonl")
    }.values())
    predictions = _filter_predictions(raw_predictions, truth)
    recognitions = read_jsonl(run_dir / "recognition.jsonl")
    visual_rows = read_jsonl(run_dir / "visual.jsonl")

    truth_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in truth:
        truth_by_video[str(row.get("video_id", ""))].append(row)

    accepted_by_archive: Counter[str] = Counter()
    correct_by_archive: Counter[str] = Counter()
    unknown_false_accept_by_archive: Counter[str] = Counter()
    wrong_known_by_archive: Counter[str] = Counter()
    candidate_mentions: Counter[str] = Counter()
    visual_mentions: Counter[str] = Counter()
    accepted_rows: list[dict[str, Any]] = []

    truth_vessel_stats: dict[tuple[str, str], dict[str, Any]] = {}
    for truth_row in truth:
        key = (str(truth_row.get("known_or_unknown", "")).lower(), str(truth_row.get("vessel_id", "")))
        truth_vessel_stats.setdefault(key, {
            "truth_status": key[0],
            "truth_vessel_id": key[1],
            "truth_entities": 0,
            "accepted_entities": 0,
            "out_of_archive_entities": 0,
            "unresolved_entities": 0,
            "accepted_identities": Counter(),
            "states": Counter(),
        })
        truth_vessel_stats[key]["truth_entities"] += 1

    for row in recognitions:
        identity = _norm_identity(row.get("archive_candidate_id", "")) or _norm_identity(row.get("verified_identity", ""))
        if identity:
            candidate_mentions[identity] += 1
    for row in visual_rows:
        identity = _norm_identity(row.get("visual_candidate_id", ""))
        if identity:
            visual_mentions[identity] += 1

    fragments_by_truth: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        truth_row = _best_truth(pred, truth_by_video)
        if truth_row:
            fragments_by_truth[str(truth_row.get("entity_id", ""))].append(pred)
        candidate = _norm_identity(pred.get("archive_candidate_id", ""))
        if candidate:
            candidate_mentions[candidate] += 1
        accepted = _accepted_identity(pred)
        if not accepted:
            continue
        accepted_by_archive[accepted] += 1
        truth_status = str(truth_row.get("known_or_unknown", "")).lower()
        truth_hull = _norm_identity(truth_row.get("hull_number", ""))
        if truth_status == "unknown":
            outcome = "unknown_false_accept"
            unknown_false_accept_by_archive[accepted] += 1
        elif accepted == truth_hull:
            outcome = "known_correct_accept"
            correct_by_archive[accepted] += 1
        else:
            outcome = "known_wrong_accept"
            wrong_known_by_archive[accepted] += 1
        accepted_rows.append({
            "outcome": outcome,
            "accepted_identity": accepted,
            "candidate_identity": candidate,
            "video_id": str(pred.get("video_id", "")),
            "entity_id": str(pred.get("entity_id", "")),
            "identity_state": str(pred.get("identity_state", "")),
            "truth_entity_id": str(truth_row.get("entity_id", "")),
            "truth_vessel_id": str(truth_row.get("vessel_id", "")),
            "truth_status": truth_status,
            "truth_hull_number": truth_hull,
            "member_track_ids": sorted(_members(pred)),
            "archive_similarity_score": float(pred.get("archive_similarity_score", 0.0) or 0.0),
        })

    for truth_row in truth:
        key = (str(truth_row.get("known_or_unknown", "")).lower(), str(truth_row.get("vessel_id", "")))
        item = truth_vessel_stats[key]
        fragments = fragments_by_truth.get(str(truth_row.get("entity_id", "")), [])
        accepted = [_accepted_identity(pred) for pred in fragments if _accepted_identity(pred)]
        states = [str(pred.get("identity_state", "unknown")) for pred in fragments]
        item["accepted_entities"] += int(bool(accepted))
        item["out_of_archive_entities"] += int(bool(states) and all(state == "out_of_archive" for state in states))
        item["unresolved_entities"] += int(not accepted and not (states and all(state == "out_of_archive" for state in states)))
        item["accepted_identities"].update(accepted)
        item["states"].update(states)

    archive_summary: list[dict[str, Any]] = []
    for identity in sorted(set(accepted_by_archive) | set(candidate_mentions) | set(visual_mentions)):
        accepted = accepted_by_archive.get(identity, 0)
        false_accepts = unknown_false_accept_by_archive.get(identity, 0)
        wrong_known = wrong_known_by_archive.get(identity, 0)
        archive_summary.append({
            "archive_identity": identity,
            "accepted_entities": accepted,
            "correct_known_accepts": correct_by_archive.get(identity, 0),
            "unknown_false_accepts": false_accepts,
            "wrong_known_accepts": wrong_known,
            "wrong_or_false_accepts": false_accepts + wrong_known,
            "candidate_mentions": candidate_mentions.get(identity, 0),
            "visual_top1_mentions": visual_mentions.get(identity, 0),
            "false_accept_share": round(false_accepts / max(1, accepted), 6),
        })
    archive_summary.sort(key=lambda row: (row["accepted_entities"], row["wrong_or_false_accepts"], row["candidate_mentions"]), reverse=True)

    truth_summary: list[dict[str, Any]] = []
    for item in truth_vessel_stats.values():
        total = max(1, int(item["truth_entities"]))
        truth_summary.append({
            "truth_status": item["truth_status"],
            "truth_vessel_id": item["truth_vessel_id"],
            "truth_entities": item["truth_entities"],
            "accepted_entities": item["accepted_entities"],
            "acceptance_rate": round(item["accepted_entities"] / total, 6),
            "out_of_archive_entities": item["out_of_archive_entities"],
            "unresolved_entities": item["unresolved_entities"],
            "accepted_identities": dict(item["accepted_identities"]),
            "states": dict(item["states"]),
        })
    truth_summary.sort(key=lambda row: (row["acceptance_rate"], row["truth_entities"]), reverse=True)
    accepted_rows.sort(key=lambda row: (row["outcome"] not in {"unknown_false_accept", "known_wrong_accept"}, row["accepted_identity"], row["video_id"]))
    return {
        "archive_summary": archive_summary,
        "truth_vessel_summary": truth_summary,
        "accepted_entity_rows": accepted_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    result = audit_archive_tendency(args.run_dir, args.annotations)
    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "archive_tendency_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "archive_tendency_by_archive.csv", result["archive_summary"])
    _write_csv(output_dir / "archive_tendency_by_truth_vessel.csv", result["truth_vessel_summary"])
    _write_csv(output_dir / "archive_tendency_accepted_entities.csv", result["accepted_entity_rows"])

    print(json.dumps({
        "run_dir": str(args.run_dir),
        "annotations": str(args.annotations),
        "archive_rows": len(result["archive_summary"]),
        "truth_vessel_rows": len(result["truth_vessel_summary"]),
        "accepted_entity_rows": len(result["accepted_entity_rows"]),
        "outputs": [
            str(output_dir / "archive_tendency_by_archive.csv"),
            str(output_dir / "archive_tendency_by_truth_vessel.csv"),
            str(output_dir / "archive_tendency_accepted_entities.csv"),
        ],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    print("top_archive_identities")
    for row in result["archive_summary"][:args.top]:
        print(row)
    print("top_truth_vessels_by_acceptance_rate")
    for row in result["truth_vessel_summary"][:args.top]:
        print(row)


if __name__ == "__main__":
    main()
