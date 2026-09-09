"""Calibrate visual reranker acceptance thresholds on a non-test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _norm_identity(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    base = text.split("_", 1)[0]
    return base.zfill(3) if base.isdigit() else base


def _truth_by_track(annotations: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for row in annotations:
        video_id = str(row.get("video_id", ""))
        for value in row.get("member_track_ids", []) or []:
            key = (video_id, int(value))
            if key in index and index[key] is not row:
                raise ValueError(f"Track {key} belongs to multiple truth entities")
            index[key] = row
    return index


def align_visual_observations(
    visual_rows: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    truth_index = _truth_by_track(annotations)
    rows: list[dict[str, Any]] = []
    unmatched = 0
    for visual in visual_rows:
        key = (str(visual.get("video_id", "")), int(visual.get("track_id", -1)))
        truth = truth_index.get(key)
        if truth is None:
            unmatched += 1
            continue
        rows.append(
            {
                "video_id": key[0],
                "track_id": key[1],
                "entity_id": str(
                    truth.get("entity_id") or f"{key[0]}:trackset:{','.join(str(v) for v in truth.get('member_track_ids', []) or [])}"
                ),
                "known": str(truth.get("known_or_unknown", "")).strip().lower() == "known",
                "truth_identity": _norm_identity(truth.get("hull_number", "")),
                "predicted_identity": _norm_identity(visual.get("visual_candidate_id", "")),
                "score": float(visual.get("visual_similarity_score", 0.0) or 0.0),
                "margin": float(visual.get("visual_margin", 0.0) or 0.0),
            }
        )
    return rows, unmatched


def evaluate_threshold(rows: list[dict[str, Any]], score_threshold: float, margin_threshold: float) -> dict[str, Any]:
    known = [row for row in rows if row["known"]]
    unknown = [row for row in rows if not row["known"]]
    accepted = [
        row
        for row in rows
        if row["score"] >= score_threshold and row["margin"] >= margin_threshold
    ]
    known_correct = sum(
        row["known"] and row["predicted_identity"] == row["truth_identity"]
        for row in accepted
    )
    known_wrong = sum(
        row["known"] and row["predicted_identity"] != row["truth_identity"]
        for row in accepted
    )
    unknown_false_accepts = sum(not row["known"] for row in accepted)
    unknown_rejections = len(unknown) - unknown_false_accepts
    known_rejections = len(known) - known_correct - known_wrong
    kacc = known_correct / max(1, len(known))
    urec = unknown_rejections / max(1, len(unknown))
    return {
        "score_threshold": round(score_threshold, 6),
        "margin_threshold": round(margin_threshold, 6),
        "known_correct_accepts": known_correct,
        "known_wrong_accepts": known_wrong,
        "known_rejections": known_rejections,
        "unknown_false_accepts": unknown_false_accepts,
        "unknown_rejections": unknown_rejections,
        "KAcc": round(kacc, 6),
        "URec": round(urec, 6),
        "UFAR": round(unknown_false_accepts / max(1, len(unknown)), 6),
        "identity_precision": round(known_correct / max(1, len(accepted)), 6),
        "balanced_accuracy": round((kacc + urec) / 2.0, 6),
        "ATS": round((known_correct + unknown_rejections) / max(1, len(rows)), 6),
    }


def _grid(start: float, stop: float, step: float) -> list[float]:
    if step <= 0 or stop < start:
        raise ValueError("Invalid threshold grid")
    count = int(round((stop - start) / step))
    return [round(start + index * step, 10) for index in range(count + 1)]


def calibrate_thresholds(
    run_dir: Path,
    annotations_path: Path,
    *,
    max_ufar: float = 0.02,
    score_start: float = 0.0,
    score_stop: float = 1.0,
    score_step: float = 0.01,
    margin_start: float = 0.0,
    margin_stop: float = 1.0,
    margin_step: float = 0.01,
    top_n: int = 20,
    objective: str = "kacc",
) -> dict[str, Any]:
    if objective not in {"kacc", "balanced", "ats"}:
        raise ValueError("objective must be one of: kacc, balanced, ats")
    visual_rows = read_jsonl(run_dir / "visual.jsonl")
    annotations = read_jsonl(annotations_path)
    rows, unmatched = align_visual_observations(visual_rows, annotations)
    known_count = sum(row["known"] for row in rows)
    unknown_count = len(rows) - known_count
    if not rows or not known_count or not unknown_count:
        raise ValueError(
            "Calibration requires matched visual observations from both known and unknown vessels"
        )
    candidates = [
        evaluate_threshold(rows, score, margin)
        for score in _grid(score_start, score_stop, score_step)
        for margin in _grid(margin_start, margin_stop, margin_step)
    ]
    feasible = [row for row in candidates if row["UFAR"] <= max_ufar]
    ranking = {
        "kacc": ("KAcc", "URec", "identity_precision", "ATS"),
        "balanced": ("balanced_accuracy", "KAcc", "URec", "identity_precision"),
        "ats": ("ATS", "KAcc", "identity_precision", "URec"),
    }[objective]
    ranked = sorted(
        feasible or candidates,
        key=lambda row: tuple(row[name] for name in ranking)
        + (row["score_threshold"], row["margin_threshold"]),
        reverse=True,
    )
    known_rows = [row for row in rows if row["known"]]
    raw_top1_correct = sum(
        row["predicted_identity"] == row["truth_identity"] for row in known_rows
    )
    known_entities = {row["entity_id"] for row in known_rows if row["entity_id"]}
    unknown_entities = {
        row["entity_id"] for row in rows if not row["known"] and row["entity_id"]
    }
    return {
        "run_dir": str(run_dir),
        "annotations": str(annotations_path),
        "visual_observations": len(visual_rows),
        "matched_observations": len(rows),
        "unmatched_observations": unmatched,
        "known_observations": known_count,
        "unknown_observations": unknown_count,
        "known_entities": len(known_entities),
        "unknown_entities": len(unknown_entities),
        "raw_known_top1_correct": raw_top1_correct,
        "raw_known_top1_accuracy": round(raw_top1_correct / known_count, 6),
        "objective": objective,
        "selection_rule": {
            "kacc": "maximize KAcc under the UFAR constraint, then URec, precision, and ATS",
            "balanced": "maximize balanced accuracy under the UFAR constraint, then KAcc, URec, and precision",
            "ats": "maximize ATS under the UFAR constraint, then KAcc, precision, and URec",
        }[objective],
        "max_ufar": max_ufar,
        "constraint_satisfied": bool(feasible),
        "selected": ranked[0],
        "top_candidates": ranked[: max(1, top_n)],
    }


def write_frozen_config(base_config: Path, output_config: Path, selected: dict[str, Any]) -> None:
    with base_config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    visual = config.setdefault("experiment", {}).setdefault("visual_archive", {})
    score = float(selected["score_threshold"])
    margin = float(selected["margin_threshold"])
    visual.update(
        {
            "backend": "qwen_vl_reranker",
            "min_support_score": score,
            "min_support_margin": margin,
            "conflict_min_score": score,
            "conflict_min_margin": margin,
            "out_of_archive_score": score,
            "out_of_archive_margin": margin,
        }
    )
    output_config.parent.mkdir(parents=True, exist_ok=True)
    with output_config.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate Qwen visual reranker thresholds on policy-training data.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--output-config", type=Path)
    parser.add_argument("--max-ufar", type=float, default=0.02)
    parser.add_argument("--score-start", type=float, default=0.0)
    parser.add_argument("--score-stop", type=float, default=1.0)
    parser.add_argument("--score-step", type=float, default=0.01)
    parser.add_argument("--margin-start", type=float, default=0.0)
    parser.add_argument("--margin-stop", type=float, default=1.0)
    parser.add_argument("--margin-step", type=float, default=0.01)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--objective", choices=["kacc", "balanced", "ats"], default="kacc")
    args = parser.parse_args()
    if bool(args.base_config) != bool(args.output_config):
        parser.error("--base-config and --output-config must be supplied together")
    return args


def main() -> None:
    args = parse_args()
    summary = calibrate_thresholds(
        args.run_dir,
        args.annotations,
        max_ufar=args.max_ufar,
        score_start=args.score_start,
        score_stop=args.score_stop,
        score_step=args.score_step,
        margin_start=args.margin_start,
        margin_stop=args.margin_stop,
        margin_step=args.margin_step,
        top_n=args.top,
        objective=args.objective,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.base_config and args.output_config:
        write_frozen_config(args.base_config, args.output_config, summary["selected"])
        summary["output_config"] = str(args.output_config)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
