"""Paper-facing open-set curves, video bootstrap CIs, and error breakdowns."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ACCEPT_STATES = {"confirmed", "structure_verified"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(1.0, max(0.0, probability)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def trapezoid_auc(points: Iterable[tuple[float, float]]) -> float:
    grouped: dict[float, float] = defaultdict(float)
    for x_value, y_value in points:
        grouped[float(x_value)] = max(grouped[float(x_value)], float(y_value))
    ordered = sorted(grouped.items())
    return sum(
        (right_x - left_x) * (left_y + right_y) * 0.5
        for (left_x, left_y), (right_x, right_y) in zip(ordered, ordered[1:])
    )


def write_oscr_svg(path: Path, curve: list[dict[str, Any]], auc: float) -> None:
    width, height = 640, 500
    left, right, top, bottom = 82, 28, 54, 72
    plot_width = width - left - right
    plot_height = height - top - bottom

    def point(x_value: float, y_value: float) -> tuple[float, float]:
        return left + x_value * plot_width, top + (1.0 - y_value) * plot_height

    ordered = sorted(curve, key=lambda row: (row["unknown_fpr"], row["ccr"]))
    polyline = " ".join(
        f"{x_value:.2f},{y_value:.2f}"
        for x_value, y_value in (point(row["unknown_fpr"], row["ccr"]) for row in ordered)
    )
    grid = []
    labels = []
    for index in range(6):
        value = index / 5
        x_value, y_value = point(value, value)
        grid.extend([
            f'<line x1="{x_value:.2f}" y1="{top}" x2="{x_value:.2f}" y2="{top + plot_height}" class="grid"/>',
            f'<line x1="{left}" y1="{y_value:.2f}" x2="{left + plot_width}" y2="{y_value:.2f}" class="grid"/>',
        ])
        labels.extend([
            f'<text x="{x_value:.2f}" y="{top + plot_height + 26}" text-anchor="middle">{value:.1f}</text>',
            f'<text x="{left - 14}" y="{y_value + 5:.2f}" text-anchor="end">{value:.1f}</text>',
        ])
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: Arial, sans-serif; font-size: 15px; fill: #111; }}
.grid {{ stroke: #d9d9d9; stroke-width: 1; }}
.axis {{ stroke: #111; stroke-width: 1.5; }}
</style>
<rect width="100%" height="100%" fill="white"/>
<text x="{width / 2}" y="30" text-anchor="middle" font-size="20">DINOv2 OSCR (AUC={auc:.3f})</text>
{''.join(grid)}
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" class="axis"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" class="axis"/>
{''.join(labels)}
<polyline points="{polyline}" fill="none" stroke="#1769aa" stroke-width="3"/>
<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle">Unknown false-positive rate</text>
<text x="22" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 22 {top + plot_height / 2})">Correct classification rate</text>
</svg>'''
    path.write_text(svg, encoding="utf-8")


def norm_identity(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    base = text.split("_", 1)[0]
    return base.zfill(3) if base.isdigit() else base


def members(row: dict[str, Any]) -> set[int]:
    return {int(value) for value in row.get("member_track_ids", []) if str(value).strip()}


def accepted_identity(row: dict[str, Any]) -> str:
    if str(row.get("identity_state", "")) not in ACCEPT_STATES:
        return ""
    return norm_identity(row.get("verified_identity", "")) or norm_identity(row.get("archive_candidate_id", ""))


def load_predictions(run_dir: Path, truth: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = list({
        (str(row.get("video_id", "")), str(row.get("entity_id", ""))): row
        for row in read_jsonl(run_dir / "entities.jsonl")
    }.values())
    annotated: dict[str, set[int]] = defaultdict(set)
    for row in truth:
        annotated[str(row.get("video_id", ""))].update(members(row))
    output = []
    for row in raw:
        video_id = str(row.get("video_id", ""))
        filtered_members = sorted(members(row) & annotated.get(video_id, set()))
        if filtered_members:
            output.append({**row, "member_track_ids": filtered_members})
    return output


def entity_outcomes(run_dir: Path, annotations: Path) -> list[dict[str, Any]]:
    truth = read_jsonl(annotations)
    predictions = load_predictions(run_dir, truth)
    predictions_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        predictions_by_video[str(row.get("video_id", ""))].append(row)

    output: list[dict[str, Any]] = []
    for truth_row in truth:
        video_id = str(truth_row.get("video_id", ""))
        truth_members = members(truth_row)
        fragments = [
            row for row in predictions_by_video.get(video_id, [])
            if members(row) & truth_members
        ]
        accepted = [identity for row in fragments if (identity := accepted_identity(row))]
        states = [str(row.get("identity_state", "unknown")) for row in fragments]
        truth_status = str(truth_row.get("known_or_unknown", "")).lower()
        truth_identity = norm_identity(truth_row.get("hull_number", ""))

        if truth_status == "unknown":
            if accepted:
                category = "unknown_to_known"
                correct = False
            elif states and all(state == "out_of_archive" for state in states):
                category = "correct_unknown_rejection"
                correct = True
            else:
                category = "unknown_unresolved_or_review"
                correct = False
        else:
            if any(identity != truth_identity for identity in accepted):
                category = "wrong_known_assignment"
                correct = False
            elif any(identity == truth_identity for identity in accepted):
                category = "correct_known_assignment"
                correct = True
            elif states and all(state == "out_of_archive" for state in states):
                category = "known_to_ooa"
                correct = False
            else:
                category = "known_unresolved_or_review"
                correct = False

        output.append({
            "video_id": video_id,
            "truth_entity_id": str(truth_row.get("entity_id", "")),
            "truth_status": truth_status,
            "truth_identity": truth_identity,
            "category": category,
            "correct": int(correct),
            "accepted_identities": "|".join(accepted),
            "predicted_states": "|".join(states),
            "truth_tracklets": len(truth_members),
            "predicted_fragments": len(fragments),
        })
    return output


def command_oscr(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.visual_matches)
    known = [row for row in rows if str(row.get("known_or_unknown", "")).lower() == "known"]
    unknown = [row for row in rows if str(row.get("known_or_unknown", "")).lower() == "unknown"]
    if not known or not unknown:
        raise RuntimeError("OSCR requires both known and unknown visual queries")

    scores = sorted({float(row.get("top_score", 0.0)) for row in rows}, reverse=True)
    thresholds = [math.inf, *scores, -math.inf]
    curve: list[dict[str, Any]] = []
    for threshold in thresholds:
        correct_known = sum(
            float(row.get("top_score", 0.0)) >= threshold
            and norm_identity(row.get("predicted_hull_number", "")) == norm_identity(row.get("hull_number", ""))
            for row in known
        )
        accepted_known = sum(float(row.get("top_score", 0.0)) >= threshold for row in known)
        false_accepted_unknown = sum(float(row.get("top_score", 0.0)) >= threshold for row in unknown)
        rejected_unknown = len(unknown) - false_accepted_unknown
        curve.append({
            "threshold": threshold,
            "ccr": correct_known / len(known),
            "known_acceptance_tpr": accepted_known / len(known),
            "unknown_fpr": false_accepted_unknown / len(unknown),
            "unknown_rejection_recall": rejected_unknown / len(unknown),
            "ats_proxy": (correct_known + rejected_unknown) / (len(known) + len(unknown)),
        })

    oscr_auc = trapezoid_auc((row["unknown_fpr"], row["ccr"]) for row in curve)
    roc_auc = trapezoid_auc((row["unknown_fpr"], row["known_acceptance_tpr"]) for row in curve)
    fpr95_candidates = [row for row in curve if row["known_acceptance_tpr"] >= 0.95]
    fpr95 = min((row["unknown_fpr"] for row in fpr95_candidates), default=1.0)
    closed_set_top1 = sum(
        norm_identity(row.get("predicted_hull_number", "")) == norm_identity(row.get("hull_number", ""))
        for row in known
    ) / len(known)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "oscr_curve.csv", curve)
    summary = {
        "visual_matches": str(args.visual_matches),
        "known_queries": len(known),
        "unknown_queries": len(unknown),
        "closed_set_top1_accuracy": round(closed_set_top1, 6),
        "oscr_auc": round(oscr_auc, 6),
        "known_unknown_auroc": round(roc_auc, 6),
        "fpr_at_95_known_acceptance_tpr": round(fpr95, 6),
    }
    (output_dir / "oscr_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if not args.no_plot:
        write_oscr_svg(output_dir / "oscr_curve.svg", curve, oscr_auc)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def metric_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = defaultdict(int)
    for row in rows:
        counts["entities"] += 1
        category = str(row["category"])
        if str(row["truth_status"]) == "known":
            counts["known"] += 1
        else:
            counts["unknown"] += 1
        counts[category] += 1
    return dict(counts)


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    correct_known = counts.get("correct_known_assignment", 0)
    rejected_unknown = counts.get("correct_unknown_rejection", 0)
    accepted_unknown = counts.get("unknown_to_known", 0)
    known = counts.get("known", 0)
    unknown = counts.get("unknown", 0)
    total = known + unknown
    return {
        "kacc": correct_known / max(1, known),
        "urec": rejected_unknown / max(1, unknown),
        "ufar": accepted_unknown / max(1, unknown),
        "ats": (correct_known + rejected_unknown) / max(1, total),
    }


def parse_named_runs(values: list[str]) -> list[tuple[str, Path]]:
    output = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"run must use NAME=PATH: {value}")
        name, raw_path = value.split("=", 1)
        output.append((name.strip(), Path(raw_path.strip())))
    return output


def command_bootstrap(args: argparse.Namespace) -> None:
    named_runs = parse_named_runs(args.run)
    outcomes = {name: entity_outcomes(path, args.annotations) for name, path in named_runs}
    video_ids = sorted({row["video_id"] for rows in outcomes.values() for row in rows})
    if not video_ids:
        raise RuntimeError("No annotated videos found")
    by_run_video: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for name, rows in outcomes.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["video_id"])].append(row)
        by_run_video[name] = grouped

    rng = random.Random(args.seed)
    samples: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in ("kacc", "urec", "ufar", "ats")}
        for name, _ in named_runs
    }
    differences: dict[str, dict[str, list[float]]] = {}
    reference = args.reference or named_runs[0][0]
    for name, _ in named_runs:
        if name != reference:
            differences[name] = {metric: [] for metric in ("kacc", "urec", "ufar", "ats")}

    for _ in range(args.iterations):
        sampled_videos = [rng.choice(video_ids) for _ in video_ids]
        replicate_metrics: dict[str, dict[str, float]] = {}
        for name, _ in named_runs:
            replicate_rows = [
                row
                for video_id in sampled_videos
                for row in by_run_video[name].get(video_id, [])
            ]
            replicate_metrics[name] = metrics_from_counts(metric_counts(replicate_rows))
            for metric, value in replicate_metrics[name].items():
                samples[name][metric].append(value)
        for name in differences:
            for metric in differences[name]:
                differences[name][metric].append(
                    replicate_metrics[name][metric] - replicate_metrics[reference][metric]
                )

    rows = []
    summary: dict[str, Any] = {
        "annotations": str(args.annotations),
        "videos": len(video_ids),
        "iterations": args.iterations,
        "seed": args.seed,
        "reference": reference,
        "runs": {},
        "paired_differences": {},
    }
    for name, _ in named_runs:
        point = metrics_from_counts(metric_counts(outcomes[name]))
        summary["runs"][name] = {}
        for metric, value in point.items():
            lower = percentile(samples[name][metric], 0.025)
            upper = percentile(samples[name][metric], 0.975)
            summary["runs"][name][metric] = {
                "estimate": round(value, 6),
                "ci95": [round(lower, 6), round(upper, 6)],
            }
            rows.append({"run": name, "metric": metric, "estimate": value, "ci95_low": lower, "ci95_high": upper})
    for name, metric_samples in differences.items():
        summary["paired_differences"][f"{name}-{reference}"] = {}
        for metric, values in metric_samples.items():
            lower = percentile(values, 0.025)
            upper = percentile(values, 0.975)
            estimate = summary["runs"][name][metric]["estimate"] - summary["runs"][reference][metric]["estimate"]
            summary["paired_differences"][f"{name}-{reference}"][metric] = {
                "estimate": round(estimate, 6),
                "ci95": [round(lower, 6), round(upper, 6)],
                "excludes_zero": bool(lower > 0 or upper < 0),
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "video_bootstrap_ci.csv", rows)
    (args.output_dir / "video_bootstrap_ci.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_errors(args: argparse.Namespace) -> None:
    rows = entity_outcomes(args.run_dir, args.annotations)
    counts = metric_counts(rows)
    summary = {
        "run_dir": str(args.run_dir),
        "annotations": str(args.annotations),
        "counts": counts,
        "metrics": {key: round(value, 6) for key, value in metrics_from_counts(counts).items()},
        "known_check": {
            "total": counts.get("known", 0),
            "sum_categories": sum(counts.get(name, 0) for name in (
                "correct_known_assignment", "wrong_known_assignment", "known_to_ooa", "known_unresolved_or_review"
            )),
        },
        "unknown_check": {
            "total": counts.get("unknown", 0),
            "sum_categories": sum(counts.get(name, 0) for name in (
                "correct_unknown_rejection", "unknown_to_known", "unknown_unresolved_or_review"
            )),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "open_set_error_entities.csv", rows)
    (args.output_dir / "open_set_error_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    oscr = subparsers.add_parser("oscr", help="Generate an OSCR curve from DINOv2 probe matches")
    oscr.add_argument("--visual-matches", required=True, type=Path)
    oscr.add_argument("--output-dir", required=True, type=Path)
    oscr.add_argument("--no-plot", action="store_true")
    oscr.set_defaults(func=command_oscr)

    bootstrap = subparsers.add_parser("bootstrap", help="Video-level bootstrap CIs and paired differences")
    bootstrap.add_argument("--annotations", required=True, type=Path)
    bootstrap.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR; repeat for each method")
    bootstrap.add_argument("--reference", help="Run name used as the paired-difference reference")
    bootstrap.add_argument("--iterations", type=int, default=10000)
    bootstrap.add_argument("--seed", type=int, default=7)
    bootstrap.add_argument("--output-dir", required=True, type=Path)
    bootstrap.set_defaults(func=command_bootstrap)

    errors = subparsers.add_parser("errors", help="Open-set entity error breakdown")
    errors.add_argument("--run-dir", required=True, type=Path)
    errors.add_argument("--annotations", required=True, type=Path)
    errors.add_argument("--output-dir", required=True, type=Path)
    errors.set_defaults(func=command_errors)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
