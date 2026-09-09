"""Evaluate all definitive paper-suite runs and calculate method gains."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments.evaluate import evaluate, load_annotations
from experiments.evaluate_entities import evaluate_entities
from experiments.run_paper_suite import ABLATION_RUNS, MAIN_RUNS, NETWORK_RUNS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="experiment_outputs", type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--group", choices=["main", "ablation", "network", "all"], default="all")
    parser.add_argument("--output", default="experiment_outputs/paper_suite_summary.csv", type=Path)
    parser.add_argument("--entity-annotations", type=Path)
    args = parser.parse_args()
    annotations = load_annotations(args.annotations)
    specs = []
    if args.group in {"main", "all"}:
        specs.extend(MAIN_RUNS)
    if args.group in {"ablation", "all"}:
        specs.extend(ABLATION_RUNS)
    if args.group in {"network", "all"}:
        specs.extend(NETWORK_RUNS)
    rows = []
    by_name = {}
    for spec in specs:
        run_dir = args.root / spec.name
        if not run_dir.exists():
            print(f"skip missing run: {run_dir}")
            continue
        metrics = evaluate(run_dir, annotations)
        if args.entity_annotations:
            entity_metrics = evaluate_entities(run_dir, args.entity_annotations)
            metrics.update(entity_metrics)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        row = {"run_name": spec.name, **metrics}
        rows.append(row)
        by_name[spec.name] = metrics
    if rows:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    def difference(left: str, right: str, metric: str) -> float | None:
        if left not in by_name or right not in by_name:
            return None
        return round(float(by_name[left].get(metric, 0.0)) - float(by_name[right].get(metric, 0.0)), 6)
    gains = {
        "entity_active_vs_single_match_success_gain": difference("main_active_tel", "main_single", "entity_archive_matching_success_rate"),
        "entity_active_vs_fixed_match_success_gain": difference("main_active_tel", "main_fixed", "entity_archive_matching_success_rate"),
        "entity_full_vs_active_match_success_gain": difference("main_full", "main_active_tel", "entity_archive_matching_success_rate"),
        "entity_active_vs_single_call_change": difference("main_active_tel", "main_single", "avg_vlm_calls_per_entity"),
        "entity_active_vs_fixed_call_change": difference("main_active_tel", "main_fixed", "avg_vlm_calls_per_entity"),
        "entity_full_vs_active_unknown_far_change": difference("main_full", "main_active_tel", "entity_unknown_false_acceptance_rate"),
        "track_active_vs_single_match_success_gain": difference("main_active_tel", "main_single", "archive_matching_success_rate"),
        "track_active_vs_fixed_call_change": difference("main_active_tel", "main_fixed", "avg_vlm_calls_per_track"),
    }
    gains_path = args.output.with_name("paper_suite_gains.json")
    gains_path.write_text(json.dumps(gains, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"runs": len(rows), "summary": str(args.output), "gains": str(gains_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
