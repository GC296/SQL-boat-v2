"""Audit experiment outputs for explicit identity-conflict evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.generate_case_studies import (
    _bundle_summary,
    _maximum_archive_score,
    _truth_for_bundle,
    _visual_evidence,
    collect_bundles,
    load_entity_annotations,
)


def _candidate_rows(root: Path, annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    run_dirs = [root] if (root / "recognition.jsonl").exists() else sorted(path for path in root.iterdir() if path.is_dir())
    for run_dir in run_dirs:
        for key, bundle in collect_bundles(run_dir).items():
            summary = _bundle_summary(bundle)
            if not summary["conflict"]:
                continue
            truth = _truth_for_bundle(key, bundle, annotations)
            visual_score, visual_margin, visual_identity = _visual_evidence(bundle)
            recognition = bundle.get("recognition", [])
            rows.append({
                "run": run_dir.name,
                "video_id": key[0],
                "entity_id": key[1],
                "truth": str(truth.get("known_or_unknown", "unmatched") or "unmatched"),
                "state": summary["state"] or "unknown",
                "review": summary["review"],
                "archive_score": _maximum_archive_score(bundle),
                "visual_score": visual_score,
                "visual_margin": visual_margin,
                "visual_identity": visual_identity,
                "recognitions": len(recognition),
                "frames": sorted({int(row.get("frame_id", 0) or 0) for row in recognition}),
                "tracks": summary["member_track_ids"],
                "signals": summary["conflict_signals"],
            })
    return sorted(
        rows,
        key=lambda row: (
            row["truth"] == "unknown",
            row["review"],
            row["visual_score"] is not None,
            row["archive_score"],
            row["recognitions"],
        ),
        reverse=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("experiment_outputs"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/test_entities.jsonl"))
    args = parser.parse_args()
    rows = _candidate_rows(args.root, load_entity_annotations(args.annotations))
    print(f"true_conflict_candidates={len(rows)}")
    for row in rows:
        visual = "N/R" if row["visual_score"] is None else f"{row['visual_score']:.4f}"
        margin = "N/R" if row["visual_margin"] is None else f"{row['visual_margin']:.4f}"
        print(
            f"run={row['run']} video={row['video_id']} entity={row['entity_id']} "
            f"truth={row['truth']} state={row['state']} review={str(row['review']).lower()} "
            f"archive={row['archive_score']:.4f} visual={visual} margin={margin} "
            f"recognitions={row['recognitions']} tracks={row['tracks']} frames={row['frames']} "
            f"signals={row['signals']}"
        )


if __name__ == "__main__":
    main()
