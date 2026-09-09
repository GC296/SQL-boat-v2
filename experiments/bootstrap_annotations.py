"""Build track-level annotation drafts from automatically generated tracker logs."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_drafts(run_dir: Path) -> list[dict[str, Any]]:
    observations = read_jsonl(run_dir / "observations.jsonl")
    recognitions = read_jsonl(run_dir / "recognition.jsonl")
    by_track: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    recognition_by_track: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in observations:
        key = (str(item.get("video_id", "")), int(item["track_id"]))
        by_track[key].append(item)
    for item in recognitions:
        key = (str(item.get("video_id", "")), int(item["track_id"]))
        recognition_by_track[key].append(item)

    drafts: list[dict[str, Any]] = []
    for (video_id, track_id), items in sorted(by_track.items()):
        items.sort(key=lambda item: int(item["frame_id"]))
        best = max(items, key=lambda item: (float(item.get("observation_quality", 0.0)), float(item.get("confidence", 0.0))))
        recs = sorted(recognition_by_track.get((video_id, track_id), []), key=lambda item: int(item["frame_id"]))
        latest = recs[-1] if recs else {}
        risk_counts: dict[str, int] = defaultdict(int)
        for rec in recs:
            risk_counts[str(rec.get("risk_level", "low"))] += 1
        predicted_risk = max(risk_counts, key=risk_counts.get) if risk_counts else "low"
        drafts.append({
            "video_id": video_id,
            "track_id": track_id,
            "frame_start": int(items[0]["frame_id"]),
            "frame_end": int(items[-1]["frame_id"]),
            "observation_count": len(items),
            "best_frame_id": int(best["frame_id"]),
            "best_observation_quality": round(float(best.get("observation_quality", 0.0)), 4),
            "best_bbox": best.get("bbox", []),
            "predicted_hull_number": latest.get("fused_hull_number", ""),
            "predicted_identity_state": latest.get("identity_state", "unknown"),
            "predicted_risk_label": predicted_risk,
            "vessel_id": "",
            "hull_number": "",
            "known_or_unknown": "",
            "scene_tags": [],
            "risk_label": "",
            "risk_reasons": [],
            "annotation_status": "pending",
            "annotator": "",
            "notes": "",
        })
    return drafts


def load_manifest(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    return {str(item["video_id"]): Path(item["video_path"]) for item in read_jsonl(path)}


def extract_representative_crops(records: list[dict[str, Any]], videos: dict[str, Path], crop_dir: Path) -> None:
    if not videos:
        return
    import cv2
    crop_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        grouped[item["video_id"]].append(item)
    for video_id, tracks in grouped.items():
        video_path = videos.get(video_id)
        if video_path is None or not video_path.exists():
            continue
        capture = cv2.VideoCapture(str(video_path))
        try:
            for item in tracks:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(item["best_frame_id"]))
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                x1, y1, x2, y2 = [int(value) for value in item.get("best_bbox", [0, 0, 0, 0])]
                height, width = frame.shape[:2]
                x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
                y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = frame[y1:y2, x1:x2]
                output = crop_dir / f"{video_id}_track_{item['track_id']}_frame_{item['best_frame_id']}.jpg"
                if cv2.imwrite(str(output), crop):
                    item["representative_crop"] = str(output)
        finally:
            capture.release()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(records[0]) if records else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in records:
            row = dict(item)
            row["best_bbox"] = json.dumps(row["best_bbox"], ensure_ascii=False)
            row["scene_tags"] = json.dumps(row["scene_tags"], ensure_ascii=False)
            row["risk_reasons"] = json.dumps(row["risk_reasons"], ensure_ascii=False)
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate human-correction drafts from automatic ByteTrack trajectories")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--manifest", type=Path, help="Optional video manifest used to export representative crops")
    parser.add_argument("--crop-dir", type=Path, help="Directory for automatically selected best-view crops")
    args = parser.parse_args()
    records = build_drafts(args.run_dir)
    output = args.output or args.run_dir / "tracks_to_annotate.jsonl"
    csv_output = args.csv or output.with_suffix(".csv")
    crop_dir = args.crop_dir or output.parent / "representative_crops"
    extract_representative_crops(records, load_manifest(args.manifest), crop_dir)
    write_jsonl(output, records)
    write_csv(csv_output, records)
    print(json.dumps({"tracks": len(records), "jsonl": str(output), "csv": str(csv_output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
