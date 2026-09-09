"""Probe DINOv2 visual retrieval without changing production decisions."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_manifest(path: Path) -> dict[str, Path]:
    videos = {}
    for item in read_jsonl(path):
        video_id = str(item.get("video_id", "")).strip()
        video_path = Path(str(item.get("video_path", ""))).expanduser()
        if video_id and str(video_path):
            videos[video_id] = video_path if video_path.is_absolute() else (path.parent / video_path).resolve()
    return videos


def load_archive(root: Path) -> list[dict[str, Any]]:
    records = []
    if not root.exists():
        raise FileNotFoundError(f"Visual archive does not exist: {root}")
    for hull_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for image_path in sorted(hull_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is not None and image.size:
                records.append({"hull_number": hull_dir.name, "prototype_id": image_path.stem, "image_path": str(image_path), "image": image})
    if not records:
        raise RuntimeError(f"No readable archive images found under: {root}")
    return records


def extract_queries(annotations: list[dict[str, Any]], videos: dict[str, Path], limit: int = 0):
    grouped = defaultdict(list)
    for item in annotations:
        grouped[str(item.get("video_id", ""))].append(item)
    queries, failures = [], []
    for video_id in sorted(grouped):
        video_path = videos.get(video_id)
        if video_path is None or not video_path.exists():
            failures.extend({"video_id": video_id, "track_id": int(item.get("track_id", 0)), "reason": "video_missing", "video_path": str(video_path or "")} for item in grouped[video_id])
            continue
        capture = cv2.VideoCapture(str(video_path))
        try:
            for item in sorted(grouped[video_id], key=lambda row: int(row.get("best_frame_id", 0))):
                if limit > 0 and len(queries) >= limit:
                    return queries, failures
                frame_id = int(item.get("best_frame_id", 0))
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ok, frame = capture.read()
                if not ok or frame is None:
                    failures.append({"video_id": video_id, "track_id": int(item.get("track_id", 0)), "frame_id": frame_id, "reason": "frame_read_failed"})
                    continue
                bbox = item.get("best_bbox", [0, 0, 0, 0])
                if isinstance(bbox, str):
                    bbox = json.loads(bbox)
                x1, y1, x2, y2 = [int(value) for value in bbox]
                height, width = frame.shape[:2]
                x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
                y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
                if x2 <= x1 or y2 <= y1:
                    failures.append({"video_id": video_id, "track_id": int(item.get("track_id", 0)), "frame_id": frame_id, "reason": "invalid_bbox"})
                    continue
                queries.append({"video_id": video_id, "track_id": int(item.get("track_id", 0)), "frame_id": frame_id, "known_or_unknown": str(item.get("known_or_unknown", "")), "hull_number": str(item.get("hull_number", "")), "image": frame[y1:y2, x1:x2].copy()})
        finally:
            capture.release()
    return queries, failures


def preprocess(image: np.ndarray, size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    scale = min(size / max(1, width), size / max(1, height))
    resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.empty((size, size, 3), dtype=np.uint8)
    canvas[...] = np.round(MEAN * 255).astype(np.uint8)
    left, top = (size - resized_width) // 2, (size - resized_height) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    normalized = (canvas.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(normalized.transpose(2, 0, 1)).contiguous()


def load_model(repo: Path, weights: Path, device: torch.device):
    if not (repo / "hubconf.py").exists():
        raise FileNotFoundError(f"hubconf.py not found: {repo}")
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    model = torch.hub.load(str(repo), "dinov2_vits14", source="local", pretrained=False)
    state = torch.load(str(weights), map_location="cpu", weights_only=True)
    for key in ("state_dict", "model", "teacher"):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            state = state[key]
            break
    cleaned = {}
    for key, value in state.items():
        normalized = str(key)
        for prefix in ("module.", "backbone."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
        cleaned[normalized] = value
    model.load_state_dict(cleaned, strict=True)
    return model.eval().to(device)


def encode(model, images: list[np.ndarray], device: torch.device, size: int, batch_size: int) -> torch.Tensor:
    features = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = torch.stack([preprocess(image, size) for image in images[start:start + batch_size]]).to(device)
            output = model(batch)
            if isinstance(output, dict):
                output = output.get("x_norm_clstoken", output.get("features"))
            features.append(F.normalize(output.float(), dim=1).cpu())
    return torch.cat(features)


def rank_identities(query: torch.Tensor, archive_features: torch.Tensor, archive: list[dict[str, Any]]):
    scores = torch.mv(archive_features, query).tolist()
    best = {}
    for record, score in zip(archive, scores):
        candidate = {"hull_number": record["hull_number"], "score": round(float(score), 6), "prototype_id": record["prototype_id"], "image_path": record["image_path"]}
        hull = candidate["hull_number"]
        if hull not in best or candidate["score"] > best[hull]["score"]:
            best[hull] = candidate
    return sorted(best.values(), key=lambda item: (-item["score"], item["hull_number"]))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    root = Path.cwd()
    torch_home = Path(os.environ.get("TORCH_HOME", root / "models" / "dinov2_torch_home"))
    parser = argparse.ArgumentParser(description="Probe DINOv2 retrieval on annotated test-video crops")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--archive-root", type=Path, default=Path("data/archive/visual_prototypes"))
    parser.add_argument("--model-repo", type=Path, default=torch_home / "hub" / "dinov2-main")
    parser.add_argument("--weights", type=Path, default=torch_home / "hub" / "checkpoints" / "dinov2_vits14_pretrain.pth")
    parser.add_argument("--output-dir", type=Path, default=Path("experiment_outputs/dinov2_probe"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-queries", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    archive = load_archive(args.archive_root)
    queries, failures = extract_queries(read_jsonl(args.annotations), load_manifest(args.manifest), max(0, args.max_queries))
    if not queries:
        raise RuntimeError("No query crops could be extracted")
    model = load_model(args.model_repo, args.weights, device)
    archive_features = encode(model, [item["image"] for item in archive], device, args.input_size, max(1, args.batch_size))
    query_features = encode(model, [item["image"] for item in queries], device, args.input_size, max(1, args.batch_size))

    results = []
    for query, feature in zip(queries, query_features):
        matches = rank_identities(feature, archive_features, archive)
        top, second = matches[0], matches[1]["score"] if len(matches) > 1 else 0.0
        correct = next((item for item in matches if item["hull_number"] == query["hull_number"]), None)
        results.append({"video_id": query["video_id"], "track_id": query["track_id"], "frame_id": query["frame_id"], "known_or_unknown": query["known_or_unknown"], "hull_number": query["hull_number"], "predicted_hull_number": top["hull_number"], "top_score": top["score"], "second_score": round(float(second), 6), "margin": round(top["score"] - float(second), 6), "correct_identity_score": correct["score"] if correct else None, "best_prototype_id": top["prototype_id"], "best_prototype_image": top["image_path"], "possible_archive_duplicate": top["score"] >= 0.9995, "matches": matches})

    known = [item for item in results if item["known_or_unknown"] == "known"]
    unknown = [item for item in results if item["known_or_unknown"] == "unknown"]
    known_scores = [float(item["correct_identity_score"]) for item in known if item["correct_identity_score"] is not None]
    summary = {"archive_prototypes": len(archive), "archive_prototypes_by_hull": dict(Counter(item["hull_number"] for item in archive)), "queries": len(results), "query_failures": len(failures), "known_queries": len(known), "unknown_queries": len(unknown), "known_top1_accuracy": round(sum(item["predicted_hull_number"] == item["hull_number"] for item in known) / max(1, len(known)), 6), "known_correct_identity_score_mean": round(mean(known_scores), 6) if known_scores else 0.0, "known_top_score_mean": round(mean(item["top_score"] for item in known), 6) if known else 0.0, "known_margin_mean": round(mean(item["margin"] for item in known), 6) if known else 0.0, "unknown_top_score_mean": round(mean(item["top_score"] for item in unknown), 6) if unknown else 0.0, "unknown_margin_mean": round(mean(item["margin"] for item in unknown), 6) if unknown else 0.0, "possible_duplicate_queries": sum(item["possible_archive_duplicate"] for item in results), "device": str(device), "output": str(args.output_dir / "visual_matches.jsonl")}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "visual_matches.jsonl", results)
    write_jsonl(args.output_dir / "failures.jsonl", failures)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
