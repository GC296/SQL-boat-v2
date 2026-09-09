from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from identity_schema import IDENTITY_FIELDS, vlm_identity_prompt
from training.common import (
    as_bool,
    compact_target_json,
    identity_target,
    read_jsonl,
    resolve_image_path,
    sample_group_id,
    write_jsonl,
)


IMAGE_FIELDS = ("image", "image_path", "representative_crop", "crop_path")


def _image_value(record: dict[str, Any]) -> str:
    return next((str(record.get(field, "") or "").strip() for field in IMAGE_FIELDS if record.get(field)), "")


def _source_split(annotation_path: Path, default_split: str) -> str:
    stem = annotation_path.stem.lower()
    if stem.startswith("train") or "_train" in stem:
        return "train"
    if stem.startswith("val") or "_val" in stem or "valid" in stem:
        return "val"
    if stem.startswith("test") or "_test" in stem:
        return "test"
    return default_split


def build_samples(
    annotation_paths: list[Path],
    image_root: Path,
    default_split: str,
    require_confirmed: bool = True,
    require_attributes: bool = False,
    strict_images: bool = True,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for annotation_path in annotation_paths:
        source_split = _source_split(annotation_path, default_split)
        for index, record in enumerate(read_jsonl(annotation_path), start=1):
            if require_confirmed and str(record.get("annotation_status", "confirmed")).lower() != "confirmed":
                continue
            raw_image = _image_value(record)
            if not raw_image:
                if strict_images:
                    raise ValueError(f"{annotation_path}:{index} has no image path")
                continue
            image_path = resolve_image_path(raw_image, annotation_path, image_root)
            if not image_path.exists():
                if strict_images:
                    raise FileNotFoundError(f"{annotation_path}:{index} image not found: {image_path}")
                continue
            target = identity_target(record)
            if require_attributes and not any(target["identity_features"].values()):
                continue
            video_id = str(record.get("video_id", "") or "")
            track_id = str(record.get("track_id", "") or "")
            sample_id = str(record.get("sample_id", "") or f"{annotation_path.stem}:{video_id}:{track_id}:{index}")
            split = str(record.get("split", source_split) or source_split).lower()
            samples.append({
                "sample_id": sample_id,
                "image": str(image_path),
                "split": split,
                "group_id": sample_group_id(record),
                "video_id": video_id,
                "track_id": track_id,
                "hull_visible": as_bool(record.get("hull_visible"), default=bool(record.get("hull_number"))),
                "prompt": vlm_identity_prompt(),
                "target": target,
                "target_text": compact_target_json(target),
                "source_annotations": str(annotation_path.resolve()),
            })
    return samples


def validate_group_splits(samples: list[dict[str, Any]]) -> dict[str, list[str]]:
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        splits_by_group[str(sample["group_id"])].add(str(sample["split"]))
    return {
        group_id: sorted(splits)
        for group_id, splits in splits_by_group.items()
        if len(splits) > 1 and group_id != "ungrouped"
    }


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = Counter(str(sample["split"]) for sample in samples)
    groups = {str(sample["group_id"]) for sample in samples}
    visible = sum(bool(sample["hull_visible"]) for sample in samples)
    attribute_samples = sum(any(sample["target"]["identity_features"].values()) for sample in samples)
    return {
        "samples": len(samples),
        "groups": len(groups),
        "splits": dict(sorted(split_counts.items())),
        "hull_visible_samples": visible,
        "hull_unreadable_samples": len(samples) - visible,
        "samples_with_attributes": attribute_samples,
        "identity_fields": list(IDENTITY_FIELDS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-checked Qwen3-VL LoRA JSONL data")
    parser.add_argument("--annotations", action="append", required=True, type=Path)
    parser.add_argument("--image-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--default-split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--allow-pending", action="store_true")
    parser.add_argument("--require-attributes", action="store_true")
    parser.add_argument("--skip-missing-images", action="store_true")
    parser.add_argument("--allow-group-overlap", action="store_true")
    args = parser.parse_args()

    samples = build_samples(
        args.annotations,
        args.image_root,
        args.default_split,
        require_confirmed=not args.allow_pending,
        require_attributes=args.require_attributes,
        strict_images=not args.skip_missing_images,
    )
    leakage = validate_group_splits(samples)
    if leakage and not args.allow_group_overlap:
        preview = json.dumps(dict(list(leakage.items())[:10]), ensure_ascii=False)
        raise SystemExit(f"physical-entity leakage across splits: {preview}")
    write_jsonl(args.output, samples)
    summary = {**summarize(samples), "group_split_overlap": leakage, "output": str(args.output.resolve())}
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
