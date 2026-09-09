"""Generate self-contained one-page case-study reports from an Agent run."""
from __future__ import annotations

import argparse
import base64
import html
import json
import mimetypes
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2


CASE_DEFINITIONS = {
    "known": {
        "number": "I",
        "title": "Cross-Tracklet Known Vessel",
        "subtitle": "Persistent identity belief survives fragmented observations",
        "accent": "#147d64",
    },
    "unknown": {
        "number": "II",
        "title": "Open-Set Unknown Vessel",
        "subtitle": "Archive rejection is supported by attributed negative evidence",
        "accent": "#1769aa",
    },
    "conflict": {
        "number": "III",
        "title": "Persistent Uncertainty and Review",
        "subtitle": "Insufficient evidence is retained instead of being forced into an identity",
        "accent": "#b05a18",
    },
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _entity_key(row: dict[str, Any]) -> tuple[str, str] | None:
    video_id = str(row.get("video_id", "") or "")
    entity_id = str(row.get("entity_id", "") or "")
    if not video_id or not entity_id:
        return None
    return video_id, entity_id


def _latest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return max(
        rows,
        key=lambda row: (
            float(row.get("timestamp", 0.0) or 0.0),
            int(row.get("frame_id", row.get("last_frame", 0)) or 0),
        ),
    )


def _conflict_signals(bundle: dict[str, list[dict[str, Any]]]) -> list[str]:
    signals: set[str] = set()
    for stream in ("recognition", "entities", "episodes", "reports", "reviews"):
        for row in bundle.get(stream, []):
            package = dict(row.get("package", {}) or {})
            states = {
                str(row.get("identity_state", "") or ""),
                str(row.get("identity_state_before_review", "") or ""),
                str(row.get("terminal_state", "") or ""),
                str(package.get("identity_state", "") or ""),
            }
            if "conflicting" in states:
                signals.add(f"{stream}:conflicting_state")
            if int(row.get("structure_conflict_observations", 0) or 0) > 0:
                signals.add(f"{stream}:temporal_candidate_disagreement")
            reason_values = [
                row.get("visual_gate_reason", ""),
                row.get("identity_evidence_source", ""),
                *(row.get("uncertainty_reasons", []) or []),
                *(row.get("reasons", []) or []),
                *(package.get("reasons", []) or []),
            ]
            for reason in reason_values:
                normalized = str(reason or "").lower()
                if normalized in {
                    "visual_identity_conflict",
                    "conflicting_hull_candidates",
                    "conflicting_identity_evidence",
                    "conflicting_identity_rejected",
                }:
                    signals.add(f"{stream}:{normalized}")
    for row in bundle.get("actions", []):
        previous = str(row.get("previous_state", row.get("identity_state_before", "")) or "")
        following = str(row.get("next_state", row.get("identity_state_after", "")) or "")
        action = str(row.get("action", "") or "")
        if previous == "conflicting" or following == "conflicting":
            signals.add("actions:conflicting_state_transition")
        if action == "reobserve_conflicting_identity":
            signals.add("actions:conflict_requery")
    return sorted(signals)


def _review_signals(bundle: dict[str, list[dict[str, Any]]]) -> list[str]:
    signals: set[str] = set()
    for stream in ("entities", "episodes", "reports"):
        for row in bundle.get(stream, []):
            state = str(row.get("identity_state", row.get("terminal_state", "")) or "")
            if state == "review_requested" or bool(row.get("escalated")) or bool(row.get("review_requested")):
                signals.add(f"{stream}:review_requested")
    if bundle.get("reviews"):
        signals.add("reviews:record_retained")
    for row in bundle.get("actions", []):
        action = str(row.get("action", "") or "")
        if action in {"escalate", "request_remote_verification", "remote_report"}:
            signals.add(f"actions:{action}")
    return sorted(signals)


def load_manifest(path: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for row in read_jsonl(path):
        video_id = str(row.get("video_id", "") or "")
        raw_path = Path(str(row.get("video_path", "") or ""))
        if not video_id or not str(raw_path):
            continue
        if raw_path.is_absolute():
            result[video_id] = raw_path
        else:
            local = (path.parent / raw_path).resolve()
            result[video_id] = local if local.exists() else raw_path
    return result


def load_entity_annotations(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    return read_jsonl(path)


def collect_bundles(run_dir: Path) -> dict[tuple[str, str], dict[str, list[dict[str, Any]]]]:
    streams = {
        "recognition": "recognition.jsonl",
        "visual": "visual.jsonl",
        "observations": "observations.jsonl",
        "actions": ("entity_actions.jsonl", "actions.jsonl"),
        "links": "entity_links.jsonl",
        "episodes": ("entity_episodes.jsonl", "episodes.jsonl"),
        "entities": "entities.jsonl",
        "reports": "reports.jsonl",
        "reviews": ("review_queue.jsonl", "review_records.jsonl"),
    }
    bundles: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for stream, filenames in streams.items():
        for filename in (filenames if isinstance(filenames, tuple) else (filenames,)):
            for row in read_jsonl(run_dir / filename):
                key = _entity_key(row)
                if key is not None:
                    bundles[key][stream].append(row)
    return {key: dict(value) for key, value in bundles.items()}


def _bundle_summary(bundle: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    recognition = _latest(bundle.get("recognition", []))
    episode = _latest(bundle.get("episodes", []))
    entity = _latest(bundle.get("entities", []))
    report = _latest(bundle.get("reports", []))
    review = _latest(bundle.get("reviews", []))
    package = dict(review.get("package", {}) or review)
    merged = {**recognition, **entity, **episode, **report}
    state = str(
        merged.get("terminal_state", "")
        or merged.get("identity_state", "")
        or merged.get("cognitive_state", "")
        or merged.get("state", "")
        or package.get("identity_state", "")
    )
    member_track_ids = merged.get("member_track_ids", []) or package.get("member_track_ids", []) or []
    conflict_signals = _conflict_signals(bundle)
    review_signals = _review_signals(bundle)
    return {
        "state": state,
        "verified_identity": str(
            merged.get("resolution_identity", "")
            or merged.get("verified_identity", "")
            or package.get("verified_identity", "")
            or ""
        ),
        "candidate": str(
            merged.get("archive_candidate_id", "")
            or package.get("archive_candidate_id", "")
            or ""
        ),
        "member_track_ids": sorted({int(value) for value in member_track_ids if str(value).isdigit()}),
        "conflict": bool(conflict_signals),
        "conflict_signals": conflict_signals,
        "review": bool(review_signals),
        "review_signals": review_signals,
        "score": float(merged.get("archive_similarity_score", package.get("archive_similarity_score", 0.0)) or 0.0),
        "uncertainty": float(merged.get("uncertainty", package.get("uncertainty", 1.0)) or 1.0),
    }


def _normalize_identity(value: Any) -> str:
    identity = str(value or "").strip().split("_", 1)[0]
    return identity.zfill(3) if identity.isdigit() else identity


def _truth_for_bundle(
    key: tuple[str, str],
    bundle: dict[str, list[dict[str, Any]]],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    if not annotations:
        return {}
    summary = _bundle_summary(bundle)
    predicted_tracks = set(summary["member_track_ids"])
    candidates = [row for row in annotations if str(row.get("video_id", "")) == key[0]]
    if not candidates:
        return {}
    exact = next((row for row in candidates if str(row.get("entity_id", "")) == key[1]), None)
    if exact is not None:
        return exact
    ranked = [
        (
            len(predicted_tracks & {int(value) for value in row.get("member_track_ids", [])}),
            row,
        )
        for row in candidates
    ]
    overlap, selected = max(ranked, key=lambda item: item[0], default=(0, {}))
    return selected if overlap > 0 else {}


def _maximum_archive_score(bundle: dict[str, list[dict[str, Any]]]) -> float:
    scores: list[float] = []
    for row in bundle.get("recognition", []):
        scores.append(float(row.get("archive_similarity_score", 0.0) or 0.0))
        scores.extend(float(match.get("score", 0.0) or 0.0) for match in list(row.get("semantic_matches", []) or []))
    return max(scores, default=0.0)


def _selector_key(selector: str, bundles: dict[tuple[str, str], Any]) -> tuple[str, str]:
    selector = selector.strip()
    if ":" in selector:
        video_id, entity_id = selector.split(":", 1)
        key = video_id.strip(), entity_id.strip()
        if key in bundles:
            return key
    matches = [key for key in bundles if key[1] == selector]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Case selector does not identify one entity: {selector}")


def select_cases(
    bundles: dict[tuple[str, str], dict[str, list[dict[str, Any]]]],
    selectors: dict[str, str] | None = None,
    annotations: list[dict[str, Any]] | None = None,
    case_types: tuple[str, ...] | None = None,
) -> dict[str, tuple[str, str]]:
    selectors = selectors or {}
    annotations = annotations or []
    required_case_types = case_types or tuple(CASE_DEFINITIONS)
    selected: dict[str, tuple[str, str]] = {}
    used: set[tuple[str, str]] = set()
    summaries = {key: _bundle_summary(bundle) for key, bundle in bundles.items()}
    truths = {key: _truth_for_bundle(key, bundle, annotations) for key, bundle in bundles.items()}

    for case_type, selector in selectors.items():
        if selector:
            selected[case_type] = _selector_key(selector, bundles)
            truth = truths.get(selected[case_type], {})
            truth_class = str(truth.get("known_or_unknown", "") or "")
            if truth_class and case_type == "known" and truth_class != "known":
                raise ValueError(f"Known case selector points to a {truth_class} entity: {selector}")
            if truth_class and case_type in {"unknown", "conflict"} and truth_class != "unknown":
                raise ValueError(f"{case_type.title()} case selector points to a known entity: {selector}")
            used.add(selected[case_type])

    if "known" in required_case_types and "known" not in selected:
        candidates = [
            (key, summary) for key, summary in summaries.items()
            if key not in used and (
                summary["state"] in {"confirmed", "structure_verified", "human_resolved"}
                or bool(summary["verified_identity"])
            ) and (
                not annotations
                or (
                    str(truths[key].get("known_or_unknown", "")) == "known"
                    and _normalize_identity(summary["verified_identity"] or summary["candidate"])
                    == _normalize_identity(truths[key].get("hull_number", ""))
                )
            )
        ]
        if candidates:
            key, _ = max(
                candidates,
                key=lambda item: (
                    len(item[1]["member_track_ids"]) > 1,
                    len(item[1]["member_track_ids"]),
                    item[1]["score"],
                ),
            )
            selected["known"] = key
            used.add(key)

    if "unknown" in required_case_types and "unknown" not in selected:
        candidates = [
            (key, summary) for key, summary in summaries.items()
            if key not in used and summary["state"] == "out_of_archive"
            and (not annotations or str(truths[key].get("known_or_unknown", "")) == "unknown")
        ]
        if candidates:
            key, _ = max(
                candidates,
                key=lambda item: (
                    _maximum_archive_score(bundles[item[0]]),
                    len(item[1]["member_track_ids"]),
                    1.0 - item[1]["uncertainty"],
                ),
            )
            selected["unknown"] = key
            used.add(key)

    if "conflict" in required_case_types and "conflict" not in selected:
        candidates = [
            (key, summary) for key, summary in summaries.items()
            if key not in used and summary["conflict"]
            and (not annotations or str(truths[key].get("known_or_unknown", "")) == "unknown")
        ]
        if candidates:
            key, _ = max(
                candidates,
                key=lambda item: (
                    item[1]["state"] == "review_requested",
                    _maximum_archive_score(bundles[item[0]]),
                    item[1]["uncertainty"],
                ),
            )
            selected["conflict"] = key

    if "conflict" in required_case_types and "conflict" not in selected and annotations:
        fallback_candidates = [
            (key, summary) for key, summary in summaries.items()
            if key not in used
            and str(truths[key].get("known_or_unknown", "")) == "unknown"
            and summary["state"] in {"review_requested", "uncertain", "unknown"}
            and bool(bundles[key].get("recognition"))
        ]
        if fallback_candidates:
            key, _ = max(
                fallback_candidates,
                key=lambda item: (
                    _visual_evidence(bundles[item[0]])[0] is not None,
                    _maximum_archive_score(bundles[item[0]]),
                    len(bundles[item[0]].get("recognition", [])),
                    item[1]["uncertainty"],
                ),
            )
            selected["conflict"] = key

    missing = [case_type for case_type in required_case_types if case_type not in selected]
    if missing:
        raise ValueError(
            "Unable to auto-select case types: " + ", ".join(missing)
            + ". Supply explicit --known-case/--unknown-case/--conflict-case selectors."
        )
    return selected


def _existing_path(raw: Any, *roots: Path | None) -> Path | None:
    if not raw:
        return None
    path = Path(str(raw))
    candidates = [path] if path.is_absolute() else [root / path for root in roots if root is not None]
    candidates.append(path)
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _data_uri_from_path(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def _data_uri_from_frame(frame: Any) -> str:
    if frame is None:
        return ""
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def _nearest_observation(
    observations: list[dict[str, Any]],
    track_id: int,
    frame_id: int,
) -> dict[str, Any]:
    rows = [row for row in observations if int(row.get("track_id", 0) or 0) == track_id]
    if not rows:
        rows = observations
    if not rows:
        return {}
    return min(rows, key=lambda row: abs(int(row.get("frame_id", 0) or 0) - frame_id))


def _video_frame_uri(
    video_path: Path | None,
    frame_id: int,
    observation: dict[str, Any] | None = None,
) -> str:
    if video_path is None or not video_path.is_file():
        return ""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return ""
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        return ""
    bbox = list((observation or {}).get("bbox", []) or [])
    if len(bbox) == 4:
        x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
        x1 = max(0, min(frame.shape[1] - 1, x1))
        x2 = max(0, min(frame.shape[1] - 1, x2))
        y1 = max(0, min(frame.shape[0] - 1, y1))
        y2 = max(0, min(frame.shape[0] - 1, y2))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (28, 79, 205), max(2, frame.shape[1] // 600))
            label = f"LIVE  frame {frame_id}"
            cv2.rectangle(frame, (x1, max(0, y1 - 30)), (min(frame.shape[1], x1 + 210), y1), (28, 79, 205), -1)
            cv2.putText(frame, label, (x1 + 7, max(19, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return _data_uri_from_frame(frame)


def _direct_live_image(
    bundle: dict[str, list[dict[str, Any]]],
    run_dir: Path,
    preferred_recognition: dict[str, Any] | None = None,
) -> str:
    review = _latest(bundle.get("reviews", []))
    package = dict(review.get("package", {}) or review)
    target_view = dict(review.get("target_view", {}) or package.get("target_view", {}) or {})
    preferred = preferred_recognition or {}
    candidates = [
        preferred.get("target_image_path"),
        preferred.get("frame_image_path"),
        preferred.get("evidence_image"),
        package.get("evidence_image"),
        review.get("evidence_image"),
        target_view.get("image_path"),
    ]
    for row in reversed(bundle.get("recognition", [])):
        candidates.extend([row.get("target_image_path"), row.get("frame_image_path"), row.get("evidence_image")])
    for raw in candidates:
        path = _existing_path(raw, run_dir)
        if path is not None:
            return _data_uri_from_path(path)
    return ""


def _choose_recognition(bundle: dict[str, list[dict[str, Any]]], case_type: str) -> dict[str, Any]:
    rows = bundle.get("recognition", [])
    if not rows:
        return {}
    if case_type == "known":
        return max(
            rows,
            key=lambda row: (
                bool(row.get("verified_identity")),
                float(row.get("archive_similarity_score", 0.0) or 0.0),
                float(row.get("visual_similarity_score", 0.0) or 0.0),
                int(row.get("frame_id", 0) or 0),
            ),
        )
    if case_type == "conflict":
        return max(
            rows,
            key=lambda row: (
                str(row.get("identity_state", "") or "") == "conflicting",
                int(row.get("structure_conflict_observations", 0) or 0) > 0,
                int(row.get("structure_conflict_observations", 0) or 0),
                str(row.get("visual_gate_reason", "") or "") == "visual_identity_conflict",
                float(row.get("archive_similarity_score", 0.0) or 0.0),
                int(row.get("frame_id", 0) or 0),
            ),
        )
    return _latest(rows)


def _sample_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return rows
    indices = sorted({round(index * (len(rows) - 1) / (count - 1)) for index in range(count)})
    return [rows[index] for index in indices]


_STRUCTURE_PHRASES = {
    "黄色船体配黑色上层建筑": "a yellow hull with a black superstructure",
    "白色船体配黑色上层建筑": "a white hull with a black superstructure",
    "蓝色船体配白色上层建筑": "a blue hull with a white superstructure",
    "船首有白色浮筒": "a white fender at the bow",
    "船首有固定文字标识": "fixed lettering near the bow",
    "船首设有推进器": "a bow thruster",
    "船尾装有推进器": "propulsion equipment at the stern",
    "船尾有推进装置": "propulsion equipment at the stern",
    "船尾设有固定文字标识": "fixed lettering at the stern",
    "船尾有固定文字标识": "fixed lettering at the stern",
    "船身中部有圆形标志": "a circular marking amidships",
    "船身呈流线型设计": "a streamlined hull profile",
    "整体为小型机动船": "an overall compact motor-vessel configuration",
    "无可见文字标识": "no visible identifying text",
    "未见明显文字标识": "no clearly visible identifying text",
}


def _english_structure_description(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if not re.search(r"[\u3400-\u9fff]", text):
        return text
    translated: list[str] = []
    for raw_clause in re.split(r"[，。；;]+", text):
        clause = raw_clause.strip()
        if not clause:
            continue
        exact = _STRUCTURE_PHRASES.get(clause)
        if exact:
            translated.append(exact)
            continue
        parts: list[str] = []
        remainder = clause
        for chinese, english in sorted(_STRUCTURE_PHRASES.items(), key=lambda item: len(item[0]), reverse=True):
            if chinese in remainder:
                parts.append(english)
                remainder = remainder.replace(chinese, "")
        if parts and not re.search(r"[\u3400-\u9fff]", remainder):
            translated.extend(parts)
    if not translated:
        return "No reliable English structural description was retained for this observation."
    sentence = "; ".join(dict.fromkeys(translated)).strip(" ;")
    return sentence[:1].upper() + sentence[1:] + "."


def _visual_evidence(
    bundle: dict[str, list[dict[str, Any]]],
    target_identity: str = "",
) -> tuple[float | None, float | None, str]:
    target = _normalize_identity(target_identity)
    candidates: list[tuple[float, float | None, str]] = []
    for row in [*bundle.get("recognition", []), *bundle.get("visual", [])]:
        row_identity = str(row.get("visual_candidate_id", "") or "")
        row_score = row.get("visual_similarity_score")
        row_margin = row.get("visual_margin")
        if row_score is not None and (not target or _normalize_identity(row_identity) == target):
            candidates.append((float(row_score), float(row_margin) if row_margin is not None else None, row_identity))
        for match in list(row.get("visual_matches", []) or []):
            identity = str(match.get("hull_number", "") or match.get("archive_identity", "") or "")
            if target and _normalize_identity(identity) != target:
                continue
            score = match.get("score")
            if score is not None:
                candidates.append((float(score), float(row_margin) if row_margin is not None else None, identity))
    if not candidates:
        return None, None, ""
    score, margin, identity = max(candidates, key=lambda item: (item[0], item[1] is not None))
    return score, margin, identity


def _archive_reference_uris(
    bundle: dict[str, list[dict[str, Any]]],
    archive_root: Path | None,
    run_dir: Path,
    case_type: str,
    target_identity: str,
) -> list[tuple[str, str]]:
    if case_type == "unknown":
        return []
    target = _normalize_identity(target_identity)
    if not target:
        return []
    paths: list[tuple[float, str, Path]] = []
    for row in reversed([*bundle.get("recognition", []), *bundle.get("visual", [])]):
        for field in ("visual_matches", "semantic_matches"):
            for match in list(row.get(field, []) or []):
                identity = str(match.get("hull_number", "") or match.get("archive_identity", "") or "")
                if _normalize_identity(identity) != target:
                    continue
                path = _existing_path(match.get("image_path"), run_dir, archive_root)
                if path is not None:
                    paths.append((float(match.get("score", 0.0) or 0.0), identity, path))
    review = _latest(bundle.get("reviews", []))
    for raw in list(review.get("archive_image_paths", []) or []):
        path = _existing_path(raw, run_dir, archive_root)
        if path is not None and (target in path.parts or target in path.stem):
            paths.append((-0.5, target_identity or target, path))
    for candidate in list(review.get("archive_candidates", []) or []):
        identity = str(candidate.get("identity", "") or candidate.get("hull_number", "") or "")
        if _normalize_identity(identity) != target:
            continue
        path = _existing_path(candidate.get("image_path"), run_dir, archive_root)
        if path is not None:
            paths.append((float(candidate.get("score", 0.0) or 0.0), identity, path))
    if archive_root is not None and archive_root.is_dir():
        identity_dir = archive_root / target_identity
        if not identity_dir.is_dir():
            identity_dir = archive_root / target
        if identity_dir.is_dir():
            image = next((path for path in sorted(identity_dir.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES), None)
            if image is not None:
                paths.append((-1.0, target_identity or target, image))
    results: list[tuple[str, str]] = []
    seen: set[Path] = set()
    for _, identity, path in sorted(paths, key=lambda item: item[0], reverse=True):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        uri = _data_uri_from_path(path)
        if uri:
            results.append((identity, uri))
        if len(results) == 1:
            break
    return results


def _history_evidence_points(bundle: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    points: dict[tuple[int, int], dict[str, Any]] = {}
    for stream in ("recognition", "links", "actions"):
        for row in bundle.get(stream, []):
            frame_id = int(row.get("frame_id", 0) or 0)
            track_id = int(row.get("track_id", 0) or 0)
            if frame_id > 0:
                points[(frame_id, track_id)] = row
    observations = sorted(bundle.get("observations", []), key=lambda row: int(row.get("frame_id", 0) or 0))
    for row in _sample_rows(observations, min(6, len(observations))):
        frame_id = int(row.get("frame_id", 0) or 0)
        track_id = int(row.get("track_id", 0) or 0)
        if frame_id > 0:
            points[(frame_id, track_id)] = row
    return _sample_rows([points[key] for key in sorted(points)], 3)


def _text(value: Any, fallback: str = "Not available") -> str:
    rendered = str(value or "").strip()
    return rendered if rendered else fallback


def _fmt_score(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "--"


def _hull_evidence(rows: list[dict[str, Any]], case_type: str) -> str:
    observed: list[str] = []
    for row in rows:
        hull = str(row.get("raw_hull_number", "") or row.get("fused_hull_number", "") or "").strip()
        if hull:
            rendered = f"{hull} @ frame {int(row.get('frame_id', 0) or 0)}"
            if case_type != "known":
                rendered += " (unverified)"
            observed.append(rendered)
    return "; ".join(dict.fromkeys(observed)) if observed else "No reliable hull number observed"


def _decision_basis(
    bundle: dict[str, list[dict[str, Any]]],
    recognition: dict[str, Any],
    case_type: str,
) -> list[str]:
    summary = _bundle_summary(bundle)
    episode = _latest(bundle.get("episodes", []))
    report = _latest(bundle.get("reports", []))
    review = _latest(bundle.get("reviews", []))
    package = dict(review.get("package", {}) or review)
    reasons: list[str] = []
    if case_type == "known":
        reasons.append(
            f"Archive identity {_text(summary['verified_identity'] or summary['candidate'])} is supported by the accumulated entity evidence."
        )
    elif case_type == "unknown":
        reasons.append("No enrolled identity obtained sufficient, separated support across qualified observations.")
    else:
        if summary["conflict"]:
            reasons.append("Conflicting evidence persisted until autonomous resolution was no longer justified.")
        else:
            reasons.append("Evidence remained insufficient for either archive confirmation or autonomous rejection.")
    for value in (
        episode.get("terminal_reason"),
        report.get("terminal_reason"),
        recognition.get("visual_gate_reason"),
        package.get("reasons"),
    ):
        if isinstance(value, list):
            reasons.extend(str(item) for item in value if item)
        elif value:
            reasons.append(str(value))
    for action in reversed(bundle.get("actions", [])):
        rationale = str(action.get("rationale", "") or action.get("outcome_reason", "") or "")
        if rationale:
            reasons.append(rationale)
            break
    return list(dict.fromkeys(reasons))[:4]


def _timeline_rows(bundle: dict[str, list[dict[str, Any]]]) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for row in bundle.get("links", []):
        events.append({
            "frame": str(int(row.get("frame_id", 0) or 0)),
            "track": str(row.get("track_id", "")),
            "event": "Tracklet reassociated",
            "evidence": f"Inherited entity state from track {row.get('source_track_id', '--')}",
            "state": "entity linked",
        })
    for row in bundle.get("recognition", []):
        candidate = str(row.get("verified_identity", "") or row.get("archive_candidate_id", "") or "--")
        hull = str(row.get("raw_hull_number", "") or "not visible")
        score = _fmt_score(row.get("archive_similarity_score"))
        events.append({
            "frame": str(int(row.get("frame_id", 0) or 0)),
            "track": str(row.get("track_id", "")),
            "event": "Identity evidence",
            "evidence": f"hull={hull}; candidate={candidate}; archive score={score}",
            "state": str(row.get("identity_state", "unknown")),
        })
    for row in bundle.get("actions", []):
        if str(row.get("status", "")) and str(row.get("status", "")) not in {"selected", "completed"}:
            continue
        events.append({
            "frame": str(int(row.get("frame_id", 0) or 0)),
            "track": str(row.get("track_id", "")),
            "event": _text(row.get("action"), "Cognitive action").replace("_", " ").title(),
            "evidence": _text(row.get("rationale") or row.get("outcome_reason"), ", ".join(row.get("reasons", []) or [])),
            "state": str(row.get("next_state", row.get("previous_state", ""))),
        })
    events.sort(key=lambda row: (int(row["frame"]), row["event"]))
    if len(events) <= 6:
        return events
    priority = [
        event for event in events
        if event["state"] == "conflicting"
        or "Conflict" in event["event"]
        or event["event"] in {"Escalate", "Terminate"}
    ]
    selected: dict[tuple[str, str, str], dict[str, str]] = {}
    for event in [*_sample_rows(priority, 4), *_sample_rows(events, 6)]:
        key = event["frame"], event["event"], event["track"]
        if key not in selected and len(selected) < 6:
            selected[key] = event
    return sorted(selected.values(), key=lambda row: int(row["frame"]))


def _image_panel(uri: str, label: str, detail: str = "", panel_class: str = "") -> str:
    if not uri:
        content = '<div class="image-missing">IMAGE NOT AVAILABLE</div>'
    else:
        content = f'<img src="{uri}" alt="{html.escape(label)}">'
    return (
        f'<div class="image-panel {html.escape(panel_class)}">'
        f'<div class="image-label">{html.escape(label)}</div>{content}'
        f'<div class="image-detail">{html.escape(detail)}</div></div>'
    )


def build_report_html(
    case_type: str,
    key: tuple[str, str],
    bundle: dict[str, list[dict[str, Any]]],
    video_path: Path | None,
    archive_root: Path | None,
    run_dir: Path,
) -> str:
    definition = dict(CASE_DEFINITIONS[case_type])
    video_id, entity_id = key
    summary = _bundle_summary(bundle)
    if case_type == "conflict" and summary["conflict"]:
        definition["title"] = "Conflicting Identity Evidence and Review"
        definition["subtitle"] = "Incompatible qualified identity evidence triggers re-observation and review"
    recognition = _choose_recognition(bundle, case_type)
    observations = bundle.get("observations", [])
    frame_id = int(recognition.get("frame_id", 0) or 0)
    track_id = int(recognition.get("track_id", 0) or 0)
    observation = _nearest_observation(observations, track_id, frame_id)
    live_uri = _direct_live_image(bundle, run_dir, recognition) or _video_frame_uri(video_path, frame_id, observation)
    identity = summary["verified_identity"] or summary["candidate"] or ""
    visual_score_value, visual_margin_value, visual_identity = _visual_evidence(bundle, identity if case_type != "unknown" else "")
    reference_identity = identity or visual_identity
    references = _archive_reference_uris(bundle, archive_root, run_dir, case_type, reference_identity)
    reference_label = "Matched archive vessel" if case_type == "known" else "Leading archive candidate"
    reference_detail = "Registered identity" if case_type == "known" else "Candidate identity"
    reference_html = "".join(
        _image_panel(uri, reference_label, f"{reference_detail} {archive_identity or '--'}", "archive-panel")
        for archive_identity, uri in references
    )
    primary_class = "primary-images" if references else "primary-images no-reference"
    reference_block = f'<div class="reference-stack">{reference_html}</div>' if references else ""

    history_images: list[str] = []
    for row in _history_evidence_points(bundle):
        historical_frame = int(row.get("frame_id", 0) or 0)
        historical_track = int(row.get("track_id", 0) or 0)
        historical_observation = _nearest_observation(observations, historical_track, historical_frame)
        uri = _video_frame_uri(video_path, historical_frame, historical_observation)
        if uri:
            history_images.append(_image_panel(uri, f"Frame {historical_frame}", f"Track {historical_track}"))
    if not history_images and live_uri:
        history_images.append(_image_panel(live_uri, f"Frame {frame_id}", f"Track {track_id}"))
    while len(history_images) < 3:
        history_images.append(_image_panel("", "Historical view", "No retained image"))

    descriptions = list(dict.fromkeys(
        _english_structure_description(row.get("observed_structure_description", ""))
        for row in reversed(bundle.get("recognition", []))
        if str(row.get("observed_structure_description", "") or "").strip()
    ))[:2]
    features = dict(recognition.get("identity_features", {}) or {})
    structure_text = " ".join(descriptions) or "No structured description was returned."
    if features:
        feature_text = "; ".join(
            f"{str(key).replace('_', ' ')}: {value}"
            for key, value in features.items()
            if value and not re.search(r"[\u3400-\u9fff]", str(value))
        )
        if feature_text:
            structure_text += " " + feature_text

    state = summary["state"] or "unknown"
    identity = identity or "--"
    if summary["review"]:
        conflict_decision = "REVIEW REQUESTED"
    elif summary["conflict"]:
        conflict_decision = "CONFLICT RETAINED"
    else:
        conflict_decision = "UNRESOLVED / REVIEW"
    decision = {
        "known": f"KNOWN / {identity}",
        "unknown": "OUT OF ARCHIVE",
        "conflict": conflict_decision,
    }[case_type]
    member_tracks = summary["member_track_ids"] or sorted({
        int(row.get("track_id", 0) or 0) for row in bundle.get("recognition", []) if int(row.get("track_id", 0) or 0)
    })
    source = _text(recognition.get("identity_evidence_source"), "none")
    hull = _hull_evidence(bundle.get("recognition", []), case_type)
    visual_score = _fmt_score(visual_score_value) if visual_score_value is not None else "N/R"
    visual_margin = _fmt_score(visual_margin_value) if visual_margin_value is not None else "N/R"
    consistent = int(recognition.get("structure_consistent_observations", 0) or 0)
    conflicts = int(recognition.get("structure_conflict_observations", 0) or 0)
    basis = _decision_basis(bundle, recognition, case_type)
    timeline = _timeline_rows(bundle)

    basis_html = "".join(f"<li>{html.escape(reason)}</li>" for reason in basis)
    timeline_html = "".join(
        "<tr>"
        f"<td>{html.escape(row['frame'])}</td>"
        f"<td>{html.escape(row['track'])}</td>"
        f"<td>{html.escape(row['event'])}</td>"
        f"<td>{html.escape(row['evidence'])}</td>"
        f"<td>{html.escape(row['state'])}</td>"
        "</tr>"
        for row in timeline
    ) or '<tr><td colspan="5">No historical events retained</td></tr>'

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ECEM-Agent Case {definition['number']} - {html.escape(video_id)} / {html.escape(entity_id)}</title>
<style>
@page {{ size: A4 landscape; margin: 0; }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; width: 297mm; height: 210mm; background: #eef1f3; color: #152029; font-family: Arial, "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; }}
.sheet {{ width: 297mm; height: 210mm; padding: 7mm 8mm 6mm; background: #fff; display: grid; grid-template-rows: 18mm 120mm 51mm; gap: 4mm; overflow: hidden; }}
.header {{ border-bottom: 1.2mm solid {definition['accent']}; display: flex; justify-content: space-between; align-items: flex-start; }}
.eyebrow {{ color: {definition['accent']}; font-size: 8pt; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }}
h1 {{ margin: 1.2mm 0 .8mm; font-size: 19pt; line-height: 1; letter-spacing: 0; }}
.subtitle {{ font-size: 8.5pt; color: #53616d; }}
.decision {{ min-width: 62mm; padding: 2.5mm 4mm; color: #fff; background: {definition['accent']}; text-align: right; }}
.decision strong {{ display: block; font-size: 15pt; }}
.decision span {{ font-size: 8pt; opacity: .9; }}
.body-grid {{ display: grid; grid-template-columns: 171mm 102mm; gap: 5mm; min-height: 0; }}
.visual-column {{ display: grid; grid-template-rows: 79mm 36mm; gap: 4mm; min-height: 0; }}
.primary-images {{ display: grid; grid-template-columns: 1.65fr 1fr; gap: 3mm; }}
.primary-images.no-reference {{ grid-template-columns: 1fr; }}
.reference-stack {{ display: grid; grid-template-columns: 1fr; min-width: 0; }}
.history-strip {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 3mm; }}
.image-panel {{ border: .3mm solid #cbd3d8; position: relative; min-width: 0; min-height: 0; background: #f3f5f6; overflow: hidden; }}
.image-panel img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
.archive-panel img {{ object-fit: contain; background: #e8edef; }}
.image-label {{ position: absolute; z-index: 2; top: 0; left: 0; padding: 1.2mm 2mm; color: #fff; background: rgba(15, 26, 34, .86); font-size: 7pt; font-weight: 700; text-transform: uppercase; }}
.image-detail {{ position: absolute; z-index: 2; right: 0; bottom: 0; max-width: 90%; padding: 1mm 1.8mm; color: #fff; background: rgba(15, 26, 34, .78); font-size: 6.5pt; }}
.image-missing {{ height: 100%; display: grid; place-items: center; color: #87939c; font-size: 7pt; }}
.details {{ display: grid; grid-template-rows: auto auto 1fr; gap: 3mm; min-height: 0; }}
.metric-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); border: .3mm solid #cbd3d8; }}
.metric {{ padding: 2mm; border-right: .2mm solid #d8dee2; }}
.metric:last-child {{ border-right: 0; }}
.metric b {{ display: block; font-size: 11pt; color: {definition['accent']}; }}
.metric span {{ font-size: 6.5pt; color: #68757e; text-transform: uppercase; }}
.section {{ border-top: .45mm solid #253641; padding-top: 1.5mm; min-height: 0; }}
.section h2 {{ margin: 0 0 1.2mm; font-size: 8pt; text-transform: uppercase; color: #253641; }}
.section p, .section li {{ margin: 0; font-size: 7.4pt; line-height: 1.34; }}
.section ul {{ margin: 0; padding-left: 4mm; display: grid; gap: .8mm; }}
.structure {{ max-height: 34mm; overflow: hidden; }}
.hull {{ margin-top: 1.5mm !important; padding: 1.5mm 2mm; border-left: 1mm solid {definition['accent']}; background: #f1f5f4; }}
.history {{ min-height: 0; }}
.history h2 {{ margin: 0 0 1.5mm; font-size: 8.5pt; text-transform: uppercase; }}
table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
th {{ padding: 1.1mm 1.4mm; color: #fff; background: #253641; text-align: left; font-size: 6.7pt; }}
td {{ padding: 1.1mm 1.4mm; border-bottom: .2mm solid #d8dee2; vertical-align: top; font-size: 6.6pt; line-height: 1.18; overflow: hidden; }}
th:nth-child(1), td:nth-child(1) {{ width: 14mm; }}
th:nth-child(2), td:nth-child(2) {{ width: 14mm; }}
th:nth-child(3), td:nth-child(3) {{ width: 34mm; }}
th:nth-child(5), td:nth-child(5) {{ width: 27mm; }}
.footer {{ margin-top: 1mm; display: flex; justify-content: space-between; color: #6b7780; font-size: 6.2pt; }}
</style>
</head>
<body>
<article class="sheet">
  <header class="header">
    <div>
      <div class="eyebrow">ECEM-Agent / Case Study {definition['number']}</div>
      <h1>{html.escape(definition['title'])}</h1>
      <div class="subtitle">{html.escape(definition['subtitle'])}</div>
    </div>
    <div class="decision"><strong>{html.escape(decision)}</strong><span>{html.escape(video_id)} · {html.escape(entity_id)}</span></div>
  </header>
  <section class="body-grid">
    <div class="visual-column">
      <div class="{primary_class}">
        {_image_panel(live_uri, "Live observation", f"Frame {frame_id} / track {track_id}")}
        {reference_block}
      </div>
      <div class="history-strip">{''.join(history_images[:3])}</div>
    </div>
    <aside class="details">
      <div class="metric-grid">
        <div class="metric"><b>{_fmt_score(summary['score'])}</b><span>Archive score</span></div>
        <div class="metric"><b>{visual_score}</b><span>Visual score</span></div>
        <div class="metric"><b>{_fmt_score(summary['uncertainty'])}</b><span>Uncertainty</span></div>
      </div>
      <div class="section structure">
        <h2>Hull and structural evidence</h2>
        <p>{html.escape(structure_text)}</p>
        <p class="hull"><strong>Hull evidence:</strong> {html.escape(hull)}</p>
      </div>
      <div class="section">
        <h2>Decision basis</h2>
        <ul>{basis_html}</ul>
        <p class="hull"><strong>Evidence state:</strong> {html.escape(state)} | source {html.escape(source)} | visual margin {html.escape(visual_margin)} | consistent/conflict {consistent}/{conflicts}</p>
      </div>
    </aside>
  </section>
  <section class="history">
    <h2>Attributed historical evidence and cognitive actions</h2>
    <table><thead><tr><th>Frame</th><th>Track</th><th>Event</th><th>Attributed evidence / rationale</th><th>Belief after event</th></tr></thead><tbody>{timeline_html}</tbody></table>
    <div class="footer"><span>Entity tracklets: {html.escape(', '.join(map(str, member_tracks)) or '--')}</span><span>Evidence images are embedded in this report; no external image paths are required.</span></div>
  </section>
</article>
</body>
</html>"""


def _find_browser(explicit: Path | None = None) -> Path | None:
    if explicit is not None and explicit.is_file():
        return explicit
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable", "msedge"):
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved)
    windows_candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    return next((path for path in windows_candidates if path.is_file()), None)


def render_pdf(html_path: Path, pdf_path: Path, browser: Path) -> None:
    command = [
        str(browser),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={pdf_path.resolve()}",
        html_path.resolve().as_uri(),
    ]
    subprocess.run(command, check=True, timeout=120, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not pdf_path.is_file():
        raise RuntimeError(f"Browser did not create PDF: {pdf_path}")


def generate_case_studies(
    run_dir: Path,
    manifest: Path,
    output_dir: Path | None = None,
    archive_root: Path | None = None,
    selectors: dict[str, str] | None = None,
    annotations: Path | None = None,
    *,
    render_pdfs: bool = True,
    browser: Path | None = None,
    case_types: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    if not (run_dir / "entity_episodes.jsonl").exists() and not (run_dir / "episodes.jsonl").exists():
        from experiments.build_episode_records import write_episode_records

        write_episode_records(run_dir)
    bundles = collect_bundles(run_dir)
    requested_case_types = case_types or tuple(CASE_DEFINITIONS)
    selected = select_cases(
        bundles,
        selectors,
        load_entity_annotations(annotations),
        requested_case_types,
    )
    videos = load_manifest(manifest)
    destination = output_dir or (run_dir / "case_studies")
    destination.mkdir(parents=True, exist_ok=True)
    case_numbers = {"known": 1, "unknown": 2, "conflict": 3}
    if set(requested_case_types) == set(CASE_DEFINITIONS):
        stale_patterns = ("case_*.html", "case_*.pdf")
    else:
        stale_patterns = tuple(
            f"case_{case_numbers[case_type]:02d}_{case_type}_*.{extension}"
            for case_type in requested_case_types
            for extension in ("html", "pdf")
        )
    for pattern in stale_patterns:
        for stale_report in destination.glob(pattern):
            if stale_report.is_file():
                stale_report.unlink()
    renderer = _find_browser(browser) if render_pdfs else None
    results: list[dict[str, Any]] = []
    for case_type in requested_case_types:
        index = case_numbers[case_type]
        key = selected[case_type]
        html_text = build_report_html(case_type, key, bundles[key], videos.get(key[0]), archive_root, run_dir)
        stem = f"case_{index:02d}_{case_type}_{key[0]}_{key[1]}"
        html_path = destination / f"{stem}.html"
        html_path.write_text(html_text, encoding="utf-8")
        pdf_path = destination / f"{stem}.pdf"
        pdf_created = False
        if renderer is not None:
            render_pdf(html_path, pdf_path, renderer)
            pdf_created = True
        results.append({
            "case_type": case_type,
            "video_id": key[0],
            "entity_id": key[1],
            "html": html_path,
            "pdf": pdf_path if pdf_created else None,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate three self-contained ECEM-Agent case-study reports")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--archive-root", type=Path, default=Path("data/archive/visual_prototypes"))
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/test_entities.jsonl"))
    parser.add_argument("--known-case", default="", help="video_id:entity_id selector")
    parser.add_argument("--unknown-case", default="", help="video_id:entity_id selector")
    parser.add_argument("--conflict-case", default="", help="video_id:entity_id selector")
    parser.add_argument(
        "--only-case",
        choices=tuple(CASE_DEFINITIONS),
        help="Generate only one case type instead of requiring all three types",
    )
    parser.add_argument("--browser", type=Path, help="Chromium/Chrome/Edge executable for PDF export")
    parser.add_argument("--no-pdf", action="store_true", help="Generate self-contained HTML only")
    args = parser.parse_args()
    selectors = {
        "known": args.known_case,
        "unknown": args.unknown_case,
        "conflict": args.conflict_case,
    }
    results = generate_case_studies(
        args.run_dir,
        args.manifest,
        args.output_dir,
        args.archive_root,
        selectors,
        args.annotations if args.annotations.is_file() else None,
        render_pdfs=not args.no_pdf,
        browser=args.browser,
        case_types=(args.only_case,) if args.only_case else None,
    )
    for result in results:
        print(f"{result['case_type']}: {result['html']}")
        if result["pdf"] is not None:
            print(f"{result['case_type']} PDF: {result['pdf']}")
    if not args.no_pdf and all(result["pdf"] is None for result in results):
        print("PDF renderer not found; HTML reports are complete and can be printed to one-page PDF in any browser.")


if __name__ == "__main__":
    main()
