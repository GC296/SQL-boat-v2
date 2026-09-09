"""Train a lightweight learned cognitive policy from offline policy samples."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


ACTIONS = ["defer", "query", "stop_known", "stop_out_of_archive", "escalate_review"]
DEFAULT_FEATURES = [
    "observation_quality",
    "identity_uncertainty",
    "evidence_opportunity_score",
    "query_budget_used",
    "query_budget_max",
    "query_budget_fraction",
    "temporal_gap_frames",
    "quality_improvement",
    "novel_tracklet_view",
    "structure_score",
    "structure_evidence_count",
    "structure_consistent_observations",
    "structure_conflict_observations",
    "visual_similarity_score",
    "visual_margin",
    "visual_observation_count",
    "visual_consistent_observations",
    "visual_low_score_observations",
    "text_visual_agree",
    "visual_only_mode",
    "recognized_before",
    "risk_score",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _feature_names(samples: list[dict[str, Any]]) -> list[str]:
    seen = set(DEFAULT_FEATURES)
    for sample in samples:
        for key, value in (sample.get("features", {}) or {}).items():
            if isinstance(value, (int, float, bool)):
                seen.add(str(key))
    return [name for name in DEFAULT_FEATURES if name in seen] + sorted(seen - set(DEFAULT_FEATURES))


def _vectorize(samples: list[dict[str, Any]], feature_names: list[str], target: str) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    action_index = {action: index for index, action in enumerate(ACTIONS)}
    for sample in samples:
        label = str(sample.get(target, ""))
        if label not in action_index:
            continue
        features = sample.get("features", {}) or {}
        x_rows.append([_as_float(features.get(name, 0.0)) for name in feature_names])
        y_rows.append(action_index[label])
    if not x_rows:
        raise ValueError(f"No usable samples found for target field: {target}")
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.int64)


def _split_by_video(
    samples: list[dict[str, Any]],
    validation_fraction: float,
    seed: int,
    target: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    if validation_fraction <= 0.0:
        return samples, [], []
    videos = sorted({str(sample.get("video_id", "")) for sample in samples})
    if len(videos) < 2:
        return samples, [], []
    validation_count = min(len(videos) - 1, max(1, round(len(videos) * validation_fraction)))

    def label_for(sample: dict[str, Any]) -> str:
        fields = [target] if target else []
        fields.extend(["utility_action", "oracle_action", "teacher_action"])
        for field in fields:
            if field and str(sample.get(field, "")) in ACTIONS:
                return str(sample[field])
        return ""

    groups: dict[str, list[dict[str, Any]]] = {video: [] for video in videos}
    for sample in samples:
        groups[str(sample.get("video_id", ""))].append(sample)
    total_labels = Counter(label_for(sample) for sample in samples if label_for(sample))
    target_sample_count = len(samples) * validation_fraction

    def split_score(selected: set[str]) -> float:
        selected_samples = [sample for video in selected for sample in groups[video]]
        label_counts = Counter(label_for(sample) for sample in selected_samples if label_for(sample))
        sample_error = abs(len(selected_samples) - target_sample_count) / max(1.0, target_sample_count)
        class_errors: list[float] = []
        coverage_penalty = 0.0
        for action, total in total_labels.items():
            selected_count = label_counts.get(action, 0)
            class_errors.append(abs(selected_count / total - validation_fraction) / max(validation_fraction, 1e-6))
            supporting_videos = sum(
                any(label_for(sample) == action for sample in groups[video])
                for video in videos
            )
            if supporting_videos >= 2 and (selected_count == 0 or selected_count == total):
                coverage_penalty += 2.0
        class_error = sum(class_errors) / max(1, len(class_errors))
        return sample_error + 2.0 * class_error + coverage_penalty

    rng = random.Random(seed)
    candidates: list[set[str]] = []
    shuffled = list(videos)
    rng.shuffle(shuffled)
    candidates.append(set(shuffled[:validation_count]))
    for _ in range(max(500, len(videos) * 40)):
        candidates.append(set(rng.sample(videos, validation_count)))
    validation_videos = min(candidates, key=lambda selected: (split_score(selected), sorted(selected)))
    train = [sample for sample in samples if str(sample.get("video_id", "")) not in validation_videos]
    validation = [sample for sample in samples if str(sample.get("video_id", "")) in validation_videos]
    return train, validation, sorted(validation_videos)


def _standardize(train_x: np.ndarray, other_x: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    scaled_train = (train_x - mean) / std
    scaled_other = None if other_x is None else (other_x - mean) / std
    return scaled_train, scaled_other, mean, std


def _class_weights(labels: np.ndarray, mode: str) -> np.ndarray:
    weights = np.ones(len(ACTIONS), dtype=np.float64)
    if mode == "none":
        return weights
    counts = np.bincount(labels, minlength=len(ACTIONS)).astype(np.float64)
    present = counts > 0
    if not np.any(present):
        return weights
    balanced = labels.size / (max(1, int(present.sum())) * np.maximum(counts, 1.0))
    if mode == "sqrt":
        balanced = np.sqrt(balanced)
    weights[present] = balanced[present]
    return weights


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def _train_softmax(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    l2: float,
    balance: str,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    weights = np.zeros((train_x.shape[1], len(ACTIONS)), dtype=np.float64)
    bias = np.zeros(len(ACTIONS), dtype=np.float64)
    one_hot = np.eye(len(ACTIONS), dtype=np.float64)[train_y]
    class_weights = _class_weights(train_y, balance)
    sample_weights = class_weights[train_y]
    sample_weights = sample_weights / max(float(sample_weights.mean()), 1e-12)
    losses: list[float] = []
    for _ in range(epochs):
        logits = train_x @ weights + bias
        probs = _softmax(logits)
        weighted_error = (probs - one_hot) * sample_weights[:, None]
        grad_w = train_x.T @ weighted_error / train_x.shape[0] + l2 * weights
        grad_b = weighted_error.mean(axis=0)
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b
        if len(losses) < 5 or (_ + 1) % max(1, epochs // 20) == 0:
            likelihood = -np.log(np.maximum(probs[np.arange(train_y.size), train_y], 1e-12))
            losses.append(float(np.mean(likelihood * sample_weights) + 0.5 * l2 * float((weights * weights).sum())))
    return weights, bias, losses


def _metrics(labels: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    counts = Counter(int(value) for value in labels.tolist())
    matrix = np.zeros((len(ACTIONS), len(ACTIONS)), dtype=int)
    for truth, prediction in zip(labels, pred):
        matrix[int(truth), int(prediction)] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, action in enumerate(ACTIONS):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        if counts.get(index, 0):
            f1_values.append(f1)
        per_class[action] = {
            "support": int(counts.get(index, 0)),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    return {
        "accuracy": round(float((labels == pred).mean()) if labels.size else 0.0, 6),
        "macro_f1": round(float(np.mean(f1_values)) if f1_values else 0.0, 6),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def train_policy(
    dataset_path: Path,
    output_path: Path,
    *,
    target: str = "oracle_action",
    validation_fraction: float = 0.20,
    seed: int = 7,
    epochs: int = 1200,
    learning_rate: float = 0.05,
    l2: float = 0.001,
    balance: str = "balanced",
) -> dict[str, Any]:
    samples = read_jsonl(dataset_path)
    train_samples, validation_samples, validation_videos = _split_by_video(samples, validation_fraction, seed)
    feature_names = _feature_names(samples)
    train_x, train_y = _vectorize(train_samples, feature_names, target)
    validation_x = validation_y = None
    if validation_samples:
        validation_x, validation_y = _vectorize(validation_samples, feature_names, target)
    train_x_scaled, validation_x_scaled, mean, std = _standardize(train_x, validation_x)
    weights, bias, losses = _train_softmax(
        train_x_scaled,
        train_y,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        balance=balance,
    )
    train_pred = np.argmax(_softmax(train_x_scaled @ weights + bias), axis=1)
    validation_metrics: dict[str, Any] | None = None
    if validation_x_scaled is not None and validation_y is not None:
        validation_pred = np.argmax(_softmax(validation_x_scaled @ weights + bias), axis=1)
        validation_metrics = _metrics(validation_y, validation_pred)
    model = {
        "version": 1,
        "model_type": "softmax_regression",
        "target": target,
        "actions": ACTIONS,
        "feature_names": feature_names,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "weights": weights.tolist(),
        "bias": bias.tolist(),
        "training": {
            "dataset": str(dataset_path),
            "samples": len(samples),
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "validation_videos": validation_videos,
            "seed": seed,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "l2": l2,
            "class_balance": balance,
            "loss_tail": losses[-5:],
        },
        "train_metrics": _metrics(train_y, train_pred),
        "validation_metrics": validation_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "dataset": str(dataset_path),
        "output": str(output_path),
        "target": target,
        "samples": len(samples),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "class_counts": {action: int(sum(1 for sample in samples if sample.get(target) == action)) for action in ACTIONS},
        "train_metrics": model["train_metrics"],
        "validation_metrics": validation_metrics,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a learned cognitive policy from policy JSONL samples.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target", choices=["oracle_action", "teacher_action"], default="oracle_action")
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.001)
    parser.add_argument("--balance", choices=["balanced", "sqrt", "none"], default="balanced")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_policy(
        args.dataset,
        args.output,
        target=args.target,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        balance=args.balance,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
