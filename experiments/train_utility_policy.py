"""Train a linear utility policy from counterfactual action utilities."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.train_learned_policy import ACTIONS, DEFAULT_FEATURES, _metrics, _split_by_video


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


def _vectorize(samples: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows: list[list[float]] = []
    utility_rows: list[list[float]] = []
    labels: list[int] = []
    for sample in samples:
        utilities = sample.get("utility_by_action", {}) or {}
        if not all(action in utilities for action in ACTIONS):
            continue
        features = sample.get("features", {}) or {}
        x_rows.append([_as_float(features.get(name, 0.0)) for name in feature_names])
        utility_rows.append([_as_float(utilities.get(action, 0.0)) for action in ACTIONS])
        labels.append(ACTIONS.index(max(ACTIONS, key=lambda action: utilities[action])))
    if not x_rows:
        raise ValueError("No usable utility samples found")
    return (
        np.asarray(x_rows, dtype=np.float64),
        np.asarray(utility_rows, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
    )


def _standardize(train_x: np.ndarray, other_x: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    train_scaled = (train_x - mean) / std
    other_scaled = None if other_x is None else (other_x - mean) / std
    return train_scaled, other_scaled, mean, std


def _sample_weights(labels: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones(labels.size, dtype=np.float64)
    counts = np.bincount(labels, minlength=len(ACTIONS)).astype(np.float64)
    present = counts > 0
    weights_by_class = np.ones(len(ACTIONS), dtype=np.float64)
    balanced = labels.size / (max(1, int(present.sum())) * np.maximum(counts, 1.0))
    if mode == "sqrt":
        balanced = np.sqrt(balanced)
    weights_by_class[present] = balanced[present]
    weights = weights_by_class[labels]
    return weights / max(float(weights.mean()), 1e-12)


def _fit_ridge(train_x: np.ndarray, train_y: np.ndarray, labels: np.ndarray, l2: float, balance: str) -> tuple[np.ndarray, np.ndarray]:
    weights = _sample_weights(labels, balance)
    design = np.concatenate([train_x, np.ones((train_x.shape[0], 1), dtype=np.float64)], axis=1)
    weighted_design = design * weights[:, None]
    penalty = np.eye(design.shape[1], dtype=np.float64) * l2
    penalty[-1, -1] = 0.0
    coef = np.linalg.pinv(design.T @ weighted_design + penalty) @ (design.T @ (train_y * weights[:, None]))
    return coef[:-1, :], coef[-1, :]


def _regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - target
    return {
        "mae": round(float(np.mean(np.abs(error))), 6),
        "rmse": round(float(np.sqrt(np.mean(error * error))), 6),
    }


def train_utility_policy(
    dataset_path: Path,
    output_path: Path,
    *,
    validation_fraction: float = 0.20,
    seed: int = 7,
    l2: float = 0.01,
    balance: str = "sqrt",
) -> dict[str, Any]:
    samples = read_jsonl(dataset_path)
    train_samples, validation_samples, validation_videos = _split_by_video(
        samples, validation_fraction, seed, target="utility_action"
    )
    feature_names = _feature_names(samples)
    train_x, train_y, train_labels = _vectorize(train_samples, feature_names)
    validation_x = validation_y = validation_labels = None
    if validation_samples:
        validation_x, validation_y, validation_labels = _vectorize(validation_samples, feature_names)
    train_x_scaled, validation_x_scaled, mean, std = _standardize(train_x, validation_x)
    weights, bias = _fit_ridge(train_x_scaled, train_y, train_labels, l2, balance)

    train_scores = train_x_scaled @ weights + bias
    train_pred = np.argmax(train_scores, axis=1)
    validation_metrics: dict[str, Any] | None = None
    if validation_x_scaled is not None and validation_y is not None and validation_labels is not None:
        validation_scores = validation_x_scaled @ weights + bias
        validation_pred = np.argmax(validation_scores, axis=1)
        validation_metrics = {
            **_metrics(validation_labels, validation_pred),
            "utility_regression": _regression_metrics(validation_y, validation_scores),
        }

    model = {
        "version": 1,
        "model_type": "linear_utility_regression",
        "target": "utility_by_action",
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
            "l2": l2,
            "sample_balance": balance,
        },
        "train_metrics": {
            **_metrics(train_labels, train_pred),
            "utility_regression": _regression_metrics(train_y, train_scores),
        },
        "validation_metrics": validation_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts = Counter(ACTIONS[int(value)] for value in np.concatenate([
        train_labels,
        validation_labels if validation_labels is not None else np.asarray([], dtype=np.int64),
    ]).tolist())
    summary = {
        "dataset": str(dataset_path),
        "output": str(output_path),
        "samples": len(samples),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "utility_action_counts": {action: counts.get(action, 0) for action in ACTIONS},
        "train_metrics": model["train_metrics"],
        "validation_metrics": validation_metrics,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a utility-regression cognitive policy.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--balance", choices=["balanced", "sqrt", "none"], default="sqrt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_utility_policy(
        args.dataset,
        args.output,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        l2=args.l2,
        balance=args.balance,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
