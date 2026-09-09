"""Train a compact nonlinear MLP utility policy.

The exported model is a JSON artifact that is evaluated by NumPy at runtime, so
the online experiment path does not need PyTorch. PyTorch is only required for
this training command.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.train_learned_policy import ACTIONS, _metrics, _split_by_video
from experiments.train_utility_policy import (
    _feature_names,
    _regression_metrics,
    _sample_weights,
    _standardize,
    _vectorize,
    read_jsonl,
)


def _parse_hidden_sizes(value: str) -> list[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--hidden-sizes must contain positive integers, e.g. 32,16")
    return sizes


def _import_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required for MLP training. Install the train dependencies or run on the server environment.") from exc
    return torch, nn, DataLoader, TensorDataset


def _make_model(nn: Any, input_dim: int, output_dim: int, hidden_sizes: list[int], dropout: float, activation: str):
    layers: list[Any] = []
    current = input_dim
    activation_layer = nn.GELU if activation == "gelu" else nn.ReLU
    for size in hidden_sizes:
        layers.append(nn.Linear(current, size))
        layers.append(activation_layer())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        current = size
    layers.append(nn.Linear(current, output_dim))
    return nn.Sequential(*layers)


def _predict_numpy(model: Any, torch: Any, x: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            tensor = torch.tensor(x[start:start + batch_size], dtype=torch.float32)
            chunks.append(model(tensor).cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, len(ACTIONS)), dtype=np.float64)


def _export_layers(model: Any, activation: str) -> list[dict[str, Any]]:
    layers: list[dict[str, Any]] = []
    linear_layers = [module for module in model if module.__class__.__name__ == "Linear"]
    for index, layer in enumerate(linear_layers):
        is_last = index == len(linear_layers) - 1
        layers.append({
            "weights": layer.weight.detach().cpu().numpy().T.tolist(),
            "bias": layer.bias.detach().cpu().numpy().tolist(),
            "activation": "linear" if is_last else activation,
        })
    return layers


def _ranking_loss(torch: Any, prediction: Any, labels: Any, sample_weights: Any, margin: float) -> Any:
    best = prediction.gather(1, labels[:, None])
    raw = torch.relu(margin - best + prediction)
    mask = torch.ones_like(raw)
    mask.scatter_(1, labels[:, None], 0.0)
    return ((raw * mask).sum(dim=1) / max(1, prediction.shape[1] - 1) * sample_weights).mean()


def train_mlp_utility_policy(
    dataset_path: Path,
    output_path: Path,
    *,
    validation_fraction: float = 0.20,
    seed: int = 7,
    hidden_sizes: list[int] | None = None,
    activation: str = "relu",
    dropout: float = 0.10,
    epochs: int = 500,
    batch_size: int = 128,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    balance: str = "sqrt",
    ranking_weight: float = 0.25,
    ranking_margin: float = 0.15,
    early_stopping_patience: int = 50,
) -> dict[str, Any]:
    torch, nn, DataLoader, TensorDataset = _import_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    hidden_sizes = hidden_sizes or [32, 16]

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
    weights = _sample_weights(train_labels, balance)

    train_dataset = TensorDataset(
        torch.tensor(train_x_scaled, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long),
        torch.tensor(weights, dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model = _make_model(nn, train_x_scaled.shape[1], len(ACTIONS), hidden_sizes, dropout, activation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses: list[float] = []
    best_state = None
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    validation_tensors = None
    if validation_x_scaled is not None and validation_y is not None and validation_labels is not None:
        validation_weights = _sample_weights(validation_labels, balance)
        validation_tensors = (
            torch.tensor(validation_x_scaled, dtype=torch.float32),
            torch.tensor(validation_y, dtype=torch.float32),
            torch.tensor(validation_labels, dtype=torch.long),
            torch.tensor(validation_weights, dtype=torch.float32),
        )
    for epoch in range(epochs):
        model.train()
        running = 0.0
        count = 0
        for batch_x, batch_y, batch_labels, batch_weights in loader:
            optimizer.zero_grad()
            prediction = model(batch_x)
            mse = (((prediction - batch_y) ** 2).mean(dim=1) * batch_weights).mean()
            rank = _ranking_loss(torch, prediction, batch_labels, batch_weights, ranking_margin)
            loss = mse + ranking_weight * rank
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu()) * batch_x.shape[0]
            count += batch_x.shape[0]
        if len(losses) < 5 or (epoch + 1) % max(1, epochs // 20) == 0:
            losses.append(round(running / max(1, count), 8))
        if validation_tensors is not None:
            model.eval()
            with torch.no_grad():
                validation_prediction = model(validation_tensors[0])
                validation_mse = (
                    ((validation_prediction - validation_tensors[1]) ** 2).mean(dim=1)
                    * validation_tensors[3]
                ).mean()
                validation_rank = _ranking_loss(
                    torch,
                    validation_prediction,
                    validation_tensors[2],
                    validation_tensors[3],
                    ranking_margin,
                )
                validation_loss = float((validation_mse + ranking_weight * validation_rank).cpu())
            if validation_loss < best_validation_loss - 1e-6:
                best_validation_loss = validation_loss
                best_epoch = epoch + 1
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                stale_epochs = 0
            else:
                stale_epochs += 1
                if early_stopping_patience > 0 and stale_epochs >= early_stopping_patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)

    train_scores = _predict_numpy(model, torch, train_x_scaled, batch_size)
    train_pred = np.argmax(train_scores, axis=1)
    validation_metrics: dict[str, Any] | None = None
    if validation_x_scaled is not None and validation_y is not None and validation_labels is not None:
        validation_scores = _predict_numpy(model, torch, validation_x_scaled, batch_size)
        validation_pred = np.argmax(validation_scores, axis=1)
        validation_metrics = {
            **_metrics(validation_labels, validation_pred),
            "utility_regression": _regression_metrics(validation_y, validation_scores),
        }

    model_payload = {
        "version": 1,
        "model_type": "mlp_utility_regression",
        "target": "utility_by_action",
        "actions": ACTIONS,
        "feature_names": feature_names,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "layers": _export_layers(model, activation),
        "training": {
            "dataset": str(dataset_path),
            "samples": len(samples),
            "train_samples": len(train_samples),
            "validation_samples": len(validation_samples),
            "validation_videos": validation_videos,
            "seed": seed,
            "hidden_sizes": hidden_sizes,
            "activation": activation,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "sample_balance": balance,
            "ranking_weight": ranking_weight,
            "ranking_margin": ranking_margin,
            "early_stopping_patience": early_stopping_patience,
            "epochs_completed": epoch + 1,
            "best_epoch": best_epoch,
            "best_validation_loss": None if best_state is None else round(best_validation_loss, 8),
            "loss_tail": losses[-5:],
        },
        "train_metrics": {
            **_metrics(train_labels, train_pred),
            "utility_regression": _regression_metrics(train_y, train_scores),
        },
        "validation_metrics": validation_metrics,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "dataset": str(dataset_path),
        "output": str(output_path),
        "samples": len(samples),
        "train_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "train_metrics": model_payload["train_metrics"],
        "validation_metrics": validation_metrics,
        "training": model_payload["training"],
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact nonlinear MLP utility policy.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hidden-sizes", default="32,16")
    parser.add_argument("--activation", choices=["relu", "gelu"], default="relu")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--balance", choices=["balanced", "sqrt", "none"], default="sqrt")
    parser.add_argument("--ranking-weight", type=float, default=0.25)
    parser.add_argument("--ranking-margin", type=float, default=0.15)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = train_mlp_utility_policy(
        args.dataset,
        args.output,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        hidden_sizes=_parse_hidden_sizes(args.hidden_sizes),
        activation=args.activation,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        balance=args.balance,
        ranking_weight=args.ranking_weight,
        ranking_margin=args.ranking_margin,
        early_stopping_patience=args.early_stopping_patience,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
