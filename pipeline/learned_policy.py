"""Inference helper for the learned cognitive policy."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class LearnedPolicyModel:
    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.model_type = str(payload.get("model_type", "linear_utility_regression"))
        self.actions = list(payload.get("actions", []))
        self.feature_names = list(payload.get("feature_names", []))
        self.mean = np.asarray(payload.get("feature_mean", []), dtype=np.float64)
        self.std = np.asarray(payload.get("feature_std", []), dtype=np.float64)
        self.weights = np.asarray(payload.get("weights", []), dtype=np.float64)
        self.bias = np.asarray(payload.get("bias", []), dtype=np.float64)
        self.layers = list(payload.get("layers", []))
        if not self.actions or not self.feature_names:
            raise ValueError("Learned policy payload is missing actions or feature_names")
        if self.layers:
            self._validate_layers()
        else:
            if self.weights.shape != (len(self.feature_names), len(self.actions)):
                raise ValueError("Learned policy weight shape does not match feature/action metadata")
            if self.bias.shape != (len(self.actions),):
                raise ValueError("Learned policy bias shape does not match actions")

    def _validate_layers(self) -> None:
        expected_in = len(self.feature_names)
        for index, layer in enumerate(self.layers):
            weight = np.asarray(layer.get("weights", []), dtype=np.float64)
            bias = np.asarray(layer.get("bias", []), dtype=np.float64)
            if weight.ndim != 2 or weight.shape[0] != expected_in:
                raise ValueError(f"MLP layer {index} weight shape does not match previous width")
            if bias.shape != (weight.shape[1],):
                raise ValueError(f"MLP layer {index} bias shape does not match output width")
            expected_in = weight.shape[1]
        if expected_in != len(self.actions):
            raise ValueError("MLP final layer width does not match actions")

    @classmethod
    def load(cls, path: str | Path) -> "LearnedPolicyModel":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def vectorize(self, features: dict[str, Any]) -> np.ndarray:
        values = []
        for name in self.feature_names:
            value = features.get(name, 0.0)
            if isinstance(value, bool):
                value = float(value)
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                values.append(0.0)
        x = np.asarray(values, dtype=np.float64)
        return (x - self.mean) / np.maximum(self.std, 1e-8)

    @staticmethod
    def _activate(values: np.ndarray, activation: str) -> np.ndarray:
        if activation == "linear":
            return values
        if activation == "gelu":
            return 0.5 * values * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (values + 0.044715 * np.power(values, 3))))
        return np.maximum(values, 0.0)

    def predict_scores(self, features: dict[str, Any]) -> dict[str, float]:
        x = self.vectorize(features)
        if self.layers:
            values = x
            for layer in self.layers:
                weight = np.asarray(layer.get("weights", []), dtype=np.float64)
                bias = np.asarray(layer.get("bias", []), dtype=np.float64)
                activation = str(layer.get("activation", "relu"))
                values = self._activate(values @ weight + bias, activation)
            logits = values
        else:
            logits = x @ self.weights + self.bias
        return {action: round(float(score), 8) for action, score in zip(self.actions, logits)}

    def predict_proba(self, features: dict[str, Any]) -> dict[str, float]:
        scores = self.predict_scores(features)
        logits = np.asarray([scores[action] for action in self.actions], dtype=np.float64)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        probs = exp / max(float(exp.sum()), 1e-12)
        return {action: round(float(prob), 8) for action, prob in zip(self.actions, probs)}

    def predict(self, features: dict[str, Any]) -> tuple[str, float, dict[str, float]]:
        probabilities = self.predict_proba(features)
        action = max(probabilities, key=probabilities.get)
        return action, probabilities[action], probabilities
