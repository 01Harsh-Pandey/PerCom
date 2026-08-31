from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from .data import Trace
from .train import TrainConfig, make_loader


@torch.inference_mode()
def _predictions_and_features(
    model: nn.Module,
    records: Sequence[Trace],
    loader_config: TrainConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = make_loader(records, loader_config, shuffle=False)
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    features: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        logits, embedding = model(x, return_features=True)
        labels.append(batch["y"].numpy())
        predictions.append(logits.argmax(dim=1).cpu().numpy())
        features.append(embedding.cpu().numpy())
    return np.concatenate(labels), np.concatenate(predictions), np.concatenate(features)


def classification_accuracy(
    model: nn.Module,
    records: Sequence[Trace],
    loader_config: TrainConfig,
    device: torch.device,
) -> float:
    if not records:
        return float("nan")
    labels, predictions, _ = _predictions_and_features(
        model, records, loader_config, device
    )
    return float(np.mean(labels == predictions))


def contextual_probe(
    model: nn.Module,
    train_records: Sequence[Trace],
    test_records: Sequence[Trace],
    forget_label: int,
    loader_config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """Link a source-context identity to its held-condition traces."""

    train_y, _, train_x = _predictions_and_features(
        model, train_records, loader_config, device
    )
    test_y, _, test_x = _predictions_and_features(
        model, test_records, loader_config, device
    )
    train_binary = (train_y == forget_label).astype(np.int64)
    test_binary = (test_y == forget_label).astype(np.int64)
    probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=loader_config.seed,
        ),
    )
    probe.fit(train_x, train_binary)
    score = probe.predict_proba(test_x)[:, 1]
    prediction = (score >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(test_binary, score)),
        "balanced_accuracy": float(
            balanced_accuracy_score(test_binary, prediction)
        ),
        "train_examples": float(len(train_binary)),
        "test_examples": float(len(test_binary)),
    }


def evaluate_method(
    model: nn.Module,
    protocol,
    loader_config: TrainConfig,
    device: torch.device,
) -> dict[str, object]:
    return {
        "forget_seen_label_accuracy": classification_accuracy(
            model, protocol.forget_seen_test, loader_config, device
        ),
        "forget_held_label_accuracy": classification_accuracy(
            model, protocol.forget_held_test, loader_config, device
        ),
        "retain_accuracy": classification_accuracy(
            model, protocol.retain_test, loader_config, device
        ),
        "contextual_probe": contextual_probe(
            model,
            protocol.probe_train,
            protocol.probe_test,
            protocol.forget_label,
            loader_config,
            device,
        ),
    }

