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
def labels_logits_features(
    model: nn.Module,
    records: Sequence[Trace],
    loader_config: TrainConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        raise ValueError("Cannot evaluate an empty record sequence")
    # Evaluation loaders are consumed once. Persistent multiprocessing workers
    # would outlive each short-lived loader and accumulate file descriptors
    # across calibration candidates and held-context audits.
    loader = make_loader(
        records,
        loader_config,
        shuffle=False,
        persistent_workers=False,
    )
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    features: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        batch_logits, embedding = model(x, return_features=True)
        labels.append(batch["y"].numpy())
        logits.append(batch_logits.cpu().numpy())
        features.append(embedding.cpu().numpy())
    return np.concatenate(labels), np.concatenate(logits), np.concatenate(features)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def classification_accuracy(
    model: nn.Module,
    records: Sequence[Trace],
    loader_config: TrainConfig,
    device: torch.device,
) -> float:
    if not records:
        return float("nan")
    labels, logits, _ = labels_logits_features(model, records, loader_config, device)
    return float(np.mean(labels == logits.argmax(axis=1)))


def target_confidence(
    model: nn.Module,
    records: Sequence[Trace],
    forget_label: int,
    loader_config: TrainConfig,
    device: torch.device,
) -> float:
    _, logits, _ = labels_logits_features(model, records, loader_config, device)
    return float(_softmax(logits)[:, forget_label].mean())


def loss_membership_auc(
    model: nn.Module,
    former_member_records: Sequence[Trace],
    nonmember_records: Sequence[Trace],
    forget_label: int,
    loader_config: TrainConfig,
    device: torch.device,
) -> float:
    """Simple loss-based audit of former-member distinguishability."""

    _, member_logits, _ = labels_logits_features(
        model, former_member_records, loader_config, device
    )
    _, nonmember_logits, _ = labels_logits_features(
        model, nonmember_records, loader_config, device
    )
    scores = np.concatenate(
        [
            np.log(_softmax(member_logits)[:, forget_label] + 1e-12),
            np.log(_softmax(nonmember_logits)[:, forget_label] + 1e-12),
        ]
    )
    labels = np.concatenate(
        [np.ones(len(member_logits)), np.zeros(len(nonmember_logits))]
    )
    return float(roc_auc_score(labels, scores))


def reidentification_probe(
    model: nn.Module,
    train_records: Sequence[Trace],
    test_records: Sequence[Trace],
    forget_label: int,
    loader_config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """Measure generic identity separability, not residual training influence."""

    train_y, _, train_x = labels_logits_features(
        model, train_records, loader_config, device
    )
    test_y, _, test_x = labels_logits_features(
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


def predictive_equivalence(
    model: nn.Module,
    retrained_model: nn.Module,
    records: Sequence[Trace],
    loader_config: TrainConfig,
    device: torch.device,
    drop_label: int | None = None,
) -> dict[str, float]:
    """Compare an unlearned model directly with exact retraining."""

    _, model_logits, _ = labels_logits_features(model, records, loader_config, device)
    _, reference_logits, _ = labels_logits_features(
        retrained_model, records, loader_config, device
    )
    p = np.clip(_softmax(model_logits), 1e-12, 1.0)
    q = np.clip(_softmax(reference_logits), 1e-12, 1.0)
    if drop_label is not None:
        p = np.delete(p, drop_label, axis=1)
        q = np.delete(q, drop_label, axis=1)
        p = p / p.sum(axis=1, keepdims=True)
        q = q / q.sum(axis=1, keepdims=True)
    midpoint = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / midpoint), axis=1)
    js += 0.5 * np.sum(q * np.log(q / midpoint), axis=1)
    return {
        "mean_js_divergence": float(js.mean()),
        "prediction_agreement": float(
            np.mean(p.argmax(axis=1) == q.argmax(axis=1))
        ),
    }


def evaluate_method(
    model: nn.Module,
    protocol,
    loader_config: TrainConfig,
    device: torch.device,
) -> dict[str, object]:
    return {
        "forget_validation_label_accuracy": classification_accuracy(
            model, protocol.forget_validation, loader_config, device
        ),
        "retain_validation_accuracy": classification_accuracy(
            model, protocol.retain_validation, loader_config, device
        ),
        "forget_seen_label_accuracy": classification_accuracy(
            model, protocol.forget_seen_test, loader_config, device
        ),
        "forget_held_label_accuracy": classification_accuracy(
            model, protocol.forget_held_test, loader_config, device
        ),
        "retain_seen_accuracy": classification_accuracy(
            model, protocol.retain_seen_test, loader_config, device
        ),
        "retain_held_accuracy": classification_accuracy(
            model, protocol.retain_held_test, loader_config, device
        ),
        "retain_accuracy": classification_accuracy(
            model, protocol.retain_test, loader_config, device
        ),
        "forget_train_target_confidence": target_confidence(
            model, protocol.forget_train, protocol.forget_label, loader_config, device
        ),
        "forget_held_target_confidence": target_confidence(
            model,
            protocol.forget_held_test,
            protocol.forget_label,
            loader_config,
            device,
        ),
        "loss_membership_auc": loss_membership_auc(
            model,
            protocol.forget_train,
            protocol.forget_held_test,
            protocol.forget_label,
            loader_config,
            device,
        ),
        "reidentification_probe_diagnostic": reidentification_probe(
            model,
            protocol.probe_train,
            protocol.probe_test,
            protocol.forget_label,
            loader_config,
            device,
        ),
    }
