import hashlib

import numpy as np
import pytest
import torch

from contextguard.mechanism import (
    class_accuracy,
    confusion_matrix_counts,
    correlation,
    feature_drift,
    gradient_conflict,
    layer_parameter_drift,
    probability_redirection,
    verify_checkpoint_immutability,
    verify_checkpoint_manifest,
)


def test_confusion_matrix_and_per_class_accuracy():
    labels = np.asarray([0, 0, 1, 1, 2])
    predictions = np.asarray([0, 1, 1, 1, 0])
    matrix = confusion_matrix_counts(labels, predictions, 3)
    assert matrix.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 0]]
    assert class_accuracy(matrix, 0) == pytest.approx(0.5)
    assert class_accuracy(matrix, 1) == pytest.approx(1.0)


def test_probability_redirection_excludes_forgotten_label():
    original = np.asarray([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1]])
    unsir = np.asarray([[0.1, 0.6, 0.3], [0.2, 0.5, 0.3]])
    retrained = np.asarray([[0.0, 0.4, 0.6], [0.0, 0.5, 0.5]])
    result = probability_redirection(original, unsir, retrained, forget_label=0)
    assert result["top_redirected_label"] == 1
    assert result["top_redirected_probability"] == pytest.approx(0.55)


def test_parameter_and_feature_drift_metrics():
    reference = {"layer.weight": torch.tensor([1.0, 0.0])}
    compared = {"layer.weight": torch.tensor([0.0, 1.0])}
    layer = layer_parameter_drift(reference, compared)["layer"]
    assert layer["delta_l2"] == pytest.approx(2**0.5)
    assert layer["cosine_similarity"] == pytest.approx(0.0)
    features = feature_drift(
        np.asarray([[1.0, 0.0], [0.0, 1.0]]),
        np.asarray([[0.0, 1.0], [0.0, 1.0]]),
    )
    assert features["mean_l2"] == pytest.approx(2**0.5 / 2)
    assert features["mean_cosine_similarity"] == pytest.approx(0.5)


def test_gradient_conflict_sign_and_cosine():
    forgotten = {"layer.weight": torch.tensor([1.0, 0.0])}
    retained = {"layer.weight": torch.tensor([-1.0, 0.0])}
    result = gradient_conflict(forgotten, retained)["layer"]
    assert result["cosine_similarity"] == pytest.approx(-1.0)
    assert result["conflict"] is True


def test_correlation_correctness():
    result = correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    assert result["pearson_r"] == pytest.approx(1.0)
    assert result["spearman_rho"] == pytest.approx(1.0)


def test_checkpoint_verification_is_read_only(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"immutable checkpoint bytes")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    before = verify_checkpoint_manifest(tmp_path, {"model.pt": digest})
    verify_checkpoint_immutability(tmp_path, before)
    assert checkpoint.read_bytes() == b"immutable checkpoint bytes"
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checkpoint_manifest(tmp_path, {"model.pt": "0" * 64})
