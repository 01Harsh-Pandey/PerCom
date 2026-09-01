from pathlib import Path

import pytest
import torch

from contextguard.data import Trace
from contextguard.utility_pilot import (
    context_balanced_retained,
    masked_probabilities,
    select_source_checkpoint,
    top_drifted_users,
)


def trace(user, label, condition, index):
    return Trace(
        path=Path(f"/{user}/{condition}{index}.mat"),
        user=user,
        label=label,
        condition=condition,
        sample_index=index,
        source_folder="synthetic",
    )


def test_deleted_label_is_masked_and_distribution_renormalized():
    probabilities = masked_probabilities(
        torch.tensor([[3.0, 2.0, 1.0], [0.0, 1.0, 2.0]]), deleted_label=1
    )
    assert probabilities[:, 1].tolist() == [0.0, 0.0]
    assert probabilities.sum(dim=1).tolist() == pytest.approx([1.0, 1.0])


def test_context_balanced_sampling_uses_every_retained_example():
    records = tuple(
        trace(user, label, condition, index)
        for label, user in enumerate(("001", "002"))
        for condition in ("a", "b")
        for index in range(3)
    )
    selected = context_balanced_retained(records, held_condition="c")
    assert len(selected) == len(records)
    assert {row.path for row in selected} == {row.path for row in records}


def test_context_balanced_sampling_rejects_imbalance():
    records = (
        trace("001", 0, "a", 1),
        trace("001", 0, "b", 1),
        trace("002", 1, "a", 1),
    )
    with pytest.raises(ValueError, match="not user/context balanced"):
        context_balanced_retained(records, held_condition="c")


def test_top_three_weighting_is_deterministic():
    drift = {"001": 0.1, "002": 0.5, "003": 0.3, "004": 0.4, "005": 0.2}
    assert top_drifted_users(drift) == ("002", "004", "003")


def test_source_only_selection_ignores_held_metrics_and_uses_js_tiebreaker():
    candidates = [
        {
            "checkpoint_index": 1,
            "source_forget_accuracy": 0.0,
            "source_retain_drop": 0.04,
            "source_retained_label_js": 0.03,
            "held_accuracy": 1.0,
        },
        {
            "checkpoint_index": 2,
            "source_forget_accuracy": 0.1,
            "source_retain_drop": 0.05,
            "source_retained_label_js": 0.01,
            "held_accuracy": 0.0,
        },
    ]
    assert select_source_checkpoint(candidates)["checkpoint_index"] == 2
    candidates[0]["held_accuracy"], candidates[1]["held_accuracy"] = 0.0, 1.0
    assert select_source_checkpoint(candidates)["checkpoint_index"] == 2


def test_selection_abstains_when_no_checkpoint_is_safe():
    candidates = [
        {
            "checkpoint_index": 1,
            "source_forget_accuracy": 0.11,
            "source_retain_drop": 0.0,
            "source_retained_label_js": 0.0,
        },
        {
            "checkpoint_index": 2,
            "source_forget_accuracy": 0.0,
            "source_retain_drop": 0.051,
            "source_retained_label_js": 0.0,
        },
    ]
    assert select_source_checkpoint(candidates) is None
