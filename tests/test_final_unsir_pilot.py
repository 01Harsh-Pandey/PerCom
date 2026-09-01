import pytest
import torch

from contextguard.final_unsir_pilot import (
    MAX_REPAIR_EPOCHS,
    forgotten_anchor_loss,
    repair_checkpoint_schedule,
    require_frozen_source_selections,
)


def test_every_repair_minibatch_has_a_checkpoint_slot():
    schedule = repair_checkpoint_schedule(example_count=286, batch_size=32)
    assert len(schedule) == 45
    assert schedule[0] == (1, 1)
    assert schedule[-1] == (5, 9)
    assert len(set(schedule)) == len(schedule)
    assert len(repair_checkpoint_schedule(example_count=308, batch_size=32)) == 50


def test_repair_limit_is_exactly_five_epochs():
    assert MAX_REPAIR_EPOCHS == 5
    assert {epoch for epoch, _ in repair_checkpoint_schedule(286, 32)} == {
        1,
        2,
        3,
        4,
        5,
    }


def test_forgotten_anchor_penalizes_deleted_probability_only():
    low_deleted = torch.tensor([[-4.0, 3.0, 2.0]])
    high_deleted = torch.tensor([[4.0, 3.0, 2.0]])
    assert forgotten_anchor_loss(high_deleted, 0).item() > forgotten_anchor_loss(
        low_deleted, 0
    ).item()
    # Swapping retained identities without changing their log-sum-exp leaves
    # the deleted-label penalty unchanged; no retained identity is targeted.
    swapped = low_deleted[:, [0, 2, 1]]
    assert forgotten_anchor_loss(swapped, 0).item() == pytest.approx(
        forgotten_anchor_loss(low_deleted, 0).item()
    )


def test_held_context_requires_both_source_selections_to_be_frozen():
    with pytest.raises(RuntimeError, match="Both method variants"):
        require_frozen_source_selections(
            {"fine_retain_kd": {"status": "selected"}}
        )
    with pytest.raises(RuntimeError, match="before source selection froze"):
        require_frozen_source_selections(
            {
                "fine_retain_kd": {"status": "selected"},
                "anchored_retain_kd": {"status": "running"},
            }
        )
    require_frozen_source_selections(
        {
            "fine_retain_kd": {"status": "selected"},
            "anchored_retain_kd": {"status": "abstained"},
        }
    )
