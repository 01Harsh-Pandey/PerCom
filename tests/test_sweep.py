from pathlib import Path

import pytest

from contextguard.data import Trace, make_protocol
from contextguard.sweep import (
    EXPECTED_CASE_COUNT,
    SweepCase,
    aggregate_records,
    assert_matrix_complete,
    build_matrix,
    frozen_unsir_config,
    split_integrity,
    training_pool_signature,
    verify_original_reuse,
)


def synthetic_records(users=tuple(f"{index:03d}" for index in range(1, 15))):
    return tuple(
        Trace(
            path=Path(f"/{user}/{condition}{index}.mat"),
            user=user,
            label=label,
            condition=condition,
            sample_index=index,
            source_folder="synthetic",
        )
        for label, user in enumerate(users)
        for condition in ("a", "b", "c")
        for index in range(20)
    )


def test_split_integrity_and_no_held_leakage():
    protocol = make_protocol(synthetic_records(), "001", "c")
    checks = split_integrity(protocol)
    assert checks["passed"] is True
    assert checks["pairwise_disjoint"] is True
    assert checks["held_absent_from_source"] is True
    assert checks["base_train_partition_exact"] is True


def test_matrix_is_complete_and_rejects_missing_case():
    users = tuple(f"{index:03d}" for index in range(1, 15))
    matrix = build_matrix(users)
    assert len(matrix) == EXPECTED_CASE_COUNT == 126
    assert len({case.case_id for case in matrix}) == 126
    with pytest.raises(ValueError, match="Incomplete sweep matrix"):
        assert_matrix_complete(matrix[:-1], users, ("a", "b", "c"), (1, 2, 3))


def test_candidate_15_configuration_is_frozen():
    config = frozen_unsir_config(seed=2, workers=8)
    assert config.impair_learning_rate == 0.003
    assert config.impair_epochs == 1
    assert config.repair_learning_rate == 0.001
    assert config.repair_epochs == 3
    assert config.noise_learning_rate == 0.1
    assert config.noise_steps == 40
    assert config.noise_samples == 16
    assert config.noise_l2 == 0.1
    assert config.retain_per_class == 10
    assert config.batch_size == 32
    assert config.seed == 2


def test_original_pool_reuse_is_valid_by_held_condition_and_seed():
    records = synthetic_records()
    users = tuple(f"{index:03d}" for index in range(1, 15))
    for held in ("a", "b", "c"):
        expected = verify_original_reuse(records, users, held)
        for user in users:
            protocol = make_protocol(records, user, held)
            assert training_pool_signature(protocol.base_train) == expected


def _aggregate_case(user, held, seed, value, eligible=True, success=True):
    return {
        "case_id": SweepCase(user, held, seed).case_id,
        "forget_user": user,
        "held_condition": held,
        "seed": seed,
        "status": "completed",
        "eligible": eligible,
        "success": success,
        "failure_reasons": [] if success else ["synthetic_failure"],
        "metrics": {"unsir": {"retain_accuracy": value}},
        "base_gate": {
            "original_forget_validation_accuracy": 1.0,
            "original_retain_validation_accuracy": 1.0,
        },
        "frozen_gate": {
            "unsir_forget_validation_accuracy": 0.0,
            "unsir_retain_validation_accuracy": value,
            "retain_validation_drop": 1.0 - value,
        },
    }


def test_aggregation_statistics_and_grouping():
    records = [
        _aggregate_case("001", "a", 1, 0.8),
        _aggregate_case("001", "b", 2, 1.0),
        _aggregate_case("002", "a", 1, 0.6, success=False),
    ]
    aggregate = aggregate_records(records)
    metric = aggregate["overall"]["metrics"]["metrics.unsir.retain_accuracy"]
    assert metric["count"] == 3
    assert metric["mean"] == pytest.approx(0.8)
    assert metric["median"] == pytest.approx(0.8)
    assert metric["sample_std"] == pytest.approx(0.2)
    assert aggregate["overall"]["eligible_success_rate"] == pytest.approx(2 / 3)
    assert set(aggregate["by_user"]) == {"001", "002"}
    assert set(aggregate["by_held_condition"]) == {"a", "b"}
    assert set(aggregate["by_seed"]) == {"1", "2"}
    assert aggregate["failure_map"][0]["case_id"] == records[2]["case_id"]
