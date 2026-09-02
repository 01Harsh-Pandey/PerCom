from pathlib import Path

import pytest

from contextguard.data import Trace, make_protocol
from contextguard.fineretainkd_sweep import (
    DEVELOPMENT_USERS,
    MAX_REPAIR_EPOCHS,
    METHOD,
    assert_frozen_config,
    is_development_case,
    matched_subset_aggregate,
    require_selection_frozen,
)
from contextguard.utility_pilot import context_balanced_retained


def synthetic_records():
    users = tuple(f"{index:03d}" for index in range(1, 14)) + ("015",)
    return tuple(
        Trace(Path(f"/{user}/{condition}{index}.mat"), user, label, condition, index, "synthetic")
        for label, user in enumerate(users)
        for condition in ("a", "b", "c")
        for index in range(20)
    )


def test_frozen_fineretainkd_configuration():
    config = assert_frozen_config(seed=2, workers=8)
    assert config["noise_steps"] == 40
    assert config["impair_learning_rate"] == 0.003
    assert config["repair_learning_rate"] == 0.001
    assert config["maximum_repair_epochs"] == MAX_REPAIR_EPOCHS == 5
    assert config["kd_weight"] == 1.0
    assert config["damage_aware_weighting"] is False
    assert config["forgotten_example_anchor"] is False


def test_full_retained_pool_is_context_balanced():
    protocol = make_protocol(synthetic_records(), "001", "c")
    retained = context_balanced_retained(protocol.retain_train, "c")
    assert len(retained) == len(protocol.retain_train)
    counts = {(user, condition): 0 for user in {r.user for r in retained} for condition in ("a", "b")}
    for row in retained:
        counts[(row.user, row.condition)] += 1
    assert set(counts.values()) == {11}


def test_development_partition_is_exactly_twelve_cases():
    users = tuple(f"{index:03d}" for index in range(1, 14)) + ("015",)
    cases = [
        (user, held, seed)
        for user in users
        for held in ("a", "b", "c")
        for seed in (1, 2, 3)
    ]
    development = [case for case in cases if is_development_case(case[0], case[2])]
    assert len(DEVELOPMENT_USERS) == 4
    assert len(development) == 12
    assert len(cases) - len(development) == 114


def test_held_access_requires_frozen_source_selection():
    with pytest.raises(RuntimeError, match="before source selection froze"):
        require_selection_frozen({"status": "running", "held_context_used_for_selection": False})
    with pytest.raises(RuntimeError, match="leakage"):
        require_selection_frozen({"status": "selected", "held_context_used_for_selection": True})
    require_selection_frozen({"status": "selected", "held_context_used_for_selection": False})
    require_selection_frozen({"status": "abstained", "held_context_used_for_selection": False})


def _case(case_id, selected, fine_value, original_value):
    methods = {
        "original": {
            "forget_accuracy": 1.0,
            "retain_accuracy": original_value,
            "retain_accuracy_drop_from_original": 0.0,
            "js_to_exact_retraining": original_value,
        },
        "retrained": {
            "forget_accuracy": 0.0,
            "retain_accuracy": 1.0,
            "retain_accuracy_drop_from_original": original_value - 1.0,
            "js_to_exact_retraining": 0.0,
        },
    }
    if selected:
        methods[METHOD] = {
            "forget_accuracy": 0.0,
            "retain_accuracy": fine_value,
            "retain_accuracy_drop_from_original": original_value - fine_value,
            "js_to_exact_retraining": fine_value,
        }
    return {
        "case_id": case_id,
        "status": "completed",
        "selection": {"status": "selected" if selected else "abstained"},
        "evaluation": {context: methods for context in ("source", "seen", "held")},
    }


def test_matched_subset_aggregation_uses_identical_case_ids_for_every_method():
    cases = [_case("selected-a", True, 0.8, 0.9), _case("abstained-b", False, 0.1, 0.2)]
    aggregate = matched_subset_aggregate(cases)
    assert aggregate["selected_case_ids"] == ["selected-a"]
    assert aggregate["source_selection_coverage"]["rate"] == pytest.approx(0.5)
    for context in ("source", "seen", "held"):
        for metric in aggregate["metrics"][context].values():
            assert metric["matched_case_ids"] == ["selected-a"]
            assert {value["count"] for value in metric["methods"].values()} == {1}
