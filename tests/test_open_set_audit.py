import inspect

import numpy as np
import pytest

from contextguard.open_set_audit import (
    calibrate_thresholds,
    operating_point,
    partition_cases,
)


def test_held_data_can_never_affect_thresholds():
    assert tuple(inspect.signature(calibrate_thresholds).parameters) == (
        "source_scores",
        "targets",
    )
    source = {
        "max_softmax": np.array([0.2, 0.4, 0.6, 0.8]),
        "energy": np.array([-2.0, -1.0, 0.0, 1.0]),
        "prototype_distance": np.array([-4.0, -3.0, -2.0, -1.0]),
    }
    first = calibrate_thresholds(source)
    # Arbitrarily changing a held-only array cannot enter the calibration API.
    held = np.array([-1e9, 1e9])
    held[:] = 7.0
    assert calibrate_thresholds(source) == first


def test_deleted_user_assigned_retained_identity_is_false_acceptance():
    result = operating_point(
        threshold=0.5,
        deleted_scores=np.array([0.9]),
        retained_scores=np.array([0.8]),
        deleted_predictions=np.array([3]),
        retained_predictions=np.array([3]),
        retained_true_labels=np.array([3]),
        deleted_assigned_confidence=np.array([0.9]),
    )
    assert result["deleted_false_acceptance_rate"] == 1.0
    assert result["deleted_rejection_rate"] == 0.0
    assert result["accepted_deleted_identity_counts"] == {"3": 1}


def test_development_cases_never_enter_primary_aggregates():
    cases = [
        {
            "case_id": f"case-{index}",
            "development_case": index < 12,
        }
        for index in range(126)
    ]
    development, primary = partition_cases(cases)
    assert len(development) == 12
    assert len(primary) == 114
    assert not ({case["case_id"] for case in development} & {case["case_id"] for case in primary})


def test_threshold_meets_empirical_source_tar():
    values = np.arange(100, dtype=float)
    source = {name: values for name in ("max_softmax", "energy", "prototype_distance")}
    calibrated = calibrate_thresholds(source)
    for score in calibrated.values():
        assert score["tar_90"]["source_retained_tar_achieved"] >= 0.90
        assert score["tar_95"]["source_retained_tar_achieved"] >= 0.95
        assert score["tar_99"]["source_retained_tar_achieved"] >= 0.99
