from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev, variance
from typing import Iterable, Mapping, Sequence

import numpy as np
import scipy
import sklearn
import torch

from .data import EXPECTED_CONDITIONS, Protocol, Trace, make_protocol, scan_ntu_humanid
from .evaluate import classification_accuracy, evaluate_method, predictive_equivalence
from .model import CSILeNet
from .train import TrainConfig, save_checkpoint, seed_everything, train_model
from .unlearning import UNSIRConfig, unsir_unlearn


FROZEN_SEEDS = (1, 2, 3)
EXPECTED_USER_COUNT = 14
EXPECTED_CASE_COUNT = EXPECTED_USER_COUNT * len(EXPECTED_CONDITIONS) * len(FROZEN_SEEDS)
MIN_BASE_FORGET_VALIDATION = 0.75
MIN_BASE_RETAIN_VALIDATION = 0.85
MAX_FORGET_VALIDATION = 0.10
MAX_RETAIN_VALIDATION_DROP = 0.05


@dataclass(frozen=True)
class SweepCase:
    forget_user: str
    held_condition: str
    seed: int

    @property
    def case_id(self) -> str:
        return (
            f"user-{self.forget_user}__held-{self.held_condition}__seed-{self.seed}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen candidate-15 robustness audit over 126 cases."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="outputs/frozen-robustness-audit")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def frozen_unsir_config(seed: int, workers: int = 8) -> UNSIRConfig:
    """Return candidate 15 exactly; only the predeclared seed varies."""

    return UNSIRConfig(
        noise_samples=16,
        noise_steps=40,
        noise_learning_rate=0.1,
        noise_l2=0.1,
        retain_per_class=10,
        impair_learning_rate=0.003,
        repair_learning_rate=0.001,
        impair_epochs=1,
        repair_epochs=3,
        batch_size=32,
        workers=workers,
        seed=seed,
    )


def build_matrix(
    users: Sequence[str],
    held_conditions: Sequence[str] = EXPECTED_CONDITIONS,
    seeds: Sequence[int] = FROZEN_SEEDS,
) -> tuple[SweepCase, ...]:
    cases = tuple(
        SweepCase(user, condition, seed)
        for condition in held_conditions
        for seed in seeds
        for user in users
    )
    assert_matrix_complete(cases, users, held_conditions, seeds)
    return cases


def assert_matrix_complete(
    cases: Sequence[SweepCase],
    users: Sequence[str],
    held_conditions: Sequence[str],
    seeds: Sequence[int],
) -> None:
    expected = {
        (user, condition, seed)
        for user in users
        for condition in held_conditions
        for seed in seeds
    }
    observed = {
        (case.forget_user, case.held_condition, case.seed) for case in cases
    }
    if len(cases) != len(expected) or observed != expected:
        missing = sorted(expected - observed)
        duplicates = len(cases) - len(observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"Incomplete sweep matrix: missing={missing}, extra={extra}, "
            f"duplicate_count={duplicates}"
        )


def trace_key(record: Trace) -> tuple[str, str, int, str]:
    return (record.user, record.condition, record.sample_index, str(record.path))


def training_pool_signature(records: Iterable[Trace]) -> tuple[tuple[str, str, int, str], ...]:
    return tuple(sorted(trace_key(record) for record in records))


def split_integrity(protocol: Protocol) -> dict[str, object]:
    groups = {
        "forget_train": {trace_key(row) for row in protocol.forget_train},
        "retain_train": {trace_key(row) for row in protocol.retain_train},
        "forget_validation": {
            trace_key(row) for row in protocol.forget_validation
        },
        "retain_validation": {
            trace_key(row) for row in protocol.retain_validation
        },
        "forget_seen_test": {
            trace_key(row) for row in protocol.forget_seen_test
        },
        "forget_held_test": {
            trace_key(row) for row in protocol.forget_held_test
        },
        "retain_seen_test": {
            trace_key(row) for row in protocol.retain_seen_test
        },
        "retain_held_test": {
            trace_key(row) for row in protocol.retain_held_test
        },
    }
    source_names = (
        "forget_train",
        "retain_train",
        "forget_validation",
        "retain_validation",
    )
    held_names = ("forget_held_test", "retain_held_test")
    source_union = set().union(*(groups[name] for name in source_names))
    held_union = set().union(*(groups[name] for name in held_names))
    pairwise_disjoint = True
    names = tuple(groups)
    overlaps: dict[str, int] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = len(groups[left] & groups[right])
            if overlap:
                pairwise_disjoint = False
                overlaps[f"{left}__{right}"] = overlap
    base_train = {trace_key(row) for row in protocol.base_train}
    checks = {
        "pairwise_disjoint": pairwise_disjoint,
        "overlaps": overlaps,
        "held_absent_from_source": not bool(source_union & held_union),
        "base_train_partition_exact": base_train == source_union,
        "base_train_excludes_held_condition": all(
            row.condition != protocol.held_condition for row in protocol.base_train
        ),
        "held_test_uses_only_held_condition": all(
            row.condition == protocol.held_condition
            for row in (*protocol.forget_held_test, *protocol.retain_held_test)
        ),
        "validation_uses_only_source_conditions": all(
            row.condition != protocol.held_condition
            for row in (*protocol.forget_validation, *protocol.retain_validation)
        ),
    }
    checks["passed"] = all(
        value for key, value in checks.items() if key not in {"overlaps", "passed"}
    )
    return checks


def dataset_counts(protocol: Protocol) -> dict[str, int]:
    return {
        "base_train_pool": len(protocol.base_train),
        "retrain_pool": len(protocol.retrain),
        "forget_train": len(protocol.forget_train),
        "retain_train": len(protocol.retain_train),
        "forget_validation": len(protocol.forget_validation),
        "retain_validation": len(protocol.retain_validation),
        "forget_seen_test": len(protocol.forget_seen_test),
        "forget_held_test": len(protocol.forget_held_test),
        "retain_seen_test": len(protocol.retain_seen_test),
        "retain_held_test": len(protocol.retain_held_test),
        "retain_test": len(protocol.retain_test),
        "probe_train": len(protocol.probe_train),
        "probe_test": len(protocol.probe_test),
    }


def verify_original_reuse(
    records: Sequence[Trace], users: Sequence[str], held_condition: str
) -> tuple[tuple[str, str, int, str], ...]:
    signatures = {
        training_pool_signature(
            make_protocol(records, user, held_condition).base_train
        )
        for user in users
    }
    if len(signatures) != 1:
        raise ValueError(
            f"Original training pool depends on forgotten user for held={held_condition}"
        )
    return signatures.pop()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    temporary.replace(path)


def flatten_for_csv(
    value: object, prefix: str = "", output: dict[str, object] | None = None
) -> dict[str, object]:
    output = {} if output is None else output
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flatten_for_csv(child, child_prefix, output)
    elif isinstance(value, (list, tuple)):
        output[prefix] = json.dumps(value, sort_keys=True)
    else:
        output[prefix] = value
    return output


def write_cases(output: Path, records: Sequence[dict[str, object]]) -> None:
    write_json(output / "cases.json", list(records))
    rows = [flatten_for_csv(record) for record in records]
    fieldnames = sorted({key for row in rows for key in row})
    temporary = output / "cases.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "cases.csv")


def numeric_metrics(record: Mapping[str, object]) -> dict[str, float]:
    flattened: dict[str, object] = {}
    flatten_for_csv(record.get("metrics", {}), "metrics", flattened)
    for key in (
        "base_gate.original_forget_validation_accuracy",
        "base_gate.original_retain_validation_accuracy",
        "frozen_gate.unsir_forget_validation_accuracy",
        "frozen_gate.unsir_retain_validation_accuracy",
        "frozen_gate.retain_validation_drop",
    ):
        current: object = record
        for part in key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                current = None
                break
            current = current[part]
        flattened[key] = current
    return {
        key: float(value)
        for key, value in flattened.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    }


def summarize_values(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    count = len(values)
    sample_std = stdev(values) if count > 1 else 0.0
    sample_variance = variance(values) if count > 1 else 0.0
    margin = 1.96 * sample_std / math.sqrt(count) if count > 1 else 0.0
    center = mean(values)
    return {
        "count": count,
        "mean": center,
        "median": median(values),
        "sample_std": sample_std,
        "sample_variance": sample_variance,
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "ci95_low": center - margin,
        "ci95_high": center + margin,
        "ci95_method": "normal_approximation_mean_plus_minus_1.96_se",
    }


def aggregate_subset(records: Sequence[dict[str, object]]) -> dict[str, object]:
    completed = [record for record in records if record.get("status") == "completed"]
    metric_names = sorted(
        {key for record in completed for key in numeric_metrics(record)}
    )
    metrics: dict[str, object] = {}
    for metric_name in metric_names:
        values = [
            numeric_metrics(record)[metric_name]
            for record in completed
            if metric_name in numeric_metrics(record)
        ]
        if values:
            metrics[metric_name] = summarize_values(values)
    eligible = [record for record in records if record.get("eligible") is True]
    eligible_success = [record for record in eligible if record.get("success") is True]
    return {
        "case_count": len(records),
        "completed_count": len(completed),
        "eligible_count": len(eligible),
        "eligible_success_count": len(eligible_success),
        "eligible_success_rate": (
            len(eligible_success) / len(eligible) if eligible else None
        ),
        "metrics": metrics,
    }


def grouped_aggregates(
    records: Sequence[dict[str, object]], key: str
) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        groups.setdefault(str(record[key]), []).append(record)
    return {name: aggregate_subset(group) for name, group in sorted(groups.items())}


def aggregate_records(records: Sequence[dict[str, object]]) -> dict[str, object]:
    return {
        "overall": aggregate_subset(records),
        "eligible_only": aggregate_subset(
            [record for record in records if record.get("eligible") is True]
        ),
        "by_user": grouped_aggregates(records, "forget_user"),
        "by_held_condition": grouped_aggregates(records, "held_condition"),
        "by_seed": grouped_aggregates(records, "seed"),
        "failure_map": [
            {
                "case_id": record["case_id"],
                "forget_user": record["forget_user"],
                "held_condition": record["held_condition"],
                "seed": record["seed"],
                "eligible": record.get("eligible"),
                "success": record.get("success"),
                "failure_reasons": record.get("failure_reasons", []),
            }
            for record in records
            if record.get("success") is not True
        ],
    }


def provenance(device: torch.device) -> dict[str, object]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "primary_success_definition": (
            "Among source-validation-eligible cases only: frozen UNSIR forget "
            "validation accuracy <= 0.10 and retain validation accuracy drop <= 0.05."
        ),
        "held_context_policy": (
            "Held-context metrics are evaluated and reported only after the frozen "
            "configuration and source-validation gates are fixed. They never affect "
            "selection, tuning, eligibility, exclusion, or rerunning."
        ),
        "diagnostic_metric_policy": (
            "Reidentification probe and loss-membership metrics are non-primary diagnostics."
        ),
        "ci95_method": "normal approximation: mean +/- 1.96 * sample_std / sqrt(n)",
    }


def equivalence_metrics(
    model: torch.nn.Module,
    retrained: torch.nn.Module,
    protocol: Protocol,
    train_config: TrainConfig,
    device: torch.device,
) -> dict[str, object]:
    return {
        "forget_seen": predictive_equivalence(
            model,
            retrained,
            protocol.forget_seen_test,
            train_config,
            device,
            protocol.forget_label,
        ),
        "forget_held": predictive_equivalence(
            model,
            retrained,
            protocol.forget_held_test,
            train_config,
            device,
            protocol.forget_label,
        ),
        "retain_seen": predictive_equivalence(
            model,
            retrained,
            protocol.retain_seen_test,
            train_config,
            device,
            protocol.forget_label,
        ),
        "retain_held": predictive_equivalence(
            model,
            retrained,
            protocol.retain_held_test,
            train_config,
            device,
            protocol.forget_label,
        ),
    }


def run_case(
    case: SweepCase,
    protocol: Protocol,
    original: torch.nn.Module,
    original_checkpoint: Path,
    original_training: Mapping[str, object],
    train_config: TrainConfig,
    device: torch.device,
    output: Path,
) -> dict[str, object]:
    case_dir = output / "case-artifacts" / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    integrity = split_integrity(protocol)
    if not integrity["passed"]:
        raise ValueError(f"Split-integrity failure: {integrity}")

    # Freeze eligibility from source validation before evaluating held context.
    original_forget_validation = classification_accuracy(
        original,
        protocol.forget_validation,
        train_config,
        device,
    )
    original_retain_validation = classification_accuracy(
        original,
        protocol.retain_validation,
        train_config,
        device,
    )
    eligible = (
        original_forget_validation >= MIN_BASE_FORGET_VALIDATION
        and original_retain_validation >= MIN_BASE_RETAIN_VALIDATION
    )

    config = frozen_unsir_config(case.seed, train_config.workers)
    selected, unsir_training = unsir_unlearn(
        original, protocol.retain_train, protocol.forget_label, config, device
    )
    retrained, retrained_training = train_model(
        CSILeNet(original.num_classes), protocol.retrain, train_config, device
    )

    selected_checkpoint = case_dir / "unsir-selected.pt"
    retrained_checkpoint = case_dir / "retrained.pt"
    save_checkpoint(
        selected_checkpoint,
        selected,
        {
            "frozen_config": asdict(config),
            "training": unsir_training,
            "case": asdict(case),
        },
    )
    save_checkpoint(
        retrained_checkpoint,
        retrained,
        {"training": retrained_training, "case": asdict(case)},
    )

    # Freeze the selected method's source-validation gates before any held
    # metric is evaluated. Held values cannot affect control flow.
    unsir_forget_validation = classification_accuracy(
        selected,
        protocol.forget_validation,
        train_config,
        device,
    )
    unsir_retain_validation = classification_accuracy(
        selected,
        protocol.retain_validation,
        train_config,
        device,
    )
    retain_drop = original_retain_validation - unsir_retain_validation
    forgetting_pass = unsir_forget_validation <= MAX_FORGET_VALIDATION
    utility_pass = retain_drop <= MAX_RETAIN_VALIDATION_DROP
    success = eligible and forgetting_pass and utility_pass
    failure_reasons: list[str] = []
    if not eligible:
        failure_reasons.append("source_validation_base_eligibility_gate_failed")
    if eligible and not forgetting_pass:
        failure_reasons.append("source_validation_forgetting_gate_failed")
    if eligible and not utility_pass:
        failure_reasons.append("source_validation_utility_gate_failed")

    # Reporting-only held-context evaluation begins after all decisions above.
    original_metrics = evaluate_method(original, protocol, train_config, device)
    unsir_metrics = evaluate_method(selected, protocol, train_config, device)
    retrained_metrics = evaluate_method(retrained, protocol, train_config, device)
    original_metrics["equivalence_to_retraining"] = equivalence_metrics(
        original, retrained, protocol, train_config, device
    )
    unsir_metrics["equivalence_to_retraining"] = equivalence_metrics(
        selected, retrained, protocol, train_config, device
    )
    retrained_metrics["equivalence_to_retraining"] = {
        name: {"mean_js_divergence": 0.0, "prediction_agreement": 1.0}
        for name in ("forget_seen", "forget_held", "retain_seen", "retain_held")
    }

    return {
        "case_id": case.case_id,
        "forget_user": case.forget_user,
        "held_condition": case.held_condition,
        "seed": case.seed,
        "status": "completed",
        "eligible": eligible,
        "success": success,
        "failure_reasons": failure_reasons,
        "frozen_candidate_index": 15,
        "frozen_unsir_config": asdict(config),
        "base_gate": {
            "original_forget_validation_accuracy": original_forget_validation,
            "original_retain_validation_accuracy": original_retain_validation,
            "minimum_forget_validation_accuracy": MIN_BASE_FORGET_VALIDATION,
            "minimum_retain_validation_accuracy": MIN_BASE_RETAIN_VALIDATION,
            "passed": eligible,
            "data_scope": "source_validation_only",
        },
        "frozen_gate": {
            "unsir_forget_validation_accuracy": unsir_forget_validation,
            "unsir_retain_validation_accuracy": unsir_retain_validation,
            "retain_validation_drop": retain_drop,
            "maximum_forget_validation_accuracy": MAX_FORGET_VALIDATION,
            "maximum_retain_validation_drop": MAX_RETAIN_VALIDATION_DROP,
            "forgetting_passed": forgetting_pass,
            "utility_passed": utility_pass,
            "data_scope": "source_validation_only",
        },
        "dataset_counts": dataset_counts(protocol),
        "split_integrity": integrity,
        "held_context_used_for_control_flow": False,
        "held_context_evaluated_after_frozen_model": True,
        "metrics": {
            "original": original_metrics,
            "unsir": unsir_metrics,
            "retrained": retrained_metrics,
        },
        "training": {
            "original_reused": True,
            "original_cache_key": f"held-{case.held_condition}__seed-{case.seed}",
            "original": original_training,
            "unsir": unsir_training,
            "retrained": retrained_training,
        },
        "artifacts": {
            "original_checkpoint": str(original_checkpoint),
            "original_checkpoint_sha256": sha256_file(original_checkpoint),
            "unsir_checkpoint": str(selected_checkpoint),
            "unsir_checkpoint_sha256": sha256_file(selected_checkpoint),
            "retrained_checkpoint": str(retrained_checkpoint),
            "retrained_checkpoint_sha256": sha256_file(retrained_checkpoint),
        },
        "diagnostic_metrics_are_non_primary": True,
    }


def checksums(output: Path) -> dict[str, str]:
    return {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and path.name not in {"checksums.json"}
        and not path.name.endswith(".tmp")
    }


def main() -> None:
    args = parse_args()
    if args.batch_size != 32:
        raise ValueError("Frozen candidate 15 requires batch size 32")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    device = torch.device(args.device)
    records = scan_ntu_humanid(args.data_root)
    users = tuple(sorted({record.user for record in records}))
    if len(users) != EXPECTED_USER_COUNT:
        raise ValueError(f"Expected 14 users, found {len(users)}: {users}")
    cases = build_matrix(users)
    if len(cases) != EXPECTED_CASE_COUNT:
        raise AssertionError(f"Expected 126 cases, found {len(cases)}")

    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device_provenance = provenance(device)
    matrix_payload = {
        "provenance": device_provenance,
        "users": users,
        "held_conditions": EXPECTED_CONDITIONS,
        "seeds": FROZEN_SEEDS,
        "expected_case_count": EXPECTED_CASE_COUNT,
        "cases": [asdict(case) | {"case_id": case.case_id} for case in cases],
        "frozen_candidate_index": 15,
        "frozen_config_seed_1": asdict(frozen_unsir_config(1, args.workers)),
    }
    write_json(output / "matrix.json", matrix_payload)

    num_classes = len(users)
    records_out: list[dict[str, object]] = []
    original_signatures = {
        condition: verify_original_reuse(records, users, condition)
        for condition in EXPECTED_CONDITIONS
    }

    for held_condition in EXPECTED_CONDITIONS:
        for seed in FROZEN_SEEDS:
            train_config = TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                workers=args.workers,
                seed=seed,
                patience=args.patience,
            )
            seed_everything(seed)
            reference_protocol = make_protocol(records, users[0], held_condition)
            observed_signature = training_pool_signature(reference_protocol.base_train)
            if observed_signature != original_signatures[held_condition]:
                raise AssertionError("Original reuse signature changed unexpectedly")
            print(
                f"=== ORIGINAL held={held_condition} seed={seed} ===", flush=True
            )
            original, original_training = train_model(
                CSILeNet(num_classes),
                reference_protocol.base_train,
                train_config,
                device,
            )
            original_checkpoint = (
                output
                / "originals"
                / f"original-held-{held_condition}-seed-{seed}.pt"
            )
            save_checkpoint(
                original_checkpoint,
                original,
                {
                    "training": original_training,
                    "held_condition": held_condition,
                    "seed": seed,
                    "training_pool_signature_sha256": hashlib.sha256(
                        json.dumps(observed_signature).encode("utf-8")
                    ).hexdigest(),
                    "reuse_verified_across_all_users": True,
                },
            )

            for user in users:
                case = SweepCase(user, held_condition, seed)
                print(f"=== CASE {case.case_id} ===", flush=True)
                protocol = make_protocol(records, user, held_condition)
                base_record = {
                    "case_id": case.case_id,
                    "forget_user": user,
                    "held_condition": held_condition,
                    "seed": seed,
                }
                try:
                    record = run_case(
                        case,
                        protocol,
                        original,
                        original_checkpoint,
                        original_training,
                        train_config,
                        device,
                        output,
                    )
                except Exception as exc:  # Preserve and report every failed case.
                    record = base_record | {
                        "status": "failed",
                        "eligible": None,
                        "success": False,
                        "failure_reasons": ["execution_failure"],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "dataset_counts": dataset_counts(protocol),
                        "split_integrity": split_integrity(protocol),
                        "held_context_used_for_control_flow": False,
                        "diagnostic_metrics_are_non_primary": True,
                    }
                    print(traceback.format_exc(), flush=True)
                finally:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                records_out.append(record)
                write_cases(output, records_out)
                write_json(
                    output / "progress.json",
                    {
                        "completed_or_failed_cases": len(records_out),
                        "expected_cases": EXPECTED_CASE_COUNT,
                        "last_case": case.case_id,
                    },
                )

    assert_matrix_complete(
        [
            SweepCase(
                str(record["forget_user"]),
                str(record["held_condition"]),
                int(record["seed"]),
            )
            for record in records_out
        ],
        users,
        EXPECTED_CONDITIONS,
        FROZEN_SEEDS,
    )
    aggregates = aggregate_records(records_out)
    write_json(output / "aggregates.json", aggregates)
    write_json(
        output / "failure-map.json",
        aggregates["failure_map"],
    )
    write_json(
        output / "summary.json",
        {
            "status": "completed",
            "provenance": device_provenance,
            "matrix": {
                "users": users,
                "held_conditions": EXPECTED_CONDITIONS,
                "seeds": FROZEN_SEEDS,
                "case_count": len(records_out),
            },
            "frozen_candidate_index": 15,
            "frozen_config_seed_1": asdict(frozen_unsir_config(1, args.workers)),
            "aggregates": aggregates,
        },
    )
    write_json(output / "checksums.json", checksums(output))
    print(f"SWEEP_OUTPUT={output}", flush=True)


if __name__ == "__main__":
    main()
