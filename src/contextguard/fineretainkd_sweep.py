from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import subprocess
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import EXPECTED_CONDITIONS, Protocol, Trace, make_protocol, scan_ntu_humanid
from .mechanism import (
    load_model,
    sha256_file,
    verify_checkpoint_immutability,
    verify_checkpoint_manifest,
    write_json,
)
from .sweep import (
    EXPECTED_CASE_COUNT,
    FROZEN_SEEDS,
    SweepCase,
    assert_matrix_complete,
    build_matrix,
    dataset_counts,
    frozen_unsir_config,
    split_integrity,
)
from .train import TrainConfig, seed_everything
from .utility_pilot import (
    FORGET_MAX,
    KD_WEIGHT,
    RETAIN_DROP_MAX,
    batched_logits,
    context_balanced_retained,
    evaluate_context,
    masked_kl_per_example,
    materialize,
    save_stage_checkpoint,
    select_source_checkpoint,
    source_metrics,
)


METHOD = "fine_retain_kd"
MAX_REPAIR_EPOCHS = 5
DEVELOPMENT_USERS = frozenset({"001", "005", "009", "015"})
DEVELOPMENT_SEED = 1
PRIMARY_CASE_COUNT = 114
INDIVIDUAL_GATE_RATE = 0.90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen FineRetainKD 126-case audit.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sweep-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def is_development_case(user: str, seed: int) -> bool:
    return user in DEVELOPMENT_USERS and seed == DEVELOPMENT_SEED


def assert_frozen_config(seed: int, workers: int) -> dict[str, object]:
    config = frozen_unsir_config(seed, workers)
    observed = asdict(config)
    expected = {
        "noise_samples": 16,
        "noise_steps": 40,
        "noise_learning_rate": 0.1,
        "noise_l2": 0.1,
        "retain_per_class": 10,
        "impair_learning_rate": 0.003,
        "repair_learning_rate": 0.001,
        "impair_epochs": 1,
        "repair_epochs": 3,
        "batch_size": 32,
        "workers": workers,
        "seed": seed,
    }
    if observed != expected:
        raise RuntimeError(f"Candidate-15 configuration drift: {observed}")
    return observed | {
        "maximum_repair_epochs": MAX_REPAIR_EPOCHS,
        "kd_weight": KD_WEIGHT,
        "retained_pool": "all retained training examples; balanced by user/context",
        "damage_aware_weighting": False,
        "forgotten_example_anchor": False,
    }


def generate_noise(
    original: nn.Module, forget_label: int, seed: int, workers: int, device: torch.device
) -> tuple[torch.Tensor, list[float]]:
    config = frozen_unsir_config(seed, workers)
    torch.manual_seed(seed)
    original.eval()
    for parameter in original.parameters():
        parameter.requires_grad_(False)
    noise = nn.Parameter(
        torch.randn(config.noise_samples, 3, 114, 500, device=device)
    )
    labels = torch.full(
        (config.noise_samples,), forget_label, dtype=torch.long, device=device
    )
    optimizer = torch.optim.Adam([noise], lr=config.noise_learning_rate)
    history: list[float] = []
    for _ in range(config.noise_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = -F.cross_entropy(original(noise), labels)
        loss = loss + config.noise_l2 * noise.square().mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    for parameter in original.parameters():
        parameter.requires_grad_(True)
    return noise.detach().cpu(), history


def candidate_record(
    *,
    index: int,
    stage: str,
    epoch: int,
    batch: int,
    path: Path,
    loss: torch.Tensor,
    ce: torch.Tensor,
    kd: torch.Tensor,
    elapsed: float,
    metrics: Mapping[str, float],
) -> dict[str, object]:
    return {
        "checkpoint_index": index,
        "stage": stage,
        "epoch": epoch,
        "batch_index": batch,
        "checkpoint": str(path),
        "loss": float(loss.detach()),
        "cross_entropy": float(ce.detach()),
        "distillation_loss": float(kd.detach()),
        "cumulative_optimizer_steps": index,
        "runtime_seconds_to_checkpoint": elapsed,
        "source_evaluation_only": True,
        **metrics,
    }


def train_frozen_fineretainkd(
    original: nn.Module,
    protocol: Protocol,
    case: SweepCase,
    output: Path,
    workers: int,
    device: torch.device,
) -> dict[str, object]:
    frozen = assert_frozen_config(case.seed, workers)
    config = frozen_unsir_config(case.seed, workers)
    train_config = TrainConfig(batch_size=32, workers=workers, seed=case.seed)
    seed_everything(case.seed)
    model = copy.deepcopy(original).to(device)
    started = time.perf_counter()

    retained_records = context_balanced_retained(
        protocol.retain_train, protocol.held_condition
    )
    retain_x, retain_y = materialize(retained_records, train_config, device)
    teacher_logits = batched_logits(original, retain_x, 32, device)
    forget_val_x, forget_val_y = materialize(
        protocol.forget_validation, train_config, device
    )
    retain_val_x, retain_val_y = materialize(
        protocol.retain_validation, train_config, device
    )
    original_retain_logits = batched_logits(original, retain_val_x, 32, device)
    original_retain_accuracy = float(
        (original_retain_logits.argmax(1) == retain_val_y).float().mean()
    )

    noise, noise_history = generate_noise(
        original, protocol.forget_label, case.seed, workers, device
    )
    noise_y = torch.full((len(noise),), protocol.forget_label, dtype=torch.long)
    impair_x = torch.cat((noise, retain_x))
    impair_y = torch.cat((noise_y, retain_y))
    retained_mask = torch.cat(
        (torch.zeros(len(noise), dtype=torch.bool), torch.ones(len(retain_x), dtype=torch.bool))
    )
    dummy_teacher = torch.zeros(len(noise), original_retain_logits.shape[1])
    impair_teacher = torch.cat((dummy_teacher, teacher_logits))
    impair_loader = DataLoader(
        TensorDataset(impair_x, impair_y, retained_mask, impair_teacher),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(case.seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.impair_learning_rate)
    trajectory: list[dict[str, object]] = []
    step = 0
    for batch, (x, y, is_retained, batch_teacher) in enumerate(impair_loader, 1):
        model.train()
        x, y = x.to(device), y.to(device)
        is_retained = is_retained.to(device)
        batch_teacher = batch_teacher.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        ce = F.cross_entropy(logits, y)
        kd = (
            masked_kl_per_example(
                logits[is_retained], batch_teacher[is_retained], protocol.forget_label
            ).mean()
            if is_retained.any()
            else torch.zeros((), device=device)
        )
        loss = ce + KD_WEIGHT * kd
        loss.backward()
        optimizer.step()
        step += 1
        metrics = source_metrics(
            model,
            original,
            forget_val_x,
            forget_val_y,
            retain_val_x,
            retain_val_y,
            original_retain_logits,
            original_retain_accuracy,
            protocol.forget_label,
            32,
            device,
        )
        path = output / f"impair-batch-{batch:03d}.pt"
        candidate = candidate_record(
            index=step,
            stage="impair",
            epoch=1,
            batch=batch,
            path=path,
            loss=loss,
            ce=ce,
            kd=kd,
            elapsed=time.perf_counter() - started,
            metrics=metrics,
        )
        save_stage_checkpoint(path, model, {"case": asdict(case), "candidate": candidate})
        trajectory.append(candidate)

    repair_dataset = TensorDataset(retain_x, retain_y, teacher_logits)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.repair_learning_rate)
    for epoch in range(1, MAX_REPAIR_EPOCHS + 1):
        loader = DataLoader(
            repair_dataset,
            batch_size=32,
            shuffle=True,
            generator=torch.Generator().manual_seed(case.seed + epoch),
        )
        for batch, (x, y, batch_teacher) in enumerate(loader, 1):
            model.train()
            x, y, batch_teacher = x.to(device), y.to(device), batch_teacher.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            ce = F.cross_entropy(logits, y)
            kd = masked_kl_per_example(
                logits, batch_teacher, protocol.forget_label
            ).mean()
            loss = ce + KD_WEIGHT * kd
            loss.backward()
            optimizer.step()
            step += 1
            metrics = source_metrics(
                model,
                original,
                forget_val_x,
                forget_val_y,
                retain_val_x,
                retain_val_y,
                original_retain_logits,
                original_retain_accuracy,
                protocol.forget_label,
                32,
                device,
            )
            path = output / f"repair-epoch-{epoch:02d}-batch-{batch:03d}.pt"
            candidate = candidate_record(
                index=step,
                stage="repair",
                epoch=epoch,
                batch=batch,
                path=path,
                loss=loss,
                ce=ce,
                kd=kd,
                elapsed=time.perf_counter() - started,
                metrics=metrics,
            )
            save_stage_checkpoint(
                path, model, {"case": asdict(case), "candidate": candidate}
            )
            trajectory.append(candidate)

    selected = select_source_checkpoint(trajectory, FORGET_MAX, RETAIN_DROP_MAX)
    selected_copy = dict(selected) if selected is not None else None
    return {
        "method": METHOD,
        "status": "selected" if selected_copy else "abstained",
        "selected": selected_copy,
        "trajectory": trajectory,
        "trajectory_checkpoint_count": len(trajectory),
        "total_optimizer_steps": step,
        "total_runtime_seconds": time.perf_counter() - started,
        "selected_cumulative_optimizer_steps": (
            selected_copy["cumulative_optimizer_steps"] if selected_copy else None
        ),
        "selected_runtime_seconds": (
            selected_copy["runtime_seconds_to_checkpoint"] if selected_copy else None
        ),
        "noise_final_loss": noise_history[-1],
        "frozen_configuration": frozen,
        "retained_training_examples": len(retained_records),
        "retained_pool_balanced": True,
        "held_context_used_for_selection": False,
        "selection_frozen_before_held_access": True,
    }


def require_selection_frozen(training: Mapping[str, object]) -> None:
    if training.get("status") not in {"selected", "abstained"}:
        raise RuntimeError("Held context accessed before source selection froze")
    if training.get("held_context_used_for_selection") is not False:
        raise RuntimeError("Selection provenance permits held-context leakage")


def summarize(values: Sequence[float]) -> dict[str, float | int | str]:
    count = len(values)
    if not count:
        return {"count": 0}
    center = mean(values)
    spread = stdev(values) if count > 1 else 0.0
    margin = 1.96 * spread / math.sqrt(count) if count > 1 else 0.0
    return {
        "count": count,
        "mean": center,
        "median": median(values),
        "sample_std": spread,
        "ci95_low": center - margin,
        "ci95_high": center + margin,
        "ci95_method": "normal approximation",
    }


def proportion_summary(successes: int, total: int) -> dict[str, float | int | None | str]:
    if total == 0:
        return {"successes": successes, "total": total, "rate": None}
    z = 1.96
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return {
        "successes": successes,
        "total": total,
        "rate": rate,
        "ci95_low": max(0.0, center - margin),
        "ci95_high": min(1.0, center + margin),
        "ci95_method": "Wilson score",
    }


def selected_case_records(cases: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [
        case
        for case in cases
        if case.get("status") == "completed"
        and case.get("selection", {}).get("status") == "selected"
    ]


def matched_subset_aggregate(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    selected = selected_case_records(cases)
    metrics: dict[str, object] = {}
    for context in ("source", "seen", "held"):
        context_metrics: dict[str, object] = {}
        for metric in (
            "forget_accuracy",
            "retain_accuracy",
            "retain_accuracy_drop_from_original",
            "js_to_exact_retraining",
        ):
            by_method = {}
            for method in ("original", METHOD, "retrained"):
                values = [
                    float(case["evaluation"][context][method][metric])
                    for case in selected
                ]
                by_method[method] = summarize(values)
            paired = [
                float(case["evaluation"][context][METHOD][metric])
                - float(case["evaluation"][context]["original"][metric])
                for case in selected
            ]
            context_metrics[metric] = {
                "matched_case_ids": [case["case_id"] for case in selected],
                "methods": by_method,
                "fine_minus_original": summarize(paired),
            }
        metrics[context] = context_metrics
    individually_passed = [
        case
        for case in selected
        if case["evaluation"]["held"][METHOD]["forget_accuracy"] <= 0.10
        and case["evaluation"]["held"][METHOD]["retain_accuracy_drop_from_original"] <= 0.05
    ]
    return {
        "case_count": len(cases),
        "source_selection_coverage": proportion_summary(len(selected), len(cases)),
        "selected_case_count": len(selected),
        "abstention_or_failure_count": len(cases) - len(selected),
        "matched_subset_rule": "all methods use exactly the FineRetainKD-selected case IDs",
        "selected_case_ids": [case["case_id"] for case in selected],
        "individual_held_gate_pass_rate": proportion_summary(
            len(individually_passed), len(selected)
        ),
        "metrics": metrics,
    }


def grouped(cases: Sequence[Mapping[str, object]], key: str) -> dict[str, object]:
    names = sorted({str(case[key]) for case in cases})
    return {
        name: matched_subset_aggregate(
            [case for case in cases if str(case[key]) == name]
        )
        for name in names
    }


def subgroup_catastrophic(aggregate: Mapping[str, object]) -> bool:
    selected = int(aggregate["selected_case_count"])
    if selected == 0:
        return True
    held = aggregate["metrics"]["held"]
    return bool(
        held["forget_accuracy"]["methods"][METHOD]["mean"] > 0.50
        or held["retain_accuracy_drop_from_original"]["methods"][METHOD]["mean"] > 0.20
    )


def primary_gate(primary: Sequence[Mapping[str, object]], aggregate: Mapping[str, object]) -> dict[str, object]:
    held = aggregate["metrics"]["held"]
    seen = aggregate["metrics"]["seen"]
    coverage = aggregate["source_selection_coverage"]["rate"]
    individual = aggregate["individual_held_gate_pass_rate"]["rate"]
    subgroup_aggregates = {
        "held_condition": grouped(primary, "held_condition"),
        "seed": grouped(primary, "seed"),
    }
    catastrophic = {
        kind: [name for name, value in values.items() if subgroup_catastrophic(value)]
        for kind, values in subgroup_aggregates.items()
    }
    checks = {
        "source_selection_coverage_at_least_0_70": coverage is not None and coverage >= 0.70,
        "mean_held_forget_accuracy_at_most_0_10": held["forget_accuracy"]["methods"][METHOD].get("mean", math.inf) <= 0.10,
        "mean_seen_retain_drop_at_most_0_05": seen["retain_accuracy_drop_from_original"]["methods"][METHOD].get("mean", math.inf) <= 0.05,
        "mean_held_retain_drop_at_most_0_05": held["retain_accuracy_drop_from_original"]["methods"][METHOD].get("mean", math.inf) <= 0.05,
        "individual_held_gate_rate_at_least_0_90": individual is not None and individual >= INDIVIDUAL_GATE_RATE,
        "mean_seen_js_lower_than_matched_original": seen["js_to_exact_retraining"]["methods"][METHOD].get("mean", math.inf) < seen["js_to_exact_retraining"]["methods"]["original"].get("mean", -math.inf),
        "mean_held_js_lower_than_matched_original": held["js_to_exact_retraining"]["methods"][METHOD].get("mean", math.inf) < held["js_to_exact_retraining"]["methods"]["original"].get("mean", -math.inf),
        "no_held_condition_or_seed_catastrophic_collapse": not any(catastrophic.values()),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "catastrophic_collapse_definition": "zero selected cases, mean held forgetting >50%, or mean held retained-accuracy drop >20%",
        "catastrophic_subgroups": catastrophic,
        "if_failed": "Stop; do not tune FineRetainKD.",
        "if_passed": "Freeze FineRetainKD for external-dataset and baseline evaluation.",
    }


def flatten(value: object, prefix: str = "", output=None):
    output = {} if output is None else output
    if isinstance(value, Mapping):
        for key, child in value.items():
            flatten(child, f"{prefix}.{key}" if prefix else str(key), output)
    elif isinstance(value, (list, tuple)):
        output[prefix] = json.dumps(value, sort_keys=True)
    else:
        output[prefix] = value
    return output


def write_cases(output: Path, cases: Sequence[Mapping[str, object]]) -> None:
    write_json(output / "fineretainkd-cases.json", list(cases))
    rows = [flatten(case) for case in cases]
    fields = sorted({key for row in rows for key in row})
    temporary = output / "fineretainkd-cases.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output / "fineretainkd-cases.csv")


def artifact_checksums(output: Path) -> dict[str, str]:
    return {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "checksums.json" and not path.name.endswith(".tmp")
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    source = Path(args.sweep_output).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((source / "checksums.json").read_text())
    source_snapshot = verify_checkpoint_manifest(source, manifest)
    records = scan_ntu_humanid(args.data_root)
    users = tuple(sorted({record.user for record in records}))
    matrix = build_matrix(users)
    if len(matrix) != EXPECTED_CASE_COUNT:
        raise RuntimeError(f"Expected 126 cases, found {len(matrix)}")
    assert sum(is_development_case(case.forget_user, case.seed) for case in matrix) == 12
    results: list[dict[str, object]] = []
    train_config_by_seed = {
        seed: TrainConfig(batch_size=32, workers=args.workers, seed=seed)
        for seed in FROZEN_SEEDS
    }

    for index, case in enumerate(matrix, 1):
        print(f"FINERETAINKD {index:03d}/126 {case.case_id}", flush=True)
        protocol = make_protocol(records, case.forget_user, case.held_condition)
        base = {
            "case_id": case.case_id,
            "forget_user": case.forget_user,
            "held_condition": case.held_condition,
            "seed": case.seed,
            "development_case": is_development_case(case.forget_user, case.seed),
        }
        try:
            integrity = split_integrity(protocol)
            if not integrity["passed"]:
                raise RuntimeError(f"Split integrity failed: {integrity}")
            original_path = source / "originals" / f"original-held-{case.held_condition}-seed-{case.seed}.pt"
            retrained_path = source / "case-artifacts" / case.case_id / "retrained.pt"
            original = load_model(original_path, len(users), device)
            retrained = load_model(retrained_path, len(users), device)
            selection = train_frozen_fineretainkd(
                original,
                protocol,
                case,
                output / "checkpoints" / case.case_id,
                args.workers,
                device,
            )
            require_selection_frozen(selection)
            models: dict[str, nn.Module] = {"original": original, "retrained": retrained}
            if selection["selected"] is not None:
                models[METHOD] = load_model(
                    Path(selection["selected"]["checkpoint"]), len(users), device
                )
            contexts = {
                "source": (protocol.forget_validation, protocol.retain_validation),
                "seen": (protocol.forget_seen_test, protocol.retain_seen_test),
                "held": (protocol.forget_held_test, protocol.retain_held_test),
            }
            evaluation = {
                context: evaluate_context(
                    models,
                    retrained,
                    forget_records,
                    retain_records,
                    train_config_by_seed[case.seed],
                    device,
                )
                for context, (forget_records, retain_records) in contexts.items()
            }
            selected = selection["status"] == "selected"
            held_pass = bool(
                selected
                and evaluation["held"][METHOD]["forget_accuracy"] <= 0.10
                and evaluation["held"][METHOD]["retain_accuracy_drop_from_original"] <= 0.05
            )
            record = base | {
                "status": "completed",
                "selection": selection,
                "evaluation": evaluation,
                "selected": selected,
                "individual_held_gate_passed": held_pass,
                "dataset_counts": dataset_counts(protocol),
                "split_integrity": integrity,
                "held_context_used_for_selection": False,
                "held_context_accessed_after_selection_froze": True,
                "source_artifacts": {
                    "original_checkpoint": str(original_path),
                    "original_sha256": sha256_file(original_path),
                    "retrained_checkpoint": str(retrained_path),
                    "retrained_sha256": sha256_file(retrained_path),
                },
            }
        except Exception as exc:
            record = base | {
                "status": "failed",
                "selected": False,
                "failure_reason": "execution_failure",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "dataset_counts": dataset_counts(protocol),
                "split_integrity": split_integrity(protocol),
                "held_context_used_for_selection": False,
            }
            print(traceback.format_exc(), flush=True)
        results.append(record)
        write_cases(output, results)
        write_json(output / "progress.json", {"processed": len(results), "expected": 126})
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert_matrix_complete(
        [SweepCase(str(row["forget_user"]), str(row["held_condition"]), int(row["seed"])) for row in results],
        users,
        EXPECTED_CONDITIONS,
        FROZEN_SEEDS,
    )
    verify_checkpoint_immutability(source, source_snapshot)
    development = [case for case in results if case["development_case"]]
    primary = [case for case in results if not case["development_case"]]
    if len(development) != 12 or len(primary) != PRIMARY_CASE_COUNT:
        raise RuntimeError("Development/primary partition is not 12/114")
    overall_aggregate = matched_subset_aggregate(results)
    development_aggregate = matched_subset_aggregate(development)
    primary_aggregate = matched_subset_aggregate(primary)
    aggregates = {
        "all_126": overall_aggregate,
        "development_12": development_aggregate,
        "primary_nondevelopment_114": primary_aggregate,
        "primary_by_user": grouped(primary, "forget_user"),
        "primary_by_held_condition": grouped(primary, "held_condition"),
        "primary_by_seed": grouped(primary, "seed"),
    }
    gate = primary_gate(primary, primary_aggregate)
    provenance = {
        "study": "frozen FineRetainKD robustness audit",
        "git_commit": git_commit(),
        "parent_commit": "567d0aaa40e9e4b1486940682e8be298b6082f90",
        "source_sweep_output": str(source),
        "verified_source_checkpoint_count": len(source_snapshot),
        "original_and_exact_retrained_reused": True,
        "fineretainkd_rerun_from_original_for_every_case": True,
        "case_count": 126,
        "development_case_count": 12,
        "primary_case_count": 114,
        "held_context_policy": "inaccessible until source-only selection freezes",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
    }
    write_json(output / "aggregates.json", aggregates)
    write_json(
        output / "failure-map.json",
        [
            {
                "case_id": case["case_id"],
                "development_case": case["development_case"],
                "status": case["status"],
                "selection_status": case.get("selection", {}).get("status", "failed"),
                "individual_held_gate_passed": case.get("individual_held_gate_passed"),
                "failure_reason": case.get("failure_reason"),
            }
            for case in results
            if case.get("individual_held_gate_passed") is not True
        ],
    )
    write_json(
        output / "summary.json",
        {"status": "completed", "provenance": provenance, "primary_gate": gate, "aggregates": aggregates},
    )
    write_json(output / "checksums.json", artifact_checksums(output))
    print(f"FINERETAINKD_SWEEP_OUTPUT={output}", flush=True)


if __name__ == "__main__":
    main()
