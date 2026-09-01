from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
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
from .model import CSILeNet
from .sweep import frozen_unsir_config
from .train import TrainConfig, make_loader, seed_everything


PANEL_USERS = ("001", "005", "009", "015")
PANEL_SEED = 1
METHOD_VARIANTS = ("retain_kd", "damage_aware_kd")
FORGET_MAX = 0.10
RETAIN_DROP_MAX = 0.05
KD_WEIGHT = 1.0
TOP_DRIFTED_COUNT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exploratory ContextGuard utility-preservation pilot."
    )
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


def masked_probabilities(logits: torch.Tensor, deleted_label: int) -> torch.Tensor:
    masked = logits.clone()
    masked[:, deleted_label] = -torch.inf
    return torch.softmax(masked, dim=1)


def masked_kl_per_example(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    deleted_label: int,
) -> torch.Tensor:
    teacher = masked_probabilities(teacher_logits.detach(), deleted_label)
    student = masked_probabilities(student_logits, deleted_label).clamp_min(1e-12)
    teacher_safe = teacher.clamp_min(1e-12)
    return torch.sum(teacher * (teacher_safe.log() - student.log()), dim=1)


def masked_js_per_example(
    left_logits: torch.Tensor,
    right_logits: torch.Tensor,
    deleted_label: int,
) -> torch.Tensor:
    left = masked_probabilities(left_logits, deleted_label).clamp_min(1e-12)
    right = masked_probabilities(right_logits, deleted_label).clamp_min(1e-12)
    middle = 0.5 * (left + right)
    return 0.5 * torch.sum(left * (left.log() - middle.log()), dim=1) + 0.5 * torch.sum(
        right * (right.log() - middle.log()), dim=1
    )


def js_per_example(left_logits: torch.Tensor, right_logits: torch.Tensor) -> torch.Tensor:
    left = torch.softmax(left_logits, dim=1).clamp_min(1e-12)
    right = torch.softmax(right_logits, dim=1).clamp_min(1e-12)
    middle = 0.5 * (left + right)
    return 0.5 * torch.sum(left * (left.log() - middle.log()), dim=1) + 0.5 * torch.sum(
        right * (right.log() - middle.log()), dim=1
    )


def context_balanced_retained(
    records: Sequence[Trace], held_condition: str
) -> tuple[Trace, ...]:
    retained = tuple(record for record in records if record.condition != held_condition)
    counts = Counter((record.user, record.condition) for record in retained)
    users = sorted({record.user for record in retained})
    source_contexts = sorted({record.condition for record in retained})
    expected_keys = {(user, context) for user in users for context in source_contexts}
    if set(counts) != expected_keys or len(set(counts.values())) != 1:
        raise ValueError(f"Retained source pool is not user/context balanced: {counts}")
    return tuple(
        sorted(retained, key=lambda row: (row.user, row.condition, row.sample_index))
    )


def top_drifted_users(
    drift_by_user: Mapping[str, float], count: int = TOP_DRIFTED_COUNT
) -> tuple[str, ...]:
    return tuple(
        user
        for user, _ in sorted(
            drift_by_user.items(), key=lambda item: (-item[1], item[0])
        )[:count]
    )


def select_source_checkpoint(
    candidates: Sequence[Mapping[str, object]],
    forget_max: float = FORGET_MAX,
    retain_drop_max: float = RETAIN_DROP_MAX,
) -> Mapping[str, object] | None:
    feasible = [
        candidate
        for candidate in candidates
        if float(candidate["source_forget_accuracy"]) <= forget_max
        and float(candidate["source_retain_drop"]) <= retain_drop_max
    ]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda candidate: (
            float(candidate["source_retained_label_js"]),
            int(candidate["checkpoint_index"]),
        ),
    )


def materialize(
    records: Sequence[Trace], config: TrainConfig, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = make_loader(records, config, shuffle=False, persistent_workers=False)
    x_parts: list[torch.Tensor] = []
    y_parts: list[torch.Tensor] = []
    for batch in loader:
        x_parts.append(batch["x"])
        y_parts.append(batch["y"])
    if not x_parts:
        raise ValueError("Cannot materialize an empty record sequence")
    return torch.cat(x_parts), torch.cat(y_parts)


@torch.inference_mode()
def batched_logits(
    model: nn.Module,
    x: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    parts = []
    model.eval()
    for start in range(0, len(x), batch_size):
        parts.append(model(x[start : start + batch_size].to(device)).cpu())
    return torch.cat(parts)


def source_metrics(
    model: nn.Module,
    original: nn.Module,
    forget_x: torch.Tensor,
    forget_y: torch.Tensor,
    retain_x: torch.Tensor,
    retain_y: torch.Tensor,
    original_retain_logits: torch.Tensor,
    original_retain_accuracy: float,
    forget_label: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model_forget_logits = batched_logits(model, forget_x, batch_size, device)
    model_retain_logits = batched_logits(model, retain_x, batch_size, device)
    forget_accuracy = float(
        (model_forget_logits.argmax(dim=1) == forget_y).float().mean()
    )
    retain_accuracy = float(
        (model_retain_logits.argmax(dim=1) == retain_y).float().mean()
    )
    retained_js = float(
        masked_js_per_example(
            model_retain_logits, original_retain_logits, forget_label
        ).mean()
    )
    return {
        "source_forget_accuracy": forget_accuracy,
        "source_retain_accuracy": retain_accuracy,
        "source_retain_drop": original_retain_accuracy - retain_accuracy,
        "source_retained_label_js": retained_js,
    }


def retained_user_drift(
    model: nn.Module,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    teacher_logits: torch.Tensor,
    label_to_user: Mapping[int, str],
    forget_label: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    student_logits = batched_logits(model, validation_x, batch_size, device)
    drift = masked_js_per_example(student_logits, teacher_logits, forget_label)
    return {
        label_to_user[label]: float(drift[validation_y == label].mean())
        for label in sorted(set(validation_y.tolist()))
    }


def save_stage_checkpoint(
    path: Path,
    model: nn.Module,
    metadata: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metadata": dict(metadata)}, path)


def generate_forgetting_noise(
    original: nn.Module,
    forget_label: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    config = frozen_unsir_config(PANEL_SEED, workers=8)
    torch.manual_seed(config.seed)
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


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * weights) / weights.sum().clamp_min(1e-12)


def train_kd_variant(
    method: str,
    original: CSILeNet,
    protocol: Protocol,
    label_to_user: Mapping[int, str],
    output: Path,
    workers: int,
    device: torch.device,
) -> dict[str, object]:
    if method not in METHOD_VARIANTS:
        raise ValueError(f"Unknown method {method}")
    damage_aware = method == "damage_aware_kd"
    config = frozen_unsir_config(PANEL_SEED, workers=workers)
    train_config = TrainConfig(batch_size=config.batch_size, workers=workers, seed=1)
    seed_everything(PANEL_SEED)
    model = copy.deepcopy(original).to(device)

    retained_records = context_balanced_retained(
        protocol.retain_train, protocol.held_condition
    )
    retain_x, retain_y = materialize(retained_records, train_config, device)
    retain_teacher_logits = batched_logits(
        original, retain_x, config.batch_size, device
    )
    forget_validation_x, forget_validation_y = materialize(
        protocol.forget_validation, train_config, device
    )
    retain_validation_x, retain_validation_y = materialize(
        protocol.retain_validation, train_config, device
    )
    original_retain_validation_logits = batched_logits(
        original, retain_validation_x, config.batch_size, device
    )
    original_retain_accuracy = float(
        (
            original_retain_validation_logits.argmax(dim=1)
            == retain_validation_y
        ).float().mean()
    )

    noise, noise_history = generate_forgetting_noise(
        original, protocol.forget_label, device
    )
    noise_y = torch.full((len(noise),), protocol.forget_label, dtype=torch.long)
    combined_x = torch.cat((noise, retain_x))
    combined_y = torch.cat((noise_y, retain_y))
    is_retained = torch.cat(
        (torch.zeros(len(noise), dtype=torch.bool), torch.ones(len(retain_x), dtype=torch.bool))
    )
    dummy_teacher = torch.zeros(len(noise), original.num_classes)
    combined_teacher = torch.cat((dummy_teacher, retain_teacher_logits))
    generator = torch.Generator().manual_seed(PANEL_SEED)
    impair_loader = DataLoader(
        TensorDataset(combined_x, combined_y, is_retained, combined_teacher),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    impair_optimizer = torch.optim.Adam(
        model.parameters(), lr=config.impair_learning_rate
    )

    trajectory: list[dict[str, object]] = []
    weighted_users: tuple[str, ...] = ()
    checkpoint_index = 0
    for batch_index, (x, y, retained_mask, teacher_logits) in enumerate(
        impair_loader, start=1
    ):
        x = x.to(device)
        y = y.to(device)
        retained_mask = retained_mask.to(device)
        teacher_logits = teacher_logits.to(device)
        weights = torch.ones(len(y), device=device)
        if weighted_users:
            weighted_labels = {
                label for label, user in label_to_user.items() if user in weighted_users
            }
            for label in weighted_labels:
                weights[(y == label) & retained_mask] = 3.0
        impair_optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        ce = weighted_mean(F.cross_entropy(logits, y, reduction="none"), weights)
        if retained_mask.any():
            kd_values = masked_kl_per_example(
                logits[retained_mask], teacher_logits[retained_mask], protocol.forget_label
            )
            kd = weighted_mean(kd_values, weights[retained_mask])
        else:
            kd = torch.zeros((), device=device)
        loss = ce + KD_WEIGHT * kd
        loss.backward()
        impair_optimizer.step()
        checkpoint_index += 1
        metrics = source_metrics(
            model,
            original,
            forget_validation_x,
            forget_validation_y,
            retain_validation_x,
            retain_validation_y,
            original_retain_validation_logits,
            original_retain_accuracy,
            protocol.forget_label,
            config.batch_size,
            device,
        )
        drift = retained_user_drift(
            model,
            retain_validation_x,
            retain_validation_y,
            original_retain_validation_logits,
            label_to_user,
            protocol.forget_label,
            config.batch_size,
            device,
        )
        next_weighted = top_drifted_users(drift) if damage_aware else ()
        checkpoint_path = output / method / f"impair-batch-{batch_index:03d}.pt"
        candidate = {
            "checkpoint_index": checkpoint_index,
            "stage": "impair",
            "stage_index": batch_index,
            "checkpoint": str(checkpoint_path),
            "loss": float(loss.detach()),
            "cross_entropy": float(ce.detach()),
            "distillation_loss": float(kd.detach()),
            "weighted_users_for_stage": weighted_users,
            "drift_by_user_after_stage": drift,
            "weighted_users_for_next_stage": next_weighted,
            **metrics,
        }
        save_stage_checkpoint(
            checkpoint_path,
            model,
            {"method": method, "candidate": candidate, "source_only": True},
        )
        trajectory.append(candidate)
        weighted_users = next_weighted

    repair_optimizer = torch.optim.Adam(
        model.parameters(), lr=config.repair_learning_rate
    )
    repair_dataset = TensorDataset(retain_x, retain_y, retain_teacher_logits)
    for epoch in range(1, config.repair_epochs + 1):
        repair_generator = torch.Generator().manual_seed(PANEL_SEED + epoch)
        repair_loader = DataLoader(
            repair_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            generator=repair_generator,
        )
        total_loss = 0.0
        total_examples = 0
        for x, y, teacher_logits in repair_loader:
            x = x.to(device)
            y = y.to(device)
            teacher_logits = teacher_logits.to(device)
            weights = torch.ones(len(y), device=device)
            if weighted_users:
                weighted_labels = {
                    label for label, user in label_to_user.items() if user in weighted_users
                }
                for label in weighted_labels:
                    weights[y == label] = 3.0
            repair_optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            ce_values = F.cross_entropy(logits, y, reduction="none")
            kd_values = masked_kl_per_example(
                logits, teacher_logits, protocol.forget_label
            )
            loss = weighted_mean(ce_values, weights) + KD_WEIGHT * weighted_mean(
                kd_values, weights
            )
            loss.backward()
            repair_optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            total_examples += len(y)
        checkpoint_index += 1
        metrics = source_metrics(
            model,
            original,
            forget_validation_x,
            forget_validation_y,
            retain_validation_x,
            retain_validation_y,
            original_retain_validation_logits,
            original_retain_accuracy,
            protocol.forget_label,
            config.batch_size,
            device,
        )
        drift = retained_user_drift(
            model,
            retain_validation_x,
            retain_validation_y,
            original_retain_validation_logits,
            label_to_user,
            protocol.forget_label,
            config.batch_size,
            device,
        )
        next_weighted = top_drifted_users(drift) if damage_aware else ()
        checkpoint_path = output / method / f"repair-epoch-{epoch:03d}.pt"
        candidate = {
            "checkpoint_index": checkpoint_index,
            "stage": "repair",
            "stage_index": epoch,
            "checkpoint": str(checkpoint_path),
            "loss": total_loss / total_examples,
            "weighted_users_for_stage": weighted_users,
            "drift_by_user_after_stage": drift,
            "weighted_users_for_next_stage": next_weighted,
            **metrics,
        }
        save_stage_checkpoint(
            checkpoint_path,
            model,
            {"method": method, "candidate": candidate, "source_only": True},
        )
        trajectory.append(candidate)
        weighted_users = next_weighted

    selected = select_source_checkpoint(trajectory)
    return {
        "method": method,
        "status": "selected" if selected is not None else "abstained",
        "selected": dict(selected) if selected is not None else None,
        "trajectory": trajectory,
        "noise_final_loss": noise_history[-1],
        "frozen_forgetting_config": asdict(config),
        "kd_weight": KD_WEIGHT,
        "retained_training_examples": len(retained_records),
        "retained_pool_balanced": True,
        "held_context_used_for_weighting_or_selection": False,
    }


@torch.inference_mode()
def evaluate_context(
    models: Mapping[str, nn.Module],
    retrained: nn.Module,
    forget_records: Sequence[Trace],
    retain_records: Sequence[Trace],
    config: TrainConfig,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    forget_x, forget_y = materialize(forget_records, config, device)
    retain_x, retain_y = materialize(retain_records, config, device)
    combined_x = torch.cat((forget_x, retain_x))
    retrained_logits = batched_logits(retrained, combined_x, config.batch_size, device)
    results: dict[str, dict[str, float]] = {}
    original_retain_accuracy = None
    logits_by_method: dict[str, torch.Tensor] = {}
    for method, model in models.items():
        logits = batched_logits(model, combined_x, config.batch_size, device)
        logits_by_method[method] = logits
        forget_logits = logits[: len(forget_x)]
        retain_logits = logits[len(forget_x) :]
        retain_accuracy = float((retain_logits.argmax(1) == retain_y).float().mean())
        if method == "original":
            original_retain_accuracy = retain_accuracy
        results[method] = {
            "forget_accuracy": float((forget_logits.argmax(1) == forget_y).float().mean()),
            "retain_accuracy": retain_accuracy,
            "js_to_exact_retraining": float(js_per_example(logits, retrained_logits).mean()),
        }
    assert original_retain_accuracy is not None
    for metrics in results.values():
        metrics["retain_accuracy_drop_from_original"] = (
            original_retain_accuracy - metrics["retain_accuracy"]
        )
    return results


def mean_available(
    cases: Sequence[Mapping[str, object]], method: str, context: str, metric: str
) -> float | None:
    values = [
        case["evaluation"][context][method][metric]
        for case in cases
        if method in case["evaluation"][context]
    ]
    return float(np.mean(values)) if values else None


def pilot_gate(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    damage_selected = sum(
        case["variants"]["damage_aware_kd"]["status"] == "selected"
        for case in cases
    )
    retain_selected = sum(
        case["variants"]["retain_kd"]["status"] == "selected" for case in cases
    )
    damage_held_forget = mean_available(
        cases, "damage_aware_kd", "held", "forget_accuracy"
    )
    damage_seen_drop = mean_available(
        cases, "damage_aware_kd", "seen", "retain_accuracy_drop_from_original"
    )
    damage_held_drop = mean_available(
        cases, "damage_aware_kd", "held", "retain_accuracy_drop_from_original"
    )
    damage_js_values = [
        value
        for context in ("seen", "held")
        if (
            value := mean_available(
                cases, "damage_aware_kd", context, "js_to_exact_retraining"
            )
        )
        is not None
    ]
    original_js_values = [
        value
        for context in ("seen", "held")
        if (
            value := mean_available(cases, "original", context, "js_to_exact_retraining")
        )
        is not None
    ]
    damage_js = float(np.mean(damage_js_values)) if damage_js_values else None
    original_js = float(np.mean(original_js_values)) if original_js_values else None
    retain_seen_drop = mean_available(
        cases, "retain_kd", "seen", "retain_accuracy_drop_from_original"
    )
    retain_held_drop = mean_available(
        cases, "retain_kd", "held", "retain_accuracy_drop_from_original"
    )
    clear_improvement = bool(
        damage_selected >= retain_selected + 2
        and damage_seen_drop is not None
        and retain_seen_drop is not None
        and damage_held_drop is not None
        and retain_held_drop is not None
        and damage_seen_drop <= retain_seen_drop - 0.01
        and damage_held_drop <= retain_held_drop - 0.01
    )
    criteria = {
        "source_valid_success_at_least_9_of_12": damage_selected >= 9,
        "mean_held_forgetting_at_most_0_10": (
            damage_held_forget is not None and damage_held_forget <= 0.10
        ),
        "mean_seen_retain_drop_at_most_0_05": (
            damage_seen_drop is not None and damage_seen_drop <= 0.05
        ),
        "mean_held_retain_drop_at_most_0_05": (
            damage_held_drop is not None and damage_held_drop <= 0.05
        ),
        "mean_js_not_worse_than_original": bool(
            damage_js is not None
            and original_js is not None
            and damage_js <= original_js
        ),
        "clear_improvement_over_retain_kd": clear_improvement,
    }
    return {
        "proceed_to_larger_experiment": all(criteria.values()),
        "criteria": criteria,
        "observed": {
            "damage_aware_selected_cases": damage_selected,
            "retain_kd_selected_cases": retain_selected,
            "damage_aware_mean_held_forget_accuracy": damage_held_forget,
            "damage_aware_mean_seen_retain_drop": damage_seen_drop,
            "damage_aware_mean_held_retain_drop": damage_held_drop,
            "damage_aware_mean_js_to_retraining_seen_held": damage_js,
            "original_mean_js_to_retraining_seen_held": original_js,
            "retain_kd_mean_seen_retain_drop": retain_seen_drop,
            "retain_kd_mean_held_retain_drop": retain_held_drop,
        },
        "clear_improvement_operationalization": (
            "DamageAwareKD must select at least two more cases and reduce both seen "
            "and held mean retained-accuracy drop by at least one percentage point."
        ),
    }


def flatten(value: object, prefix: str = "", out: dict[str, object] | None = None) -> dict[str, object]:
    out = {} if out is None else out
    if isinstance(value, Mapping):
        for key, child in value.items():
            flatten(child, f"{prefix}.{key}" if prefix else str(key), out)
    elif isinstance(value, (list, tuple)):
        out[prefix] = json.dumps(value, sort_keys=True)
    else:
        out[prefix] = value
    return out


def write_csv(path: Path, cases: Sequence[Mapping[str, object]]) -> None:
    rows = [flatten(case) for case in cases]
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def output_checksums(output: Path) -> dict[str, str]:
    return {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact-checksums.json"
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    sweep_root = Path(args.sweep_output).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((sweep_root / "checksums.json").read_text())
    source_snapshot = verify_checkpoint_manifest(sweep_root, manifest)
    sweep_cases = json.loads((sweep_root / "cases.json").read_text())
    panel_keys = {
        (user, held, PANEL_SEED) for user in PANEL_USERS for held in EXPECTED_CONDITIONS
    }
    panel = [
        case
        for case in sweep_cases
        if (case["forget_user"], case["held_condition"], case["seed"]) in panel_keys
    ]
    if len(panel) != 12:
        raise ValueError(f"Expected 12 development cases, found {len(panel)}")

    records = scan_ntu_humanid(args.data_root)
    users = sorted({record.user for record in records})
    label_to_user = {record.label: record.user for record in records}
    num_classes = len(users)
    evaluation_config = TrainConfig(batch_size=32, workers=args.workers, seed=1)
    results: list[dict[str, object]] = []

    for case_index, sweep_case in enumerate(panel, start=1):
        case_id = str(sweep_case["case_id"])
        print(f"PANEL {case_index:02d}/12 {case_id}", flush=True)
        protocol = make_protocol(
            records, str(sweep_case["forget_user"]), str(sweep_case["held_condition"])
        )
        original_path = (
            sweep_root
            / "originals"
            / f"original-held-{sweep_case['held_condition']}-seed-1.pt"
        )
        case_dir = sweep_root / "case-artifacts" / case_id
        baseline_path = case_dir / "unsir-selected.pt"
        retrained_path = case_dir / "retrained.pt"
        original = load_model(original_path, num_classes, device)
        baseline = load_model(baseline_path, num_classes, device)
        retrained = load_model(retrained_path, num_classes, device)

        variant_results: dict[str, dict[str, object]] = {}
        for method in METHOD_VARIANTS:
            variant_results[method] = train_kd_variant(
                method,
                original,
                protocol,
                label_to_user,
                output / "checkpoints" / case_id,
                args.workers,
                device,
            )

        # Held/seen records are first accessed only after both source-only
        # trajectories have selected a feasible checkpoint or abstained.
        models: dict[str, nn.Module] = {
            "original": original,
            "candidate15": baseline,
            "retrained": retrained,
        }
        for method, variant in variant_results.items():
            if variant["selected"] is not None:
                models[method] = load_model(
                    Path(variant["selected"]["checkpoint"]), num_classes, device
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
                evaluation_config,
                device,
            )
            for context, (forget_records, retain_records) in contexts.items()
        }
        results.append(
            {
                "case_id": case_id,
                "forget_user": sweep_case["forget_user"],
                "held_condition": sweep_case["held_condition"],
                "seed": sweep_case["seed"],
                "variants": variant_results,
                "evaluation": evaluation,
                "source_checkpoint_paths": {
                    "original": str(original_path),
                    "candidate15": str(baseline_path),
                    "retrained": str(retrained_path),
                },
                "held_context_used_for_weighting_or_selection": False,
            }
        )
        del models, original, baseline, retrained
        torch.cuda.empty_cache()

    verify_checkpoint_immutability(sweep_root, source_snapshot)
    gate = pilot_gate(results)
    provenance = {
        "study_type": "exploratory method-development pilot; not final paper evidence",
        "audit_git_commit": git_commit(),
        "branch": "codex/contextguard-utility-pilot",
        "source_checkpoint_commit": "9c9c630ddb04a1398d2e2fcdb282745a005b00f1",
        "parent_commit": "54c6402f23978ad1709e4a035f8ba47bc3eb5597",
        "source_sweep_output": str(sweep_root),
        "source_checkpoint_count_verified": len(source_snapshot),
        "source_checkpoints_retrained_or_modified": False,
        "panel_users": PANEL_USERS,
        "held_conditions": EXPECTED_CONDITIONS,
        "seed": PANEL_SEED,
        "case_count": 12,
        "kd_weight": KD_WEIGHT,
        "deleted_label_masked_and_renormalized": True,
        "damage_aware_top_user_count": TOP_DRIFTED_COUNT,
        "damage_aware_weight": 3.0,
        "selection_data": "source validation only",
        "held_context_policy": "not accessed until both variant selections were frozen",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
    }
    write_json(output / "utility-pilot-cases.json", results)
    write_csv(output / "utility-pilot-cases.csv", results)
    write_json(
        output / "utility-pilot-summary.json",
        {"status": "completed", "provenance": provenance, "primary_gate": gate},
    )
    write_json(output / "artifact-checksums.json", output_checksums(output))
    print(f"UTILITY_PILOT_OUTPUT={output}", flush=True)


if __name__ == "__main__":
    main()
