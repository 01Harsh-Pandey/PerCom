from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import EXPECTED_CONDITIONS, Trace, make_protocol, scan_ntu_humanid
from .mechanism import (
    load_model,
    sha256_file,
    verify_checkpoint_immutability,
    verify_checkpoint_manifest,
    write_json,
)
from .train import TrainConfig, seed_everything
from .utility_pilot import (
    FORGET_MAX,
    KD_WEIGHT,
    PANEL_SEED,
    PANEL_USERS,
    RETAIN_DROP_MAX,
    batched_logits,
    context_balanced_retained,
    evaluate_context,
    masked_kl_per_example,
    materialize,
    save_stage_checkpoint,
    select_source_checkpoint,
    source_metrics,
    weighted_mean,
)


METHODS = ("fine_retain_kd", "anchored_retain_kd")
MAX_REPAIR_EPOCHS = 5
ANCHOR_WEIGHT = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Final exploratory UNSIR-derived utility pilot."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sweep-output", required=True)
    parser.add_argument("--previous-pilot-output", required=True)
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


def repair_checkpoint_schedule(
    example_count: int, batch_size: int, epochs: int = MAX_REPAIR_EPOCHS
) -> tuple[tuple[int, int], ...]:
    if example_count <= 0 or batch_size <= 0 or epochs <= 0:
        raise ValueError("example_count, batch_size, and epochs must be positive")
    batches = math.ceil(example_count / batch_size)
    return tuple(
        (epoch, batch_index)
        for epoch in range(1, epochs + 1)
        for batch_index in range(1, batches + 1)
    )


def forgotten_anchor_loss(logits: torch.Tensor, deleted_label: int) -> torch.Tensor:
    deleted_probability = torch.softmax(logits, dim=1)[:, deleted_label]
    return -torch.log1p(-deleted_probability.clamp(max=1.0 - 1e-7)).mean()


def require_frozen_source_selections(
    variants: Mapping[str, Mapping[str, object]]
) -> None:
    if set(variants) != set(METHODS):
        raise RuntimeError("Both method variants must finish before held evaluation")
    allowed = {"selected", "abstained"}
    if any(variant.get("status") not in allowed for variant in variants.values()):
        raise RuntimeError("Held evaluation requested before source selection froze")


def previous_case_map(cases: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    result = {str(case["case_id"]): case for case in cases}
    if len(result) != 12:
        raise ValueError(f"Expected 12 previous pilot cases, found {len(result)}")
    return result


def load_impairment_model(
    previous_root: Path,
    previous_case: Mapping[str, object],
    num_classes: int,
    device: torch.device,
) -> tuple[nn.Module, Mapping[str, object]]:
    trajectory = previous_case["variants"]["retain_kd"]["trajectory"]
    impairment = [candidate for candidate in trajectory if candidate["stage"] == "impair"]
    if not impairment:
        raise ValueError(f"No retained impairment checkpoints for {previous_case['case_id']}")
    final_impairment = max(impairment, key=lambda candidate: candidate["stage_index"])
    path = Path(str(final_impairment["checkpoint"]))
    try:
        relative = path.relative_to(previous_root)
    except ValueError as exc:
        raise ValueError(f"Impairment checkpoint is outside previous output: {path}") from exc
    return load_model(previous_root / relative, num_classes, device), final_impairment


def train_repair_variant(
    method: str,
    previous_root: Path,
    previous_case: Mapping[str, object],
    original: nn.Module,
    protocol,
    output: Path,
    workers: int,
    num_classes: int,
    device: torch.device,
) -> dict[str, object]:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    anchored = method == "anchored_retain_kd"
    model, impairment_candidate = load_impairment_model(
        previous_root, previous_case, num_classes, device
    )
    config = previous_case["variants"]["retain_kd"]["frozen_forgetting_config"]
    if config["repair_learning_rate"] != 0.001 or config["batch_size"] != 32:
        raise ValueError("Previous RetainKD configuration is not the frozen configuration")
    train_config = TrainConfig(batch_size=32, workers=workers, seed=PANEL_SEED)
    seed_everything(PANEL_SEED)

    retained_records = context_balanced_retained(
        protocol.retain_train, protocol.held_condition
    )
    retain_x, retain_y = materialize(retained_records, train_config, device)
    retain_teacher_logits = batched_logits(original, retain_x, 32, device)
    forget_validation_x, forget_validation_y = materialize(
        protocol.forget_validation, train_config, device
    )
    retain_validation_x, retain_validation_y = materialize(
        protocol.retain_validation, train_config, device
    )
    original_retain_logits = batched_logits(
        original, retain_validation_x, 32, device
    )
    original_retain_accuracy = float(
        (original_retain_logits.argmax(1) == retain_validation_y).float().mean()
    )

    retained_mask = torch.ones(len(retain_x), dtype=torch.bool)
    if anchored:
        forget_x, forget_y = materialize(protocol.forget_train, train_config, device)
        x_all = torch.cat((retain_x, forget_x))
        y_all = torch.cat((retain_y, forget_y))
        retained_mask = torch.cat(
            (retained_mask, torch.zeros(len(forget_x), dtype=torch.bool))
        )
        teacher_all = torch.cat(
            (retain_teacher_logits, torch.zeros(len(forget_x), num_classes))
        )
    else:
        x_all, y_all, teacher_all = retain_x, retain_y, retain_teacher_logits

    dataset = TensorDataset(x_all, y_all, retained_mask, teacher_all)
    expected_schedule = repair_checkpoint_schedule(len(dataset), 32)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["repair_learning_rate"])
    )
    trajectory = [
        {
            **dict(candidate),
            "reused_from_job_10114": True,
            "stage": "reused_impair",
        }
        for candidate in previous_case["variants"]["retain_kd"]["trajectory"]
        if candidate["stage"] == "impair"
    ]
    new_checkpoints: list[dict[str, object]] = []
    checkpoint_index = len(trajectory)

    for epoch in range(1, MAX_REPAIR_EPOCHS + 1):
        generator = torch.Generator().manual_seed(PANEL_SEED + epoch)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, generator=generator)
        for batch_index, (x, y, is_retained, teacher_logits) in enumerate(loader, start=1):
            model.train()
            x = x.to(device)
            y = y.to(device)
            is_retained = is_retained.to(device)
            teacher_logits = teacher_logits.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            if is_retained.any():
                ce = F.cross_entropy(logits[is_retained], y[is_retained])
                kd = masked_kl_per_example(
                    logits[is_retained],
                    teacher_logits[is_retained],
                    protocol.forget_label,
                ).mean()
                retained_loss = ce + KD_WEIGHT * kd
            else:
                ce = torch.zeros((), device=device)
                kd = torch.zeros((), device=device)
                retained_loss = torch.zeros((), device=device)
            if anchored and (~is_retained).any():
                anchor = forgotten_anchor_loss(
                    logits[~is_retained], protocol.forget_label
                )
            else:
                anchor = torch.zeros((), device=device)
            loss = retained_loss + ANCHOR_WEIGHT * anchor
            loss.backward()
            optimizer.step()

            checkpoint_index += 1
            metrics = source_metrics(
                model,
                original,
                forget_validation_x,
                forget_validation_y,
                retain_validation_x,
                retain_validation_y,
                original_retain_logits,
                original_retain_accuracy,
                protocol.forget_label,
                32,
                device,
            )
            checkpoint_path = (
                output
                / method
                / f"repair-epoch-{epoch:03d}-batch-{batch_index:03d}.pt"
            )
            candidate = {
                "checkpoint_index": checkpoint_index,
                "stage": "repair_minibatch",
                "repair_epoch": epoch,
                "repair_minibatch": batch_index,
                "checkpoint": str(checkpoint_path),
                "loss": float(loss.detach()),
                "cross_entropy": float(ce.detach()),
                "distillation_loss": float(kd.detach()),
                "anchor_loss": float(anchor.detach()),
                "anchor_weight": ANCHOR_WEIGHT if anchored else 0.0,
                **metrics,
            }
            save_stage_checkpoint(
                checkpoint_path,
                model,
                {"method": method, "candidate": candidate, "source_only": True},
            )
            trajectory.append(candidate)
            new_checkpoints.append(candidate)

    observed_schedule = tuple(
        (candidate["repair_epoch"], candidate["repair_minibatch"])
        for candidate in new_checkpoints
    )
    if observed_schedule != expected_schedule:
        raise RuntimeError("Repair minibatch checkpoint schedule is incomplete")
    selected = select_source_checkpoint(trajectory, FORGET_MAX, RETAIN_DROP_MAX)
    return {
        "method": method,
        "status": "selected" if selected is not None else "abstained",
        "selected": dict(selected) if selected is not None else None,
        "trajectory": trajectory,
        "new_repair_checkpoint_count": len(new_checkpoints),
        "repair_epochs": MAX_REPAIR_EPOCHS,
        "repair_learning_rate": config["repair_learning_rate"],
        "kd_weight": KD_WEIGHT,
        "anchor_weight": ANCHOR_WEIGHT if anchored else 0.0,
        "forgotten_examples_replayed": len(protocol.forget_train) if anchored else 0,
        "impairment_checkpoint_reused": impairment_candidate["checkpoint"],
        "held_context_used_for_selection": False,
    }


def mean_metric(
    cases: Sequence[Mapping[str, object]], method: str, context: str, metric: str
) -> float | None:
    values = [
        case["evaluation"][context][method][metric]
        for case in cases
        if method in case["evaluation"][context]
    ]
    return float(np.mean(values)) if values else None


def existing_retain_kd_summary(
    previous_cases: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    selected = [
        case
        for case in previous_cases
        if case["variants"]["retain_kd"]["status"] == "selected"
    ]

    def previous_mean(context: str, metric: str) -> float | None:
        values = [
            case["evaluation"][context]["retain_kd"][metric]
            for case in selected
            if "retain_kd" in case["evaluation"][context]
        ]
        return float(np.mean(values)) if values else None

    return {
        "reused_without_rerun": True,
        "selected_cases": len(selected),
        "abstained_cases": len(previous_cases) - len(selected),
        "mean_held_forget_accuracy": previous_mean("held", "forget_accuracy"),
        "mean_seen_retain_drop": previous_mean(
            "seen", "retain_accuracy_drop_from_original"
        ),
        "mean_held_retain_drop": previous_mean(
            "held", "retain_accuracy_drop_from_original"
        ),
        "mean_js_to_retraining_seen_held": float(
            np.mean(
                [
                    previous_mean("seen", "js_to_exact_retraining"),
                    previous_mean("held", "js_to_exact_retraining"),
                ]
            )
        ),
    }


def method_gate(cases: Sequence[Mapping[str, object]], method: str) -> dict[str, object]:
    selected = sum(case["variants"][method]["status"] == "selected" for case in cases)
    held_forget = mean_metric(cases, method, "held", "forget_accuracy")
    seen_drop = mean_metric(
        cases, method, "seen", "retain_accuracy_drop_from_original"
    )
    held_drop = mean_metric(
        cases, method, "held", "retain_accuracy_drop_from_original"
    )
    method_js_values = [
        value
        for context in ("seen", "held")
        if (value := mean_metric(cases, method, context, "js_to_exact_retraining"))
        is not None
    ]
    original_js_values = [
        mean_metric(cases, "original", context, "js_to_exact_retraining")
        for context in ("seen", "held")
    ]
    method_js = float(np.mean(method_js_values)) if method_js_values else None
    original_js = float(np.mean(original_js_values))
    criteria = {
        "source_valid_success_at_least_9_of_12": selected >= 9,
        "held_forgetting_at_most_0_10": held_forget is not None and held_forget <= 0.10,
        "seen_retain_drop_at_most_0_05": seen_drop is not None and seen_drop <= 0.05,
        "held_retain_drop_at_most_0_05": held_drop is not None and held_drop <= 0.05,
        "js_not_worse_than_original": method_js is not None and method_js <= original_js,
    }
    return {
        "passes_previous_gates": all(criteria.values()),
        "criteria": criteria,
        "selected_cases": selected,
        "mean_held_forget_accuracy": held_forget,
        "mean_seen_retain_drop": seen_drop,
        "mean_held_retain_drop": held_drop,
        "mean_js_to_retraining_seen_held": method_js,
        "original_mean_js_to_retraining_seen_held": original_js,
    }


def family_decision(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    fine = method_gate(cases, "fine_retain_kd")
    anchored = method_gate(cases, "anchored_retain_kd")
    adds_two = anchored["selected_cases"] >= fine["selected_cases"] + 2
    no_utility_worsening = bool(
        anchored["mean_seen_retain_drop"] is not None
        and fine["mean_seen_retain_drop"] is not None
        and anchored["mean_held_retain_drop"] is not None
        and fine["mean_held_retain_drop"] is not None
        and anchored["mean_seen_retain_drop"] <= fine["mean_seen_retain_drop"]
        and anchored["mean_held_retain_drop"] <= fine["mean_held_retain_drop"]
    )
    no_js_worsening = bool(
        anchored["mean_js_to_retraining_seen_held"] is not None
        and fine["mean_js_to_retraining_seen_held"] is not None
        and anchored["mean_js_to_retraining_seen_held"]
        <= fine["mean_js_to_retraining_seen_held"]
    )
    anchored_alternative = adds_two and no_utility_worsening and no_js_worsening
    if fine["passes_previous_gates"]:
        selected_method = "fine_retain_kd"
        stop_family = False
        reason = "FineRetainKD passed all previous gates and is preferred."
    elif anchored["passes_previous_gates"] or anchored_alternative:
        selected_method = "anchored_retain_kd"
        stop_family = False
        reason = "FineRetainKD failed and AnchoredRetainKD met an allowed decision rule."
    else:
        selected_method = None
        stop_family = True
        reason = "Neither final UNSIR-derived variant met the decision rule."
    return {
        "selected_method": selected_method,
        "stop_unsir_derived_method_family": stop_family,
        "no_further_adjustment_allowed": stop_family,
        "reason": reason,
        "fine_retain_kd": fine,
        "anchored_retain_kd": anchored,
        "anchored_adds_two_valid_cases": adds_two,
        "anchored_no_mean_utility_worsening": no_utility_worsening,
        "anchored_no_mean_js_worsening": no_js_worsening,
        "anchored_alternative_rule_passed": anchored_alternative,
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


def write_csv(path: Path, cases: Sequence[Mapping[str, object]]) -> None:
    rows = [flatten(case) for case in cases]
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checksums(output: Path) -> dict[str, str]:
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
    previous_root = Path(args.previous_pilot_output).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output {output}")
    output.mkdir(parents=True, exist_ok=True)

    sweep_manifest = json.loads((sweep_root / "checksums.json").read_text())
    previous_manifest = json.loads((previous_root / "artifact-checksums.json").read_text())
    sweep_snapshot = verify_checkpoint_manifest(sweep_root, sweep_manifest)
    previous_snapshot = verify_checkpoint_manifest(previous_root, previous_manifest)
    previous_case_list = json.loads(
        (previous_root / "utility-pilot-cases.json").read_text()
    )
    previous_cases = previous_case_map(previous_case_list)
    records = scan_ntu_humanid(args.data_root)
    num_classes = len({record.user for record in records})
    config = TrainConfig(batch_size=32, workers=args.workers, seed=PANEL_SEED)

    results: list[dict[str, object]] = []
    for index, user in enumerate(PANEL_USERS):
        for held_index, held in enumerate(EXPECTED_CONDITIONS):
            case_number = index * len(EXPECTED_CONDITIONS) + held_index + 1
            case_id = f"user-{user}__held-{held}__seed-1"
            print(f"FINAL PANEL {case_number:02d}/12 {case_id}", flush=True)
            previous_case = previous_cases[case_id]
            protocol = make_protocol(records, user, held)
            original_path = sweep_root / "originals" / f"original-held-{held}-seed-1.pt"
            retrained_path = sweep_root / "case-artifacts" / case_id / "retrained.pt"
            original = load_model(original_path, num_classes, device)
            retrained = load_model(retrained_path, num_classes, device)

            variants = {
                method: train_repair_variant(
                    method,
                    previous_root,
                    previous_case,
                    original,
                    protocol,
                    output / "checkpoints" / case_id,
                    args.workers,
                    num_classes,
                    device,
                )
                for method in METHODS
            }
            require_frozen_source_selections(variants)

            models: dict[str, nn.Module] = {
                "original": original,
                "retrained": retrained,
            }
            for method, variant in variants.items():
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
                    config,
                    device,
                )
                for context, (forget_records, retain_records) in contexts.items()
            }
            evaluation["existing_retain_kd"] = previous_case["evaluation"]
            results.append(
                {
                    "case_id": case_id,
                    "forget_user": user,
                    "held_condition": held,
                    "seed": PANEL_SEED,
                    "existing_retain_kd_reused_without_rerun": True,
                    "existing_retain_kd": previous_case["variants"]["retain_kd"],
                    "variants": variants,
                    "evaluation": evaluation,
                    "held_context_used_before_both_selections_frozen": False,
                }
            )
            del models, original, retrained
            torch.cuda.empty_cache()

    verify_checkpoint_immutability(sweep_root, sweep_snapshot)
    verify_checkpoint_immutability(previous_root, previous_snapshot)
    decision = family_decision(results)
    decision["existing_retain_kd"] = existing_retain_kd_summary(
        previous_case_list
    )
    decision["compared_methods"] = (
        "existing_retain_kd",
        "fine_retain_kd",
        "anchored_retain_kd",
    )
    provenance = {
        "study_type": "final exploratory UNSIR-derived method pilot; not paper evidence",
        "git_commit": git_commit(),
        "branch": "codex/final-unsir-method-pilot",
        "parent_commit": "de015d001f8c99c8b38531b4ad3a2c571272aa99",
        "job_10114_preserved_and_reused": True,
        "existing_retain_kd_rerun": False,
        "source_sweep_output": str(sweep_root),
        "previous_pilot_output": str(previous_root),
        "verified_sweep_checkpoint_count": len(sweep_snapshot),
        "verified_previous_checkpoint_count": len(previous_snapshot),
        "original_or_retrained_models_trained": False,
        "panel_users": PANEL_USERS,
        "held_conditions": EXPECTED_CONDITIONS,
        "seed": PANEL_SEED,
        "case_count": 12,
        "max_repair_epochs": MAX_REPAIR_EPOCHS,
        "repair_checkpoint_frequency": "every minibatch",
        "anchor_weight": ANCHOR_WEIGHT,
        "anchor_definition": "-log(1 - deleted-label probability); no target retained identity",
        "selection_data": "source validation only",
        "held_context_policy": "inaccessible until both selections froze",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
    }
    write_json(output / "final-pilot-cases.json", results)
    write_csv(output / "final-pilot-cases.csv", results)
    write_json(
        output / "final-pilot-summary.json",
        {"status": "completed", "provenance": provenance, "decision": decision},
    )
    write_json(output / "artifact-checksums.json", checksums(output))
    print(f"FINAL_UNSIR_PILOT_OUTPUT={output}", flush=True)


if __name__ == "__main__":
    main()
