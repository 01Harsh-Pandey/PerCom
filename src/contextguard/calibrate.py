from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch

from .data import make_protocol, scan_ntu_humanid
from .evaluate import classification_accuracy, evaluate_method, predictive_equivalence
from .model import CSILeNet
from .train import TrainConfig, save_checkpoint, seed_everything, train_model
from .unlearning import UNSIRConfig, unsir_unlearn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validation-only calibration followed by one held-context audit."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="outputs/calibration")
    parser.add_argument("--forget-user", default="001")
    parser.add_argument("--held-condition", choices=("a", "b", "c"), default="c")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--noise-samples", type=int, default=16)
    parser.add_argument("--noise-steps", type=int, default=40)
    parser.add_argument("--max-retain-drop", type=float, default=0.05)
    parser.add_argument("--max-forget-accuracy", type=float, default=0.10)
    parser.add_argument("--min-base-forget-validation", type=float, default=0.75)
    parser.add_argument("--min-base-retain-validation", type=float, default=0.85)
    return parser.parse_args()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def candidate_configs(args: argparse.Namespace) -> list[UNSIRConfig]:
    # Frozen before viewing any held-condition result.
    configs = []
    for impair_lr in (3e-4, 1e-3, 3e-3):
        for repair_lr in (3e-4, 1e-3, 3e-3):
            for repair_epochs in (1, 3):
                configs.append(
                    UNSIRConfig(
                        noise_samples=args.noise_samples,
                        noise_steps=args.noise_steps,
                        impair_learning_rate=impair_lr,
                        repair_learning_rate=repair_lr,
                        repair_epochs=repair_epochs,
                        batch_size=args.batch_size,
                        workers=args.workers,
                        seed=args.seed,
                    )
                )
    return configs


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    device = torch.device(args.device)
    seed_everything(args.seed)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    records = scan_ntu_humanid(args.data_root)
    protocol = make_protocol(records, args.forget_user, args.held_condition)
    num_classes = len({record.label for record in records})
    train_config = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        patience=args.patience,
    )
    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }
    protocol_summary = {
        "forget_user": protocol.forget_user,
        "forget_label": protocol.forget_label,
        "held_condition": protocol.held_condition,
        "base_train_pool": len(protocol.base_train),
        "retrain_pool": len(protocol.retrain),
        "forget_unlearning_train": len(protocol.forget_train),
        "retain_unlearning_train": len(protocol.retain_train),
        "forget_calibration": len(protocol.forget_validation),
        "retain_calibration": len(protocol.retain_validation),
        "forget_seen_test": len(protocol.forget_seen_test),
        "forget_held_test": len(protocol.forget_held_test),
        "retain_seen_test": len(protocol.retain_seen_test),
        "retain_held_test": len(protocol.retain_held_test),
    }

    print("=== TRAIN BALANCED ORIGINAL ===", flush=True)
    original, original_training = train_model(
        CSILeNet(num_classes), protocol.base_train, train_config, device
    )
    save_checkpoint(
        output / "original.pt",
        original,
        {"training": original_training, "protocol": protocol_summary},
    )
    original_forget_validation = classification_accuracy(
        original, protocol.forget_validation, train_config, device
    )
    original_retain_validation = classification_accuracy(
        original, protocol.retain_validation, train_config, device
    )
    base_gate_passed = (
        original_forget_validation >= args.min_base_forget_validation
        and original_retain_validation >= args.min_base_retain_validation
    )
    base_gate = {
        "forget_validation_accuracy": original_forget_validation,
        "retain_validation_accuracy": original_retain_validation,
        "minimum_forget_validation_accuracy": args.min_base_forget_validation,
        "minimum_retain_validation_accuracy": args.min_base_retain_validation,
        "passed": base_gate_passed,
    }
    if not base_gate_passed:
        payload = {
            "status": "base_model_gate_failed",
            "provenance": provenance,
            "protocol": protocol_summary,
            "base_gate": base_gate,
            "train_config": asdict(train_config),
            "original_training": original_training,
            "held_context_was_evaluated": False,
        }
        write_json(output / "calibration.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
        return

    retain_floor = original_retain_validation - args.max_retain_drop
    candidates: list[dict[str, object]] = []
    best_key: tuple[float, ...] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_index: int | None = None
    best_training: dict[str, object] | None = None

    print("=== VALIDATION-ONLY UNSIR CALIBRATION ===", flush=True)
    for index, config in enumerate(candidate_configs(args)):
        print(f"candidate={index:02d} config={asdict(config)}", flush=True)
        candidate, training = unsir_unlearn(
            original, protocol.retain_train, protocol.forget_label, config, device
        )
        forget_accuracy = classification_accuracy(
            candidate, protocol.forget_validation, train_config, device
        )
        retain_accuracy = classification_accuracy(
            candidate, protocol.retain_validation, train_config, device
        )
        valid = (
            forget_accuracy <= args.max_forget_accuracy
            and retain_accuracy >= retain_floor
        )
        row = {
            "index": index,
            "config": asdict(config),
            "forget_validation_accuracy": forget_accuracy,
            "retain_validation_accuracy": retain_accuracy,
            "retain_accuracy_drop": original_retain_validation - retain_accuracy,
            "valid": valid,
            "noise_final_loss": training["noise_final_loss"],
            "impair_losses": training["impair_losses"],
            "repair_losses": training["repair_losses"],
        }
        candidates.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if valid:
            key = (
                retain_accuracy,
                -forget_accuracy,
                -config.impair_learning_rate,
                -config.repair_learning_rate,
                -float(config.repair_epochs),
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
                best_state = copy.deepcopy(candidate.state_dict())
                best_training = training

    calibration = {
        "status": "candidate_selected" if best_index is not None else "no_valid_candidate",
        "provenance": provenance,
        "protocol": protocol_summary,
        "base_gate": base_gate,
        "selection_constraints": {
            "maximum_forget_validation_accuracy": args.max_forget_accuracy,
            "maximum_retain_accuracy_drop": args.max_retain_drop,
            "absolute_retain_accuracy_floor": retain_floor,
        },
        "candidate_count": len(candidates),
        "selected_candidate_index": best_index,
        "candidates": candidates,
        "held_context_was_evaluated": False,
    }
    write_json(output / "calibration.json", calibration)
    if best_state is None or best_index is None or best_training is None:
        print(json.dumps(calibration, indent=2, sort_keys=True), flush=True)
        return

    selected = CSILeNet(num_classes).to(device)
    selected.load_state_dict(best_state)
    selected_config = candidate_configs(args)[best_index]
    save_checkpoint(
        output / "unsir-selected.pt",
        selected,
        {
            "training": best_training,
            "protocol": protocol_summary,
            "selection": candidates[best_index],
        },
    )

    print("=== TRAIN EXACT RETRAINING REFERENCE ===", flush=True)
    retrained, retrain_training = train_model(
        CSILeNet(num_classes), protocol.retrain, train_config, device
    )
    save_checkpoint(
        output / "retrained.pt",
        retrained,
        {"training": retrain_training, "protocol": protocol_summary},
    )

    print("=== SINGLE HELD-CONTEXT AUDIT ===", flush=True)
    metrics = {
        "original": evaluate_method(original, protocol, train_config, device),
        "retrained": evaluate_method(retrained, protocol, train_config, device),
        "unsir_selected": evaluate_method(selected, protocol, train_config, device),
    }
    for method_name, model in (("original", original), ("unsir_selected", selected)):
        metrics[method_name]["equivalence_to_retraining"] = {
            "forget_held": predictive_equivalence(
                model,
                retrained,
                protocol.forget_held_test,
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
    metrics["retrained"]["equivalence_to_retraining"] = {
        "forget_held": {"mean_js_divergence": 0.0, "prediction_agreement": 1.0},
        "retain_held": {"mean_js_divergence": 0.0, "prediction_agreement": 1.0},
    }
    result = {
        "status": "completed",
        "provenance": provenance,
        "protocol": protocol_summary,
        "base_gate": base_gate,
        "train_config": asdict(train_config),
        "selected_unsir_config": asdict(selected_config),
        "selected_candidate": candidates[best_index],
        "training": {
            "original": original_training,
            "retrained": retrain_training,
            "unsir_selected": best_training,
        },
        "metrics": metrics,
        "metric_note": (
            "Reidentification probe results are diagnostic identity separability, "
            "not evidence of residual training influence."
        ),
        "held_context_was_evaluated": True,
    }
    write_json(output / "result.json", result)
    calibration["held_context_was_evaluated"] = True
    write_json(output / "calibration.json", calibration)
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)
    print(f"RESULT_JSON={output / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()
