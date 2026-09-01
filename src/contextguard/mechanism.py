from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from .data import Trace, make_protocol, scan_ntu_humanid
from .evaluate import labels_logits_features
from .model import CSILeNet
from .train import TrainConfig, make_loader


METHODS = ("original", "unsir", "retrained")
CONTEXTS = ("seen", "held")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only failure-mechanism audit of frozen sweep checkpoints."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sweep-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


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


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def verify_checkpoint_manifest(
    root: Path, manifest: Mapping[str, str]
) -> dict[str, dict[str, object]]:
    checkpoint_entries = {
        relative: expected
        for relative, expected in manifest.items()
        if relative.endswith(".pt")
    }
    if not checkpoint_entries:
        raise ValueError("checksums.json contains no checkpoint entries")
    verified: dict[str, dict[str, object]] = {}
    for relative, expected in sorted(checkpoint_entries.items()):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(
                f"Checkpoint checksum mismatch for {relative}: "
                f"expected={expected}, observed={observed}"
            )
        stat = path.stat()
        verified[relative] = {
            "sha256": observed,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
    return verified


def verify_checkpoint_immutability(
    root: Path, before: Mapping[str, Mapping[str, object]]
) -> None:
    for relative, snapshot in sorted(before.items()):
        path = root / relative
        stat = path.stat()
        observed = sha256_file(path)
        if (
            observed != snapshot["sha256"]
            or stat.st_size != snapshot["size"]
            or stat.st_mtime_ns != snapshot["mtime_ns"]
        ):
            raise RuntimeError(f"Checkpoint changed during audit: {relative}")


def load_model(path: Path, num_classes: int, device: torch.device) -> CSILeNet:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = CSILeNet(num_classes)
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    return model


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def confusion_matrix_counts(
    labels: np.ndarray, predictions: np.ndarray, num_classes: int
) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (labels.astype(int), predictions.astype(int)), 1)
    return matrix


def class_accuracy(matrix: np.ndarray, label: int) -> float | None:
    total = int(matrix[label].sum())
    return float(matrix[label, label] / total) if total else None


def probability_redirection(
    original_probabilities: np.ndarray,
    unsir_probabilities: np.ndarray,
    retrained_probabilities: np.ndarray,
    forget_label: int,
) -> dict[str, object]:
    original_mean = original_probabilities.mean(axis=0)
    unsir_mean = unsir_probabilities.mean(axis=0)
    retrained_mean = retrained_probabilities.mean(axis=0)
    alternative = unsir_mean.copy()
    alternative[forget_label] = -np.inf
    redirected_label = int(alternative.argmax())
    return {
        "forgotten_label": forget_label,
        "top_redirected_label": redirected_label,
        "top_redirected_probability": float(unsir_mean[redirected_label]),
        "original_mean_probabilities": original_mean.tolist(),
        "unsir_mean_probabilities": unsir_mean.tolist(),
        "retrained_mean_probabilities": retrained_mean.tolist(),
        "unsir_minus_original": (unsir_mean - original_mean).tolist(),
        "unsir_minus_retrained": (unsir_mean - retrained_mean).tolist(),
    }


def feature_drift(reference: np.ndarray, compared: np.ndarray) -> dict[str, float]:
    delta = compared - reference
    l2 = np.linalg.norm(delta, axis=1)
    denominator = np.linalg.norm(reference, axis=1) * np.linalg.norm(compared, axis=1)
    cosine = np.divide(
        np.sum(reference * compared, axis=1),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return {
        "mean_l2": float(l2.mean()),
        "median_l2": float(np.median(l2)),
        "mean_cosine_similarity": float(cosine.mean()),
        "mean_cosine_distance": float((1.0 - cosine).mean()),
    }


def module_name(parameter_name: str) -> str:
    return parameter_name.rsplit(".", 1)[0]


def layer_parameter_drift(
    reference: Mapping[str, torch.Tensor], compared: Mapping[str, torch.Tensor]
) -> dict[str, dict[str, float]]:
    grouped_reference: dict[str, list[torch.Tensor]] = defaultdict(list)
    grouped_compared: dict[str, list[torch.Tensor]] = defaultdict(list)
    for name, tensor in reference.items():
        group = module_name(name)
        grouped_reference[group].append(tensor.detach().double().flatten().cpu())
        grouped_compared[group].append(
            compared[name].detach().double().flatten().cpu()
        )
    result: dict[str, dict[str, float]] = {}
    for group in sorted(grouped_reference):
        left = torch.cat(grouped_reference[group])
        right = torch.cat(grouped_compared[group])
        delta = right - left
        left_norm = float(torch.linalg.vector_norm(left))
        right_norm = float(torch.linalg.vector_norm(right))
        denominator = left_norm * right_norm
        result[group] = {
            "parameter_count": int(left.numel()),
            "delta_l2": float(torch.linalg.vector_norm(delta)),
            "relative_delta_l2": float(
                torch.linalg.vector_norm(delta) / max(left_norm, 1e-12)
            ),
            "cosine_similarity": (
                float(torch.dot(left, right) / denominator) if denominator else 0.0
            ),
        }
    return result


def mean_gradients(
    model: torch.nn.Module,
    records: Sequence[Trace],
    config: TrainConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    loader = make_loader(
        records, config, shuffle=False, persistent_workers=False
    )
    totals = {
        name: torch.zeros_like(parameter, device="cpu", dtype=torch.float64)
        for name, parameter in model.named_parameters()
    }
    example_count = 0
    model.eval()
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y, reduction="sum")
        loss.backward()
        example_count += y.numel()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                totals[name] += parameter.grad.detach().double().cpu()
    model.zero_grad(set_to_none=True)
    if not example_count:
        raise ValueError("Cannot compute gradients for an empty record sequence")
    return {name: value / example_count for name, value in totals.items()}


def gradient_conflict(
    forgotten: Mapping[str, torch.Tensor], retained: Mapping[str, torch.Tensor]
) -> dict[str, dict[str, float]]:
    grouped_forget: dict[str, list[torch.Tensor]] = defaultdict(list)
    grouped_retain: dict[str, list[torch.Tensor]] = defaultdict(list)
    for name, tensor in forgotten.items():
        group = module_name(name)
        grouped_forget[group].append(tensor.flatten().double())
        grouped_retain[group].append(retained[name].flatten().double())
    result: dict[str, dict[str, float]] = {}
    for group in sorted(grouped_forget):
        left = torch.cat(grouped_forget[group])
        right = torch.cat(grouped_retain[group])
        dot = float(torch.dot(left, right))
        left_norm = float(torch.linalg.vector_norm(left))
        right_norm = float(torch.linalg.vector_norm(right))
        denominator = left_norm * right_norm
        result[group] = {
            "cosine_similarity": dot / denominator if denominator else 0.0,
            "dot_product": dot,
            "forgotten_gradient_l2": left_norm,
            "retained_gradient_l2": right_norm,
            "conflict": bool(dot < 0),
        }
    return result


def correlation(x: Sequence[float], y: Sequence[float]) -> dict[str, float | int | None]:
    pairs = [
        (float(left), float(right))
        for left, right in zip(x, y)
        if math.isfinite(float(left)) and math.isfinite(float(right))
    ]
    if len(pairs) < 3:
        return {"count": len(pairs), "pearson_r": None, "spearman_rho": None}
    left = np.asarray([pair[0] for pair in pairs])
    right = np.asarray([pair[1] for pair in pairs])
    if np.std(left) == 0 or np.std(right) == 0:
        return {"count": len(pairs), "pearson_r": None, "spearman_rho": None}
    pearson_value = float(pearsonr(left, right).statistic)
    spearman_value = float(spearmanr(left, right).statistic)
    return {
        "count": len(pairs),
        "pearson_r": pearson_value if math.isfinite(pearson_value) else None,
        "spearman_rho": spearman_value if math.isfinite(spearman_value) else None,
    }


def aggregate_correlations(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    predictors = (
        "repair_loss_mean",
        "repair_loss_final",
        "retain_feature_drift_l2",
        "gradient_mean_cosine",
        "gradient_conflict_fraction",
    )
    result: dict[str, object] = {}
    for context in CONTEXTS:
        damage_key = f"utility_damage_{context}"
        result[context] = {
            predictor: correlation(
                [float(row[damage_key]) for row in rows],
                [float(row[predictor]) for row in rows],
            )
            for predictor in predictors
        }
    return result


def infer(
    model: torch.nn.Module,
    records: Sequence[Trace],
    config: TrainConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    labels, logits, features = labels_logits_features(model, records, config, device)
    return {
        "labels": labels,
        "predictions": logits.argmax(axis=1),
        "probabilities": softmax(logits),
        "features": features,
    }


def checkpoint_paths(root: Path, case: Mapping[str, object]) -> dict[str, Path]:
    case_id = str(case["case_id"])
    held = str(case["held_condition"])
    seed = int(case["seed"])
    return {
        "original": root / "originals" / f"original-held-{held}-seed-{seed}.pt",
        "unsir": root / "case-artifacts" / case_id / "unsir-selected.pt",
        "retrained": root / "case-artifacts" / case_id / "retrained.pt",
    }


def checkpoint_relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def output_checksums(output: Path) -> dict[str, str]:
    return {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact-checksums.json"
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    device = torch.device(args.device)
    sweep_root = Path(args.sweep_output).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((sweep_root / "checksums.json").read_text())
    checkpoint_snapshot = verify_checkpoint_manifest(sweep_root, manifest)
    cases = json.loads((sweep_root / "cases.json").read_text())
    if len(cases) != 126 or any(case.get("status") != "completed" for case in cases):
        raise ValueError("Mechanism audit requires all 126 completed sweep cases")

    records = scan_ntu_humanid(args.data_root)
    users = sorted({record.user for record in records})
    label_to_user = {
        record.label: record.user for record in records
    }
    num_classes = len(users)
    config = TrainConfig(
        epochs=1, batch_size=args.batch_size, workers=args.workers, seed=1
    )

    mechanism_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    redirection_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    matrices: list[np.ndarray] = []

    for case_number, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        print(f"CASE {case_number:03d}/126 {case_id}", flush=True)
        protocol = make_protocol(records, str(case["forget_user"]), str(case["held_condition"]))
        paths = checkpoint_paths(sweep_root, case)
        for path in paths.values():
            relative = checkpoint_relative(sweep_root, path)
            if relative not in checkpoint_snapshot:
                raise ValueError(f"Checkpoint absent from verified manifest: {relative}")
        models = {
            method: load_model(path, num_classes, device)
            for method, path in paths.items()
        }

        layer_drift = {
            "unsir": layer_parameter_drift(
                models["original"].state_dict(), models["unsir"].state_dict()
            ),
            "retrained": layer_parameter_drift(
                models["original"].state_dict(), models["retrained"].state_dict()
            ),
        }
        for comparison, layers in layer_drift.items():
            for layer, metrics in layers.items():
                layer_rows.append(
                    {"case_id": case_id, "comparison": comparison, "layer": layer, **metrics}
                )

        forgotten_gradients = mean_gradients(
            models["original"], protocol.forget_train, config, device
        )
        retained_gradients = mean_gradients(
            models["original"], protocol.retain_train, config, device
        )
        conflicts = gradient_conflict(forgotten_gradients, retained_gradients)
        for layer, metrics in conflicts.items():
            gradient_rows.append({"case_id": case_id, "layer": layer, **metrics})
        gradient_mean_cosine = float(
            np.mean([metrics["cosine_similarity"] for metrics in conflicts.values()])
        )
        gradient_conflict_fraction = float(
            np.mean([metrics["conflict"] for metrics in conflicts.values()])
        )

        row: dict[str, object] = {
            "case_id": case_id,
            "forget_user": case["forget_user"],
            "held_condition": case["held_condition"],
            "seed": case["seed"],
            "eligible": case["eligible"],
            "success": case["success"],
            "repair_loss_mean": float(np.mean(case["training"]["unsir"]["repair_losses"])),
            "repair_loss_final": float(case["training"]["unsir"]["repair_losses"][-1]),
            "impair_loss": float(case["training"]["unsir"]["impair_losses"][-1]),
            "gradient_mean_cosine": gradient_mean_cosine,
            "gradient_conflict_fraction": gradient_conflict_fraction,
        }

        for context in CONTEXTS:
            forget_records = getattr(protocol, f"forget_{context}_test")
            retain_records = getattr(protocol, f"retain_{context}_test")
            combined_records = (*forget_records, *retain_records)
            outputs = {
                method: infer(model, combined_records, config, device)
                for method, model in models.items()
            }
            labels = outputs["original"]["labels"]
            forget_mask = labels == protocol.forget_label
            retain_mask = ~forget_mask
            context_matrices: dict[str, np.ndarray] = {}
            for method in METHODS:
                context_matrices[method] = confusion_matrix_counts(
                    labels[retain_mask], outputs[method]["predictions"][retain_mask], num_classes
                )
                matrices.append(context_matrices[method])

            for label, user in sorted(label_to_user.items()):
                if label == protocol.forget_label:
                    continue
                accuracies = {
                    method: class_accuracy(context_matrices[method], label)
                    for method in METHODS
                }
                per_class_rows.append(
                    {
                        "case_id": case_id,
                        "context": context,
                        "retained_user": user,
                        "retained_label": label,
                        "accuracy": accuracies,
                        "unsir_minus_original": accuracies["unsir"] - accuracies["original"],
                        "retrained_minus_original": accuracies["retrained"] - accuracies["original"],
                        "unsir_minus_retrained": accuracies["unsir"] - accuracies["retrained"],
                    }
                )

            redirection = probability_redirection(
                outputs["original"]["probabilities"][forget_mask],
                outputs["unsir"]["probabilities"][forget_mask],
                outputs["retrained"]["probabilities"][forget_mask],
                protocol.forget_label,
            )
            redirection["top_redirected_user"] = label_to_user[
                redirection["top_redirected_label"]
            ]
            redirection_rows.append(
                {"case_id": case_id, "context": context, **redirection}
            )

            for group, mask in (("forgotten", forget_mask), ("retained", retain_mask)):
                for comparison in ("unsir", "retrained"):
                    metrics = feature_drift(
                        outputs["original"]["features"][mask],
                        outputs[comparison]["features"][mask],
                    )
                    feature_rows.append(
                        {
                            "case_id": case_id,
                            "context": context,
                            "group": group,
                            "comparison": comparison,
                            **metrics,
                        }
                    )
                    if group == "retained" and comparison == "unsir":
                        row[f"retain_feature_drift_l2_{context}"] = metrics["mean_l2"]

            for method in METHODS:
                predictions = outputs[method]["predictions"]
                row[f"{method}_retain_accuracy_{context}"] = float(
                    np.mean(predictions[retain_mask] == labels[retain_mask])
                )
                row[f"{method}_forget_accuracy_{context}"] = float(
                    np.mean(predictions[forget_mask] == labels[forget_mask])
                )
            row[f"utility_damage_{context}"] = (
                row[f"original_retain_accuracy_{context}"]
                - row[f"unsir_retain_accuracy_{context}"]
            )
            row[f"retain_feature_drift_l2_{context}"] = float(
                row[f"retain_feature_drift_l2_{context}"]
            )

        # Context-specific aliases make correlation inputs explicit.
        for context in CONTEXTS:
            row["retain_feature_drift_l2"] = row[f"retain_feature_drift_l2_{context}"]
        mechanism_rows.append(row)
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Repair/gradient quantities are context-independent; feature drift is not.
    relationships: dict[str, object] = {}
    for context in CONTEXTS:
        context_rows = []
        for row in mechanism_rows:
            copy = dict(row)
            copy["retain_feature_drift_l2"] = row[f"retain_feature_drift_l2_{context}"]
            context_rows.append(copy)
        relationships[context] = aggregate_correlations(context_rows)[context]

    write_csv(output / "mechanism-cases.csv", mechanism_rows)
    write_json(
        output / "per-class-damage.json",
        {"retained_user_damage": per_class_rows, "forgotten_probability_redirection": redirection_rows},
    )
    with (output / "confusion-matrices.npz.tmp").open("wb") as handle:
        np.savez_compressed(
            handle,
            matrices=np.stack(matrices).reshape(len(cases), 2, 3, num_classes, num_classes),
            case_ids=np.asarray([case["case_id"] for case in cases]),
            contexts=np.asarray(CONTEXTS),
            methods=np.asarray(METHODS),
            users=np.asarray(users),
        )
    (output / "confusion-matrices.npz.tmp").replace(output / "confusion-matrices.npz")
    write_json(output / "layer-drift.json", layer_rows)
    write_json(output / "feature-drift.json", feature_rows)
    write_json(output / "gradient-conflict.json", gradient_rows)

    verify_checkpoint_immutability(sweep_root, checkpoint_snapshot)
    provenance = {
        "audit_git_commit": git_commit(),
        "source_sweep_git_commit": "9c9c630ddb04a1398d2e2fcdb282745a005b00f1",
        "source_sweep_output": str(sweep_root),
        "source_manifest_sha256": sha256_file(sweep_root / "checksums.json"),
        "verified_checkpoint_count": len(checkpoint_snapshot),
        "checkpoint_verification": "SHA256 before analysis; SHA256, size, and mtime rechecked after analysis",
        "checkpoint_access_policy": "read-only; no checkpoint is written or replaced",
        "case_count": len(cases),
        "contexts": CONTEXTS,
        "methods": METHODS,
        "gradient_definition": "mean cross-entropy parameter gradient on original model; forgotten source train versus retained source train",
        "held_context_policy": "reporting-only mechanism analysis; no selection, tuning, exclusion, or rerunning",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    summary = {
        "status": "completed",
        "provenance": provenance,
        "relationships": relationships,
        "record_counts": {
            "mechanism_cases": len(mechanism_rows),
            "per_class_damage": len(per_class_rows),
            "probability_redirection": len(redirection_rows),
            "layer_drift": len(layer_rows),
            "feature_drift": len(feature_rows),
            "gradient_conflict": len(gradient_rows),
            "confusion_matrices": len(matrices),
        },
        "no_training_or_unlearning_performed": True,
    }
    write_json(output / "mechanism-summary.json", summary)
    write_json(output / "artifact-checksums.json", output_checksums(output))
    print(f"MECHANISM_OUTPUT={output}", flush=True)


if __name__ == "__main__":
    main()
