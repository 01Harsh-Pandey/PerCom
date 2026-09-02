from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Mapping, Sequence

import numpy as np
import torch

from .data import EXPECTED_CONDITIONS, Protocol, make_protocol, scan_ntu_humanid
from .evaluate import labels_logits_features
from .mechanism import load_model, sha256_file, write_json
from .train import TrainConfig


METHODS = ("original", "retrained", "unsir_candidate_15", "fine_retain_kd")
SCORES = ("max_softmax", "energy", "prototype_distance")
TAR_TARGETS = (0.90, 0.95, 0.99)
DEVELOPMENT_USERS = frozenset({"001", "005", "009", "015"})
DEVELOPMENT_SEED = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only open-set authentication audit.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--candidate-output", required=True)
    parser.add_argument("--fine-output", required=True)
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


def partition_cases(cases: Sequence[Mapping[str, object]]):
    development = [case for case in cases if bool(case["development_case"])]
    primary = [case for case in cases if not bool(case["development_case"])]
    if len(development) != 12 or len(primary) != 114:
        raise ValueError(
            f"Expected development/primary partition 12/114, got {len(development)}/{len(primary)}"
        )
    return development, primary


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def logsumexp(logits: np.ndarray) -> np.ndarray:
    maximum = logits.max(axis=1)
    return maximum + np.log(np.exp(logits - maximum[:, None]).sum(axis=1))


def threshold_for_target(source_retained_scores: Sequence[float], target: float) -> float:
    """Lowest source-score cutoff giving empirical acceptance of at least target."""
    scores = np.sort(np.asarray(source_retained_scores, dtype=np.float64))
    if not len(scores) or not 0.0 < target <= 1.0:
        raise ValueError("Scores must be non-empty and target must be in (0, 1]")
    rejected = math.floor((1.0 - target) * len(scores) + 1e-12)
    return float(scores[min(rejected, len(scores) - 1)])


def calibrate_thresholds(
    source_scores: Mapping[str, np.ndarray],
    targets: Sequence[float] = TAR_TARGETS,
) -> dict[str, dict[str, dict[str, float]]]:
    """Calibrate from source retained scores; held scores are not accepted here."""
    result: dict[str, dict[str, dict[str, float]]] = {}
    for score_name in SCORES:
        values = np.asarray(source_scores[score_name], dtype=np.float64)
        result[score_name] = {}
        for target in targets:
            threshold = threshold_for_target(values, target)
            result[score_name][f"tar_{int(target * 100)}"] = {
                "target_retained_tar": target,
                "threshold": threshold,
                "source_retained_tar_achieved": float(np.mean(values >= threshold)),
                "source_example_count": int(len(values)),
            }
    return result


def retained_logits(logits: np.ndarray, deleted_label: int):
    labels = np.asarray(
        [label for label in range(logits.shape[1]) if label != deleted_label],
        dtype=np.int64,
    )
    return logits[:, labels], labels


def prototypes(features: np.ndarray, labels: np.ndarray, deleted_label: int):
    retained_labels = np.asarray(
        sorted(label for label in np.unique(labels) if label != deleted_label),
        dtype=np.int64,
    )
    centers = np.stack([features[labels == label].mean(axis=0) for label in retained_labels])
    return centers, retained_labels


def prototype_scores(features: np.ndarray, centers: np.ndarray, labels: np.ndarray):
    distances = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
    nearest = distances.argmin(axis=1)
    return -distances[np.arange(len(features)), nearest], labels[nearest]


def acceptance_scores(
    logits: np.ndarray,
    features: np.ndarray,
    deleted_label: int,
    centers: np.ndarray,
    center_labels: np.ndarray,
):
    kept_logits, kept_labels = retained_logits(logits, deleted_label)
    probabilities = softmax(kept_logits)
    nearest_probability = probabilities.argmax(axis=1)
    prototype_score, prototype_prediction = prototype_scores(
        features, centers, center_labels
    )
    return {
        "scores": {
            "max_softmax": probabilities.max(axis=1),
            # Conventional energy is -logsumexp; this sign is reversed so all
            # three audit scores consistently accept when score >= threshold.
            "energy": logsumexp(kept_logits),
            "prototype_distance": prototype_score,
        },
        "predictions": {
            "max_softmax": kept_labels[nearest_probability],
            "energy": kept_labels[nearest_probability],
            "prototype_distance": prototype_prediction,
        },
        "retained_probabilities": probabilities,
        "retained_labels": kept_labels,
    }


def binary_auroc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """Tie-correct Mann-Whitney AUROC; retained samples are positive."""
    positive = np.asarray(positive_scores, dtype=np.float64)
    negative = np.asarray(negative_scores, dtype=np.float64)
    if not len(positive) or not len(negative):
        raise ValueError("Both AUROC classes must be non-empty")
    wins = (positive[:, None] > negative[None, :]).sum()
    ties = (positive[:, None] == negative[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(positive) * len(negative)))


def operating_point(
    threshold: float,
    deleted_scores: np.ndarray,
    retained_scores: np.ndarray,
    deleted_predictions: np.ndarray,
    retained_predictions: np.ndarray,
    retained_true_labels: np.ndarray,
    deleted_assigned_confidence: np.ndarray,
) -> dict[str, object]:
    deleted_accepted = np.asarray(deleted_scores) >= threshold
    retained_accepted = np.asarray(retained_scores) >= threshold
    accepted_confidence = deleted_assigned_confidence[deleted_accepted]
    return {
        "deleted_false_acceptance_rate": float(deleted_accepted.mean()),
        "deleted_rejection_rate": float(1.0 - deleted_accepted.mean()),
        "retained_true_acceptance_rate": float(retained_accepted.mean()),
        "retained_false_rejection_rate": float(1.0 - retained_accepted.mean()),
        "retained_correct_identity_acceptance_rate": float(
            np.mean(retained_accepted & (retained_predictions == retained_true_labels))
        ),
        "retained_misidentification_acceptance_rate": float(
            np.mean(retained_accepted & (retained_predictions != retained_true_labels))
        ),
        "deleted_accepted_count": int(deleted_accepted.sum()),
        "retained_accepted_count": int(retained_accepted.sum()),
        "mean_incorrect_retained_identity_confidence_when_accepted": (
            float(accepted_confidence.mean()) if len(accepted_confidence) else None
        ),
        "accepted_deleted_identity_counts": dict(
            sorted(Counter(map(str, deleted_predictions[deleted_accepted])).items())
        ),
    }


def calibrate_model(
    model: torch.nn.Module,
    protocol: Protocol,
    config: TrainConfig,
    device: torch.device,
):
    train_y, _, train_features = labels_logits_features(
        model, protocol.retain_train, config, device
    )
    centers, center_labels = prototypes(train_features, train_y, protocol.forget_label)
    val_y, val_logits, val_features = labels_logits_features(
        model, protocol.retain_validation, config, device
    )
    source = acceptance_scores(
        val_logits,
        val_features,
        protocol.forget_label,
        centers,
        center_labels,
    )
    thresholds = calibrate_thresholds(source["scores"])
    frozen = json.loads(json.dumps(thresholds, allow_nan=False))
    return frozen, centers, center_labels, {
        "source_retained_examples": int(len(val_y)),
        "prototype_training_examples": int(len(train_y)),
        "prototype_source": "source retained training data only",
        "threshold_source": "source retained validation data only",
        "held_context_used": False,
    }


def evaluate_held(
    model: torch.nn.Module,
    protocol: Protocol,
    config: TrainConfig,
    device: torch.device,
    frozen_thresholds: Mapping[str, Mapping[str, Mapping[str, float]]],
    centers: np.ndarray,
    center_labels: np.ndarray,
) -> dict[str, object]:
    deleted_y, deleted_logits, deleted_features = labels_logits_features(
        model, protocol.forget_held_test, config, device
    )
    retained_y, retained_logits_values, retained_features = labels_logits_features(
        model, protocol.retain_held_test, config, device
    )
    deleted = acceptance_scores(
        deleted_logits,
        deleted_features,
        protocol.forget_label,
        centers,
        center_labels,
    )
    retained = acceptance_scores(
        retained_logits_values,
        retained_features,
        protocol.forget_label,
        centers,
        center_labels,
    )
    deleted_probability_prediction = deleted["predictions"]["max_softmax"]
    probability_index = {
        int(label): index for index, label in enumerate(deleted["retained_labels"])
    }
    top_confidence = deleted["retained_probabilities"].max(axis=1)
    scores_out: dict[str, object] = {}
    for score_name in SCORES:
        deleted_prediction = deleted["predictions"][score_name]
        assigned_confidence = np.asarray(
            [
                deleted["retained_probabilities"][row, probability_index[int(label)]]
                for row, label in enumerate(deleted_prediction)
            ]
        )
        operating_points = {}
        for target_name, calibration in frozen_thresholds[score_name].items():
            operating_points[target_name] = {
                **dict(calibration),
                **operating_point(
                    float(calibration["threshold"]),
                    deleted["scores"][score_name],
                    retained["scores"][score_name],
                    deleted_prediction,
                    retained["predictions"][score_name],
                    retained_y,
                    assigned_confidence,
                ),
            }
        scores_out[score_name] = {
            "auroc_retained_vs_deleted": binary_auroc(
                retained["scores"][score_name], deleted["scores"][score_name]
            ),
            "operating_points": operating_points,
        }
    return {
        "deleted_examples": int(len(deleted_y)),
        "retained_examples": int(len(retained_y)),
        "deleted_label_removed_from_authentication_identities": True,
        "deleted_assignment": {
            "mean_incorrect_retained_identity_confidence": float(top_confidence.mean()),
            "median_incorrect_retained_identity_confidence": float(np.median(top_confidence)),
            "maximum_incorrect_retained_identity_confidence": float(top_confidence.max()),
            "assigned_retained_identity_counts": dict(
                sorted(Counter(map(str, deleted_probability_prediction)).items())
            ),
        },
        "scores": scores_out,
    }


class SourceVerifier:
    def __init__(self, roots: Sequence[Path]):
        self.roots = tuple(root.resolve() for root in roots)
        self.manifests = {
            root: json.loads((root / "checksums.json").read_text()) for root in self.roots
        }
        self.snapshots: dict[Path, dict[str, object]] = {}

    def verify(self, path: Path) -> dict[str, object]:
        path = path.resolve()
        if path in self.snapshots:
            return self.snapshots[path]
        root = next((root for root in self.roots if path.is_relative_to(root)), None)
        if root is None:
            raise ValueError(f"Checkpoint is outside verified roots: {path}")
        relative = path.relative_to(root).as_posix()
        expected = self.manifests[root].get(relative)
        if expected is None:
            raise KeyError(f"Checkpoint absent from checksum manifest: {relative}")
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"Checksum mismatch for {path}")
        stat = path.stat()
        snapshot = {
            "path": str(path),
            "sha256": observed,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        self.snapshots[path] = snapshot
        return snapshot

    def assert_immutable(self) -> None:
        for path, before in self.snapshots.items():
            stat = path.stat()
            if (
                stat.st_size != before["size"]
                or stat.st_mtime_ns != before["mtime_ns"]
                or sha256_file(path) != before["sha256"]
            ):
                raise RuntimeError(f"Source checkpoint changed during audit: {path}")


def summarize(values: Sequence[float]) -> dict[str, float | int | str]:
    if not values:
        return {"count": 0}
    spread = stdev(values) if len(values) > 1 else 0.0
    center = mean(values)
    margin = 1.96 * spread / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": center,
        "median": median(values),
        "sample_std": spread,
        "ci95_low": center - margin,
        "ci95_high": center + margin,
        "ci95_method": "normal approximation",
    }


def numeric_paths(value: object, prefix: str = "", output=None):
    output = {} if output is None else output
    if isinstance(value, Mapping):
        for key, child in value.items():
            numeric_paths(child, f"{prefix}.{key}" if prefix else str(key), output)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if math.isfinite(float(value)):
            output[prefix] = float(value)
    return output


def aggregate_subset(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    models: dict[str, object] = {}
    for method in METHODS:
        available = [
            case for case in cases if method in case.get("models", {})
        ]
        flattened = [numeric_paths(case["models"][method]["held"]) for case in available]
        paths = sorted({path for row in flattened for path in row})
        models[method] = {
            "available_cases": len(available),
            "coverage": len(available) / len(cases) if cases else None,
            "case_ids": [case["case_id"] for case in available],
            "metrics": {
                path: summarize([row[path] for row in flattened if path in row])
                for path in paths
            },
        }
    return {"case_count": len(cases), "models": models}


def paired_fine_vs_retrained(cases: Sequence[Mapping[str, object]]) -> dict[str, object]:
    matched = [
        case
        for case in cases
        if "fine_retain_kd" in case.get("models", {})
        and "retrained" in case.get("models", {})
    ]
    fine = [numeric_paths(case["models"]["fine_retain_kd"]["held"]) for case in matched]
    retrained = [numeric_paths(case["models"]["retrained"]["held"]) for case in matched]
    paths = sorted(set.intersection(*(set(row) for row in fine + retrained))) if matched else []
    return {
        "matched_case_count": len(matched),
        "matched_case_ids": [case["case_id"] for case in matched],
        "exact_retraining_is_not_treated_as_a_security_oracle": True,
        "fine_minus_retrained": {
            path: summarize([left[path] - right[path] for left, right in zip(fine, retrained)])
            for path in paths
        },
    }


def grouped(cases: Sequence[Mapping[str, object]], key: str) -> dict[str, object]:
    return {
        name: aggregate_subset([case for case in cases if str(case[key]) == name])
        for name in sorted({str(case[key]) for case in cases})
    }


def flatten_csv(value: object, prefix: str = "", output=None):
    output = {} if output is None else output
    if isinstance(value, Mapping):
        for key, child in value.items():
            flatten_csv(child, f"{prefix}.{key}" if prefix else str(key), output)
    elif isinstance(value, (list, tuple)):
        output[prefix] = json.dumps(value, sort_keys=True)
    else:
        output[prefix] = value
    return output


def write_cases(path: Path, cases: Sequence[Mapping[str, object]]) -> None:
    rows = [flatten_csv(case) for case in cases]
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checksums(output: Path) -> dict[str, str]:
    return {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "checksums.json"
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    candidate_root = Path(args.candidate_output).expanduser().resolve()
    fine_root = Path(args.fine_output).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output {output}")
    output.mkdir(parents=True, exist_ok=True)

    candidate_cases = {
        str(case["case_id"]): case
        for case in json.loads((candidate_root / "cases.json").read_text())
        if case.get("status") == "completed"
    }
    fine_cases = [
        case
        for case in json.loads((fine_root / "fineretainkd-cases.json").read_text())
        if case.get("status") == "completed"
    ]
    if len(candidate_cases) != 126 or len(fine_cases) != 126:
        raise RuntimeError("Source audits do not contain 126 completed cases each")
    verifier = SourceVerifier((candidate_root, fine_root))
    records = scan_ntu_humanid(args.data_root)
    users = tuple(sorted({record.user for record in records}))
    num_classes = len(users)
    results: list[dict[str, object]] = []

    for index, fine_case in enumerate(fine_cases, 1):
        case_id = str(fine_case["case_id"])
        print(f"OPEN_SET {index:03d}/126 {case_id}", flush=True)
        user = str(fine_case["forget_user"])
        held = str(fine_case["held_condition"])
        seed = int(fine_case["seed"])
        protocol = make_protocol(records, user, held)
        case_out: dict[str, object] = {
            "case_id": case_id,
            "forget_user": user,
            "held_condition": held,
            "seed": seed,
            "development_case": is_development_case(user, seed),
            "status": "completed",
            "models": {},
            "missing_models": [],
        }
        try:
            candidate_case = candidate_cases[case_id]
            paths: dict[str, Path] = {
                "original": candidate_root / "originals" / f"original-held-{held}-seed-{seed}.pt",
                "retrained": candidate_root / "case-artifacts" / case_id / "retrained.pt",
                "unsir_candidate_15": candidate_root / "case-artifacts" / case_id / "unsir-selected.pt",
            }
            selected = fine_case.get("selection", {}).get("selected")
            if selected is not None:
                paths["fine_retain_kd"] = Path(str(selected["checkpoint"]))
            else:
                case_out["missing_models"].append("fine_retain_kd_source_abstention")

            config = TrainConfig(batch_size=32, workers=args.workers, seed=seed)
            for method, path in paths.items():
                checkpoint = verifier.verify(path)
                model = load_model(path, num_classes, device)
                frozen, centers, center_labels, calibration = calibrate_model(
                    model, protocol, config, device
                )
                # The held split is loaded only after the JSON-safe threshold
                # snapshot above has been finalized.
                held_result = evaluate_held(
                    model,
                    protocol,
                    config,
                    device,
                    frozen,
                    centers,
                    center_labels,
                )
                case_out["models"][method] = {
                    "checkpoint": checkpoint,
                    "thresholds_frozen_before_held_access": frozen,
                    "calibration": calibration,
                    "held": held_result,
                }
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        except Exception as exc:
            case_out["status"] = "failed"
            case_out["error_type"] = type(exc).__name__
            case_out["error"] = str(exc)
            case_out["traceback"] = traceback.format_exc()
            print(traceback.format_exc(), flush=True)
        results.append(case_out)
        write_json(output / "open-set-cases.json", results)
        write_json(output / "progress.json", {"processed": len(results), "expected": 126})

    verifier.assert_immutable()
    development, primary = partition_cases(results)
    aggregates = {
        "all_126": aggregate_subset(results),
        "development_12": aggregate_subset(development),
        "primary_114": aggregate_subset(primary),
        "primary_by_user": grouped(primary, "forget_user"),
        "primary_by_condition": grouped(primary, "held_condition"),
        "primary_by_seed": grouped(primary, "seed"),
        "fine_vs_exact_retraining_primary_matched": paired_fine_vs_retrained(primary),
    }
    failure_map = []
    for case in results:
        if case["status"] != "completed" or case["missing_models"]:
            failure_map.append(
                {
                    "case_id": case["case_id"],
                    "development_case": case["development_case"],
                    "status": case["status"],
                    "missing_models": case["missing_models"],
                    "error": case.get("error"),
                }
            )
        for method, model in case.get("models", {}).items():
            for score, score_result in model["held"]["scores"].items():
                for target, metrics in score_result["operating_points"].items():
                    if metrics["deleted_false_acceptance_rate"] > 0.0:
                        failure_map.append(
                            {
                                "case_id": case["case_id"],
                                "development_case": case["development_case"],
                                "model": method,
                                "score": score,
                                "target": target,
                                "reason": "deleted_user_false_acceptance_observed",
                                "deleted_false_acceptance_rate": metrics[
                                    "deleted_false_acceptance_rate"
                                ],
                            }
                        )

    provenance = {
        "study": "read-only open-set authentication security audit",
        "git_commit": git_commit(),
        "candidate_output": str(candidate_root),
        "fine_output": str(fine_root),
        "used_checkpoint_count": len(verifier.snapshots),
        "checkpoint_updates": 0,
        "model_training_or_tuning": False,
        "threshold_targets": TAR_TARGETS,
        "threshold_data": "source retained validation only",
        "prototype_data": "source retained training only",
        "held_data_access": "after thresholds frozen",
        "energy_acceptance_score": "logsumexp over retained-label logits (negative conventional energy)",
        "deleted_label_removed_before_authentication_scoring": True,
        "exact_retraining_security_oracle_assumption": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
    }
    write_cases(output / "open-set-cases.csv", results)
    write_json(output / "open-set-aggregates.json", aggregates)
    write_json(output / "failure-map.json", failure_map)
    write_json(
        output / "summary.json",
        {
            "status": "completed",
            "provenance": provenance,
            "case_counts": {
                "all": len(results),
                "development": len(development),
                "primary": len(primary),
                "failed": sum(case["status"] != "completed" for case in results),
            },
            "aggregates": aggregates,
        },
    )
    write_json(output / "checksums.json", checksums(output))
    print(f"OPEN_SET_AUDIT_OUTPUT={output}", flush=True)


if __name__ == "__main__":
    main()
