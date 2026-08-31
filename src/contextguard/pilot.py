from __future__ import annotations

import argparse
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
from .evaluate import evaluate_method
from .model import CSILeNet
from .train import TrainConfig, save_checkpoint, seed_everything, train_model
from .unlearning import UNSIRConfig, unsir_unlearn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one exact-retraining vs CIU-L/UNSIR cross-context pilot."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="outputs/pilot")
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
    return parser.parse_args()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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
        "base_train": len(protocol.base_train),
        "retrain": len(protocol.retrain),
        "forget_train": len(protocol.forget_train),
        "retain_train": len(protocol.retain_train),
        "forget_seen_test": len(protocol.forget_seen_test),
        "forget_held_test": len(protocol.forget_held_test),
        "retain_test": len(protocol.retain_test),
    }
    print(json.dumps({"provenance": provenance, "protocol": protocol_summary}, indent=2))

    print("\n=== TRAIN ORIGINAL ===", flush=True)
    original, original_training = train_model(
        CSILeNet(num_classes), protocol.base_train, train_config, device
    )
    save_checkpoint(
        output / "original.pt",
        original,
        {"training": original_training, "protocol": protocol_summary},
    )

    print("\n=== EXACT RETRAIN WITHOUT USER ===", flush=True)
    retrained, retrain_training = train_model(
        CSILeNet(num_classes), protocol.retrain, train_config, device
    )
    save_checkpoint(
        output / "retrained.pt",
        retrained,
        {"training": retrain_training, "protocol": protocol_summary},
    )

    print("\n=== CIU-L / UNSIR UNLEARNING ===", flush=True)
    unsir_config = UNSIRConfig(
        noise_samples=args.noise_samples,
        noise_steps=args.noise_steps,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
    )
    unsir, unsir_training = unsir_unlearn(
        original, protocol.retain_train, protocol.forget_label, unsir_config, device
    )
    save_checkpoint(
        output / "unsir.pt",
        unsir,
        {"training": unsir_training, "protocol": protocol_summary},
    )

    print("\n=== AUDIT ===", flush=True)
    metrics = {
        "original": evaluate_method(original, protocol, train_config, device),
        "retrained": evaluate_method(retrained, protocol, train_config, device),
        "unsir": evaluate_method(unsir, protocol, train_config, device),
    }
    retrain_auc = metrics["retrained"]["contextual_probe"]["auc"]
    for method in ("original", "unsir"):
        metrics[method]["contextual_unlearning_gap_auc"] = (
            metrics[method]["contextual_probe"]["auc"] - retrain_auc
        )
    metrics["retrained"]["contextual_unlearning_gap_auc"] = 0.0

    result = {
        "status": "completed",
        "provenance": provenance,
        "protocol": protocol_summary,
        "train_config": asdict(train_config),
        "unsir_config": asdict(unsir_config),
        "training": {
            "original": original_training,
            "retrained": retrain_training,
            "unsir": unsir_training,
        },
        "metrics": metrics,
    }
    with (output / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"RESULT_JSON={output / 'result.json'}", flush=True)


if __name__ == "__main__":
    main()

