from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import NTUHumanIDDataset, Trace, stratified_train_validation


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    workers: int = 4
    seed: int = 1
    patience: int = 8


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loader(
    records: Sequence[Trace], config: TrainConfig, shuffle: bool
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        NTUHumanIDDataset(records),
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.workers > 0,
        generator=generator,
    )


@torch.inference_mode()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        prediction = model(x).argmax(dim=1)
        correct += int((prediction == y).sum())
        total += y.numel()
    return correct / total if total else float("nan")


def train_model(
    model: nn.Module,
    records: Sequence[Trace],
    config: TrainConfig,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    seed_everything(config.seed)
    train_records, validation_records = stratified_train_validation(records)
    train_loader = make_loader(train_records, config, shuffle=True)
    validation_loader = make_loader(validation_records, config, shuffle=False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    best_state = copy.deepcopy(model.state_dict())
    best_accuracy = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        example_count = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * y.numel()
            example_count += y.numel()

        validation_accuracy = accuracy(model, validation_loader, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": loss_sum / max(example_count, 1),
                "validation_accuracy": validation_accuracy,
            }
        )
        print(
            f"epoch={epoch:03d} loss={history[-1]['train_loss']:.5f} "
            f"val_acc={validation_accuracy:.4f}",
            flush=True,
        )
        if validation_accuracy > best_accuracy + 1e-8:
            best_accuracy = validation_accuracy
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    model.load_state_dict(best_state)
    return model, {
        "config": asdict(config),
        "best_validation_accuracy": best_accuracy,
        "epochs_ran": len(history),
        "history": history,
    }


def save_checkpoint(
    path: str | Path, model: nn.Module, metadata: dict[str, object]
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "metadata": metadata}, path)

