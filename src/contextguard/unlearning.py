from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import NTUHumanIDDataset, Trace


@dataclass(frozen=True)
class UNSIRConfig:
    noise_samples: int = 16
    noise_steps: int = 40
    noise_learning_rate: float = 0.1
    noise_l2: float = 0.1
    retain_per_class: int = 10
    impair_learning_rate: float = 0.02
    repair_learning_rate: float = 0.01
    batch_size: int = 32
    workers: int = 4
    seed: int = 1


def _sample_retained(
    records: Sequence[Trace], per_class: int, seed: int
) -> tuple[Trace, ...]:
    by_label: dict[int, list[Trace]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record)
    rng = random.Random(seed)
    sampled: list[Trace] = []
    for label in sorted(by_label):
        candidates = list(by_label[label])
        rng.shuffle(candidates)
        sampled.extend(candidates[:per_class])
    return tuple(sampled)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total = 0
    for batch in loader:
        if isinstance(batch, dict):
            x, y = batch["x"], batch["y"]
        else:
            x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * y.numel()
        total += y.numel()
    return total_loss / max(total, 1)


def unsir_unlearn(
    source_model: nn.Module,
    retain_records: Sequence[Trace],
    forget_label: int,
    config: UNSIRConfig,
    device: torch.device,
) -> tuple[nn.Module, dict[str, object]]:
    """CIU-L/UNSIR-style targeted-noise impair-and-repair unlearning."""

    torch.manual_seed(config.seed)
    model = copy.deepcopy(source_model).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    noise = nn.Parameter(
        torch.randn(config.noise_samples, 3, 114, 500, device=device)
    )
    labels = torch.full(
        (config.noise_samples,), forget_label, dtype=torch.long, device=device
    )
    noise_optimizer = torch.optim.Adam([noise], lr=config.noise_learning_rate)
    noise_history: list[float] = []
    for step in range(config.noise_steps):
        noise_optimizer.zero_grad(set_to_none=True)
        # Gradient descent on negative CE maximizes the target-class error,
        # matching the public UNSIR implementation used by CIU-L.
        loss = -nn.functional.cross_entropy(model(noise), labels)
        loss = loss + config.noise_l2 * noise.square().mean()
        loss.backward()
        noise_optimizer.step()
        noise_history.append(float(loss.detach()))
        if (step + 1) % 10 == 0:
            print(f"noise_step={step + 1:03d} loss={noise_history[-1]:.5f}", flush=True)

    for parameter in model.parameters():
        parameter.requires_grad_(True)

    sampled_retained = _sample_retained(
        retain_records, config.retain_per_class, config.seed
    )
    retained_dataset = NTUHumanIDDataset(sampled_retained)
    retained_loader = DataLoader(
        retained_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Materialize only the small retained replay set for the one-epoch impair step.
    retained_x: list[torch.Tensor] = []
    retained_y: list[torch.Tensor] = []
    for batch in retained_loader:
        retained_x.append(batch["x"])
        retained_y.append(batch["y"])
    impair_x = torch.cat([noise.detach().cpu(), *retained_x], dim=0)
    impair_y = torch.cat([labels.detach().cpu(), *retained_y], dim=0)
    impair_loader = DataLoader(
        TensorDataset(impair_x, impair_y),
        batch_size=config.batch_size,
        shuffle=True,
    )
    impair_optimizer = torch.optim.Adam(
        model.parameters(), lr=config.impair_learning_rate
    )
    impair_loss = _train_epoch(model, impair_loader, impair_optimizer, device)

    repair_optimizer = torch.optim.Adam(
        model.parameters(), lr=config.repair_learning_rate
    )
    repair_loss = _train_epoch(model, retained_loader, repair_optimizer, device)
    return model, {
        "config": asdict(config),
        "noise_final_loss": noise_history[-1],
        "noise_history": noise_history,
        "impair_loss": impair_loss,
        "repair_loss": repair_loss,
        "retained_replay_examples": len(sampled_retained),
    }

