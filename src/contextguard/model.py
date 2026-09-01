from __future__ import annotations

import torch
from torch import nn


class CSILeNet(nn.Module):
    """Compact SenseFi-style encoder with an explicit audit representation."""

    def __init__(self, num_classes: int = 14, embedding_dim: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(15, 23), stride=9),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=(7, 3), stride=(1, 3)),
            nn.ReLU(inplace=True),
        )
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 6, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(self.encoder(x))

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits

