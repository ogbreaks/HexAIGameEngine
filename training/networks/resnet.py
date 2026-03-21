"""
resnet.py — ResNet trunk for the AlphaZero PolicyValueNetwork.

Architecture
------------
  Stem: Conv2d(1 → num_channels, 3×3, pad=1) → BN → ReLU
  Tower: N × ResidualBlock(num_channels)
  Output: [B, num_channels, 11, 11]  — spatial feature map, no pooling

Default: 10 residual blocks, 128 channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Two-layer residual block with batch normalisation."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ResNetTrunk(nn.Module):
    """
    Full ResNet trunk.

    Input:  [B, 1, 11, 11]  (board as a single greyscale channel)
    Output: [B, num_channels, 11, 11]
    """

    def __init__(self, num_blocks: int = 10, num_channels: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 1, 11, 11]  →  [B, num_channels, 11, 11]"""
        return self.tower(self.stem(x))
