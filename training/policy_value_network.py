"""
policy_value_network.py — PolicyValueNetwork for AlphaZero Hex.

Wraps a pluggable trunk (ResNet or ViT) with shared policy and value heads.

Input:  [B, 122]  — 121 board cells (float32: -1/0/+1) + player scalar (-1/+1)
Output: (policy_logits [B, 121], value [B, 1])
  - policy_logits: raw logits over 121 actions; softmax applied externally (MCTS)
  - value: in [-1, +1] via tanh; current player's win probability estimate

ONNX export: policy head only (value head discarded by export.py).
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

from training.networks.resnet import ResNetTrunk
from training.networks.vit import ViTTrunk


class PolicyValueNetwork(nn.Module):
    """
    Joint policy + value network.

    config keys (all optional, defaults match the training YAML):
        trunk           : 'resnet' | 'vit'   (default: 'resnet')
        num_channels    : int                 (default: 128)
        num_res_blocks  : int                 (default: 10, ResNet only)
        vit_heads       : int                 (default: 8,  ViT only)
        vit_layers      : int                 (default: 6,  ViT only)
        vit_ff_dim      : int                 (default: 512, ViT only)
        vit_dropout     : float               (default: 0.1, ViT only)
    """

    def __init__(self, config: dict | None = None) -> None:
        super().__init__()
        if config is None:
            config = {}

        trunk_type: str = config.get("trunk", "resnet")
        num_channels: int = config.get("num_channels", 128)
        self.trunk_type = trunk_type

        if trunk_type == "vit":
            self.trunk = ViTTrunk(
                embed_dim=num_channels,
                num_heads=config.get("vit_heads", 8),
                num_layers=config.get("vit_layers", 6),
                dim_feedforward=config.get("vit_ff_dim", 512),
                dropout=config.get("vit_dropout", 0.1),
            )
        else:
            self.trunk = ResNetTrunk(
                num_blocks=config.get("num_res_blocks", 10),
                num_channels=num_channels,
            )

        # Policy head: [B, C, 11, 11] → [B, 121] (raw logits)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 11 * 11, 121),
        )

        # Value head: [B, C, 11, 11] → [B, 1] (tanh → [-1, +1])
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(11 * 11, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        obs: [B, 122]  — board_flat (121) + player scalar (1)
        Returns: (policy_logits [B, 121], value [B, 1])
        """
        board_flat = obs[:, :121]  # [B, 121]
        player = obs[:, 121:]  # [B, 1]
        board_2d = board_flat.view(-1, 1, 11, 11)  # [B, 1, 11, 11]

        if self.trunk_type == "vit":
            trunk_out = self.trunk(board_2d, player)
        else:
            trunk_out = self.trunk(board_2d)  # [B, num_channels, 11, 11]

        policy_logits = self.policy_head(trunk_out)  # [B, 121]
        value = self.value_head(trunk_out)  # [B, 1]
        return policy_logits, value
