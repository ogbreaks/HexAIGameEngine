"""
vit.py — Vision Transformer trunk for the AlphaZero PolicyValueNetwork.

Architecture
------------
  Each of the 121 board cells is its own patch (no subdivision).
  A CLS-style player token prepends the sequence.
  Transformer layers process all 122 tokens together.
  The player token is dropped after the transformer; the 121 cell tokens
  are reshaped to [B, embed_dim, 11, 11] to match the ResNet trunk output
  contract used by the shared policy/value heads.

Why ViT for 11×11 Hex
----------------------
  Self-attention computes pairwise relationships between all 121 cells
  simultaneously. A winning Hex chain spans all 11 rows — the long-range
  dependency that CNN layers model poorly but attention handles directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ViTTrunk(nn.Module):
    """
    Vision Transformer trunk for 11×11 Hex.

    Input:
        board:  [B, 1, 11, 11]
        player: [B, 1]          — +1.0 for P1, -1.0 for P2

    Output: [B, embed_dim, 11, 11]  — same spatial contract as ResNetTrunk
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Project each cell scalar to embed_dim
        self.patch_embed = nn.Linear(1, embed_dim)

        # Learnable positional embedding for 121 board cells
        self.pos_embed = nn.Parameter(torch.zeros(1, 121, embed_dim))

        # Player token (CLS-style) — bias toward current player perspective
        self.player_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.player_token, std=0.02)

    def forward(self, board: torch.Tensor, player: torch.Tensor) -> torch.Tensor:
        B = board.shape[0]

        # Flatten board: [B, 1, 11, 11] → [B, 121, 1] → embed → [B, 121, embed_dim]
        x = board.view(B, 121, 1)
        x = self.patch_embed(x)
        x = x + self.pos_embed

        # Build player token: broadcast base token and bias with player scalar
        # player: [B, 1] → [B, 1, embed_dim] via expand
        player_bias = player.unsqueeze(-1).expand(B, 1, self.embed_dim)
        player_tok = self.player_token.expand(B, -1, -1) + player_bias

        # Prepend player token: [B, 122, embed_dim]
        x = torch.cat([player_tok, x], dim=1)

        # Transformer encoder
        x = self.transformer(x)

        # Drop player token, keep cell tokens: [B, 121, embed_dim]
        # Reshape to spatial map: [B, embed_dim, 11, 11]
        out = x[:, 1:, :].permute(0, 2, 1).contiguous().view(B, self.embed_dim, 11, 11)
        return out
