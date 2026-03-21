"""
replay_buffer.py — Fixed-size replay buffer for AlphaZero training data.

Each entry is a (state_vector, policy_target, value_target) tuple where:
    state_vector   : list[float]  — 122 floats (hex_state_vector format)
    policy_target  : np.ndarray   — [121] visit-count policy from MCTS
    value_target   : float        — +1.0 (win) or -1.0 (loss) from that player's view
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    FIFO replay buffer with random sampling.

    When full, the oldest entries are evicted automatically (deque maxlen).
    """

    def __init__(self, max_size: int = 500_000) -> None:
        self.buffer: deque[tuple] = deque(maxlen=max_size)

    def add_game(self, game_data: list[tuple]) -> None:
        """Add all (state, policy, value) tuples from one game."""
        self.buffer.extend(game_data)

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample batch_size entries uniformly at random (without replacement).

        Returns
        -------
        states   : float32 [batch_size, 122]
        policies : float32 [batch_size, 121]
        values   : float32 [batch_size]
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, policies, values = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(policies, dtype=np.float32),
            np.array(values, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
