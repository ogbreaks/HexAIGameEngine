"""
hex_symmetry.py — Board symmetry transforms for 4× data augmentation.

11×11 Hex has a symmetry group of order 4:
  1. Identity
  2. 180° rotation:       (r,c) → (10-r, 10-c)           — no colour swap
  3. Diagonal transpose:  (r,c) → (c,r)                   — swap colours
  4. Anti-diagonal:       (r,c) → (10-c, 10-r)            — swap colours

All transforms operate on the raw training tuple format:
  state_vector : list[float]   — 122 floats (board[0:121] + player[121])
  policy       : np.ndarray    — [121] visit-count distribution
  value_target : float         — +1.0 / -1.0

The identity transform is NOT included — callers already have the original.
"""

from __future__ import annotations

import numpy as np

SIZE = 11


def _cell_index(r: int, c: int) -> int:
    return r * SIZE + c


# Pre-computed index permutation tables (built once at import time)
_ROT180_MAP = [
    _cell_index(SIZE - 1 - (i // SIZE), SIZE - 1 - (i % SIZE))
    for i in range(SIZE * SIZE)
]
_DIAG_MAP = [_cell_index(i % SIZE, i // SIZE) for i in range(SIZE * SIZE)]
_ANTI_DIAG_MAP = [
    _cell_index(SIZE - 1 - (i % SIZE), SIZE - 1 - (i // SIZE))
    for i in range(SIZE * SIZE)
]


def _apply_board_perm(
    state: list[float], perm: list[int], swap_colours: bool
) -> list[float]:
    """Permute board cells and optionally negate (swap P1/P2)."""
    board = [state[perm[i]] for i in range(SIZE * SIZE)]
    if swap_colours:
        board = [-v for v in board]
    player = -state[121] if swap_colours else state[121]
    return board + [player]


def _apply_policy_perm(policy: np.ndarray, perm: list[int]) -> np.ndarray:
    """Permute the policy vector using the same index mapping."""
    out = np.empty_like(policy)
    for i in range(SIZE * SIZE):
        out[i] = policy[perm[i]]
    return out


def augment_game_data(
    game_data: list[tuple],
) -> list[tuple]:
    """
    Augment a list of (state, policy, value) tuples with all 3 non-identity
    symmetry transforms, returning 3× additional tuples (caller keeps originals).

    Returns
    -------
    list of (state_vector, policy, value_target) — the 3 augmented copies
    """
    augmented: list[tuple] = []
    for state_vec, policy, value_target in game_data:
        # 180° rotation — no colour swap
        augmented.append(
            (
                _apply_board_perm(state_vec, _ROT180_MAP, swap_colours=False),
                _apply_policy_perm(policy, _ROT180_MAP),
                value_target,
            )
        )
        # Diagonal transpose — swap colours
        augmented.append(
            (
                _apply_board_perm(state_vec, _DIAG_MAP, swap_colours=True),
                _apply_policy_perm(policy, _DIAG_MAP),
                value_target,
            )
        )
        # Anti-diagonal transpose — swap colours
        augmented.append(
            (
                _apply_board_perm(state_vec, _ANTI_DIAG_MAP, swap_colours=True),
                _apply_policy_perm(policy, _ANTI_DIAG_MAP),
                value_target,
            )
        )
    return augmented
