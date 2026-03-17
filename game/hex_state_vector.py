"""
hex_state_vector.py — canonical state vector for training and inference.

CONTRACT — do not change the format without updating ARCHITECTURE.md
and the Unity consumer.

Vector layout (122 floats)
--------------------------
  [0 – 120]  Board cells in row-major order:
                 1.0  = Player 1 piece
                -1.0  = Player 2 piece
                 0.0  = empty
  [121]      Current player:
                 1.0  = Player 1 to move
                -1.0  = Player 2 to move
"""

from __future__ import annotations

from game.hex_game import HexGame

STATE_VECTOR_SIZE: int = 122

_CELL_ENCODING: dict[int, float] = {0: 0.0, 1: 1.0, 2: -1.0}


def get_state_vector(game_state: HexGame) -> list[float]:
    """
    Return the canonical 122-float state vector.

    The returned list is a fresh object on every call; mutations
    by the caller do not affect game_state.
    """
    board = game_state.get_board()
    vec: list[float] = [_CELL_ENCODING[v] for v in board]
    vec.append(1.0 if game_state.get_current_player() == 1 else -1.0)
    return vec
