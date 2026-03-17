"""
hex_actions.py — legal move enumeration and action-space utilities.

Action N corresponds to the cell at row-major index N:
  row = N // 11
  col = N % 11
"""

from __future__ import annotations

from game.hex_game import HexGame, SIZE


def get_legal_actions(game_state: HexGame) -> list[int]:
    """
    Return a list of all empty cell indices in row-major order.

    Returns an empty list if the game is terminal.
    """
    if game_state.is_terminal():
        return []
    board = game_state.get_board()
    return [i for i, v in enumerate(board) if v == 0]


def action_to_rowcol(action: int) -> tuple[int, int]:
    """Convert a row-major action index to (row, col)."""
    return (action // SIZE, action % SIZE)


def rowcol_to_action(row: int, col: int) -> int:
    """Convert (row, col) to a row-major action index."""
    return row * SIZE + col
