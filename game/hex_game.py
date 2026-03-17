"""
hex_game.py — immutable Hex game state and rules.

Board layout
------------
11 × 11 grid stored as a flat list of 121 ints.
  0 = empty   1 = Player 1   2 = Player 2

Win conditions
--------------
Player 1 connects Row 0 (top) to Row 10 (bottom).
Player 2 connects Col 0 (left) to Col 10 (right).

Hex-grid adjacency (pointy-top, row-major indexing)
----------------------------------------------------
Neighbours of (r, c):
  (r-1, c)   upper-left
  (r-1, c+1) upper-right
  (r,   c-1) left
  (r,   c+1) right
  (r+1, c-1) lower-left
  (r+1, c)   lower-right
"""

from __future__ import annotations

from collections import deque

SIZE = 11

_DIRECTIONS: list[tuple[int, int]] = [
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
]


class HexGame:
    """Immutable snapshot of a Hex game."""

    def __init__(self) -> None:
        self._board: list[int] = [0] * (SIZE * SIZE)
        self._current_player: int = 1
        self._terminal: bool = False
        self._winner: int | None = None

    # ── Public API ──────────────────────────────────────────────────────────

    def get_board(self) -> list[int]:
        """Return a copy of the flat board (121 ints: 0 / 1 / 2)."""
        return list(self._board)

    def get_current_player(self) -> int:
        """Return the player to move: 1 or 2."""
        return self._current_player

    def is_terminal(self) -> bool:
        """Return True if the game has ended."""
        return self._terminal

    def get_winner(self) -> int | None:
        """Return 1, 2, or None if the game is still in progress."""
        return self._winner

    def apply_action(self, action: int) -> HexGame:
        """
        Place the current player's piece on *action* and return a new state.

        Parameters
        ----------
        action : int
            Row-major cell index 0–120  (row = action // 11, col = action % 11).

        Returns
        -------
        HexGame
            New game state; the original is never mutated.

        Raises
        ------
        ValueError
            If the action is illegal (out of range, occupied, or game over).
        """
        if self._terminal:
            raise ValueError("Cannot apply action: game is already over.")
        if not (0 <= action < SIZE * SIZE):
            raise ValueError(f"Action {action} is out of range [0, {SIZE * SIZE - 1}].")
        if self._board[action] != 0:
            raise ValueError(f"Cell {action} is already occupied.")

        new_state = HexGame()
        new_state._board = list(self._board)
        new_state._board[action] = self._current_player
        new_state._current_player = 3 - self._current_player  # 1 ↔ 2

        winner = _check_winner(new_state._board, self._current_player)
        if winner is not None:
            new_state._terminal = True
            new_state._winner = winner

        return new_state

    def print_board(self) -> None:
        """
        Print the board to stdout with a parallelogram indent.

        Row 0 is the top edge (Player 1's goal).
        Row 10 is the bottom edge.
        Column 0 is the left edge (Player 2's goal).
        Column 10 is the right edge.
        """
        symbols = {0: ".", 1: "1", 2: "2"}
        for r in range(SIZE):
            row_str = " ".join(symbols[self._board[r * SIZE + c]] for c in range(SIZE))
            print(" " * r + row_str)


# ── Internal helpers ────────────────────────────────────────────────────────


def _neighbors(r: int, c: int):
    """Yield all valid (row, col) neighbours of (r, c)."""
    for dr, dc in _DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < SIZE and 0 <= nc < SIZE:
            yield nr, nc


def _check_winner(board: list[int], player: int) -> int | None:
    """
    BFS flood fill to determine whether *player* has just won.

    Returns *player* if a winning connection exists, None otherwise.
    """
    if player == 1:
        # Seeds: P1 pieces in row 0.  Goal: reach row 10.
        seeds = [(0, c) for c in range(SIZE) if board[c] == 1]

        def goal(r: int, _c: int) -> bool:
            return r == SIZE - 1

    else:
        # Seeds: P2 pieces in col 0.  Goal: reach col 10.
        seeds = [(r, 0) for r in range(SIZE) if board[r * SIZE] == 2]

        def goal(_r: int, c: int) -> bool:
            return c == SIZE - 1

    visited: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    for cell in seeds:
        if cell not in visited:
            visited.add(cell)
            queue.append(cell)

    while queue:
        r, c = queue.popleft()
        if goal(r, c):
            return player
        for nr, nc in _neighbors(r, c):
            if (nr, nc) not in visited and board[nr * SIZE + nc] == player:
                visited.add((nr, nc))
                queue.append((nr, nc))

    return None
