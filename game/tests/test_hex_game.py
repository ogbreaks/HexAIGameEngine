"""
test_hex_game.py — unit tests for HexGame state and rules.
"""

from __future__ import annotations

import pytest

from game.hex_game import HexGame

SIZE = 11


# ── Shared helpers ─────────────────────────────────────────────────────────


def _p1_win_state() -> HexGame:
    """P1 fills column 0 (rows 0–10) — a straight top-to-bottom connection."""
    game = HexGame()
    for r in range(SIZE):
        game = game.apply_action(r * SIZE + 0)  # P1 at (r, 0)
        if not game.is_terminal():
            game = game.apply_action(r * SIZE + 2)  # P2 at (r, 2)  — safe bystander
    return game


def _p2_win_state() -> HexGame:
    """P2 fills row 0 (cols 0–10) — a straight left-to-right connection."""
    game = HexGame()
    for c in range(SIZE):
        game = game.apply_action(1 * SIZE + c)  # P1 at (1, c)  — safe bystander
        if not game.is_terminal():
            game = game.apply_action(0 * SIZE + c)  # P2 at (0, c)
    return game


# ── Initialisation ─────────────────────────────────────────────────────────


def test_empty_board_initialises_correctly():
    board = HexGame().get_board()
    assert len(board) == SIZE * SIZE
    assert all(v == 0 for v in board)


def test_initial_current_player_is_one():
    assert HexGame().get_current_player() == 1


def test_initial_not_terminal():
    assert not HexGame().is_terminal()


def test_initial_no_winner():
    assert HexGame().get_winner() is None


# ── apply_action — placement ───────────────────────────────────────────────


def test_apply_action_places_piece_on_correct_cell():
    game = HexGame().apply_action(0)
    assert game.get_board()[0] == 1


def test_apply_action_places_piece_at_arbitrary_index():
    game = HexGame().apply_action(60)  # centre cell (5, 5)
    assert game.get_board()[60] == 1


def test_apply_action_p2_places_correctly():
    game = HexGame().apply_action(0).apply_action(1)
    assert game.get_board()[1] == 2


# ── apply_action — player switching ───────────────────────────────────────


def test_apply_action_switches_player_to_two():
    game = HexGame().apply_action(0)
    assert game.get_current_player() == 2


def test_apply_action_switches_player_back_to_one():
    game = HexGame().apply_action(0).apply_action(1)
    assert game.get_current_player() == 1


# ── apply_action — immutability ────────────────────────────────────────────


def test_apply_action_does_not_mutate_original_board():
    original = HexGame()
    original_board = original.get_board()
    _ = original.apply_action(60)
    assert original.get_board() == original_board


def test_apply_action_does_not_mutate_original_player():
    original = HexGame()
    _ = original.apply_action(60)
    assert original.get_current_player() == 1


def test_apply_action_does_not_mutate_terminal_flag():
    original = HexGame()
    _ = original.apply_action(60)
    assert not original.is_terminal()


# ── apply_action — illegal moves ───────────────────────────────────────────


def test_apply_action_raises_on_occupied_cell():
    g = HexGame().apply_action(0).apply_action(1)
    with pytest.raises(ValueError):
        g.apply_action(0)  # cell 0 already holds P1's piece


def test_apply_action_raises_when_game_is_terminal():
    terminal = _p1_win_state()
    assert terminal.is_terminal()
    with pytest.raises(ValueError):
        terminal.apply_action(60)


def test_apply_action_raises_on_negative_index():
    with pytest.raises(ValueError):
        HexGame().apply_action(-1)


def test_apply_action_raises_on_out_of_bounds_index():
    with pytest.raises(ValueError):
        HexGame().apply_action(SIZE * SIZE)


# ── Win conditions ─────────────────────────────────────────────────────────


def test_p1_wins_top_to_bottom():
    game = _p1_win_state()
    assert game.is_terminal()
    assert game.get_winner() == 1


def test_p2_wins_left_to_right():
    game = _p2_win_state()
    assert game.is_terminal()
    assert game.get_winner() == 2


def test_no_false_win_on_partial_board():
    # P1 at (0,0) and (1,0) — connected but nowhere near row 10.
    game = HexGame().apply_action(0).apply_action(5).apply_action(11)
    assert not game.is_terminal()
    assert game.get_winner() is None


def test_winner_is_not_other_player_after_p1_wins():
    game = _p1_win_state()
    assert game.get_winner() != 2


def test_winner_is_not_other_player_after_p2_wins():
    game = _p2_win_state()
    assert game.get_winner() != 1


def test_win_detected_on_non_straight_path():
    """P1 wins via a bent (L-shaped) path rather than a straight column.

    Path: (0,0) → (0,1) → (1,1) → (2,1) → … → (10,1)
    Adjacency proof (using _DIRECTIONS = [(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0)]):
      (0,0)→(0,1) : right-neighbour  (0, +1)
      (0,1)→(1,1) : lower-right      (+1, 0)
      (r,1)→(r+1,1): lower-right     (+1, 0)  for every r 1..9
    """
    # 12 P1 cells: detour via col 0 at row 0, then straight down col 1
    p1_path = [(0, 0), (0, 1)] + [(r, 1) for r in range(1, SIZE)]  # len 12
    # 11 P2 cells in col 5 — no overlap, cannot win
    p2_cells = [(r, 5) for r in range(SIZE)]  # len 11

    game = HexGame()
    # zip plays first 11 P1 moves paired with all 11 P2 moves
    for p1_cell, p2_cell in zip(p1_path, p2_cells):
        game = game.apply_action(p1_cell[0] * SIZE + p1_cell[1])
        if not game.is_terminal():
            game = game.apply_action(p2_cell[0] * SIZE + p2_cell[1])

    # Play the 12th (final) P1 move: (10, 1)
    if not game.is_terminal():
        game = game.apply_action(p1_path[-1][0] * SIZE + p1_path[-1][1])

    assert game.is_terminal()
    assert game.get_winner() == 1


def test_get_board_returns_copy():
    """Mutating the returned board must not affect the game state."""
    game = HexGame()
    board = game.get_board()
    board[0] = 99
    assert game.get_board()[0] == 0
