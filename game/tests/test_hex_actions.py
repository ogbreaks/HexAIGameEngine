"""
test_hex_actions.py — unit tests for legal-move enumeration and
action-space conversion utilities.
"""

from __future__ import annotations

import pytest

from game.hex_game import HexGame
from game.hex_actions import action_to_rowcol, get_legal_actions, rowcol_to_action

SIZE = 11


# ── Shared helper ──────────────────────────────────────────────────────────


def _p1_win_state() -> HexGame:
    """P1 fills column 0 top-to-bottom; P2 is a harmless bystander."""
    game = HexGame()
    for r in range(SIZE):
        game = game.apply_action(r * SIZE + 0)
        if not game.is_terminal():
            game = game.apply_action(r * SIZE + 2)
    return game


# ── get_legal_actions ──────────────────────────────────────────────────────


def test_legal_actions_returns_121_on_empty_board():
    assert len(get_legal_actions(HexGame())) == SIZE * SIZE


def test_legal_actions_contains_all_indices_on_empty_board():
    assert get_legal_actions(HexGame()) == list(range(SIZE * SIZE))


def test_legal_actions_reduces_by_one_after_each_move():
    game = HexGame()
    for move_count in range(1, 10):
        game = game.apply_action(move_count - 1)
        assert len(get_legal_actions(game)) == SIZE * SIZE - move_count


def test_legal_actions_excludes_occupied_cell():
    game = HexGame().apply_action(60)
    assert 60 not in get_legal_actions(game)


def test_legal_actions_only_excludes_occupied_cells():
    game = HexGame().apply_action(0).apply_action(1).apply_action(2)
    legal = get_legal_actions(game)
    assert 0 not in legal
    assert 1 not in legal
    assert 2 not in legal
    assert len(legal) == SIZE * SIZE - 3


def test_legal_actions_returns_empty_list_when_terminal():
    terminal = _p1_win_state()
    assert get_legal_actions(terminal) == []


# ── action_to_rowcol ───────────────────────────────────────────────────────


def test_action_to_rowcol_top_left_corner():
    assert action_to_rowcol(0) == (0, 0)


def test_action_to_rowcol_top_right_corner():
    assert action_to_rowcol(SIZE - 1) == (0, SIZE - 1)


def test_action_to_rowcol_bottom_left_corner():
    assert action_to_rowcol((SIZE - 1) * SIZE) == (SIZE - 1, 0)


def test_action_to_rowcol_bottom_right_corner():
    assert action_to_rowcol(SIZE * SIZE - 1) == (SIZE - 1, SIZE - 1)


def test_action_to_rowcol_centre():
    assert action_to_rowcol(60) == (5, 5)


def test_action_to_rowcol_arbitrary():
    assert action_to_rowcol(23) == (2, 1)  # 23 = 2*11 + 1


# ── rowcol_to_action ───────────────────────────────────────────────────────


def test_rowcol_to_action_top_left_corner():
    assert rowcol_to_action(0, 0) == 0


def test_rowcol_to_action_top_right_corner():
    assert rowcol_to_action(0, SIZE - 1) == SIZE - 1


def test_rowcol_to_action_bottom_left_corner():
    assert rowcol_to_action(SIZE - 1, 0) == (SIZE - 1) * SIZE


def test_rowcol_to_action_bottom_right_corner():
    assert rowcol_to_action(SIZE - 1, SIZE - 1) == SIZE * SIZE - 1


def test_rowcol_to_action_centre():
    assert rowcol_to_action(5, 5) == 60


def test_rowcol_to_action_arbitrary():
    assert rowcol_to_action(2, 1) == 23


# ── Round-trip ─────────────────────────────────────────────────────────────


def test_round_trip_all_actions():
    for n in range(SIZE * SIZE):
        r, c = action_to_rowcol(n)
        assert rowcol_to_action(r, c) == n


def test_round_trip_all_rowcols():
    for r in range(SIZE):
        for c in range(SIZE):
            action = rowcol_to_action(r, c)
            assert action_to_rowcol(action) == (r, c)
