"""
test_game_manager.py — unit tests for GameManager.
"""

from __future__ import annotations

import pytest

from server.game_manager import GameManager

SIZE = 11
TOTAL_CELLS = SIZE * SIZE  # 121


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_terminal_manager() -> GameManager:
    """Return a GameManager where P1 has won (column 0, top to bottom)."""
    gm = GameManager()
    for r in range(SIZE):
        gm.apply_move(r * SIZE + 0)  # P1 at (r, 0)
        if not gm.game.is_terminal():
            gm.apply_move(r * SIZE + 2)  # P2 at (r, 2)
    return gm


# ── Initial state structure ───────────────────────────────────────────────


def test_initial_state_has_all_required_keys():
    state = GameManager().get_state()
    assert set(state.keys()) == {
        "state_vector",
        "legal_actions",
        "current_player",
        "is_terminal",
        "winner",
    }


def test_initial_state_vector_is_122_floats():
    sv = GameManager().get_state()["state_vector"]
    assert len(sv) == 122
    assert all(isinstance(v, float) for v in sv)


def test_initial_legal_actions_has_121_items():
    assert len(GameManager().get_state()["legal_actions"]) == TOTAL_CELLS


def test_initial_current_player_is_1():
    assert GameManager().get_state()["current_player"] == 1


def test_initial_is_terminal_is_false():
    assert GameManager().get_state()["is_terminal"] is False


def test_initial_winner_is_none():
    assert GameManager().get_state()["winner"] is None


# ── apply_move ────────────────────────────────────────────────────────────


def test_apply_move_reduces_legal_actions_by_1():
    gm = GameManager()
    gm.apply_move(0)
    assert len(gm.get_state()["legal_actions"]) == TOTAL_CELLS - 1


def test_apply_move_switches_current_player_to_2():
    gm = GameManager()
    gm.apply_move(0)
    assert gm.get_state()["current_player"] == 2


def test_apply_move_switches_current_player_back_to_1():
    gm = GameManager()
    gm.apply_move(0)
    gm.apply_move(1)
    assert gm.get_state()["current_player"] == 1


def test_apply_move_returns_state_dict():
    gm = GameManager()
    result = gm.apply_move(60)
    assert isinstance(result, dict)
    assert "state_vector" in result
    assert "legal_actions" in result


def test_apply_move_raises_on_occupied_cell():
    gm = GameManager()
    gm.apply_move(0)
    gm.apply_move(1)
    with pytest.raises(ValueError):
        gm.apply_move(0)  # cell 0 already occupied


def test_apply_move_raises_on_out_of_range_action():
    gm = GameManager()
    with pytest.raises(ValueError):
        gm.apply_move(TOTAL_CELLS)  # one past the end


def test_apply_move_raises_when_game_is_terminal():
    gm = _make_terminal_manager()
    assert gm.get_state()["is_terminal"] is True
    with pytest.raises(ValueError):
        gm.apply_move(60)


def test_apply_move_excludes_played_cell_from_legal_actions():
    gm = GameManager()
    gm.apply_move(60)
    assert 60 not in gm.get_state()["legal_actions"]


# ── Terminal state ────────────────────────────────────────────────────────


def test_winner_is_1_after_p1_wins():
    gm = _make_terminal_manager()
    assert gm.get_state()["winner"] == 1


def test_winner_is_not_2_after_p1_wins():
    gm = _make_terminal_manager()
    assert gm.get_state()["winner"] != 2


def test_is_terminal_true_after_win():
    gm = _make_terminal_manager()
    assert gm.get_state()["is_terminal"] is True


def test_legal_actions_empty_when_terminal():
    gm = _make_terminal_manager()
    assert gm.get_state()["legal_actions"] == []


# ── reset ─────────────────────────────────────────────────────────────────


def test_reset_returns_fresh_state():
    gm = GameManager()
    gm.apply_move(0)
    gm.apply_move(1)
    state = gm.reset()
    assert len(state["legal_actions"]) == TOTAL_CELLS


def test_reset_clears_winner():
    gm = _make_terminal_manager()
    state = gm.reset()
    assert state["winner"] is None


def test_reset_clears_terminal_flag():
    gm = _make_terminal_manager()
    state = gm.reset()
    assert state["is_terminal"] is False


def test_reset_restores_current_player_to_1():
    gm = GameManager()
    gm.apply_move(0)  # now P2's turn
    state = gm.reset()
    assert state["current_player"] == 1


def test_reset_returns_122_float_state_vector():
    gm = _make_terminal_manager()
    sv = gm.reset()["state_vector"]
    assert len(sv) == 122
    assert all(isinstance(v, float) for v in sv)
