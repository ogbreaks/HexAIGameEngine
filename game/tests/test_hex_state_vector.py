"""
test_hex_state_vector.py — unit tests for the canonical state vector.

The 122-float layout is the API contract between Python training and Unity.
These tests lock it down so a format change cannot go undetected.
"""

from __future__ import annotations

import pytest

from game.hex_game import HexGame
from game.hex_state_vector import STATE_VECTOR_SIZE, get_state_vector

SIZE = 11


# ── Constants ──────────────────────────────────────────────────────────────


def test_state_vector_size_constant_is_122():
    assert STATE_VECTOR_SIZE == 122


# ── Length ─────────────────────────────────────────────────────────────────


def test_vector_length_is_122_on_empty_board():
    assert len(get_state_vector(HexGame())) == STATE_VECTOR_SIZE


def test_vector_length_is_122_mid_game():
    game = HexGame().apply_action(0).apply_action(1).apply_action(2)
    assert len(get_state_vector(game)) == STATE_VECTOR_SIZE


# ── Empty board ────────────────────────────────────────────────────────────


def test_empty_board_first_121_positions_are_zero():
    vec = get_state_vector(HexGame())
    assert all(v == 0.0 for v in vec[: SIZE * SIZE])


def test_empty_board_position_121_is_1_for_player_1():
    vec = get_state_vector(HexGame())
    assert vec[121] == 1.0


# ── Cell encoding ──────────────────────────────────────────────────────────


def test_p1_piece_encodes_as_positive_one():
    game = HexGame().apply_action(0)  # P1 places at cell 0
    assert get_state_vector(game)[0] == 1.0


def test_p2_piece_encodes_as_negative_one():
    game = HexGame().apply_action(0).apply_action(1)  # P2 places at cell 1
    assert get_state_vector(game)[1] == -1.0


def test_empty_cell_encodes_as_zero():
    game = HexGame().apply_action(0)
    assert get_state_vector(game)[1] == 0.0  # cell 1 not yet occupied


def test_p1_and_p2_cells_encoded_correctly_together():
    game = HexGame().apply_action(5).apply_action(10)
    vec = get_state_vector(game)
    assert vec[5] == 1.0  # P1
    assert vec[10] == -1.0  # P2
    assert vec[0] == 0.0  # empty


# ── Current player encoding ────────────────────────────────────────────────


def test_current_player_1_encodes_as_positive_one():
    vec = get_state_vector(HexGame())  # P1 to move
    assert vec[121] == 1.0


def test_current_player_2_encodes_as_negative_one():
    game = HexGame().apply_action(0)  # P2 to move now
    assert get_state_vector(game)[121] == -1.0


def test_current_player_toggles_correctly():
    game = HexGame()
    for i in range(6):
        expected = 1.0 if game.get_current_player() == 1 else -1.0
        assert get_state_vector(game)[121] == expected
        game = game.apply_action(i)


# ── Determinism and identity ───────────────────────────────────────────────


def test_identical_board_states_produce_identical_vectors():
    g1 = HexGame().apply_action(5).apply_action(10)
    g2 = HexGame().apply_action(5).apply_action(10)
    assert get_state_vector(g1) == get_state_vector(g2)


def test_vector_is_deterministic_across_multiple_calls():
    game = HexGame().apply_action(0)
    assert get_state_vector(game) == get_state_vector(game)


def test_different_board_states_produce_different_vectors():
    g1 = HexGame().apply_action(0)
    g2 = HexGame().apply_action(1)
    assert get_state_vector(g1) != get_state_vector(g2)


# ── Type correctness ───────────────────────────────────────────────────────


def test_all_elements_are_floats():
    vec = get_state_vector(HexGame().apply_action(0).apply_action(1))
    assert all(isinstance(v, float) for v in vec)


def test_only_valid_values_in_vector():
    game = HexGame().apply_action(0).apply_action(1)
    for v in get_state_vector(game):
        assert v in {-1.0, 0.0, 1.0}


# ── Isolation ──────────────────────────────────────────────────────────────


def test_mutating_returned_vector_does_not_affect_game_state():
    game = HexGame()
    vec = get_state_vector(game)
    vec[0] = 99.0
    assert get_state_vector(game)[0] == 0.0
