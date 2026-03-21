"""
players.py — Player types for Arena evaluation.

All players share a single interface:
    player.get_action(game: HexGame) -> int

RandomPlayer   : uniform random legal move
MCTSPlayer     : MCTS search (using PolicyValueNetwork)
HumanPlayer    : CLI input — for interactive play / debugging
"""

from __future__ import annotations

import os
import random
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from game.hex_game import HexGame
from game.hex_actions import get_legal_actions
from training.mcts import MCTS


class RandomPlayer:
    """Selects a uniformly random legal move."""

    def get_action(self, game: HexGame) -> int:
        return random.choice(get_legal_actions(game))


class MCTSPlayer:
    """
    MCTS-backed player.

    Uses the same MCTS as training, but without Dirichlet noise and
    with the arena simulation count (typically lower than self-play).
    """

    def __init__(self, network, num_simulations: int = 200) -> None:
        self.mcts = MCTS(network, num_simulations=num_simulations)

    def get_action(self, game: HexGame) -> int:
        # temperature=0.01 ≈ greedy; no Dirichlet noise (evaluation mode)
        policy = self.mcts.get_policy(game, temperature=0.01, is_self_play=False)
        return int(np.argmax(policy))


class HumanPlayer:
    """
    Interactive CLI player.

    Input format: "row col"  (0-indexed, space-separated)
    Validates the move and reprompts on invalid input.
    """

    def get_action(self, game: HexGame) -> int:
        legal = get_legal_actions(game)
        game.print_board()
        print(f"Player {game.get_current_player()}'s turn.")
        print(f"Legal actions: {len(legal)} available. Enter move as 'row col':")

        while True:
            try:
                raw = input("> ").strip().split()
                if len(raw) != 2:
                    print("Enter exactly two numbers (row col).")
                    continue
                row, col = int(raw[0]), int(raw[1])
                action = row * 11 + col
                if action not in legal:
                    print(f"Illegal move ({row}, {col}). Try again.")
                    continue
                return action
            except (ValueError, EOFError):
                print("Invalid input. Enter row col as integers.")
