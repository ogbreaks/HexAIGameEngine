"""
arena.py — Tournament evaluation between two players.

Usage
-----
    arena = Arena(player1, player2, num_games=40)
    p1_wins, p2_wins, draws = arena.play_games()

Half the games are played with player1 moving first; half with player2 first.
This removes first-player advantage bias from the win-rate measurement.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from game.hex_game import HexGame


class Arena:
    """
    Run a fixed-game tournament between two players.

    Both players must implement: get_action(game: HexGame) -> int
    """

    def __init__(self, player1, player2, num_games: int = 40) -> None:
        self.player1 = player1
        self.player2 = player2
        self.num_games = num_games

    def play_games(self) -> tuple[int, int, int]:
        """
        Play num_games games; return (p1_wins, p2_wins, draws).

        Half the games start with player1 as P1 (first mover),
        half start with player1 as P2 (second mover).
        In Hex there are no draws — the draw count will always be 0.
        """
        p1_wins = 0
        p2_wins = 0

        half = self.num_games // 2

        # Round 1: player1 plays as P1 (first mover)
        for _ in range(half):
            w = self._play_one(first=self.player1, second=self.player2)
            if w == 1:
                p1_wins += 1
            else:
                p2_wins += 1

        # Round 2: player1 plays as P2 (second mover)
        for _ in range(self.num_games - half):
            w = self._play_one(first=self.player2, second=self.player1)
            if w == 1:
                # first mover won — that's player2
                p2_wins += 1
            else:
                p1_wins += 1

        return p1_wins, p2_wins, 0

    @staticmethod
    def _play_one(first, second) -> int:
        """
        Play one game. Returns 1 if the first-mover player wins, 2 otherwise.
        """
        game = HexGame()
        players = {1: first, 2: second}

        while not game.is_terminal():
            current = game.get_current_player()
            action = players[current].get_action(game)
            game = game.apply_action(action)

        winner = game.get_winner()
        # winner is 1 (P1/first-mover wins) or 2 (P2/second-mover wins)
        return winner  # type: ignore[return-value]
