"""
main.py — Hex game command-line interface.

Usage
-----
  python main.py --mode human_vs_human
  python main.py --mode random_vs_random
  python main.py --mode random_vs_random --games 1000
  python main.py --mode server
"""

from __future__ import annotations

import argparse
import random
import sys

from game.hex_game import HexGame
from game.hex_actions import get_legal_actions
from server.hex_server import run_server


# ── Game modes ──────────────────────────────────────────────────────────────


def play_human_vs_human() -> None:
    game = HexGame()
    print("Hex  —  Player 1 connects top↔bottom  |  Player 2 connects left↔right")
    print("Enter moves as:  row col  (0-indexed, space-separated)\n")

    while not game.is_terminal():
        game.print_board()
        player = game.get_current_player()
        print(f"\nPlayer {player}'s turn:")

        while True:
            try:
                raw = input("> ").strip().split()
                if len(raw) != 2:
                    raise ValueError("Enter exactly two numbers (row col).")
                row, col = int(raw[0]), int(raw[1])
                game = game.apply_action(row * 11 + col)
                break
            except ValueError as exc:
                print(f"  Invalid: {exc}  — try again.")

    print()
    game.print_board()
    print(f"\nPlayer {game.get_winner()} wins!")


def play_random_vs_random(num_games: int) -> None:
    p1_wins = 0
    p2_wins = 0
    errors = 0

    for game_index in range(num_games):
        try:
            game = HexGame()
            while not game.is_terminal():
                actions = get_legal_actions(game)
                game = game.apply_action(random.choice(actions))

            winner = game.get_winner()
            if winner == 1:
                p1_wins += 1
            elif winner == 2:
                p2_wins += 1
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"Error in game {game_index}: {exc}", file=sys.stderr)

    print(f"Games completed : {num_games}")
    print(f"Player 1 wins   : {p1_wins}")
    print(f"Player 2 wins   : {p2_wins}")
    print(f"Errors          : {errors}")


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Hex game CLI")
    parser.add_argument(
        "--mode",
        choices=["human_vs_human", "random_vs_random", "server"],
        required=True,
        help="Game mode",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1000,
        help="Number of games for random_vs_random (default: 1000)",
    )
    args = parser.parse_args()

    if args.mode == "human_vs_human":
        play_human_vs_human()
    elif args.mode == "random_vs_random":
        play_random_vs_random(args.games)
    elif args.mode == "server":
        run_server()


if __name__ == "__main__":
    main()
