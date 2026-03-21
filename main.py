"""
main.py — Hex game command-line interface.

Usage
-----
  python main.py --mode human_vs_human
  python main.py --mode random_vs_random
  python main.py --mode random_vs_random --games 1000
  python main.py --mode server
  python main.py --mode train_az
  python main.py --mode train_az --config config/hex11_default.yaml
  python main.py --mode train_az --config config/hex11_default.yaml --resume training/checkpoints/hex_az_100.pth
  python main.py --mode export_az --model training/models/hex_az_best.pth --output training/models/hex_az_best.onnx
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


def train_az(config_path: str, resume_path: str | None = None) -> None:
    """Launch AlphaZero training with the given config."""
    import yaml  # type: ignore[import]
    from training.coach import Coach

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if resume_path:
        print(f"[AZ] Resuming from checkpoint: {resume_path}")
        coach, start_iteration = Coach.load_checkpoint(resume_path, config)
    else:
        coach = Coach(config)
        start_iteration = 0

    coach.train(start_iteration=start_iteration)


def export_az(
    model_path: str, output_path: str, config_path: str | None = None
) -> None:
    """Export a trained AZ network to ONNX."""
    import torch
    from training.policy_value_network import PolicyValueNetwork
    from training.export import export_onnx

    config: dict = {}
    if config_path:
        import yaml  # type: ignore[import]

        with open(config_path) as f:
            config = yaml.safe_load(f)

    network = PolicyValueNetwork(config)
    network.load_state_dict(torch.load(model_path, map_location="cpu"))
    network.eval()
    export_onnx(network, output_path)


# ── Entry point ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Hex game CLI")
    parser.add_argument(
        "--mode",
        choices=[
            "human_vs_human",
            "random_vs_random",
            "server",
            "train_az",
            "export_az",
        ],
        required=True,
        help="Mode to run",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1000,
        help="Number of games for random_vs_random (default: 1000)",
    )
    parser.add_argument(
        "--config",
        default="config/hex11_default.yaml",
        help="Path to YAML config for train_az (default: config/hex11_default.yaml)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to checkpoint .pth to resume AZ training from",
    )
    parser.add_argument(
        "--model",
        default="training/models/hex_az_best.pth",
        help="Path to model weights for export_az",
    )
    parser.add_argument(
        "--output",
        default="training/models/hex_az_best.onnx",
        help="Output ONNX path for export_az",
    )
    args = parser.parse_args()

    if args.mode == "human_vs_human":
        play_human_vs_human()
    elif args.mode == "random_vs_random":
        play_random_vs_random(args.games)
    elif args.mode == "server":
        run_server()
    elif args.mode == "train_az":
        train_az(args.config, resume_path=args.resume)
    elif args.mode == "export_az":
        export_az(args.model, args.output, config_path=args.config)


if __name__ == "__main__":
    main()
