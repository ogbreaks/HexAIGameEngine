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

import multiprocessing
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from game.hex_game import HexGame


# ── Worker function (top-level for pickling) ─────────────────────────────────


def _arena_worker(
    net_config: dict,
    challenger_sd: dict,
    champion_sd: dict,
    num_sims: int,
    games_as_p1: int,
    games_as_p2: int,
    result_queue: multiprocessing.Queue,
) -> None:
    """Play a slice of arena games in a subprocess and report results."""
    import numpy as np

    from training.mcts import MCTS
    from training.policy_value_network import PolicyValueNetwork

    # Reconstruct networks from state dicts (CPU only)
    challenger_net = PolicyValueNetwork(net_config)
    challenger_net.load_state_dict(challenger_sd)
    challenger_net.eval()

    champion_net = PolicyValueNetwork(net_config)
    champion_net.load_state_dict(champion_sd)
    champion_net.eval()

    challenger_mcts = MCTS(challenger_net, num_simulations=num_sims)
    champion_mcts = MCTS(champion_net, num_simulations=num_sims)

    p1_wins = 0
    p2_wins = 0

    # Games where challenger moves first
    for _ in range(games_as_p1):
        w = _play_one_game(challenger_mcts, champion_mcts, np)
        if w == 1:
            p1_wins += 1
        else:
            p2_wins += 1

    # Games where challenger moves second
    for _ in range(games_as_p2):
        w = _play_one_game(champion_mcts, challenger_mcts, np)
        if w == 1:
            p2_wins += 1  # first mover (champion) won
        else:
            p1_wins += 1

    result_queue.put((p1_wins, p2_wins))


def _play_one_game(first_mcts, second_mcts, np) -> int:
    """Play one game between two MCTS instances. Returns 1 or 2."""
    game = HexGame()
    mcts_map = {1: first_mcts, 2: second_mcts}

    while not game.is_terminal():
        cur = game.get_current_player()
        policy = mcts_map[cur].get_policy(game, temperature=0.01, is_self_play=False)
        game = game.apply_action(int(np.argmax(policy)))

    return game.get_winner()


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
    def play_games_parallel(
        net_config: dict,
        challenger_sd: dict,
        champion_sd: dict,
        num_sims: int,
        num_games: int = 40,
        num_workers: int = 4,
    ) -> tuple[int, int, int]:
        """
        Parallel arena: distribute games across worker processes.

        Parameters
        ----------
        net_config : dict   — network architecture config (trunk, channels, etc.)
        challenger_sd : dict — challenger network state_dict
        champion_sd : dict  — champion network state_dict
        num_sims : int      — MCTS simulations per move
        num_games : int     — total games to play
        num_workers : int   — number of worker processes

        Returns (p1_wins, p2_wins, 0)
        """
        half = num_games // 2
        remainder = num_games - half

        # Distribute p1-first and p2-first games evenly across workers
        p1_per_worker = [half // num_workers] * num_workers
        p2_per_worker = [remainder // num_workers] * num_workers
        for i in range(half % num_workers):
            p1_per_worker[i] += 1
        for i in range(remainder % num_workers):
            p2_per_worker[i] += 1

        ctx = multiprocessing.get_context("spawn")
        result_queue = ctx.Queue()

        processes = []
        for w in range(num_workers):
            p = ctx.Process(
                target=_arena_worker,
                args=(
                    net_config,
                    challenger_sd,
                    champion_sd,
                    num_sims,
                    p1_per_worker[w],
                    p2_per_worker[w],
                    result_queue,
                ),
            )
            p.start()
            processes.append(p)

        # Collect results
        total_p1 = 0
        total_p2 = 0
        for _ in range(num_workers):
            p1w, p2w = result_queue.get(timeout=600)
            total_p1 += p1w
            total_p2 += p2w

        for p in processes:
            p.join(timeout=30)

        return total_p1, total_p2, 0

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
