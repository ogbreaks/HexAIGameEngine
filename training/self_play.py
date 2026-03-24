"""
self_play.py — Parallel self-play data generation for AlphaZero Hex.

Public API
----------
play_one_game(mcts, temperature_threshold)
    Play a single Hex game to completion; return labelled training tuples.

generate_self_play_data(network, num_workers, games_per_worker)
    Spawn CPU worker processes; collect and return all game data.

Value labelling convention
--------------------------
value_target is from the CURRENT PLAYER'S perspective at the time that state
was recorded:
    +1.0 = the player who moved at this step won the game
    -1.0 = the player who moved at this step lost the game

HexGame always starts with Player 1 to move, so step_index % 2 == 0 means
Player 1 moved at that step. Do NOT use game.get_current_player() for labelling
— by the time the game ends, get_current_player() reflects the terminal state.

Temperature schedule (Fix 12)
------------------------------
temperature = 1.0  for move indices 0 .. temperature_threshold-1  (exploration)
temperature = 0.01 for move indices >= temperature_threshold        (exploitation)
0.01 approximates argmax without division-by-zero.

Worker safety (Fix 6)
---------------------
Workers run network inference on CPU only. CUDA contexts must not cross
process boundaries with multiprocessing. Use 'spawn' start method.
All workers must be TOP-LEVEL module functions — not lambdas or nested fns.
"""

from __future__ import annotations

import copy
import multiprocessing
import multiprocessing.queues
import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from game.hex_game import HexGame
from game.hex_actions import get_legal_actions
from game.hex_state_vector import get_state_vector
from training.mcts import MCTS


# ---------------------------------------------------------------------------
# Single-game self-play
# ---------------------------------------------------------------------------


def play_one_game(
    mcts: MCTS,
    temperature_threshold: int = 30,
) -> list[tuple]:
    """
    Play one Hex game to completion using MCTS; return labelled training data.

    Returns
    -------
    list of (state_vector [122], policy_target [121], value_target float)

    Value convention: value_target is from the CURRENT PLAYER'S perspective
      at the time that state was recorded.
        +1.0 = the player who moved at this step won the game
        -1.0 = the player who moved at this step lost the game

    Implementation note: HexGame always starts with Player 1 to move,
    so step_index % 2 == 0 → Player 1 moved, step_index % 2 == 1 → Player 2 moved.
    Do NOT change this to use game.get_current_player() — by the time labelling
    happens the game is terminal and get_current_player() reflects the last state.
    """
    game = HexGame()
    history: list[tuple[list[float], np.ndarray]] = []

    step_index = 0
    while not game.is_terminal():
        state_vec = get_state_vector(game)

        # Temperature schedule: explore early, exploit later
        temperature = 1.0 if step_index < temperature_threshold else 0.01

        policy = mcts.get_policy(game, temperature=temperature, is_self_play=True)

        # Sample action from policy distribution
        policy = policy.astype(np.float64)
        policy = policy / policy.sum()  # renormalise
        policy = np.clip(policy, 0, None)  # remove any tiny negatives
        policy = policy / policy.sum()  # renormalise again after clip
        action = int(np.random.choice(121, p=policy))

        history.append((state_vec, policy))
        game = game.apply_action(action)
        step_index += 1

    winner = game.get_winner()

    # Label each position with outcome from that player's perspective
    labeled: list[tuple] = []
    for i, (state_vec, policy) in enumerate(history):
        # Player 1 moves at even steps (0, 2, 4...), Player 2 at odd steps
        player_at_step = 1 if i % 2 == 0 else 2
        value_target = 1.0 if winner == player_at_step else -1.0
        labeled.append((state_vec, policy, value_target))

    return labeled


# ---------------------------------------------------------------------------
# Parallel self-play orchestrator
# ---------------------------------------------------------------------------


def generate_self_play_data(
    network,
    num_workers: int = 4,
    games_per_worker: int = 25,
    temperature_threshold: int = 30,
    num_simulations: int = 800,
    use_inference_server: bool = False,
    inference_batch_size: int = 64,
    inference_max_wait_ms: float = 5.0,
    virtual_loss_k: int = 1,
) -> list[list[tuple]]:
    """
    Spawn num_workers CPU processes; each plays games_per_worker games.

    When use_inference_server=True, a GPU inference server handles all neural
    network evaluations in batches, dramatically improving throughput for large
    networks or high worker counts.

    Returns
    -------
    list of game_data lists (one list per completed game)

    Raises
    ------
    RuntimeError
        If all workers die before the expected number of games is collected.
    """
    total_games = num_workers * games_per_worker

    # Fast path: single worker — skip multiprocessing entirely, run in main process
    if num_workers == 1 and not use_inference_server:
        mcts = MCTS(
            network, num_simulations=num_simulations, virtual_loss_k=virtual_loss_k
        )
        results: list[list[tuple]] = []
        for i in range(total_games):
            results.append(
                play_one_game(mcts, temperature_threshold=temperature_threshold)
            )
            print(f"  Game {i + 1}/{total_games} complete")
        return results

    # ── Inference server path ────────────────────────────────────────────────
    if use_inference_server:
        from training.inference_server import InferenceServer
        from training.worker import run_worker_with_server

        server = InferenceServer(
            network=copy.deepcopy(network),
            num_workers=num_workers,
            batch_size=inference_batch_size,
            max_wait_ms=inference_max_wait_ms,
        )
        server.start()

        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()

        processes = []
        for worker_id in range(num_workers):
            p = ctx.Process(
                target=run_worker_with_server,
                args=(
                    worker_id,
                    games_per_worker,
                    num_simulations,
                    temperature_threshold,
                    queue,
                    server.request_queue,
                    server.result_queues[worker_id],
                    virtual_loss_k,
                ),
            )
            p.start()
            processes.append(p)

        results = _collect_results(processes, queue, total_games)

        server.stop()
        return results

    # ── CPU-only multi-worker path ───────────────────────────────────────────
    from training.worker import run_worker

    # Copy network weights to CPU before dispatching to workers
    cpu_network = copy.deepcopy(network).to("cpu")
    cpu_network.eval()

    # 'spawn' start method — required for CUDA-safe multiprocessing on all platforms
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()

    processes = []
    for _ in range(num_workers):
        p = ctx.Process(
            target=run_worker,
            args=(
                cpu_network,
                games_per_worker,
                num_simulations,
                temperature_threshold,
                queue,
            ),
        )
        p.start()
        processes.append(p)

    return _collect_results(processes, queue, total_games)


def _collect_results(
    processes: list,
    queue: multiprocessing.Queue,
    total_expected: int,
) -> list[list[tuple]]:
    """Collect game results from worker processes via queue."""
    results: list[list[tuple]] = []

    while len(results) < total_expected:
        alive = [p for p in processes if p.is_alive()]
        if not alive and len(results) < total_expected:
            raise RuntimeError(
                f"All workers died. Collected {len(results)}/{total_expected} games. "
                f"Exit codes: {[p.exitcode for p in processes]}"
            )

        try:
            game_data = queue.get(
                timeout=600
            )  # 10 min — CPU inference with large nets is slow
            results.append(game_data)
        except multiprocessing.queues.Empty:
            continue

    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    return results
