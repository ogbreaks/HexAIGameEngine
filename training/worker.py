"""
worker.py — Top-level worker function for multiprocessing self-play.

Must live in its own importable module so that Windows 'spawn' can pickle it
by import path (training.worker.run_worker). Functions defined inside scripts
run with `python -c` or inside closures are not picklable under spawn.

Supports two modes:
  1. Local CPU inference (default) — network copied to each worker
  2. GPU inference server — worker sends requests to shared InferenceServer
"""

from __future__ import annotations

import multiprocessing

import torch

from training.mcts import MCTS
from training.self_play import play_one_game


def run_worker(
    network,
    num_games: int,
    num_simulations: int,
    temperature_threshold: int,
    queue: multiprocessing.Queue,
) -> None:
    """
    Worker process: play num_games games and push each result to queue.

    Runs entirely on CPU — network weights are copied before dispatch.
    """
    # Prevent OpenMP/MKL thread contention across spawned workers
    torch.set_num_threads(1)

    mcts = MCTS(network, num_simulations=num_simulations)
    for _ in range(num_games):
        game_data = play_one_game(mcts, temperature_threshold=temperature_threshold)
        queue.put(game_data)


def run_worker_with_server(
    worker_id: int,
    num_games: int,
    num_simulations: int,
    temperature_threshold: int,
    queue: multiprocessing.Queue,
    request_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
) -> None:
    """
    Worker process using GPU inference server — no local network needed.

    Instead of running forward passes locally, sends evaluation requests to
    the inference server via queues. Much faster for large networks.
    """
    torch.set_num_threads(1)

    from training.inference_server import InferenceClient

    client = InferenceClient(worker_id, request_queue, result_queue)

    # network=None since we use the inference client for all evaluations
    mcts = MCTS(
        network=None,
        num_simulations=num_simulations,
        inference_client=client,
    )
    for _ in range(num_games):
        game_data = play_one_game(mcts, temperature_threshold=temperature_threshold)
        queue.put(game_data)
