"""
inference_server.py — GPU batched inference server for MCTS self-play workers.

Problem
-------
Self-play workers run MCTS on CPU (CUDA contexts can't cross process boundaries).
Each MCTS simulation evaluates ONE board state → single forward pass → GPU idle 95%.

Solution
--------
A dedicated inference server runs in its own process with exclusive GPU access.
Workers send evaluation requests via shared memory queues; the server batches
them and runs vectorised GPU forward passes. Results are returned per-worker.

Architecture
------------
    Worker 0 ──┐                    ┌── Result Queue 0
    Worker 1 ──┤ Request Queue ──→  │── Result Queue 1
    Worker 2 ──┤   (shared)    GPU  │── Result Queue 2
    Worker 3 ──┘   Server      ↓    └── Result Queue 3

Each request: (worker_id, state_vector [122])
Each result:  (priors [121], value float)

The server accumulates requests up to `batch_size` or `max_wait_ms`, then
runs a single batched forward pass on GPU and dispatches results.

Usage
-----
    # In self_play.py — start server, pass queues to workers
    server = InferenceServer(network, num_workers=16, batch_size=64)
    server.start()
    ...
    server.stop()
"""

from __future__ import annotations

import multiprocessing
import multiprocessing.queues
import time
from typing import Optional

import numpy as np
import torch


class InferenceServer:
    """
    GPU-resident batched inference server for MCTS workers.

    Parameters
    ----------
    network          : PolicyValueNetwork — moved to GPU for inference
    num_workers      : int   — number of worker processes
    batch_size       : int   — max states per forward pass (default: 64)
    max_wait_ms      : float — max milliseconds to wait for batch to fill (default: 5)
    device           : str   — torch device (default: 'cuda' if available)
    """

    def __init__(
        self,
        network,
        num_workers: int,
        batch_size: int = 64,
        max_wait_ms: float = 5.0,
        device: str | None = None,
    ) -> None:
        self.network = network
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ctx = multiprocessing.get_context("spawn")

        # Shared request queue: workers push (worker_id, state_vector)
        self.request_queue: multiprocessing.Queue = ctx.Queue()

        # Per-worker result queues: server pushes (priors, value) to worker_id's queue
        self.result_queues: list[multiprocessing.Queue] = [
            ctx.Queue() for _ in range(num_workers)
        ]

        # Shutdown signal
        self._stop_event = ctx.Event()
        self._process: Optional[multiprocessing.Process] = None

    def start(self) -> None:
        """Start the inference server in a background process."""
        ctx = multiprocessing.get_context("spawn")
        self._process = ctx.Process(
            target=_server_loop,
            args=(
                self.network,
                self.request_queue,
                self.result_queues,
                self.batch_size,
                self.max_wait_ms,
                self.device,
                self._stop_event,
            ),
            daemon=True,
        )
        self._process.start()

    def stop(self) -> None:
        """Signal shutdown and join."""
        self._stop_event.set()
        if self._process is not None:
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()


class InferenceClient:
    """
    Client-side proxy used by workers to request GPU evaluations.

    Drop-in replacement for local network inference in MCTS._evaluate().
    """

    def __init__(
        self,
        worker_id: int,
        request_queue: multiprocessing.Queue,
        result_queue: multiprocessing.Queue,
    ) -> None:
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.result_queue = result_queue

    def evaluate(self, state_vector: list[float]) -> tuple[np.ndarray, float]:
        """
        Send a state to the inference server and block until result.

        Returns
        -------
        (priors [121], value float)
        """
        self.request_queue.put((self.worker_id, state_vector))
        priors, value = self.result_queue.get()  # blocks
        return priors, value


# ---------------------------------------------------------------------------
# Server loop — runs in its own process with GPU access
# ---------------------------------------------------------------------------


def _server_loop(
    network,
    request_queue: multiprocessing.Queue,
    result_queues: list[multiprocessing.Queue],
    batch_size: int,
    max_wait_ms: float,
    device: str,
    stop_event,
) -> None:
    """
    Main inference server loop. Collects requests, batches them, runs GPU forward
    passes, and dispatches results to per-worker queues.
    """
    network = network.to(device)
    network.eval()

    max_wait_sec = max_wait_ms / 1000.0

    while not stop_event.is_set():
        batch_worker_ids: list[int] = []
        batch_states: list[list[float]] = []

        # Collect up to batch_size requests (or wait max_wait_ms)
        deadline = time.monotonic() + max_wait_sec

        while len(batch_states) < batch_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                worker_id, state_vec = request_queue.get(timeout=max(remaining, 0.001))
                batch_worker_ids.append(worker_id)
                batch_states.append(state_vec)
            except multiprocessing.queues.Empty:
                break

        if not batch_states:
            continue

        # Batched GPU forward pass
        with torch.no_grad():
            obs = torch.tensor(batch_states, dtype=torch.float32, device=device)
            policy_logits, value_t = network(obs)
            priors = torch.softmax(policy_logits, dim=-1).cpu().numpy()
            values = value_t.squeeze(-1).cpu().numpy()

        # Dispatch results to each worker's queue
        for i, worker_id in enumerate(batch_worker_ids):
            result_queues[worker_id].put((priors[i], float(values[i])))

    # Drain remaining requests on shutdown
    while not request_queue.empty():
        try:
            worker_id, state_vec = request_queue.get_nowait()
            # Return zeros — workers will terminate anyway
            result_queues[worker_id].put((np.zeros(121, dtype=np.float32), 0.0))
        except multiprocessing.queues.Empty:
            break
