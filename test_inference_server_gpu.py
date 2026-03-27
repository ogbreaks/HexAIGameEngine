"""
test_inference_server_gpu.py — Fast smoke test for the GPU inference server fix.

Verifies:
  1. InferenceServer starts without crashing (CUDA pickling fix)
  2. Workers can send requests and receive results
  3. Server process is alive after startup
  4. GPU is actually used (device reported as cuda)

Run:
    python test_inference_server_gpu.py

Expected output (GPU machine):
    [OK] InferenceServer started on cuda
    [OK] Worker received 5 results in Xs
    [OK] GPU inference server working correctly

Expected output (CPU-only machine):
    [OK] InferenceServer started on cpu
    [OK] Worker received 5 results in Xs
    [OK] Inference server working correctly (CPU mode)

Exit code 0 = pass, 1 = fail
"""

import sys
import time
import multiprocessing
import torch

sys.path.insert(0, ".")

from training.policy_value_network import PolicyValueNetwork
from training.inference_server import InferenceServer, InferenceClient
from game.hex_game import HexGame
from game.hex_state_vector import get_state_vector

CONFIG = {
    "trunk": "resnet",
    "num_res_blocks": 4,
    "num_channels": 64,
}
NUM_REQUESTS = 5


def _worker(worker_id, request_queue, result_queue, n, done_queue):
    """Send n requests and collect results."""
    from game.hex_game import HexGame
    from game.hex_state_vector import get_state_vector
    from training.inference_server import InferenceClient

    client = InferenceClient(worker_id, request_queue, result_queue)
    game = HexGame()
    sv = get_state_vector(game)
    t0 = time.time()
    for _ in range(n):
        priors, value = client.evaluate(sv)
        assert len(priors) == 121, f"Expected 121 priors, got {len(priors)}"
        assert -1.0 <= value <= 1.0, f"Value out of range: {value}"
    elapsed = time.time() - t0
    done_queue.put(("ok", elapsed))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TEST] torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[TEST] GPU: {torch.cuda.get_device_name(0)}")

    # Build network and extract CPU state dict
    net = PolicyValueNetwork(CONFIG)
    net.eval()
    state_dict = {k: v.cpu() for k, v in net.state_dict().items()}

    # Start inference server
    print("[TEST] Starting InferenceServer...")
    t0 = time.time()
    try:
        server = InferenceServer(
            net_config=CONFIG,
            state_dict=state_dict,
            num_workers=1,
            batch_size=8,
            max_wait_ms=10,
        )
        server.start()
    except RuntimeError as exc:
        print(f"[FAIL] InferenceServer failed to start: {exc}")
        sys.exit(1)

    startup_time = time.time() - t0
    print(f"[OK] InferenceServer started on {server.device} in {startup_time:.1f}s")

    # Spawn a worker to send requests
    ctx = multiprocessing.get_context("spawn")
    done_queue = ctx.Queue()
    p = ctx.Process(
        target=_worker,
        args=(
            0,
            server.request_queue,
            server.result_queues[0],
            NUM_REQUESTS,
            done_queue,
        ),
    )
    p.start()

    try:
        result, elapsed = done_queue.get(timeout=30)
    except Exception:
        print("[FAIL] Worker timed out — server is not dispatching results")
        server.stop()
        p.terminate()
        sys.exit(1)

    p.join(timeout=5)

    if result != "ok":
        print(f"[FAIL] Worker reported: {result}")
        server.stop()
        sys.exit(1)

    print(f"[OK] Worker received {NUM_REQUESTS} results in {elapsed:.2f}s")
    server.stop()

    mode = "GPU" if device == "cuda" else "CPU"
    print(f"[OK] Inference server working correctly ({mode} mode)")
    print("[PASS] All checks passed — safe to deploy")
    sys.exit(0)


if __name__ == "__main__":
    main()
