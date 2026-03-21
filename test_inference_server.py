"""
test_inference_server.py — Integration test for GPU inference server.

Tests:
1. InferenceServer starts and shuts down cleanly
2. InferenceClient can evaluate individual states correctly
3. generate_self_play_data(use_inference_server=True) completes a small batch
4. Results are valid training tuples with correct shapes

Run on a machine with PyTorch installed:
    python test_inference_server.py

If CUDA is available, the server uses GPU; otherwise falls back to CPU
(still validates the batching/queue architecture).
"""

import sys
import time

sys.path.insert(0, ".")

from game.hex_game import HexGame
from game.hex_state_vector import get_state_vector
from training.policy_value_network import PolicyValueNetwork
from training.inference_server import InferenceServer, InferenceClient
from training.mcts import MCTS
from training.self_play import generate_self_play_data, play_one_game

import numpy as np

# ── Test 1: Server lifecycle ────────────────────────────────────────────────

print("Test 1: InferenceServer start/stop...")
net = PolicyValueNetwork({})
net.eval()
param_count = sum(p.numel() for p in net.parameters())
print(f"  Network params: {param_count:,}")

server = InferenceServer(net, num_workers=2, batch_size=8, max_wait_ms=10)
server.start()
time.sleep(1)

assert (
    server._process is not None and server._process.is_alive()
), "Server process not alive"
print("  Server started OK")

server.stop()
assert not server._process.is_alive(), "Server process still alive after stop"
print("  Server stopped OK")
print("PASS: Test 1\n")


# ── Test 2: Client evaluation round-trip ────────────────────────────────────

print("Test 2: InferenceClient evaluation round-trip...")
server = InferenceServer(net, num_workers=1, batch_size=4, max_wait_ms=50)
server.start()
time.sleep(1)

client = InferenceClient(
    worker_id=0,
    request_queue=server.request_queue,
    result_queue=server.result_queues[0],
)

game = HexGame()
state_vec = get_state_vector(game)

priors, value = client.evaluate(state_vec)

assert isinstance(priors, np.ndarray), f"priors should be ndarray, got {type(priors)}"
assert priors.shape == (121,), f"priors shape should be (121,), got {priors.shape}"
assert -1.0 <= value <= 1.0, f"value should be in [-1, +1], got {value}"
assert abs(priors.sum() - 1.0) < 0.01, f"priors should sum to ~1.0, got {priors.sum()}"
print(f"  priors shape: {priors.shape}, sum: {priors.sum():.4f}")
print(f"  value: {value:.4f}")

# Verify consistency with direct network evaluation
import torch

with torch.no_grad():
    obs = torch.tensor([state_vec], dtype=torch.float32)
    logits, val = net(obs)
    direct_priors = torch.softmax(logits, dim=-1).squeeze(0).numpy()
    direct_value = float(val.squeeze().item())

assert np.allclose(priors, direct_priors, atol=1e-4), "Server priors differ from direct"
assert abs(value - direct_value) < 1e-4, "Server value differs from direct"
print("  Results match direct network evaluation")

server.stop()
print("PASS: Test 2\n")


# ── Test 3: MCTS with inference client ──────────────────────────────────────

print("Test 3: MCTS with inference client plays a full game...")
server = InferenceServer(net, num_workers=1, batch_size=8, max_wait_ms=20)
server.start()
time.sleep(1)

client = InferenceClient(0, server.request_queue, server.result_queues[0])
mcts = MCTS(network=None, num_simulations=25, inference_client=client)

game_data = play_one_game(mcts, temperature_threshold=10)

assert len(game_data) > 0, "Game produced no data"
for i, (state, policy, value) in enumerate(game_data):
    assert len(state) == 122, f"Step {i}: state should be 122 floats, got {len(state)}"
    assert (
        len(policy) == 121
    ), f"Step {i}: policy should be 121 floats, got {len(policy)}"
    assert abs(policy.sum() - 1.0) < 0.01, f"Step {i}: policy sum={policy.sum()}"
    assert value in (-1.0, 1.0), f"Step {i}: value should be ±1.0, got {value}"

print(f"  Game completed: {len(game_data)} moves")
server.stop()
print("PASS: Test 3\n")


# ── Test 4: Full self-play pipeline with inference server ────────────────────

print(
    "Test 4: generate_self_play_data(use_inference_server=True), 2 workers × 2 games..."
)
t0 = time.time()

results = generate_self_play_data(
    network=net,
    num_workers=2,
    games_per_worker=2,
    temperature_threshold=10,
    num_simulations=25,
    use_inference_server=True,
    inference_batch_size=8,
    inference_max_wait_ms=20,
)

elapsed = time.time() - t0
assert len(results) == 4, f"Expected 4 games, got {len(results)}"
total_positions = sum(len(g) for g in results)

for game_idx, game_data in enumerate(results):
    assert len(game_data) > 0, f"Game {game_idx} has no data"
    for step_idx, (state, policy, value) in enumerate(game_data):
        assert len(state) == 122
        assert len(policy) == 121
        assert value in (-1.0, 1.0)

print(f"  4 games completed in {elapsed:.1f}s ({total_positions} positions)")
print(f"  Avg game length: {total_positions / 4:.0f} moves")
print("PASS: Test 4\n")


# ── Test 5: CPU-only path still works (backward compat) ─────────────────────

print("Test 5: generate_self_play_data(use_inference_server=False) still works...")
t0 = time.time()

results_cpu = generate_self_play_data(
    network=net,
    num_workers=1,
    games_per_worker=2,
    temperature_threshold=10,
    num_simulations=25,
    use_inference_server=False,
)

elapsed = time.time() - t0
assert len(results_cpu) == 2, f"Expected 2 games, got {len(results_cpu)}"
print(f"  2 games completed in {elapsed:.1f}s (CPU-only)")
print("PASS: Test 5\n")

print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
