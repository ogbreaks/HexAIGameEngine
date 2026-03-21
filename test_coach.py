"""Phase 4 — Training loop runs 5 iterations without error."""

import sys

sys.path.insert(0, ".")

from training.coach import Coach

config = {
    "trunk": "resnet",
    "num_res_blocks": 4,  # smaller network for speed
    "num_channels": 64,
    "num_iterations": 5,
    "num_workers": 1,  # single worker for local test
    "games_per_worker": 4,
    "num_simulations": 50,
    "batch_size": 32,
    "train_steps_per_iter": 10,
    "min_buffer_size": 50,
    "buffer_size": 10000,
    "lr_init": 0.001,
    "arena_freq": 999,  # skip arena during test
    "arena_games": 20,
    "arena_simulations": 50,
    "promotion_threshold": 0.55,
    "temperature_threshold": 30,
    "checkpoint_freq": 999,  # skip checkpointing during test
    "model_dir": "training/models",
    "checkpoint_dir": "training/checkpoints",
    "level": "test",
}

coach = Coach(config)
coach.train()
print("Phase 4 PASS — 5 iterations completed without error")
