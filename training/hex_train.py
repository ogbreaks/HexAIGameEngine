"""
hex_train.py — Train Hex agents with PPO (Stable Baselines 3).

Usage
-----
    python training/hex_train.py --level easy
    python training/hex_train.py --level medium
    python training/hex_train.py --level hard
    python training/hex_train.py --level expert
    python training/hex_train.py --level hard --continue

Difficulty levels
-----------------
  All levels use the same CNN architecture (stem conv64 + 6×ResidualBlock + GAP + dense[256,128]).
  Only step counts and self-play swap frequency differ.

  easy   : 1 500 000 steps  swap every  75 000
  medium : 3 000 000 steps  swap every 100 000
  hard   : 8 000 000 steps  swap every 200 000
  expert : 15 000 000 steps  swap every 300 000

Outputs
-------
  training/models/hex_{level}.zip              — final model (always latest)
  training/models/hex_{level}_YYYYMMDD_HHMM.zip — backup of previous model
  training/checkpoints/{level}/                — periodic checkpoints

--continue flag
---------------
  Loads hex_{level}.zip as starting weights, picks up the step counter
  from the model's num_timesteps, seeds SelfPlayCallback with the latest
  checkpoint, and continues to the full total_steps target.
  Falls back to fresh training if no saved model is found.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime

# Allow imports from the project root (game/) when called from any cwd.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.hex_env import HexEnv

# ── Residual block ────────────────────────────────────────────────────────


class ResidualBlock(nn.Module):
    """Two-layer residual block with batch normalisation."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


# ── Residual CNN feature extractor ───────────────────────────────────────


class HexCNNExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor for the 11×11 Hex board.

    Input observation: flat vector of shape [batch, 122]
      [0:121]  board cells (1.0=P1, -1.0=P2, 0.0=empty) in row-major order
      [121]    current player (1.0=P1, -1.0=P2)

    Architecture:
      board  → reshape [batch, 1, 11, 11]
               → Conv2d(1→64, 3×3, pad=1) → BN → ReLU   (stem)
               → 6 × ResidualBlock(64)
               → Global average pooling → [batch, 64]
      player → [batch, 1]
      concat → [batch, 65]
               → Linear(65→256) → ReLU
               → Linear(256→features_dim)
    """

    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(6)])

        self.gap = nn.AdaptiveAvgPool2d(1)  # global average pooling → [B, 64, 1, 1]

        self.head = nn.Sequential(
            nn.Linear(64 + 1, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        board = obs[:, :121].view(-1, 1, 11, 11)  # [B, 1, 11, 11]
        player = obs[:, 121:122]  # [B, 1]
        x = self.stem(board)  # [B, 64, 11, 11]
        x = self.res_blocks(x)  # [B, 64, 11, 11]
        x = self.gap(x).flatten(1)  # [B, 64]
        combined = torch.cat([x, player], dim=1)  # [B, 65]
        return self.head(combined)  # [B, features_dim]


# ── Configuration ─────────────────────────────────────────────────────────

LEVELS: dict[str, dict] = {
    "easy": {
        "total_steps": 1_500_000,
        "swap_freq": 75_000,
    },
    "medium": {
        "total_steps": 3_000_000,
        "swap_freq": 100_000,
    },
    "hard": {
        "total_steps": 8_000_000,
        "swap_freq": 200_000,
    },
    "expert": {
        "total_steps": 15_000_000,
        "swap_freq": 300_000,
    },
}

CHECKPOINT_FREQ = 50_000  # steps between checkpoint saves


# ── Self-play update callback ─────────────────────────────────────────────

from stable_baselines3.common.callbacks import BaseCallback


class SelfPlayCallback(BaseCallback):
    """
    Every `swap_freq` steps, loads the latest checkpoint into the
    opponent model so the agent trains against progressively stronger play.
    """

    def __init__(self, env: HexEnv, checkpoint_dir: str, swap_freq: int = 100_000):
        super().__init__()
        self._hex_env = env
        self._checkpoint_dir = checkpoint_dir
        self._swap_freq = swap_freq
        self._last_swap = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_swap >= self._swap_freq:
            self._last_swap = self.num_timesteps
            # Find the checkpoint with the highest step count
            latest = _latest_checkpoint(self._checkpoint_dir)
            if latest:
                try:
                    opponent = PPO.load(latest)
                    self._hex_env.set_opponent_model(opponent)
                    print(
                        f"\n[SelfPlay] step={self.num_timesteps:,} — "
                        f"opponent upgraded → {os.path.basename(latest)}"
                    )
                except Exception as e:
                    self.logger.warn(f"[SelfPlay] Could not load checkpoint: {e}")
        return True


# ── Helpers ──────────────────────────────────────────────────────────────


def _latest_checkpoint(checkpoint_dir: str) -> str | None:
    """Return the path of the checkpoint with the highest step count, or None.

    Filenames are expected to contain the step count as a numeric token, e.g.
    hex_easy_50000_steps.zip.  Falls back to lexicographic order if no digits
    are found in a filename.
    """
    import re

    if not os.path.isdir(checkpoint_dir):
        return None
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]
    if not checkpoints:
        return None

    def _step_key(name: str) -> int:
        nums = re.findall(r"\d+", name)
        return max(int(n) for n in nums) if nums else 0

    latest = max(checkpoints, key=_step_key)
    return os.path.join(checkpoint_dir, latest)


def _backup_model(model_path: str) -> None:
    """Copy model_path to a timestamped backup in the same directory."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    base, ext = os.path.splitext(model_path)
    backup_path = f"{base}_{stamp}{ext}"
    shutil.copy2(model_path, backup_path)
    print(f"[hex_train] Backed up existing model → {backup_path}")


# ── Main ──────────────────────────────────────────────────────────────────


def train(level: str, resume: bool = False) -> None:
    if level not in LEVELS:
        raise ValueError(f"Unknown level '{level}'. Choose from: {list(LEVELS)}")

    cfg = LEVELS[level]
    total_steps: int = cfg["total_steps"]
    swap_freq: int = cfg["swap_freq"]

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints", level)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Build env
    base_env = HexEnv(agent_player=1)
    env = DummyVecEnv([lambda: base_env])

    output_path = os.path.join(models_dir, f"hex_{level}.zip")
    existing_model_path = output_path if os.path.exists(output_path) else None

    steps_already_done = 0
    initial_opponent_ckpt: str | None = None

    if resume and existing_model_path:
        print(f"[hex_train] Loading existing model: {existing_model_path}")
        model = PPO.load(existing_model_path, env=env)
        steps_already_done = int(model.num_timesteps)
        print(f"[hex_train] Resuming from step {steps_already_done:,}")

        # Seed the self-play opponent with the best checkpoint available.
        initial_opponent_ckpt = _latest_checkpoint(checkpoint_dir)
        if initial_opponent_ckpt:
            print(f"[hex_train] Initial opponent: {initial_opponent_ckpt}")
            try:
                base_env.set_opponent_model(PPO.load(initial_opponent_ckpt))
            except Exception as e:
                print(f"[hex_train] Warning: could not load opponent checkpoint: {e}")
    else:
        if resume:
            print("[hex_train] No existing model found — starting fresh.")
        policy_kwargs = dict(
            features_extractor_class=HexCNNExtractor,
            features_extractor_kwargs=dict(features_dim=128),
        )
        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            policy_kwargs=policy_kwargs,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log=os.path.join(os.path.dirname(__file__), "results", level),
        )

    remaining_steps = max(0, total_steps - steps_already_done)
    if remaining_steps == 0:
        print(
            f"[hex_train] Model already at {steps_already_done:,} steps — nothing to do."
        )
        return

    callbacks = [
        CheckpointCallback(
            save_freq=CHECKPOINT_FREQ,
            save_path=checkpoint_dir,
            name_prefix=f"hex_{level}",
            verbose=1,
        ),
        SelfPlayCallback(
            env=base_env,
            checkpoint_dir=checkpoint_dir,
            swap_freq=swap_freq,
        ),
    ]

    print(
        f"\n[hex_train] Level={level}  arch=HexCNN(conv64x3+dense[256,128])  "
        f"target={total_steps:,}  remaining={remaining_steps:,}\n"
    )
    model.learn(
        total_timesteps=remaining_steps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=not resume,
    )

    # Backup any existing saved model before overwriting.
    if os.path.exists(output_path):
        _backup_model(output_path)

    model.save(output_path)
    print(f"\n[hex_train] Saved model → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Hex PPO agent.")
    parser.add_argument(
        "--level",
        choices=list(LEVELS),
        required=True,
        help="Difficulty level to train.",
    )
    parser.add_argument(
        "--continue",
        dest="resume",
        action="store_true",
        help="Resume training from the existing model and latest checkpoint.",
    )
    args = parser.parse_args()
    train(args.level, resume=args.resume)
