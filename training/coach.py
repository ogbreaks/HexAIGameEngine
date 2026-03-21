"""
coach.py — AlphaZero training loop for 11×11 Hex.

Usage
-----
    from training.coach import Coach
    import yaml

    with open("config/hex11_default.yaml") as f:
        config = yaml.safe_load(f)
    coach = Coach(config)
    coach.train()

Metrics
-------
Writes training/metrics.json after every iteration in the same schema as
the PPO callback — the existing dashboard works without changes.
Progress unit: iterations (one full self-play + train cycle = one iteration).

Cost persistence
----------------
training/cost_state.json stores {"total_hours": float} across restarts.
Loaded on __init__, updated on every metrics write. Crash-safe (atomic write).
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Optional

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F

from training.policy_value_network import PolicyValueNetwork
from training.replay_buffer import ReplayBuffer
from training.self_play import generate_self_play_data
from training.arena import Arena
from training.players import MCTSPlayer
from training.export import export_onnx


class Coach:
    """
    AlphaZero training loop.

    Parameters
    ----------
    config : dict  — loaded from config/hex11_default.yaml (or equivalent)
    """

    def __init__(self, config: dict) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Network + optimiser
        self.network = PolicyValueNetwork(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.get("lr_init", 0.001),
            weight_decay=config.get("weight_decay", 0.0),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("num_iterations", 1000),
        )

        # Best network (kept on CPU for self-play dispatch)
        self.best_network = copy.deepcopy(self.network).to("cpu")
        self.best_network.eval()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.get("buffer_size", 500_000))

        # Training state
        self.iteration: int = 0

        # Metrics tracking
        self._start_time: float = time.time()
        self._last_iter_per_sec: float = 0.0
        self._last_policy_loss: float = 0.0
        self._last_value_loss: float = 0.0
        self._last_arena_win_rate: Optional[float] = None
        self._promotion_count: int = 0
        self._recent_outcomes: deque[int] = deque(maxlen=200)  # 1=win, 0=loss

        # Cost tracking
        self._prior_session_hours: float = self._load_prior_hours()
        self._hourly_rate: float = float(os.environ.get("HOURLY_RATE", "0.0"))
        self._tz_offset: int = int(os.environ.get("TIMEZONE_OFFSET", "0"))

        # Paths
        _training_dir = os.path.dirname(os.path.abspath(__file__))
        self._metrics_path = os.path.join(_training_dir, "metrics.json")
        self._cost_state_path = os.path.join(_training_dir, "cost_state.json")
        self._model_dir = config.get("model_dir", os.path.join(_training_dir, "models"))
        self._checkpoint_dir = config.get(
            "checkpoint_dir", os.path.join(_training_dir, "checkpoints")
        )

        os.makedirs(self._model_dir, exist_ok=True)
        os.makedirs(self._checkpoint_dir, exist_ok=True)

    # ── Cost persistence ────────────────────────────────────────────────────

    def _load_prior_hours(self) -> float:
        """Load accumulated training hours from cost_state.json (0 if missing)."""
        _training_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(_training_dir, "cost_state.json")
        try:
            with open(path) as f:
                return float(json.load(f).get("total_hours", 0.0))
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return 0.0

    def _save_total_hours(self, total_hours: float) -> None:
        """Atomically persist total accumulated training hours."""
        tmp = self._cost_state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"total_hours": total_hours}, f)
        os.replace(tmp, self._cost_state_path)

    # ── Win rate helper ─────────────────────────────────────────────────────

    def _win_rate(self) -> Optional[float]:
        """Rolling win rate over last 200 self-play games (None if insufficient)."""
        if len(self._recent_outcomes) < 10:
            return None
        return round(sum(self._recent_outcomes) / len(self._recent_outcomes), 4)

    # ── Training steps ──────────────────────────────────────────────────────

    def _train_epoch(self) -> None:
        """Run train_steps_per_iter gradient updates on sampled replay buffer data."""
        self.network.train()
        steps = self.config.get("train_steps_per_iter", 100)
        batch_size = self.config.get("batch_size", 512)
        total_policy_loss = 0.0
        total_value_loss = 0.0

        for _ in range(steps):
            states, pi_targets, v_targets = self.replay_buffer.sample(batch_size)

            states_t = torch.FloatTensor(states).to(self.device)
            pi_targets_t = torch.FloatTensor(pi_targets).to(self.device)
            v_targets_t = torch.FloatTensor(v_targets).to(self.device)

            pi_logits, v_pred = self.network(states_t)

            # Illegal action masking: zero out logits for illegal positions
            # pi_targets already has 0 for illegal actions (MCTS visit counts)
            legal_mask = (pi_targets_t > 0).float()  # [B, 121]
            masked_logits = pi_logits - 1e9 * (1.0 - legal_mask)

            policy_loss = F.cross_entropy(masked_logits, pi_targets_t)
            value_loss = F.mse_loss(v_pred.squeeze(-1), v_targets_t)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        self._last_policy_loss = total_policy_loss / steps
        self._last_value_loss = total_value_loss / steps

    def _evaluate_and_promote(self) -> None:
        """
        Run Arena between network (challenger) and best_network (champion).
        Promote challenger if win_rate > promotion_threshold.
        """
        self.network.eval()
        arena_sims = self.config.get("arena_simulations", 200)
        arena_games = self.config.get("arena_games", 40)
        threshold = self.config.get("promotion_threshold", 0.55)

        # Both players run on CPU
        challenger_cpu = copy.deepcopy(self.network).to("cpu")
        challenger_cpu.eval()

        challenger = MCTSPlayer(challenger_cpu, num_simulations=arena_sims)
        champion = MCTSPlayer(self.best_network, num_simulations=arena_sims)

        arena = Arena(challenger, champion, num_games=arena_games)
        p1_wins, p2_wins, _ = arena.play_games()

        total = p1_wins + p2_wins
        win_rate = p1_wins / total if total > 0 else 0.0
        self._last_arena_win_rate = round(win_rate, 4)

        print(
            f"[Arena] iter={self.iteration}  challenger={p1_wins}  "
            f"champion={p2_wins}  win_rate={win_rate:.3f}"
        )

        if win_rate > threshold:
            print(
                f"[Arena] Promoting challenger (win_rate={win_rate:.3f} > {threshold})"
            )
            self.best_network = copy.deepcopy(self.network).to("cpu")
            self.best_network.eval()
            self._promotion_count += 1

            # Export ONNX checkpoint
            onnx_path = os.path.join(self._model_dir, f"hex_az_{self.iteration}.onnx")
            try:
                export_onnx(self.network, onnx_path)
                print(f"[Arena] Exported ONNX to {onnx_path}")
            except Exception as exc:
                print(f"[Arena] ONNX export failed: {exc}")

            # Save promoted best weights
            best_path = os.path.join(self._model_dir, "hex_az_best.pth")
            torch.save(self.network.state_dict(), best_path)

    def _save_checkpoint(self, tag: str) -> None:
        """Save full training checkpoint (network + optimiser + iteration)."""
        path = os.path.join(self._checkpoint_dir, f"hex_az_{tag}.pth")
        torch.save(
            {
                "iteration": self.iteration,
                "model_state_dict": self.network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.config,
            },
            path,
        )
        # Keep best weights separately for server loading (weights only)
        best_path = os.path.join(self._model_dir, "hex_az_best.pth")
        torch.save(self.network.state_dict(), best_path)

    # ── Metrics ─────────────────────────────────────────────────────────────

    def _write_metrics(self, status: str = "training") -> None:
        """
        Write training/metrics.json in the same schema as the PPO callback.
        Progress unit: iterations (one full self-play + train cycle = one iteration).
        """
        now = time.time()
        elapsed_session = now - self._start_time

        iterations_completed = self.iteration
        iterations_total = self.config.get("num_iterations", 1000)

        # iterations/sec — stable after 10s warmup
        if elapsed_session > 10 and iterations_completed > 0:
            iter_per_sec = iterations_completed / elapsed_session
            self._last_iter_per_sec = iter_per_sec
        else:
            iter_per_sec = self._last_iter_per_sec

        remaining_iters = max(iterations_total - iterations_completed, 0)
        eta_seconds = (remaining_iters / iter_per_sec) if iter_per_sec > 0 else 0

        total_elapsed_hours = self._prior_session_hours + (elapsed_session / 3600)
        if iterations_completed > 0:
            secs_per_iter = elapsed_session / iterations_completed
            total_estimated_hours = (
                self._prior_session_hours + secs_per_iter * iterations_total / 3600
            )
        else:
            total_estimated_hours = 0.0

        cost_so_far = round(total_elapsed_hours * self._hourly_rate, 4)
        cost_estimate_total = round(total_estimated_hours * self._hourly_rate, 4)

        utc_finish = datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)
        local_finish = utc_finish + timedelta(hours=self._tz_offset)
        eta_label = local_finish.strftime("%a %d %b at %H:%M")

        percent = (
            round(iterations_completed / iterations_total * 100, 2)
            if iterations_total > 0
            else 0.0
        )

        is_complete = self.iteration >= self.config["num_iterations"]
        status = "complete" if is_complete else status
        percent = 100.0 if is_complete else percent

        metrics = {
            "status": status,
            "progress": {
                "steps_completed": iterations_completed,
                "steps_total": iterations_total,
                "percent": percent,
                "fps": round(iter_per_sec, 3),  # iterations/sec
            },
            "time": {
                "elapsed_session_hours": round(elapsed_session / 3600, 3),
                "elapsed_total_hours": round(total_elapsed_hours, 3),
                "eta_hours": round(eta_seconds / 3600, 2),
                "eta_label": eta_label,
            },
            "cost": {
                "hourly_rate": self._hourly_rate,
                "cost_so_far": cost_so_far,
                "cost_estimate_total": cost_estimate_total,
                "currency": "USD",
            },
            "training": {
                "policy_loss": round(self._last_policy_loss, 4),
                "value_loss": round(self._last_value_loss, 4),
                "win_rate_last_200": self._win_rate(),
                "arena_win_rate": self._last_arena_win_rate,
                "network_promotions": self._promotion_count,
                "buffer_size": len(self.replay_buffer),
                "iteration": self.iteration,
            },
            "meta": {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "level": self.config.get("level", "az"),
            },
        }

        # Atomic write
        tmp = self._metrics_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(metrics, f, indent=2)
        os.replace(tmp, self._metrics_path)

        # Persist cost state
        self._save_total_hours(total_elapsed_hours)

    # ── Main training loop ──────────────────────────────────────────────────

    def train(self, start_iteration: int = 0) -> None:
        """
        Run the full AlphaZero training loop.

        Parameters
        ----------
        start_iteration : int  — resume from this iteration (loaded from checkpoint)
        """
        self.iteration = start_iteration
        num_iterations = self.config.get("num_iterations", 1000)
        min_buffer = self.config.get("min_buffer_size", 10_000)
        arena_freq = self.config.get("arena_freq", 10)
        checkpoint_freq = self.config.get("checkpoint_freq", 50)
        num_workers = self.config.get("num_workers", 4)
        games_per_worker = self.config.get("games_per_worker", 25)
        temperature_threshold = self.config.get("temperature_threshold", 30)
        num_simulations = self.config.get("num_simulations", 800)
        use_inference_server = self.config.get("use_inference_server", False)
        inference_batch_size = self.config.get("inference_batch_size", 64)
        inference_max_wait_ms = self.config.get("inference_max_wait_ms", 5.0)

        print(f"[Coach] Starting training on {self.device}")
        print(f"[Coach] Iterations: {start_iteration} → {num_iterations}")

        # Guard: warn if cloud-scale settings are used without inference server
        if not use_inference_server:
            if num_simulations > 800:
                print(
                    f"[Coach] WARNING: num_simulations={num_simulations} without "
                    f"inference server — self-play will be very slow on CPU."
                )
            if self.config.get("batch_size", 512) > 512:
                print(
                    f"[Coach] WARNING: batch_size={self.config['batch_size']} without "
                    f"inference server — may OOM on CPU-only training."
                )
        else:
            print(
                f"[Coach] GPU inference server enabled (batch={inference_batch_size})"
            )

        # Write initial metrics immediately so the dashboard shows "training" at 0%
        self.iteration = start_iteration
        self._write_metrics(status="training")

        for i in range(start_iteration, num_iterations):
            iter_start = time.time()

            print(f"\n[Coach] Iteration {i}/{num_iterations}")

            # 1. Self-play data generation
            print(
                f"[Coach] Generating self-play data ({num_workers}w × {games_per_worker}g)…"
            )
            data = generate_self_play_data(
                self.network,
                num_workers=num_workers,
                games_per_worker=games_per_worker,
                temperature_threshold=temperature_threshold,
                num_simulations=num_simulations,
                use_inference_server=use_inference_server,
                inference_batch_size=inference_batch_size,
                inference_max_wait_ms=inference_max_wait_ms,
            )
            for game_data in data:
                self.replay_buffer.add_game(game_data)

                # Track win/loss outcomes for rolling window
                # Each game ends with a winner; the last entry's value_target reveals who won
                if game_data:
                    last_value = game_data[-1][2]
                    # Last move player: len(game_data) - 1 steps → player = 1 if even else 2
                    last_step = len(game_data) - 1
                    last_player = 1 if last_step % 2 == 0 else 2
                    # Winner is the player whose value is +1.0 at the last step
                    winner = last_player if last_value > 0 else 3 - last_player
                    # Record from player-1 perspective
                    self._recent_outcomes.append(1 if winner == 1 else 0)

            print(f"[Coach] Buffer size: {len(self.replay_buffer)}")

            # 2. Train on buffer
            if len(self.replay_buffer) >= min_buffer:
                print("[Coach] Training network…")
                self._train_epoch()
                print(
                    f"[Coach] policy_loss={self._last_policy_loss:.4f}  "
                    f"value_loss={self._last_value_loss:.4f}"
                )
            else:
                print(
                    f"[Coach] Buffer too small ({len(self.replay_buffer)}/{min_buffer}), skipping train."
                )

            # 3. Arena evaluation
            if i > 0 and i % arena_freq == 0:
                print("[Coach] Running arena evaluation…")
                self._evaluate_and_promote()

            # 4. Write metrics — increment first so count reflects completed iterations
            self.iteration = i + 1
            self._write_metrics(status="training")

            # 5. Step LR scheduler
            self.scheduler.step()

            # 6. Periodic checkpoint
            if i > 0 and i % checkpoint_freq == 0:
                self._save_checkpoint(str(i))
                print(f"[Coach] Checkpoint saved at iteration {i}")

            elapsed = time.time() - iter_start
            print(f"[Coach] Iteration {i} complete in {elapsed:.1f}s")

        # Final save
        self._save_checkpoint("final")
        self._write_metrics(status="complete")
        print("[Coach] Training complete.")

    @classmethod
    def load_checkpoint(
        cls, checkpoint_path: str, config: dict | None = None
    ) -> "Coach":
        """
        Resume training from a saved checkpoint.

        Returns
        -------
        (coach, start_iteration)
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if config is None:
            config = checkpoint.get("config", {})

        coach = cls(config)
        coach.network.load_state_dict(checkpoint["model_state_dict"])
        coach.network.to(coach.device)
        coach.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        coach.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_iteration = checkpoint.get("iteration", 0) + 1
        return coach, start_iteration
