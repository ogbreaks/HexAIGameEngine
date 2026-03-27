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
from game.hex_symmetry import augment_game_data


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

        # Phase tracking (read-only telemetry for dashboard)
        self._phase: str = "init"
        self._sub_done: int = 0
        self._sub_total: int = 0
        self._sub_label: str = ""

        # ELO tracking
        self._elo_k: float = config.get("elo_k_factor", 32)
        self._elo_initial: float = config.get("elo_initial", 1000)
        self._champion_elo: float = self._elo_initial
        self._elo_history: list[dict] = [{"iter": 0, "elo": self._elo_initial}]

        # Event log (rolling buffer for dashboard console)
        self._events: deque[dict] = deque(maxlen=30)

        # Cost tracking
        self._prior_session_hours: float = self._load_prior_hours()
        self._hourly_rate: float = float(os.environ.get("HOURLY_RATE", "0.0"))
        self._tz_offset: int = int(os.environ.get("TIMEZONE_OFFSET", "0"))

        # Paths
        _training_dir = os.path.dirname(os.path.abspath(__file__))
        self._metrics_path = os.path.join(_training_dir, "metrics.json")
        self._events_path = os.path.join(_training_dir, "events.json")
        self._cost_state_path = os.path.join(_training_dir, "cost_state.json")
        self._elo_state_path = os.path.join(_training_dir, "elo_state.json")
        self._model_dir = config.get("model_dir", os.path.join(_training_dir, "models"))
        self._checkpoint_dir = config.get(
            "checkpoint_dir", os.path.join(_training_dir, "checkpoints")
        )

        os.makedirs(self._model_dir, exist_ok=True)
        os.makedirs(self._checkpoint_dir, exist_ok=True)

        # Load persisted ELO state (survives restarts)
        self._load_elo_state()

    # ── Event log ───────────────────────────────────────────────────────────

    def _log_event(self, msg: str) -> None:
        """Append a timestamped event to the rolling log (dashboard console)."""
        tz = timezone(timedelta(hours=self._tz_offset))
        ts = datetime.now(tz).strftime("%H:%M")
        self._events.append({"t": ts, "msg": msg})
        try:
            tmp = self._events_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(list(self._events), f)
            os.replace(tmp, self._events_path)
        except Exception:
            pass  # Never let event logging interrupt training

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

    # ── ELO helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _elo_expected(rating_a: float, rating_b: float) -> float:
        """Expected score for player A against player B."""
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

    def _update_elo(self, win_rate: float) -> None:
        """Update champion ELO after an arena evaluation.

        Challenger's provisional rating equals the champion's current ELO
        (unknown strength assumption). The match score is the observed
        win_rate (0-1) of the challenger.
        """
        challenger_elo = self._champion_elo  # provisional
        expected = self._elo_expected(challenger_elo, self._champion_elo)
        # New rating for the challenger based on arena outcome
        new_challenger_elo = challenger_elo + self._elo_k * (win_rate - expected)
        # Champion rating adjusts inversely
        new_champion_elo = self._champion_elo + self._elo_k * (
            (1 - win_rate) - expected
        )
        # The network that survives (promoted or not) carries the updated rating
        self._champion_elo = round(
            (
                new_challenger_elo
                if win_rate > self.config.get("promotion_threshold", 0.55)
                else new_champion_elo
            ),
            1,
        )
        self._elo_history.append({"iter": self.iteration, "elo": self._champion_elo})
        self._save_elo_state()

    def _load_elo_state(self) -> None:
        """Load persisted ELO ratings from elo_state.json."""
        try:
            with open(self._elo_state_path) as f:
                state = json.load(f)
            self._champion_elo = float(state.get("champion_elo", self._elo_initial))
            self._elo_history = state.get(
                "history", [{"iter": 0, "elo": self._elo_initial}]
            )
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            pass  # Keep defaults set in __init__

    def _save_elo_state(self) -> None:
        """Atomically persist ELO state."""
        state = {
            "champion_elo": self._champion_elo,
            "history": self._elo_history,
        }
        tmp = self._elo_state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self._elo_state_path)

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

        for step_i in range(steps):
            self._sub_done = step_i
            self._sub_label = f"{step_i}/{steps} steps"
            if step_i > 0 and step_i % 20 == 0:
                self._write_metrics(status="training")

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
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
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
        arena_workers = self.config.get(
            "arena_workers", self.config.get("num_workers", 4)
        )

        if arena_workers > 1:
            # Parallel arena — distribute games across worker processes
            challenger_cpu = copy.deepcopy(self.network).to("cpu")
            challenger_cpu.eval()
            p1_wins, p2_wins, _ = Arena.play_games_parallel(
                net_config=self.config,
                challenger_sd=challenger_cpu.state_dict(),
                champion_sd=self.best_network.state_dict(),
                num_sims=arena_sims,
                num_games=arena_games,
                num_workers=arena_workers,
            )
        else:
            # Sequential arena — single process
            challenger_cpu = copy.deepcopy(self.network).to("cpu")
            challenger_cpu.eval()
            challenger = MCTSPlayer(challenger_cpu, num_simulations=arena_sims)
            champion = MCTSPlayer(self.best_network, num_simulations=arena_sims)
            arena = Arena(challenger, champion, num_games=arena_games)
            p1_wins, p2_wins, _ = arena.play_games()

        total = p1_wins + p2_wins
        win_rate = p1_wins / total if total > 0 else 0.0
        self._last_arena_win_rate = round(win_rate, 4)

        # Update ELO before promotion decision (uses current champion rating)
        self._update_elo(win_rate)

        print(
            f"[Arena] iter={self.iteration}  challenger={p1_wins}  "
            f"champion={p2_wins}  win_rate={win_rate:.3f}  "
            f"elo={self._champion_elo}"
        )

        if win_rate > threshold:
            self._log_event(
                f"Arena: challenger {win_rate:.0%} vs champion \u2014 PROMOTED"
            )
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
        else:
            self._log_event(f"Arena: challenger {win_rate:.0%} vs champion \u2014 kept")

    def _save_checkpoint(self, tag: str) -> None:
        """Save full training checkpoint (network + optimiser + iteration)."""
        path = os.path.join(self._checkpoint_dir, f"hex_az_{tag}.pth")
        torch.save(
            {
                "iteration": self.iteration,
                "model_state_dict": self.network.state_dict(),
                "best_model_state_dict": self.best_network.state_dict(),
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
                "current_elo": self._champion_elo,
                "elo_history": self._elo_history,
            },
            "phase": {
                "name": self._phase,
                "sub_done": self._sub_done,
                "sub_total": self._sub_total,
                "sub_label": self._sub_label,
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
        virtual_loss_k = self.config.get("virtual_loss_k", 1)

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
            self._log_event(f"Iteration {i}/{num_iterations} started")

            # 1. Self-play data generation
            total_games = num_workers * games_per_worker
            self._phase = "self_play"
            self._sub_done = 0
            self._sub_total = total_games
            self._sub_label = f"0/{total_games} games"
            self._write_metrics(status="training")
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
                virtual_loss_k=virtual_loss_k,
                net_config=self.config,
            )
            for game_data in data:
                self.replay_buffer.add_game(game_data)
                if self.config.get("augment_symmetry", True):
                    self.replay_buffer.add_game(augment_game_data(game_data))

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

            self._sub_done = total_games
            self._sub_label = f"{total_games}/{total_games} games"
            self._log_event(
                f"Self-play complete \u2014 {total_games} games, buffer {len(self.replay_buffer):,}"
            )
            print(f"[Coach] Buffer size: {len(self.replay_buffer)}")

            # 2. Train on buffer
            if len(self.replay_buffer) >= min_buffer:
                steps = self.config.get("train_steps_per_iter", 100)
                self._phase = "training"
                self._sub_done = 0
                self._sub_total = steps
                self._sub_label = f"0/{steps} steps"
                self._write_metrics(status="training")
                print("[Coach] Training network…")
                self._train_epoch()
                self._log_event(
                    f"Training done \u2014 policy={self._last_policy_loss:.4f}, "
                    f"value={self._last_value_loss:.4f}"
                )
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
                arena_games = self.config.get("arena_games", 40)
                self._phase = "arena"
                self._sub_done = 0
                self._sub_total = arena_games
                self._sub_label = f"{arena_games} games"
                self._write_metrics(status="training")
                print("[Coach] Running arena evaluation…")
                self._evaluate_and_promote()

            # 4. Write metrics — increment first so count reflects completed iterations
            self._phase = "idle"
            self._sub_done = 0
            self._sub_total = 0
            self._sub_label = ""
            self.iteration = i + 1
            self._write_metrics(status="training")

            # 5. Step LR scheduler
            self.scheduler.step()

            # 6. Periodic checkpoint
            if i > 0 and i % checkpoint_freq == 0:
                self._phase = "checkpoint"
                self._sub_label = f"iter {i}"
                self._write_metrics(status="training")
                self._save_checkpoint(str(i))
                self._log_event(f"Checkpoint saved (iter {i})")
                print(f"[Coach] Checkpoint saved at iteration {i}")

            elapsed = time.time() - iter_start
            self._log_event(f"Iteration {i} complete in {elapsed:.0f}s")
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

        # Restore champion (best_network) if saved; fallback to challenger copy
        if "best_model_state_dict" in checkpoint:
            coach.best_network.load_state_dict(checkpoint["best_model_state_dict"])

        coach.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        coach.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_iteration = checkpoint.get("iteration", 0) + 1
        return coach, start_iteration
