"""
hex_env.py — Gymnasium environment wrapping the Hex rules engine.

Observation space : Box(-1, 1, shape=(122,), float32)
Action space      : Discrete(121)

Reward
------
  +1.0  agent wins
  -1.0  opponent wins
   0.0  game continues

Self-play opponent
------------------
  Phase 1 (model=None): opponent plays uniformly random moves.
  Phase 2 (set_opponent_model): opponent uses the supplied model's
          predict() to pick moves — call this periodically from the
          training callback once a checkpoint is available.
"""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from game.hex_game import HexGame
from game.hex_actions import get_legal_actions
from game.hex_state_vector import get_state_vector


class HexEnv(gym.Env):
    """Single-agent Gymnasium wrapper for 11×11 Hex with self-play opponent."""

    metadata = {"render_modes": []}

    def __init__(self, agent_player: int = 1) -> None:
        """
        Parameters
        ----------
        agent_player : int
            Which player the learning agent controls (1 or 2).
        """
        super().__init__()
        assert agent_player in (1, 2), "agent_player must be 1 or 2"

        self.agent_player: int = agent_player
        self.opponent_player: int = 3 - agent_player

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(122,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(121)

        self._game: Optional[HexGame] = None
        self._opponent_model = None  # None → random; otherwise SB3 model

    # ── Public helpers ────────────────────────────────────────────────────

    def set_opponent_model(self, model) -> None:
        """
        Swap in a trained SB3 model as the self-play opponent.
        Pass None to revert to random play.
        """
        self._opponent_model = model

    # ── Gymnasium API ─────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._game = HexGame()

        # If the opponent moves first, let it do so before returning control.
        if self._game.get_current_player() == self.opponent_player:
            self._apply_opponent_move()

        obs = self._obs()
        return obs, {}

    def step(self, action: int):
        assert self._game is not None, "reset() must be called before step()"

        legal = get_legal_actions(self._game)

        # Illegal action masking: substitute a random legal action.
        if action not in legal:
            action = random.choice(legal)

        # Agent move
        self._game = self._game.apply_action(action)

        if self._game.is_terminal():
            winner = self._game.get_winner()
            reward = 1.0 if winner == self.agent_player else -1.0
            return self._obs(), reward, True, False, {}

        # Opponent move
        self._apply_opponent_move()

        if self._game.is_terminal():
            winner = self._game.get_winner()
            reward = 1.0 if winner == self.agent_player else -1.0
            return self._obs(), reward, True, False, {}

        return self._obs(), 0.0, False, False, {}

    def render(self) -> None:
        pass  # No rendering needed for training

    # ── Private helpers ───────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        return np.array(get_state_vector(self._game), dtype=np.float32)

    def _apply_opponent_move(self) -> None:
        legal = get_legal_actions(self._game)
        if not legal:
            return

        if self._opponent_model is not None:
            obs = self._obs()
            opp_action, _ = self._opponent_model.predict(obs, deterministic=False)
            opp_action = int(opp_action)
            if opp_action not in legal:
                opp_action = random.choice(legal)
        else:
            opp_action = random.choice(legal)

        self._game = self._game.apply_action(opp_action)
