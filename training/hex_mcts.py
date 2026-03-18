"""
hex_mcts.py — Monte Carlo Tree Search for 11×11 Hex using a PPO policy network.

Public API
----------
MCTSNode   — a single node in the search tree
MCTS       — the search algorithm

Network contract
----------------
The MCTS expects a loaded Stable Baselines 3 PPO model whose policy
observation space is Box(-1, 1, (122,), float32) (see hex_state_vector.py).

  Prior P(s,a)  — from model.policy.get_distribution(obs).distribution.probs
  Value V(s)    — from model.policy.predict_values(obs), player-1 perspective

Value perspective convention
-----------------------------
The SB3 model was trained with reward +1 (agent/P1 wins) / -1 (P1 loses).
Values are therefore always in player-1's frame of reference and are negated
internally when the current player at a node is player 2.

Within the tree, node.value stores the *sum* of backed-up values from the
perspective of the player to move at that node.  Backpropagation flips the
sign at each level (zero-sum game).

This gives:
  Q(s, a) from node's player perspective  =  node.value / node.visits
  Q(s, a) from parent's player perspective = -child.value / child.visits

PUCT formula (select step)
--------------------------
  UCB = Q(s,a) + c_puct · P(s,a) · √max(N(s),1) / (1 + N(s,a))

Using max(N_parent, 1) ensures the exploration term is nonzero on the very
first visit, so children are ranked by network priors rather than arbitrarily.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Optional

import numpy as np
import torch

# Allow running this module directly from any working directory.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from game.hex_game import HexGame
from game.hex_actions import get_legal_actions
from game.hex_state_vector import get_state_vector


# ---------------------------------------------------------------------------
# MCTSNode
# ---------------------------------------------------------------------------


class MCTSNode:
    """A single node in the Monte Carlo Search Tree."""

    __slots__ = ("state", "parent", "action", "children", "visits", "value", "prior")

    def __init__(
        self,
        state: HexGame,
        parent: Optional[MCTSNode] = None,
        action: Optional[int] = None,
        prior: float = 0.0,
    ) -> None:
        self.state: HexGame = state
        self.parent: Optional[MCTSNode] = parent
        self.action: Optional[int] = action  # action taken from parent → this node
        self.children: dict[int, MCTSNode] = {}
        self.visits: int = 0
        self.value: float = 0.0  # sum of values from *this* node's player's perspective
        self.prior: float = prior

    @property
    def q(self) -> float:
        """Mean value from this node's player's perspective (0 if unvisited)."""
        return self.value / self.visits if self.visits > 0 else 0.0


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------


class MCTS:
    """
    Monte Carlo Tree Search powered by a PPO policy network.

    Parameters
    ----------
    model : stable_baselines3.PPO
        A fully loaded SB3 PPO model.
    num_simulations : int
        Default number of simulations to run per ``get_action`` call.
    c_puct : float
        Exploration constant in the PUCT formula.  Higher values favour
        exploration; lower values favour exploitation.
    add_noise : bool
        When True, Dirichlet noise (alpha=0.3, mix=0.25) is injected into
        the root node's child priors after expansion.  Set True during
        self-play training to encourage exploration; False for inference.
    """

    def __init__(
        self,
        model,
        num_simulations: int = 200,
        c_puct: float = 1.4,
        add_noise: bool = False,
    ) -> None:
        self._model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.add_noise = add_noise

        # Put policy in eval mode for deterministic and fast inference.
        self._model.policy.set_training_mode(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_action(self, game: HexGame) -> int:
        """
        Run ``num_simulations`` MCTS rollouts and return the most-visited action.

        Parameters
        ----------
        game : HexGame
            Current game position (must not be terminal).

        Returns
        -------
        int
            Row-major cell index (0–120) of the recommended move.
        """
        action, _, _ = self.get_action_with_stats(game)
        return action

    def get_action_with_stats(
        self,
        game: HexGame,
        simulations: Optional[int] = None,
    ) -> tuple[int, dict[int, int], float]:
        """
        Run MCTS and return the best action alongside visit-count statistics.

        Parameters
        ----------
        game : HexGame
            Current game position (must not be terminal).
        simulations : int | None
            Number of simulations to run.  Defaults to ``self.num_simulations``.

        Returns
        -------
        action : int
            Most-visited root child action.
        visits : dict[int, int]
            Mapping of action → visit count for all root children.
        time_ms : float
            Wall-clock time spent in simulations (milliseconds).
        """
        if game.is_terminal():
            raise ValueError("Cannot run MCTS on a terminal game state.")

        n_sims = simulations if simulations is not None else self.num_simulations
        root = MCTSNode(state=game)
        self._expand(root)

        # Dirichlet noise on root priors (training only) for exploration.
        if self.add_noise and root.children:
            actions = list(root.children.keys())
            noise = np.random.dirichlet([0.3] * len(actions))
            for a, eta in zip(actions, noise):
                child = root.children[a]
                child.prior = 0.75 * child.prior + 0.25 * eta

        t0 = time.perf_counter()

        for _ in range(n_sims):
            leaf = self._select(root)

            if leaf.state.is_terminal():
                value = self._terminal_value(leaf)
            else:
                self._expand(leaf)
                value = self._simulate(leaf)

            self._backpropagate(leaf, value)

        elapsed_ms = (time.perf_counter() - t0) * 1_000.0

        visits = {a: child.visits for a, child in root.children.items()}
        best_action = max(root.children, key=lambda a: root.children[a].visits)
        return best_action, visits, elapsed_ms

    # ------------------------------------------------------------------
    # Tree operations
    # ------------------------------------------------------------------

    def _select(self, node: MCTSNode) -> MCTSNode:
        """
        Descend the tree using the PUCT formula until reaching a leaf.

        A leaf is a node with no children (not yet expanded) or a terminal state.

        PUCT:
            UCB = Q(s,a) + c_puct · P(s,a) · √max(N(s),1) / (1 + N(s,a))

        Q is negated relative to the child because child.value is in the
        child's player's frame, which is the opposite of the selecting node's
        player frame (zero-sum).
        """
        while node.children and not node.state.is_terminal():
            sqrt_n = math.sqrt(max(node.visits, 1))
            best_ucb = -float("inf")
            best_child: Optional[MCTSNode] = None

            for child in node.children.values():
                q = -child.q  # from node's player's perspective
                u = self.c_puct * child.prior * sqrt_n / (1 + child.visits)
                ucb = q + u
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_child = child

            node = best_child  # type: ignore[assignment]

        return node

    def _expand(self, node: MCTSNode) -> None:
        """
        Create one child node per legal action, seeded with network priors.

        If the node is terminal or already expanded this is a no-op.
        Priors are the policy action-probability vector masked to legal moves
        and then renormalized.
        """
        if node.state.is_terminal() or node.children:
            return

        legal = get_legal_actions(node.state)
        if not legal:
            return

        priors, _value = self._get_policy_outputs(node.state)

        # Mask to legal actions and renormalize.
        legal_priors = np.array([priors[a] for a in legal], dtype=np.float64)
        total = legal_priors.sum()
        if total > 1e-8:
            legal_priors /= total
        else:
            legal_priors = np.ones(len(legal), dtype=np.float64) / len(legal)

        for a, p in zip(legal, legal_priors):
            node.children[a] = MCTSNode(
                state=node.state.apply_action(a),
                parent=node,
                action=a,
                prior=float(p),
            )

    def _simulate(self, node: MCTSNode) -> float:
        """
        Evaluate the position using the value head of the network.

        Returns the value from the *current player's* perspective at ``node``.
        The network returns V(s) from player-1's perspective; this is negated
        when it is player 2's turn.
        """
        _priors, value_p1 = self._get_policy_outputs(node.state)
        current = node.state.get_current_player()
        return value_p1 if current == 1 else -value_p1

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """
        Walk from ``node`` to the root, updating visit counts and value sums.

        The sign of ``value`` is flipped at each step because parent and child
        belong to opposite players in a zero-sum game.
        """
        while node is not None:
            node.visits += 1
            node.value += value
            value = -value
            node = node.parent  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _terminal_value(node: MCTSNode) -> float:
        """
        Return the game-over value from the current player's perspective.

        In Hex, the player who cannot move has just lost (the opponent made
        the last move and won).  ``current_player`` at a terminal node is the
        one whose turn *would* be next — i.e. the loser.
        """
        winner = node.state.get_winner()
        current = node.state.get_current_player()
        # If the opponent (3 - current) won, this is -1 from current's view.
        return -1.0 if winner == 3 - current else 1.0

    def _get_policy_outputs(self, game: HexGame) -> tuple[np.ndarray, float]:
        """
        Run a single forward pass through the PPO policy.

        Returns
        -------
        priors : np.ndarray, shape (121,)
            Softmax action-probability vector over all 121 cells.
        value_p1 : float
            State value from **player-1's** perspective (trained reward signal).
        """
        obs_np = np.array(get_state_vector(game), dtype=np.float32)[np.newaxis, :]
        obs_tensor, _ = self._model.policy.obs_to_tensor(obs_np)

        with torch.no_grad():
            dist = self._model.policy.get_distribution(obs_tensor)
            priors = dist.distribution.probs[0].cpu().numpy()  # (121,)
            value_p1 = self._model.policy.predict_values(obs_tensor).squeeze().item()

        return priors, value_p1
