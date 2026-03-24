"""
mcts.py — Monte Carlo Tree Search for AlphaZero 11×11 Hex.

Public API
----------
MCTSNode                — a single node in the search tree
MCTS                    — the search algorithm

Network contract
----------------
Expects a PolicyValueNetwork (or any callable) with:
    policy_logits, value = network(obs)
where obs is [B, 122] float32 and:
    policy_logits : [B, 121]  — raw logits (softmax applied here)
    value         : [B, 1]    — float in [-1, +1], current-player perspective

Value perspective
-----------------
All values are from the current player's perspective at each node.
Backpropagation negates the value at each level (zero-sum game).

Terminal nodes
--------------
At a terminal node, HexGame.apply_action() has already flipped _current_player
to the next player AFTER the winning move. Therefore get_current_player() at a
terminal node returns the LOSER. Value from the loser's perspective = -1.0.

Virtual loss (collapsed form)
------------------------------
_apply_virtual_loss  : visits += 1, value_sum -= 1.0 (pessimistic)
_backpropagate       : value_sum += (1.0 + real_value)  — undo -1.0, add real
visits is NOT touched in _backpropagate; net effect per node:
    visits    += 1                (from virtual loss)
    value_sum += real_value       (net after undo+real)

PUCT formula
------------
UCB(s,a) = Q(s,a) + c_puct × P(s,a) × √N(s) / (1 + N(s,a))
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Optional

import numpy as np
import torch

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

    __slots__ = [
        "state",
        "parent",
        "action",
        "children",
        "visits",
        "value_sum",
        "prior",
        "is_expanded",
    ]

    def __init__(
        self,
        state: HexGame,
        parent: Optional[MCTSNode] = None,
        action: Optional[int] = None,
        prior: float = 0.0,
    ) -> None:
        self.state: HexGame = state
        self.parent: Optional[MCTSNode] = parent
        self.action: Optional[int] = action
        self.children: dict[int, MCTSNode] = {}
        self.visits: int = 0
        self.value_sum: float = 0.0
        self.prior: float = prior
        self.is_expanded: bool = False

    @property
    def q(self) -> float:
        """Mean value from this node's player's perspective (0 if unvisited)."""
        return self.value_sum / self.visits if self.visits > 0 else 0.0


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------


class MCTS:
    """
    AlphaZero MCTS using a joint policy+value network.

    Parameters
    ----------
    network         : PolicyValueNetwork (or compatible callable)
    num_simulations : int    — simulations per get_policy() call
    c_puct          : float  — exploration constant (PUCT formula)
    dirichlet_alpha : float  — Dirichlet noise concentration (root, self-play)
    dirichlet_weight: float  — weight of noise vs prior at root
    """

    def __init__(
        self,
        network,
        num_simulations: int = 800,
        c_puct: float = 1.4,
        dirichlet_alpha: float = 0.3,
        dirichlet_weight: float = 0.25,
        inference_client=None,
        virtual_loss_k: int = 1,
    ) -> None:
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight
        self._root: Optional[MCTSNode] = None
        self._inference_client = inference_client
        self.virtual_loss_k = virtual_loss_k

    # ── Network evaluation ──────────────────────────────────────────────────

    def _evaluate(self, game: HexGame) -> tuple[np.ndarray, float]:
        """Query the network (or inference server); return (prior_probs [121], value scalar)."""
        state_vec = get_state_vector(game)

        if self._inference_client is not None:
            # Use GPU inference server — batched on the server side
            priors, v = self._inference_client.evaluate(state_vec)
            return priors, v

        with torch.no_grad():
            obs = torch.tensor([state_vec], dtype=torch.float32)
            policy_logits, value_t = self.network(obs)
            priors = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
            v = float(value_t.squeeze().cpu().item())
        return priors, v

    def _evaluate_batch(self, games: list[HexGame]) -> list[tuple[np.ndarray, float]]:
        """Evaluate multiple positions in one batched call."""
        state_vecs = [get_state_vector(g) for g in games]

        if self._inference_client is not None:
            return self._inference_client.evaluate_batch(state_vecs)

        with torch.no_grad():
            obs = torch.tensor(state_vecs, dtype=torch.float32)
            policy_logits, value_t = self.network(obs)
            priors = torch.softmax(policy_logits, dim=-1).cpu().numpy()
            values = value_t.squeeze(-1).cpu().numpy()
        return [(priors[i], float(values[i])) for i in range(len(games))]

    # ── Tree operations ─────────────────────────────────────────────────────

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Descend using PUCT until an unexpanded or terminal node."""
        while node.is_expanded and not node.state.is_terminal():
            node = self._best_child(node)
        return node

    def _best_child(self, node: MCTSNode) -> MCTSNode:
        """Return the child with the highest PUCT score."""
        sqrt_n = math.sqrt(max(node.visits, 1))
        best_score = -float("inf")
        best_child: Optional[MCTSNode] = None
        for child in node.children.values():
            score = child.q + self.c_puct * child.prior * sqrt_n / (1 + child.visits)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child  # type: ignore[return-value]

    def _expand(self, node: MCTSNode, priors: np.ndarray) -> None:
        """Create child nodes for all legal actions; renormalise priors over legal only."""
        legal = get_legal_actions(node.state)
        legal_probs = priors[legal]
        s = legal_probs.sum()
        if s > 0:
            legal_probs = legal_probs / s
        else:
            legal_probs = np.ones(len(legal), dtype=np.float32) / len(legal)

        for action, prob in zip(legal, legal_probs):
            node.children[action] = MCTSNode(
                state=node.state.apply_action(action),
                parent=node,
                action=action,
                prior=float(prob),
            )
        node.is_expanded = True

    def _apply_virtual_loss(self, node: MCTSNode) -> None:
        """Walk from node to root applying virtual loss before descent.

        visits += 1  — virtual visit; will NOT be undone in backprop
        value_sum -= 1.0  — pessimistic; will be corrected in backprop
        """
        current: Optional[MCTSNode] = node
        while current is not None:
            current.visits += 1
            current.value_sum -= 1.0
            current = current.parent

    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Walk from node to root, correcting virtual loss and applying real value.

        visits is intentionally NOT touched here — virtual loss already incremented
        it in _apply_virtual_loss and that increment represents the real visit.
        Only value_sum needs correction: undo the -1.0 virtual penalty, add real value.

        Net result per node:
            visits:    +1            (from virtual loss)
            value_sum: +real_value   (after -1.0 virtual + 1.0 undo + real_value)
        """
        current: Optional[MCTSNode] = node
        while current is not None:
            current.value_sum += 1.0 + value  # undo virtual -1.0, add real value
            value = -value  # flip perspective at each level
            current = current.parent

    def _add_dirichlet_noise(self, root: MCTSNode) -> None:
        """Add Dirichlet noise to root children priors (self-play only)."""
        actions = list(root.children.keys())
        if not actions:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        w = self.dirichlet_weight
        for i, action in enumerate(actions):
            child = root.children[action]
            child.prior = (1.0 - w) * child.prior + w * noise[i]

    # ── Core simulation loop ────────────────────────────────────────────────

    def _simulate(
        self,
        game: HexGame,
        num_sims: int,
        dirichlet: bool = False,
    ) -> MCTSNode:
        """Run num_sims simulations from game; return the root node.

        When virtual_loss_k > 1, selects K leaves per batch using virtual loss
        to force diverse paths, evaluates them in one GPU call, then expands
        and backpropagates all K together.  This dramatically improves GPU
        utilisation.
        """
        root = MCTSNode(state=game)
        priors, _ = self._evaluate(game)
        self._expand(root, priors)

        if dirichlet and root.children:
            self._add_dirichlet_noise(root)

        k = self.virtual_loss_k
        sim = 0
        while sim < num_sims:
            batch_k = min(k, num_sims - sim)

            # Phase 1: select + virtual-loss K leaves
            leaves: list[MCTSNode] = []
            terminal_leaves: list[MCTSNode] = []
            eval_leaves: list[MCTSNode] = []
            for _ in range(batch_k):
                leaf = self._select(root)
                self._apply_virtual_loss(leaf)
                leaves.append(leaf)
                if leaf.state.is_terminal():
                    terminal_leaves.append(leaf)
                else:
                    eval_leaves.append(leaf)

            # Phase 2: handle terminals immediately
            for leaf in terminal_leaves:
                self._backpropagate(leaf, -1.0)

            # Phase 3: batch-evaluate non-terminal leaves
            if eval_leaves:
                results = self._evaluate_batch([l.state for l in eval_leaves])
                for leaf, (p, v) in zip(eval_leaves, results):
                    if not leaf.is_expanded:
                        self._expand(leaf, p)
                    self._backpropagate(leaf, v)

            sim += batch_k

        return root

    # ── Public API ──────────────────────────────────────────────────────────

    def get_policy(
        self,
        game: HexGame,
        temperature: float = 1.0,
        is_self_play: bool = False,
    ) -> np.ndarray:
        """
        Run simulations and return visit-count distribution as policy vector [121].

        temperature = 1.0   → proportional to visit counts  (self-play, exploration)
        temperature → 0     → one-hot on argmax              (evaluation, greedy)
        """
        root = self._simulate(game, self.num_simulations, dirichlet=is_self_play)
        self._root = root

        visits = np.zeros(121, dtype=np.float64)
        for a, child in root.children.items():
            visits[a] = child.visits

        if temperature < 0.01:
            # Greedy: one-hot on argmax
            policy = np.zeros(121, dtype=np.float32)
            policy[int(np.argmax(visits))] = 1.0
            return policy

        # Proportional with temperature
        visits = visits ** (1.0 / temperature)
        s = visits.sum()
        if s > 0:
            return (visits / s).astype(np.float32)

        # Fallback: uniform over legal actions
        legal = get_legal_actions(game)
        policy = np.zeros(121, dtype=np.float32)
        policy[legal] = 1.0 / len(legal)
        return policy

    def get_action(self, game: HexGame) -> int:
        """Greedy action: argmax of visit counts (temperature → 0)."""
        policy = self.get_policy(game, temperature=0.0, is_self_play=False)
        return int(np.argmax(policy))

    def get_action_with_stats(
        self,
        game: HexGame,
        simulations: int | None = None,
    ) -> tuple[int, dict[int, int], float]:
        """
        Server-compatible interface matching the old hex_mcts.py signature.

        Returns (action, visits, time_ms).
            action   : int            — chosen action index
            visits   : dict[int,int]  — action → visit count (visited actions only)
            time_ms  : float          — wall-clock time in milliseconds

        Uses a local simulations variable — does NOT mutate self.num_simulations.
        """
        start = time.time()
        num_sims = simulations if simulations is not None else self.num_simulations

        root = self._simulate(game, num_sims, dirichlet=False)
        self._root = root

        # Pick action with most visits
        action = max(root.children, key=lambda a: root.children[a].visits)

        # Visited actions only (server does str(k) conversion)
        visits: dict[int, int] = {
            a: child.visits for a, child in root.children.items() if child.visits > 0
        }

        time_ms = (time.time() - start) * 1000.0
        return action, visits, time_ms
