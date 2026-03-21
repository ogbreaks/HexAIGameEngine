"""Phase 2 — MCTS plays a full game."""

import sys

sys.path.insert(0, ".")

from game.hex_game import HexGame
from game.hex_actions import get_legal_actions
from training.policy_value_network import PolicyValueNetwork
from training.mcts import MCTS

net = PolicyValueNetwork()
net.eval()
mcts = MCTS(net, num_simulations=50)
game = HexGame()

moves = 0
while not game.is_terminal():
    action, visits, time_ms = mcts.get_action_with_stats(game)
    assert action in get_legal_actions(game), f"Illegal action {action}"
    assert isinstance(visits, dict), "visits must be dict"
    assert all(
        isinstance(k, int) and isinstance(v, int) for k, v in visits.items()
    ), "visits keys/values must be int"
    game = game.apply_action(action)
    moves += 1

assert game.get_winner() in (1, 2), "Game must have a winner"
print(f"Phase 2 PASS — {moves} moves, winner: {game.get_winner()}")
