"""Phase 5 — Arena evaluation runs cleanly."""

import sys

sys.path.insert(0, ".")

from training.policy_value_network import PolicyValueNetwork
from training.arena import Arena
from training.players import MCTSPlayer, RandomPlayer

net = PolicyValueNetwork()
net.eval()

random_player = RandomPlayer()
mcts_player = MCTSPlayer(net, num_simulations=50)

arena = Arena(mcts_player, random_player, num_games=10)
wins, losses, draws = arena.play_games()

assert wins + losses == 10, f"Expected 10 games, got {wins + losses + draws}"
# Untrained network vs random: expect roughly 50/50 — no assertion on win rate
print(f"Phase 5 PASS — MCTS vs Random: {wins}W {losses}L {draws}D")
