"""Phase 1 — Network forward pass."""

import sys

sys.path.insert(0, ".")

from training.policy_value_network import PolicyValueNetwork
import torch

net = PolicyValueNetwork()
dummy = torch.zeros(4, 122)  # batch of 4
pi, v = net(dummy)

assert pi.shape == (4, 121), f"Policy shape wrong: {pi.shape}"
assert v.shape == (4, 1), f"Value shape wrong: {v.shape}"
assert v.min() >= -1.0 and v.max() <= 1.0, "Value out of tanh range"
print("Phase 1 PASS")
