"""
Phase 6 — ONNX export is valid and Unity contract is intact.

Prerequisites:
    pip install onnx
"""

import sys

sys.path.insert(0, ".")

from training.policy_value_network import PolicyValueNetwork
from training.export import export_onnx

try:
    import onnx
except ImportError:
    print("ERROR: run 'pip install onnx' first")
    sys.exit(1)

import os

os.makedirs("training/models", exist_ok=True)

net = PolicyValueNetwork()
net.eval()

out_path = "training/models/hex_az_test.onnx"
export_onnx(net, out_path)

model = onnx.load(out_path)
onnx.checker.check_model(model)

inputs = [i.name for i in model.graph.input]
outputs = [o.name for o in model.graph.output]
assert "obs_0" in inputs, f"Wrong input name: {inputs}"
assert "action_probs" in outputs, f"Wrong output name: {outputs}"
print("Phase 6 PASS — ONNX valid, contract correct")
print("Next: drop training/models/hex_az_test.onnx into Unity and run the scene.")
