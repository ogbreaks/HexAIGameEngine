"""
export.py — Export a trained AlphaZero PolicyValueNetwork to ONNX for Unity Sentis.

ONNX contract (unchanged from PPO export — Unity reads the same file)
----------------------------------------------------------------------
  Input  name : obs_0        shape : [1, 122]   dtype: float32
  Output name : action_probs shape : [1, 121]   dtype: float32
  opset_version: 17
  dynamo: False

Only the policy head is exported. The value head is discarded — Unity
uses action probabilities only, not the value estimate.

Usage
-----
    from training.export import export_onnx
    export_onnx(network, "training/models/hex_az_best.onnx")

    # Or from the CLI:
    python -m training.export --model training/models/hex_az_best.pth \\
                               --output training/models/hex_az_best.onnx
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

from training.policy_value_network import PolicyValueNetwork


class _PolicyWrapper(nn.Module):
    """Thin wrapper that exports only the policy head output."""

    def __init__(self, network: PolicyValueNetwork) -> None:
        super().__init__()
        self.network = network

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        policy_logits, _ = self.network(obs)  # discard value head
        return policy_logits  # [B, 121] raw logits


def export_onnx(network: PolicyValueNetwork, path: str) -> None:
    """
    Export network's policy head to ONNX.

    Parameters
    ----------
    network : PolicyValueNetwork  — trained network (any device)
    path    : str                  — output .onnx file path

    The exported model receives obs_0 [1, 122] and produces action_probs [1, 121].
    Dynamic batch axis is registered so Unity can call it with batch size 1.
    """
    wrapper = _PolicyWrapper(network)
    wrapper.eval()
    wrapper = wrapper.cpu()

    dummy = torch.zeros(1, 122, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy,
        path,
        opset_version=17,
        dynamo=False,
        export_params=True,
        input_names=["obs_0"],
        output_names=["action_probs"],
        dynamic_axes={
            "obs_0": {0: "batch"},
            "action_probs": {0: "batch"},
        },
    )
    print(f"[export] Saved ONNX to {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main() -> None:
    parser = argparse.ArgumentParser(description="Export AZ network to ONNX")
    parser.add_argument(
        "--model",
        default="training/models/hex_az_best.pth",
        help="Path to .pth weights file",
    )
    parser.add_argument(
        "--output",
        default="training/models/hex_az_best.onnx",
        help="Output .onnx file path",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to YAML config (uses network defaults if omitted)",
    )
    args = parser.parse_args()

    config: dict = {}
    if args.config:
        import yaml  # type: ignore[import]

        with open(args.config) as f:
            config = yaml.safe_load(f)

    network = PolicyValueNetwork(config)
    network.load_state_dict(torch.load(args.model, map_location="cpu"))
    network.eval()

    export_onnx(network, args.output)


if __name__ == "__main__":
    _main()
