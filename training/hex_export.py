"""
hex_export.py — Export a trained SB3 PPO model to ONNX for Unity Sentis.

Usage
-----
    python training/hex_export.py --level easy

The script loads training/models/hex_{level}.zip and exports the
actor network to training/models/hex_{level}.onnx.

ONNX contract (consumed by Unity Sentis / Inference Engine)
-----------------------------------------------------------
  Input  name : obs_0        shape : [1, 122]   dtype: float32
  Output name : action_probs shape : [1, 121]   dtype: float32
  opset_version: 17
  dynamo: False
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from stable_baselines3 import PPO


def export(level: str) -> None:
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    zip_path = os.path.join(models_dir, f"hex_{level}.zip")
    onnx_path = os.path.join(models_dir, f"hex_{level}.onnx")

    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Model not found: {zip_path}\n"
            f"Run: python training/hex_train.py --level {level}"
        )

    print(f"[hex_export] Loading {zip_path} …")
    model = PPO.load(zip_path)

    # Extract and freeze the actor network only.
    policy = model.policy
    policy.eval()

    # Dummy observation: batch size 1, 122 features.
    dummy_obs = torch.zeros(1, 122, dtype=torch.float32)

    # Minimal wrapper: observation → action logits via actor path only.
    class _ActorWrapper(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self._pi_features = policy.pi_features_extractor
            self._mlp_actor = policy.mlp_extractor.policy_net
            self._action_net = policy.action_net

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            features = self._pi_features(obs)
            latent_pi = self._mlp_actor(features)
            return self._action_net(latent_pi)  # [batch, 121]

    wrapper = _ActorWrapper(policy)
    wrapper.eval()

    print(f"[hex_export] Exporting to {onnx_path} (opset 17, dynamo=False) …")
    torch.onnx.export(
        wrapper,
        dummy_obs,
        onnx_path,
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

    print(f"[hex_export] Done → {onnx_path}")

    # Validate, force all weights inline, and print graph output names.
    try:
        import onnx

        model_proto = onnx.load(onnx_path)
        onnx.checker.check_model(model_proto)
        print("[hex_export] ONNX model validated ✓")

        # Rewrite as a single self-contained file with no external data sidecar.
        onnx.save(model_proto, onnx_path, save_as_external_data=False)
        print("[hex_export] Weights saved inline (no external data file)")

        output_names = [o.name for o in model_proto.graph.output]
        print(f"[hex_export] Graph outputs: {output_names}")
    except ImportError:
        print("[hex_export] onnx not installed — skipping validation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a trained Hex model to ONNX.")
    parser.add_argument(
        "--level",
        choices=["easy", "medium", "hard", "expert"],
        required=True,
        help="Difficulty level to export.",
    )
    args = parser.parse_args()
    export(args.level)
