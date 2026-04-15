import json
import os
import torch
from stable_baselines3 import PPO


def export_to_json(model: PPO, output_path: str, input_dim: int = 16):
    policy = model.policy
    weights = {}

    for layer in policy.mlp_extractor.policy_net.children():
        if isinstance(layer, torch.nn.Linear):
            idx = len(weights)
            weights[str(idx)] = {
                "weight": layer.weight.detach().cpu().numpy().tolist(),
                "bias": layer.bias.detach().cpu().tolist()
            }

    if isinstance(policy.action_net, torch.nn.Linear):
        idx = len(weights)
        weights[str(idx)] = {
            "weight": policy.action_net.weight.detach().cpu().numpy().tolist(),
            "bias": policy.action_net.bias.detach().cpu().tolist()
        }

    normalization = None
    vec_env = model.get_env()
    if hasattr(vec_env, "obs_rms") and vec_env.obs_rms is not None:
        normalization = {
            "mean": vec_env.obs_rms.mean.tolist(),
            "var": vec_env.obs_rms.var.tolist(),
            "epsilon": 1e-8
        }

    export_data = {
        "architecture": {"input_dim": input_dim, "hidden_dims": [128, 128], "output_dim": 3},
        "weights": weights,
        "normalization": normalization,
        "metadata": {"steps": int(model.num_timesteps)}
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)