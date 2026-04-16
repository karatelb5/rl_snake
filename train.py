import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from snake_env import SnakeEnv
from exporter import export_to_json


def run_training(total_timesteps: int = 500_000, grid_size: int = 20,
                 export_path: str = "docs/models/weights.json",
                 learning_rate: float = 3e-4, ent_coef: float = 0.01):
    
    env = DummyVecEnv([lambda: SnakeEnv(grid_size=grid_size)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[128, 128])
    )

    os.makedirs("models", exist_ok=True)

    eval_callback = EvalCallback(
        env,
        best_model_save_path="models/best_snake",
        log_path="logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path="models/checkpoints/",
        name_prefix="snake"
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    model.save("models/snake_final")

    if export_path:
        export_to_json(model, export_path, input_dim=16)

    return model
