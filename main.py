#!/usr/bin/env python3
import argparse
from train import run_training


def main():
    parser = argparse.ArgumentParser(description="Snake RL Training")
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--grid", type=int, default=20)
    parser.add_argument("--export", type=str, default="web/models/weights.json")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--ent", type=float, default=0.01)

    args = parser.parse_args()

    run_training(
        total_timesteps=args.steps,
        grid_size=args.grid,
        export_path=args.export,
        learning_rate=args.lr,
        ent_coef=args.ent
    )


if __name__ == "__main__":
    main()