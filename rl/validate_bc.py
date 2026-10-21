"""Two bounded, independent shaped PPO runs from the same BC init.

uv run python -m rl.validate_bc --bc-init ckpts/europa_bc/actor_init.msgpack
"""

import argparse
import json
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from rl.ppo import PPOConfig
from train import lander_bindings, run_experiment


def validate(bc_init, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "training_seed": 1,
        "runs": [],
    }
    for steps in (300_000, 500_000):
        config = PPOConfig(
            num_envs=16,
            num_steps=125,
            total_steps=steps,
            initial_action_mean=(-0.5, 0.0),
            initial_log_std=-1.5,
            reward_scale=0.01,
        )
        bindings = lander_bindings(
            OmegaConf.create(
                {
                    "reward": {"mode": "potential_profile", "potential_scale": 100.0},
                    "ppo": {"gamma": config.gamma},
                    "evaluation": {"seed_start": 2000, "episodes": 100},
                }
            )
        )
        print(f"Starting BC + shaped PPO, {steps} steps", flush=True)
        _, manifest = run_experiment(
            bindings,
            config,
            seed=1,
            bc_init=bc_init,
            output_root=output_dir / "checkpoints",
        )
        report["runs"].append(
            {
                "label": f"bc_shaping_{steps}",
                "run_id": manifest["run"]["id"],
                "run_dir": str(Path(manifest["run"]["checkpoint_path"]).parent),
                "training_steps": manifest["actual_steps"],
                "config": manifest["config"],
                "evaluation": manifest["evaluation"],
            }
        )
        (output_dir / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bc-init", required=True)
    parser.add_argument("--output-dir", default="ckpts/europa_bc_validation")
    args = parser.parse_args()
    validate(args.bc_init, args.output_dir)
