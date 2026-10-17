"""Bounded RK4 validation: PD, then four fresh PPO runs; never a full run.

Run: uv run python -m rl.validate_europa
"""

import argparse
import json
import subprocess
from pathlib import Path

import jax
from omegaconf import OmegaConf

from rl.fallback import pd_action
from rl.ppo import PPOConfig
from train import evaluate_smoke, lander_bindings, run_experiment


def validate(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "results.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    seeds = tuple(range(2000, 2100))
    report = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "training_seed": 1,
        "evaluation_seeds": list(seeds),
        "runs": [],
    }

    def record(label, evaluation, **metadata):
        row = {"label": label, **metadata, "evaluation": evaluation}
        report["runs"].append(row)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "label": label,
                    **{
                        key: evaluation[key]
                        for key in ("soft_landings", "crashed", "timeouts")
                    },
                }
            ),
            flush=True,
        )

    bindings = lander_bindings(OmegaConf.create({}))
    pd = evaluate_smoke(
        bindings.env,
        bindings.body_params,
        bindings.vehicle_params,
        None,
        None,
        seeds=seeds,
        policy_fn=lambda obs: pd_action(
            obs, bindings.body_params, bindings.vehicle_params
        ),
        policy_label="pd",
    )
    record("pd", pd, training_steps=0)
    for mode in ("baseline", "potential_profile"):
        for steps in (300_000, 500_000):
            config = PPOConfig(
                num_envs=16,
                num_steps=125,
                total_steps=steps,
                initial_action_mean=(-0.5, 0.0),
                initial_log_std=-1.0,
                reward_scale=0.01,
            )
            bindings = lander_bindings(
                OmegaConf.create(
                    {
                        "reward": {"mode": mode, "potential_scale": 100.0},
                        "ppo": {"gamma": config.gamma},
                        "evaluation": {"seed_start": 2000, "episodes": 100},
                    }
                )
            )
            print(f"Starting {mode}, {steps} steps", flush=True)
            _, manifest = run_experiment(
                bindings,
                config,
                seed=1,
                output_root=output_dir / "checkpoints",
            )
            record(
                f"{mode}_{steps}",
                manifest["evaluation"],
                training_steps=manifest["actual_steps"],
                run_id=manifest["run"]["id"],
                run_dir=str(Path(manifest["run"]["checkpoint_path"]).parent),
                config=manifest["config"],
            )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="ckpts/europa_rk4_validation")
    validate(parser.parse_args().output_dir)
