"""Compare a PPO checkpoint with observation-only control baselines.

Run: uv run python -m rl.diagnostics --run-dir ckpts/<run_id>
Uses only reset/step and the documented observations/terminal info. This does
not modify physics, rewards, training parameters, or any saved checkpoint.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

from env.lander_env import LanderEnv
from rl.fallback import pd_action
from rl.networks import ObsStats
from train import evaluate_smoke


def diagnose(run_dir, *, seeds=tuple(range(10_000, 10_100))):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    # Replay the run's recorded parameter data even if YAML configs later change.
    body, vehicle = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value, dtype=jnp.float32),
        (manifest["config"]["body_params"], manifest["config"]["vehicle_params"]),
    )
    with ocp.PyTreeCheckpointer() as checkpointer:
        saved = checkpointer.restore(str(run_dir / "checkpoint"))
    stats = ObsStats(**saved["obs_stats"])
    policies = {
        "ppo": None,
        "pd": lambda obs: pd_action(obs, body, vehicle),
        "zero_thrust": lambda obs: jnp.array([-1.0, 0.0]),
        "nominal_hover": lambda obs: jnp.array([2.0 / vehicle["twr_max"] - 1, 0.0]),
    }
    report = {"run_id": manifest["run"]["id"], "comparisons": {}}
    for name, policy in policies.items():
        result = evaluate_smoke(
            LanderEnv(),
            body,
            vehicle,
            saved["train_state"]["params"],
            stats,
            seeds=seeds,
            policy_fn=policy,
            policy_label=name,
        )
        result["mean_return"] = sum(result["reward_components_mean"].values())
        report["comparisons"][name] = result
        print(
            json.dumps(
                {k: v for k, v in result.items() if k not in ("episodes", "seeds")}
            )
        )
    (run_dir / "diagnostics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    diagnose(args.run_dir)
