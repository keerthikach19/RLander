"""Predeclared BC+PPO stability audit; no tuning or full training runs.

Three continuous 600k schedules, snapshots at 400k/500k/600k, validation on
2000..2099. Only the predeclared 500k snapshots go to W3 frozen evaluation.
"""

import argparse
import json
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from eval.run_eval import load_body_config, load_vehicle_config, run_evaluation
from rl.ppo import PPOConfig
from storage.db import init_db, insert_run, upsert_body, upsert_vehicle
from train import lander_bindings, run_experiment


def frozen_evaluation(run_dir, output_dir):
    """Register an existing checkpoint and call W3, with no custom policy."""
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    database = output_dir / "w3_frozen.sqlite"
    conn = init_db(database)
    try:
        upsert_body(conn, "europa", load_body_config("europa"))
        upsert_vehicle(conn, "europa_lander", load_vehicle_config("europa_lander"))
        run_id = insert_run(
            conn,
            train_bodies=["europa"],
            vehicle="europa_lander",
            policy_type="ppo",
            seed=manifest["config"]["seed"],
            config_hash=manifest["run"]["config_hash"],
            jax_version=manifest["run"]["jax_version"],
            checkpoint_path=manifest["run"]["checkpoint_path"],
        )
        result = run_evaluation(
            run_id,
            conn=conn,
            seeds=range(10_000, 10_100),
            traj_base_dir=str((output_dir / "w3_trajectories").resolve()),
        )
    finally:
        conn.close()
    result.update(
        training_seed=manifest["config"]["seed"],
        actual_steps=manifest["actual_steps"],
        schedule_total_steps=manifest["config"]["ppo"]["total_steps"],
        source_run_id=manifest["run"]["id"],
        soft_landings=sum(row["outcome"] == 1 for row in result["episodes"]),
        crashed=sum(row["outcome"] == 2 for row in result["episodes"]),
        timeouts=sum(row["outcome"] == 3 for row in result["episodes"]),
        seeds=list(range(10_000, 10_100)),
        reward_reporting="W3 YAML baseline components; trained with profile shaping",
    )
    return result


def audit(bc_init, output_dir):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "protocol": "seeds 1,2,3; continuous 600k schedules; fixed 400k/500k/600k validation; W3 frozen at 500k only",
        "runs": [],
        "frozen": [],
    }

    def write():
        (output_dir / "results.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    write()
    config = PPOConfig(
        num_envs=16,
        num_steps=125,
        total_steps=600_000,
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
    for seed in (1, 2, 3):
        print(f"Starting continuous 600k BC+PPO run, seed {seed}", flush=True)
        _, final = run_experiment(
            bindings,
            config,
            seed=seed,
            bc_init=bc_init,
            checkpoint_steps=(400_000, 500_000),
            output_root=output_dir / "checkpoints",
        )
        dirs = {
            **final["snapshots"],
            "600000": str(Path(final["run"]["checkpoint_path"]).parent),
        }
        for steps in (400_000, 500_000, 600_000):
            run_dir = Path(dirs[str(steps)])
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            report["runs"].append(
                {
                    "label": f"seed_{seed}_{steps}",
                    "seed": seed,
                    "training_steps": steps,
                    "schedule_total_steps": 600_000,
                    "run_dir": str(run_dir),
                    "config": manifest["config"],
                    "evaluation": manifest["evaluation"],
                }
            )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "steps": steps,
                        **{
                            key: manifest["evaluation"][key]
                            for key in ("soft_landings", "crashed", "timeouts")
                        },
                    }
                ),
                flush=True,
            )
            write()
    # All training and validation is finished before any frozen result is read.
    for row in report["runs"]:
        if row["training_steps"] == 500_000:
            report["frozen"].append(frozen_evaluation(row["run_dir"], output_dir))
            write()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bc-init", default="rl/validation/bc_20261018/actor_init.msgpack"
    )
    parser.add_argument("--output-dir", default="ckpts/europa_stability_20261018")
    args = parser.parse_args()
    audit(args.bc_init, args.output_dir)
