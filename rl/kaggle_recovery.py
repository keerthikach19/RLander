"""Bounded, diagnostic-first PPO recovery. Training is Kaggle-only.

Seed 1: reward-only 960k (128x125), then, ONLY if validation <=10%,
an independent 960k run at 16x125 with the identical revised reward.
The second arm changes update frequency, not the initial-state distribution.
No BC, threshold relaxation, checkpoint selection, or full-run waiver.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ARMS = {"reward_only": 128, "more_updates": 16}
STEPS = 960_000
SNAPSHOTS = (240_000, 480_000, 720_000)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def bindings():
    from omegaconf import OmegaConf

    from train import lander_bindings
    return lander_bindings(OmegaConf.create({
        "reward": {"mode": "landing_profile"},
        "evaluation": {"seed_start": 2000, "episodes": 100},
    }))


def register_evaluation(output, *, checkpoint=None, noise=1.0, seeds=range(10000, 10100), batched=False):
    import jax

    from eval.run_eval import load_body_config, load_vehicle_config, run_evaluation
    from storage.db import init_db, insert_run, upsert_body, upsert_vehicle
    conn = init_db(output / "evaluations.sqlite")
    try:
        upsert_body(conn, "europa", load_body_config("europa"))
        upsert_vehicle(conn, "europa_lander", load_vehicle_config("europa_lander"))
        run_id = insert_run(
            conn, train_bodies=["europa"], vehicle="europa_lander",
            policy_type="ppo" if checkpoint else "pd_fallback", seed=1,
            jax_version=jax.__version__, checkpoint_path=checkpoint,
        )
        return run_evaluation(
            run_id, conn=conn, seeds=seeds, batched=batched,
            sensor_noise_scale=noise, traj_base_dir=str(output / "trajectories"),
        )
    finally:
        conn.close()


def preflight(output):
    import jax.numpy as jnp

    from rl.fallback import pd_action
    from train import evaluate_smoke
    bind = bindings()
    # Same reporting path and frozen seeds as the failed Kaggle experiment.
    pd = register_evaluation(output)
    report = {"pd_w3": pd, "training_allowed": False}
    write_json(output / "preflight.json", report)
    if pd["success_rate"] < 0.8:
        report["pd_noise_off_diagnostic"] = register_evaluation(output, noise=0.0)
        report["decision"] = "PD below 80%; inspect environment/controller before PPO"
        write_json(output / "preflight.json", report)
        return
    policies = {
        "pd": lambda obs: pd_action(obs, bind.body_params, bind.vehicle_params),
        "zero_thrust": lambda obs: jnp.array([-1.0, 0.0]),
        "nominal_hover": lambda obs: jnp.array([2.0 / 3.0 - 1.0, 0.0]),
    }
    report["reward_checks"] = {}
    for label, policy in policies.items():
        result = evaluate_smoke(
            bind.env, bind.body_params, bind.vehicle_params, None, None,
            seeds=tuple(range(2000, 2100)), policy_fn=policy, policy_label=label,
        )
        components = result["reward_components_mean"]
        result["mean_return"] = sum(components.values())
        result["dense_cost_over_100"] = abs(sum(components[k] for k in bind.shaping_components)) / 100
        report["reward_checks"][label] = result
    ratio = report["reward_checks"]["pd"]["dense_cost_over_100"]
    report["pd_dense_cost_in_target"] = 0.2 <= ratio <= 1.0
    # A safety check of the new objective, not evidence that PPO will learn it.
    report["training_allowed"] = report["pd_dense_cost_in_target"]
    report["decision"] = (
        "Proceed to bounded reward pilot" if report["training_allowed"]
        else "PD dense/100 ratio outside 0.2--1; review reward before spending training budget"
    )
    write_json(output / "preflight.json", report)
    print(report["decision"], "PD dense/100 =", ratio, flush=True)


def train_arm(output, arm):
    import jax
    import numpy as np
    import orbax.checkpoint as ocp

    from eval.run_eval import make_policy_fn
    from rl.ppo import PPOConfig
    from train import run_experiment
    bind = bindings()
    config = PPOConfig(
        num_envs=ARMS[arm], num_steps=125, total_steps=STEPS,
        initial_action_mean=(-0.5, 0.0), initial_log_std=-1.0, reward_scale=0.01,
    )
    result, manifest = run_experiment(
        bind, config, seed=1, output_root=output / arm,
        checkpoint_steps=SNAPSHOTS,
    )
    run_dir = Path(manifest["run"]["checkpoint_path"]).parent
    # Test actual saved normalization and W3 inference against the in-memory
    # policy on the same observations, before reading frozen PPO outcomes.
    checkpoint = manifest["run"]["checkpoint_path"]
    with ocp.PyTreeCheckpointer() as reader:
        restored = reader.restore(checkpoint)
    from rl.networks import policy_action
    for name in ("mean", "var", "count"):
        np.testing.assert_array_equal(
            restored["obs_stats"][name], getattr(result.runner.obs_stats, name)
        )
    policy = make_policy_fn("ppo", checkpoint_path=checkpoint)
    observations = result.runner.obs
    live = policy_action(result.runner.train_state.params, result.runner.obs_stats, observations)
    replay = policy(observations, jax.random.key(0))
    error = float(np.max(np.abs(np.asarray(live) - np.asarray(replay))))
    if not np.isfinite(error) or error > 1e-5:
        raise RuntimeError(f"Saved policy mismatch: {error}")
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    updates = [r for r in rows if r["event"] == "update"]
    fields = ("env_steps", "episodes", "episode_return", "soft_landing_rate",
              "landed", "crashed", "timeouts", "episode_length", "entropy",
              "approx_kl", "clip_fraction", "value_loss", "shaping_to_terminal_ratio")
    with (output / f"{arm}_training.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(updates)
    validation = []
    for step, path in [*manifest["snapshots"].items(), (str(STEPS), str(run_dir))]:
        saved = json.loads((Path(path) / "manifest.json").read_text())
        validation.append({"steps": int(step), "evaluation": saved["evaluation"]})
    summary = {
        "arm": arm, "run_dir": str(run_dir), "checkpoint": checkpoint,
        "actual_steps": manifest["actual_steps"], "num_updates": config.num_updates,
        "normalization_restored_exactly": True, "max_action_reload_error": error,
        "validation": validation,
        "final_validation_rate": manifest["evaluation"]["soft_landing_rate"],
        "last_training_update": updates[-1],
    }
    write_json(output / f"{arm}.json", summary)
    print(arm, "validation:", [(v["steps"], v["evaluation"]["soft_landings"]) for v in validation], flush=True)


def plan():
    return {
        "preflight": "PD W3 10000..10099; if <80%, noise-off diagnostic and stop",
        "reward_sanity": "PD/zero-thrust/nominal-hover on 2000..2099; PD dense/100 target .2--1",
        "arms": [{"name": name, "steps": STEPS, "num_envs": n,
                  "num_steps": 125, "updates": STEPS // (n * 125)} for name, n in ARMS.items()],
        "fallback_trigger": "reward_only FINAL validation success <=10%",
        "snapshots": list(SNAPSHOTS), "training_seed": 1,
        "evaluation": "Validation 2000..2099; final checkpoints only on 10000..10099 after arm decisions",
        "full_run": False, "bc": False, "success_thresholds": {"vy": 1.0, "vx": 0.5},
    }


def launch(output, dry_run=False):
    if dry_run:
        print(json.dumps(plan(), indent=2))
        return
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Training launcher is restricted to Kaggle; use --dry-run locally")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "JAX_PLATFORMS": "cuda",
                   "XLA_PYTHON_CLIENT_PREALLOCATE": "false"}
    report = {"protocol": plan(), "status": "preflight", "runs": [], "frozen": []}
    write_json(output / "results.json", report)
    for name, command in (("packages.txt", [sys.executable, "-m", "pip", "freeze"]),
                          ("hardware.txt", ["nvidia-smi"])):
        (output / name).write_text(subprocess.check_output(command, text=True))
    if Path("source_manifest.json").exists():
        (output / "source_manifest.json").write_bytes(Path("source_manifest.json").read_bytes())

    def worker(stage):
        print("Starting", stage, "on GPU 0; log:", output / f"{stage}.log", flush=True)
        with (output / f"{stage}.log").open("w") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", "-m", "rl.kaggle_recovery", "--worker", stage,
                 "--output", str(output)], env=environment, stdout=log, stderr=subprocess.STDOUT,
            )
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(stage, "still running; progress/diagnostics saved to its log", flush=True)
            if code:
                tail = (output / f"{stage}.log").read_text(errors="replace")[-6000:]
                raise RuntimeError(f"{stage} failed (exit {code}):\n{tail}")

    try:
        worker("preflight")
        report["preflight"] = json.loads((output / "preflight.json").read_text())
        if not report["preflight"]["training_allowed"]:
            report["status"] = "diagnostic_stop"
            return
        report["status"] = "training"
        write_json(output / "results.json", report)
        worker("reward_only")
        report["runs"].append(json.loads((output / "reward_only.json").read_text()))
        write_json(output / "results.json", report)
        if report["runs"][0]["final_validation_rate"] <= 0.1:
            worker("more_updates")
            report["runs"].append(json.loads((output / "more_updates.json").read_text()))
        report["status"] = "evaluating"
        write_json(output / "results.json", report)
        # No model/arm decision is made from the historically reused frozen set.
        worker("evaluate")
        report["frozen"] = json.loads((output / "frozen.json").read_text())
        report["status"] = "complete"
        report["next_step"] = (
            "Review curves, then replicate promising arm across seeds 1--3"
            if any(r["final_validation_rate"] > 0.1 for r in report["runs"])
            else "Review trajectories; next iteration: near-ground curriculum with nominal final evaluation, or labeled BC+PPO"
        )
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(output / "results.json", report)
        print("Status:", report["status"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/kaggle/working/europa_ppo_recovery_v1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", choices=("preflight", *ARMS, "evaluate"))
    args = parser.parse_args()
    if args.worker:
        if not Path("/kaggle/working").is_dir():
            raise RuntimeError("Workers are restricted to Kaggle")
        output = Path(args.output).resolve()
        if args.worker == "preflight":
            preflight(output)
        elif args.worker in ARMS:
            train_arm(output, args.worker)
        else:
            report = json.loads((output / "results.json").read_text())
            results = []
            for run in report["runs"]:
                result = register_evaluation(output, checkpoint=run["checkpoint"])
                result["arm"] = run["arm"]
                result["reward_reporting"] = "W3 baseline totals; training used landing_profile v1"
                results.append(result)
                write_json(output / "frozen.json", results)
    else:
        launch(args.output, args.dry_run)


if __name__ == "__main__":
    main()
