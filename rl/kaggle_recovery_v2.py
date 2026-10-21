"""Kaggle-only paired PPO refinement from a validation-selected PPO checkpoint.

Two 320k schedules share a PPO parent and stabilization settings. Only reward
differs. This is PPO fine-tuning, not BC or independent from-scratch replication.
"""

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rl.kaggle_recovery import write_json

ARMS = {"stable_control": "landing_profile", "success_aligned": "landing_success"}
STEPS = 320_000
SNAPSHOTS = tuple(range(40_000, STEPS, 40_000))
FRESH_SEEDS = tuple(range(30_000, 30_200))


def configuration():
    from rl.ppo import PPOConfig
    return PPOConfig(
        num_envs=16, num_steps=125, total_steps=STEPS,
        learning_rate=3e-5, clip_coef=.1, ent_coef=0., reward_scale=.01,
        initial_log_std=math.log(.1), log_std_min=math.log(.05), log_std_max=math.log(.15),
        freeze_obs_stats=True,
    )


def bindings(mode):
    from omegaconf import OmegaConf

    from train import lander_bindings
    return lander_bindings(OmegaConf.create({
        "reward": {"mode": mode}, "evaluation": {"seed_start": 2000, "episodes": 100},
    }))


def choose_best(candidates):
    """Highest validation count, earliest on tie; initial policy is eligible."""
    return max(candidates, key=lambda row: (row["evaluation"]["soft_landings"], -row["additional_steps"]))


def preflight(parent, output):
    import jax.numpy as jnp

    from train import evaluate_smoke, load_ppo_initialization, make_smoke_rollout
    bind = bindings("landing_profile")
    params, stats, metadata = load_ppo_initialization(
        parent, body_name="europa", vehicle_name="europa_lander", initial_log_std=None
    )
    report = {"parent": metadata, "training_allowed": False}
    rollouts = {sampled: make_smoke_rollout(
        bind.env, bind.body_params, bind.vehicle_params, sample_actions=sampled
    ) for sampled in (False, True)}
    for label, sampled, std in (("deterministic", False, None),
                                ("original_sampled", True, None),
                                ("reduced_noise_sampled", True, math.log(.1))):
        weights = params if std is None else {**params, "params": {**params["params"],
            "log_std": jnp.full_like(params["params"]["log_std"], std)}}
        report[label] = evaluate_smoke(
            bind.env, bind.body_params, bind.vehicle_params, weights, stats,
            seeds=tuple(range(2000, 2100)), sample_actions=sampled, policy_label=label,
            _rollout=rollouts[sampled],
        )
        write_json(output / "preflight.json", report)
    original = json.loads((parent.parent / "manifest.json").read_text())
    expected = original["evaluation"]["soft_landings"]
    report["expected_parent_soft_landings"] = expected
    report["training_allowed"] = abs(report["deterministic"]["soft_landings"] - expected) <= 2
    report["decision"] = "proceed" if report["training_allowed"] else "stop: parent replay differs from archived result"
    write_json(output / "preflight.json", report)
    print({label: report[label]["soft_landings"] for label in (
        "deterministic", "original_sampled", "reduced_noise_sampled")}, flush=True)


def train_arm(parent, output, arm, *, config=None, bind=None, seed=1,
              checkpoint_steps=None, initial_evaluation_key="deterministic"):
    import jax
    import numpy as np

    from eval.run_eval import make_policy_fn
    from rl.networks import policy_action
    from train import run_experiment
    config = configuration() if config is None else config
    bind = bindings(ARMS[arm]) if bind is None else bind
    checkpoint_steps = SNAPSHOTS if checkpoint_steps is None else checkpoint_steps
    result, manifest = run_experiment(
        bind, config, seed=seed, ppo_init=parent, output_root=output / arm,
        checkpoint_steps=checkpoint_steps,
    )
    run_dir = Path(manifest["run"]["checkpoint_path"]).parent
    policy = make_policy_fn("ppo", checkpoint_path=str(run_dir / "checkpoint"))
    live = policy_action(result.runner.train_state.params, result.runner.obs_stats, result.runner.obs)
    replay = policy(result.runner.obs, jax.random.key(0))
    reload_error = float(np.max(np.abs(np.asarray(live) - np.asarray(replay))))
    if not np.isfinite(reload_error) or reload_error > 1e-5:
        raise RuntimeError(f"Checkpoint reload action mismatch: {reload_error}")
    initial = json.loads((output / "preflight.json").read_text())[initial_evaluation_key]
    candidates = [{"additional_steps": 0, "checkpoint": str(parent), "evaluation": initial}]
    for steps, directory in [*manifest.get("snapshots", {}).items(),
                             (str(manifest["actual_steps"]), str(run_dir))]:
        snapshot = json.loads((Path(directory) / "manifest.json").read_text())
        candidates.append({"additional_steps": int(steps),
                           "checkpoint": snapshot["run"]["checkpoint_path"],
                           "evaluation": snapshot["evaluation"]})
    rows = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    updates = [row for row in rows if row["event"] == "update"]
    for row in updates:
        row["throttle_std"], row["angle_std"] = row["action_std"]
    fields = ("env_steps", "episodes", "episode_return", "soft_landing_rate", "landed",
              "crashed", "timeouts", "approx_kl", "clip_fraction", "entropy",
              "throttle_std", "angle_std", "value_loss", "learning_rate")
    with (output / f"{arm}_training.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(updates)
    summary = {"arm": arm, "training_seed": seed,
               "execution": manifest["execution"],
               "method": "PPO fine-tuning from shared PPO parent, no BC",
               "parent": manifest["config"]["actor_initialization"],
               "max_action_reload_error": reload_error, "validation": candidates,
               "selected": choose_best(candidates), "final": candidates[-1]}
    write_json(output / f"{arm}.json", summary)
    print(arm, [(c["additional_steps"], c["evaluation"]["soft_landings"]) for c in candidates], flush=True)


def evaluate(parent, output):
    from rl.kaggle_recovery import register_evaluation
    report = json.loads((output / "results.json").read_text())
    requests = [("initial_parent", str(parent))]
    for row in report["runs"]:
        requests.extend((f"{row['arm']}_{kind}", row[kind]["checkpoint"]) for kind in ("selected", "final"))
    cache, evidence = {}, []
    for label, checkpoint in requests:
        if checkpoint not in cache:
            print("Batched fresh evaluation:", label, flush=True)
            cache[checkpoint] = register_evaluation(output, checkpoint=checkpoint, seeds=FRESH_SEEDS,
                                                    batched=True)
            cache[checkpoint]["reward_reporting"] = "W3 baseline components; use training manifests for actual reward"
        evidence.append({"label": label, "checkpoint": checkpoint, "evaluation": cache[checkpoint]})
        write_json(output / "fresh_evaluation.json", evidence)


def plan():
    return {"method": "paired PPO fine-tuning from one shared validation-selected seed-1 PPO parent",
            "arms": ARMS, "additional_steps_per_arm": STEPS, "total_new_transitions": 2 * STEPS,
            "validation": "2000..2099: initial and saved 40k snapshots evaluated after training; best success, earliest on tie",
            "execution": "one CUDA GPU required; compiled PPO/validation/W3 devices and timings recorded",
            "fresh_evaluation_seeds": [FRESH_SEEDS[0], FRESH_SEEDS[-1]],
            "fresh_evaluation": "initial, selected and final policies; ONLY after both arms selected",
            "normalization": "frozen at parent", "learning_rate": "3e-5 annealed to zero",
            "action_std": "reset to .10; constrained to [.05,.15]", "entropy_bonus": 0,
            "clip_coef": .1, "initial_states": "unchanged 2000m nominal; full sensor noise/randomization",
            "success_thresholds": {"vy": 1., "vx": .5},
            "not_claimed": "independent training-seed replication or guaranteed improvement"}


def launch(parent, output, dry_run=False, *, worker_module="rl.kaggle_recovery_v2",
           protocol=None, arms=None, diagnosis_files=("v1_diagnosis.json",)):
    protocol = plan() if protocol is None else protocol
    arms = ARMS if arms is None else arms
    if dry_run:
        print(json.dumps(protocol, indent=2))
        return
    if not Path("/kaggle/working").is_dir():
        raise RuntimeError("Launch is Kaggle-only; use --dry-run locally")
    parent, output = Path(parent).resolve(), Path(output).resolve()
    if not (parent.parent / "manifest.json").is_file():
        raise FileNotFoundError("Missing bundled parent manifest")
    output.mkdir(parents=True, exist_ok=False)
    # Keep the eligible initial policy in the returned ZIP, including when it
    # remains the best model. Never edit the original parent checkpoint.
    shutil.copytree(parent.parent, output / "parent")
    parent = output / "parent" / "checkpoint"
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "JAX_PLATFORMS": "cuda",
                   "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "JAX_DISABLE_JIT": "false",
                   "RLANDER_REQUIRE_GPU": "1",
                   "JAX_COMPILATION_CACHE_DIR": str(output.parent / "rlander_gpu_compilation_cache")}
    report = {"status": "preflight", "protocol": protocol, "runs": [], "fresh_evaluation": []}
    write_json(output / "results.json", report)
    for filename in ("source_manifest.json", "parent_provenance.json", *diagnosis_files):
        if Path(filename).exists():
            (output / filename).write_bytes(Path(filename).read_bytes())

    def worker(stage):
        print("Starting", stage, "— worker must verify compiled GPU execution.", flush=True)
        log_path = output / f"{stage}.log"
        offset = 0
        def relay():
            nonlocal offset
            with log_path.open(encoding="utf-8", errors="replace") as reader:
                reader.seek(offset)
                text = reader.read()
                offset = reader.tell()
            if text:
                print(text, end="", flush=True)
        with log_path.open("w") as log:
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", worker_module, "--worker", stage,
                 "--parent", str(parent), "--output", str(output)],
                env={**environment, "RLANDER_DEVICE_LOG": str(output / f"{stage}_execution.jsonl")},
                stdout=log, stderr=subprocess.STDOUT,
            )
            while True:
                try:
                    code = proc.wait(timeout=10)
                    break
                except subprocess.TimeoutExpired:
                    relay()
                    print(stage, "still running; details:", log_path.name, flush=True)
                    try:
                        sample = subprocess.check_output([
                            "nvidia-smi", "--query-gpu=timestamp,index,name,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits", "-i", "0",
                        ], text=True, timeout=5)
                        with (output / "gpu_telemetry.csv").open("a") as telemetry:
                            telemetry.write(sample)
                        print("GPU sample (time,index,name,util %,MiB):", sample.strip(), flush=True)
                    except (OSError, subprocess.SubprocessError):
                        pass  # Diagnostic only; executable/array device checks are mandatory.
            relay()
        if code:
            raise RuntimeError(f"{stage} exit {code}:\n{log_path.read_text(errors='replace')[-6000:]}")

    try:
        (output / "packages.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
        (output / "hardware.txt").write_text(subprocess.check_output(["nvidia-smi"], text=True))
        worker("preflight")
        report["preflight"] = json.loads((output / "preflight.json").read_text())
        if not report["preflight"]["training_allowed"]:
            report["status"] = "diagnostic_stop"
            return
        report["status"] = "training"
        write_json(output / "results.json", report)
        for arm in arms:
            worker(arm)
            report["runs"].append(json.loads((output / f"{arm}.json").read_text()))
            write_json(output / "results.json", report)
        report["status"] = "evaluating"
        write_json(output / "results.json", report)
        worker("evaluate")
        report["fresh_evaluation"] = json.loads((output / "fresh_evaluation.json").read_text())
        if (output / "selection.json").exists():
            report["selection"] = json.loads((output / "selection.json").read_text())
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(output / "results.json", report)
        print("Status:", report["status"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default="inputs/ppo_parent/checkpoint")
    parser.add_argument("--output", default="/kaggle/working/europa_ppo_recovery_v2_gpu")
    parser.add_argument("--worker", choices=("preflight", *ARMS, "evaluate"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if not Path("/kaggle/working").is_dir():
            raise RuntimeError("Workers are Kaggle-only")
        # This is enforcement for the actual worker, not a separate availability probe.
        os.environ["RLANDER_REQUIRE_GPU"] = "1"
        from rl.execution import event, require_gpu
        device = require_gpu()
        event(args.worker, "worker_backend_verified", device=str(device),
              device_kind=device.device_kind)
        parent, output = Path(args.parent).resolve(), Path(args.output).resolve()
        if args.worker == "preflight":
            preflight(parent, output)
        elif args.worker == "evaluate":
            evaluate(parent, output)
        else:
            train_arm(parent, output, args.worker)
    else:
        launch(args.parent, args.output, args.dry_run)


if __name__ == "__main__":
    main()
