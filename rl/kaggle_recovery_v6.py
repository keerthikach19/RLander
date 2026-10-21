"""Final authorized Kaggle session: PPO-only mixed-start landing curriculum."""

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from rl.kaggle_recovery import register_evaluation, write_json
from rl.kaggle_recovery_v2 import choose_best, train_arm
from rl.kaggle_recovery_v2 import launch as launch_workers
from rl.kaggle_recovery_v4 import paired
from rl.kaggle_recovery_v5 import MAX_PD_GAP, TARGET_SUCCESS
from rl.kaggle_recovery_v5 import configuration as previous_configuration

TRAINING_SEEDS = {"seed_11": 11, "seed_12": 12, "seed_13": 13}
STEPS = 640_000
SNAPSHOTS = tuple(range(80_000, STEPS, 80_000))
VALIDATION_SEEDS = tuple(range(9000, 10000))
FRESH_SEEDS = tuple(range(80000, 82000))
BANK_SEEDS = tuple(range(900000, 901024))
CURRICULUM_PROBABILITY = .75


def configuration():
    return replace(previous_configuration(), total_steps=STEPS, learning_rate=1e-5)


def bindings():
    """Always return FULL-start validation; workers separately replace training reset."""
    from omegaconf import OmegaConf

    from train import lander_bindings
    return lander_bindings(OmegaConf.create({
        "reward": {"mode": "landing_margin"},
        "evaluation": {"seed_start": VALIDATION_SEEDS[0], "episodes": len(VALIDATION_SEEDS)},
    }))


def preflight(parent, output):
    from rl.landing_curriculum import collect_bank
    from train import evaluate_smoke, load_ppo_initialization
    bind = bindings()
    params, stats, metadata = load_ppo_initialization(
        parent, body_name="europa", vehicle_name="europa_lander", initial_log_std=None,
    )
    replay = evaluate_smoke(
        bind.env, bind.body_params, bind.vehicle_params, params, stats,
        seeds=tuple(range(8000, 8800)), policy_label="archived_v5_validation_replay",
    )
    expected = json.loads((parent.parent / "manifest.json").read_text())["evaluation"]["soft_landings"]
    report = {"parent": metadata, "archived_replay": replay, "expected_soft_landings": expected,
              "training_allowed": abs(replay["soft_landings"] - expected) <= 2}
    write_json(output / "preflight.json", report)
    if report["training_allowed"]:
        report["selection_initial"] = bind.evaluate(params, stats)
        bank_path = output / "curriculum_bank.npz"
        report["curriculum_bank"] = collect_bank(
            bank_path, BANK_SEEDS, bind.body_params, bind.vehicle_params, params, stats,
        )
        report["curriculum_bank"]["sha256"] = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    report["decision"] = "proceed" if report["training_allowed"] else "stop: archived replay differs"
    write_json(output / "preflight.json", report)


def training_bindings(output):
    from rl.landing_curriculum import load_bank, mixed_start_env
    bind = bindings()
    bank_path = output / "curriculum_bank.npz"
    expected = json.loads((output / "preflight.json").read_text())["curriculum_bank"]["sha256"]
    if hashlib.sha256(bank_path.read_bytes()).hexdigest() != expected:
        raise ValueError("Curriculum bank hash changed")
    bank, outcomes = load_bank(bank_path)
    # Preserve evaluate's original LanderEnv closure; only training reset changes.
    return replace(bind, env=mixed_start_env(bank, outcomes, CURRICULUM_PROBABILITY))


def select_candidate(runs):
    rows = sorted(runs, key=lambda row: row["training_seed"])
    if [row["training_seed"] for row in rows] != list(TRAINING_SEEDS.values()):
        raise ValueError("Missing or duplicate training seed")
    for row in rows:
        if row["arm"] != f"seed_{row['training_seed']}":
            raise ValueError("Training arm and seed disagree")
        if row["selected"] != choose_best(row["validation"]):
            raise ValueError("Run candidate disagrees with validation")
    candidate = choose_best([
        {**row["selected"], "arm": row["arm"], "training_seed": row["training_seed"]}
        for row in rows
    ])
    if candidate["additional_steps"] == 0:
        candidate = {**candidate, "arm": "initial_parent", "training_seed": None}
    return candidate


def evaluate(parent, output):
    report = json.loads((output / "results.json").read_text())
    selection = {
        "candidate": select_candidate(report["runs"]),
        "selection_frozen_before_fresh_evaluation": True,
        "rule": "full-start validation landings, earliest step, then seed 11/12/13",
    }
    write_json(output / "selection.json", selection)
    cache, evidence = {}, []
    for label, checkpoint in (
        ("initial_parent", str(parent)), ("selected_candidate", selection["candidate"]["checkpoint"]),
        ("pd_baseline", None),
    ):
        if checkpoint not in cache:
            # W3 constructs original LanderEnv internally: never the curriculum wrapper.
            cache[checkpoint] = register_evaluation(
                output, checkpoint=checkpoint, seeds=FRESH_SEEDS, batched=True,
            )
            cache[checkpoint]["reward_reporting"] = "W3 baseline; training uses landing_margin"
        evidence.append({"label": label, "checkpoint": checkpoint, "evaluation": cache[checkpoint]})
        write_json(output / "fresh_evaluation.json", evidence)
    candidate = cache[selection["candidate"]["checkpoint"]]
    pd = cache[None]
    gap = pd["success_rate"] - candidate["success_rate"]
    write_json(output / "paired_comparison.json", {
        "candidate_vs_parent": paired(cache[str(parent)], candidate),
        "candidate_vs_pd": paired(pd, candidate),
        "scope": "original full starts, same GPU and seeds; no test reselection",
    })
    target = {
        "candidate_success_rate": candidate["success_rate"],
        "pd_success_rate": pd["success_rate"], "gap_to_pd_percentage_points": 100 * gap,
        "target_success_rate": TARGET_SUCCESS, "maximum_gap_percentage_points": 100 * MAX_PD_GAP,
        "target_met": candidate["success_rate"] >= TARGET_SUCCESS and gap <= MAX_PD_GAP + 1e-12,
        "scope": "full-start test point estimates; Wilson intervals in fresh_evaluation.json",
        "next_step": "Final authorized session; report results and stop. No automatic further training.",
    }
    write_json(output / "target_check.json", target)
    print("Target check:", json.dumps(target), flush=True)


def plan():
    return {
        "method": "PPO-only curriculum from own parent flight states; no BC, PD labels or action assistance",
        "session": "second and final authorized extra Kaggle session",
        "training_seeds": TRAINING_SEEDS, "additional_steps_per_seed": STEPS,
        "total_ppo_transitions": STEPS * len(TRAINING_SEEDS),
        "bank_collection": {"seed_range": [BANK_SEEDS[0], BANK_SEEDS[-1]], "episodes": len(BANK_SEEDS),
                            "extra_transitions": "measured in preflight; not included in PPO budget"},
        "training_resets": "75% from bank, 25% original full starts; proportions are resets, not timesteps",
        "bank_sampling": "half from parent failed flights, half successful; all available states if a group is empty",
        "bank_state": "first live crossing of 100/40/15/5m; preserve scales, time, mass, physics; fresh sensor noise",
        "reward": "landing_margin unchanged",
        "settings": "32x125, LR1e-5 annealed, clip .1, entropy0, std .03 [.01,.05], frozen obs stats",
        "changes_vs_v5": "mixed training starts, 640k per seed (was320k), LR1e-5 (was3e-6)",
        "validation_seeds": [VALIDATION_SEEDS[0], VALIDATION_SEEDS[-1]],
        "selection": "1000 original full-start episodes; parent plus80k snapshots; freeze one candidate",
        "fresh_seeds": [FRESH_SEEDS[0], FRESH_SEEDS[-1]],
        "test": "2000 full-start episodes each for parent/candidate/PD; same GPU, deduplicate",
        "target": ">=91% PPO and <=8 percentage points below PD; point estimates, not CI lower bounds",
        "not_claimed": "guaranteed improvement; independent from-scratch training; isolated curriculum effect",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default="inputs/ppo_parent/checkpoint")
    parser.add_argument("--output", default="/kaggle/working/europa_ppo_recovery_v6")
    parser.add_argument("--worker", choices=("preflight", *TRAINING_SEEDS, "evaluate"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.worker:
        if not Path("/kaggle/working").is_dir():
            raise RuntimeError("Workers are Kaggle-only")
        os.environ["RLANDER_REQUIRE_GPU"] = "1"
        from rl.execution import event, require_gpu
        device = require_gpu()
        event(args.worker, "worker_backend_verified", device=str(device), device_kind=device.device_kind)
        parent, output = Path(args.parent).resolve(), Path(args.output).resolve()
        if args.worker == "preflight":
            preflight(parent, output)
        elif args.worker == "evaluate":
            evaluate(parent, output)
        else:
            train_arm(parent, output, args.worker, config=configuration(), bind=training_bindings(output),
                      seed=TRAINING_SEEDS[args.worker], checkpoint_steps=SNAPSHOTS,
                      initial_evaluation_key="selection_initial")
            row = json.loads((output / f"{args.worker}.json").read_text())
            row["training_start_distribution"] = plan()["training_resets"]
            row["validation_start_distribution"] = "original full starts only"
            write_json(output / f"{args.worker}.json", row)
    else:
        launch_workers(args.parent, args.output, args.dry_run, worker_module="rl.kaggle_recovery_v6",
                       protocol=plan(), arms=TRAINING_SEEDS, diagnosis_files=("v5_diagnosis.json",))


if __name__ == "__main__":
    main()
