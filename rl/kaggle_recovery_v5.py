"""One bounded Kaggle-only PPO refinement recipe; unchanged v4 reward."""

import argparse
import json
import math
import os
from dataclasses import replace
from pathlib import Path

from rl.kaggle_recovery import register_evaluation, write_json
from rl.kaggle_recovery_v2 import choose_best, train_arm
from rl.kaggle_recovery_v2 import launch as launch_workers
from rl.kaggle_recovery_v3 import configuration as previous_configuration
from rl.kaggle_recovery_v4 import paired

TRAINING_SEEDS = {"seed_8": 8, "seed_9": 9, "seed_10": 10}
STEPS = 320_000
SNAPSHOTS = tuple(range(40_000, STEPS, 40_000))
VALIDATION_SEEDS = tuple(range(8000, 8800))
FRESH_SEEDS = tuple(range(70000, 72000))
TARGET_SUCCESS = .91
MAX_PD_GAP = .08


def configuration():
    return replace(
        previous_configuration(), num_envs=32, total_steps=STEPS, learning_rate=3e-6,
        initial_log_std=math.log(.03), log_std_min=math.log(.01), log_std_max=math.log(.05),
    )


def bindings():
    from omegaconf import OmegaConf

    from train import lander_bindings
    return lander_bindings(OmegaConf.create({
        "reward": {"mode": "landing_margin"},
        "evaluation": {"seed_start": VALIDATION_SEEDS[0], "episodes": len(VALIDATION_SEEDS)},
    }))


def preflight(parent, output):
    from train import evaluate_smoke, load_ppo_initialization
    bind = bindings()
    params, stats, metadata = load_ppo_initialization(
        parent, body_name="europa", vehicle_name="europa_lander", initial_log_std=None,
    )
    replay = evaluate_smoke(
        bind.env, bind.body_params, bind.vehicle_params, params, stats,
        seeds=tuple(range(6000, 6400)), policy_label="archived_v4_validation_replay",
    )
    expected = json.loads((parent.parent / "manifest.json").read_text())["evaluation"]["soft_landings"]
    report = {"parent": metadata, "archived_replay": replay, "expected_soft_landings": expected,
              "training_allowed": abs(replay["soft_landings"] - expected) <= 2}
    write_json(output / "preflight.json", report)
    if report["training_allowed"]:
        report["selection_initial"] = bind.evaluate(params, stats)
        # Diagnostic only: these scores do not change the fixed training recipe.
        for label, std in (("original", None), ("reduced", configuration().initial_log_std)):
            weights, frozen_stats, _ = load_ppo_initialization(
                parent, body_name="europa", vehicle_name="europa_lander", initial_log_std=std,
            )
            report[f"sampled_{label}"] = evaluate_smoke(
                bind.env, bind.body_params, bind.vehicle_params, weights, frozen_stats,
                seeds=VALIDATION_SEEDS[:200], sample_actions=True, policy_label=f"parent_sampled_{label}",
            )
    report["decision"] = "proceed" if report["training_allowed"] else "stop: archived replay differs"
    write_json(output / "preflight.json", report)


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
        "rule": "validation landings, earliest step, then training seed 8/9/10",
    }
    write_json(output / "selection.json", selection)
    requests = [("initial_parent", str(parent)),
                ("selected_candidate", selection["candidate"]["checkpoint"]),
                ("pd_baseline", None)]
    cache, evidence = {}, []
    for label, checkpoint in requests:
        if checkpoint not in cache:
            cache[checkpoint] = register_evaluation(
                output, checkpoint=checkpoint, seeds=FRESH_SEEDS, batched=True,
            )
            cache[checkpoint]["reward_reporting"] = "W3 baseline; training objective is landing_margin"
        evidence.append({"label": label, "checkpoint": checkpoint, "evaluation": cache[checkpoint]})
        write_json(output / "fresh_evaluation.json", evidence)
    parent_eval = cache[str(parent)]
    candidate_eval = cache[selection["candidate"]["checkpoint"]]
    pd_eval = cache[None]
    gap = pd_eval["success_rate"] - candidate_eval["success_rate"]
    write_json(output / "paired_comparison.json", {
        "candidate_vs_parent": paired(parent_eval, candidate_eval),
        "candidate_vs_pd": paired(pd_eval, candidate_eval),
        "scope": "same GPU, seeds, noise, thresholds and evaluator; no test reselection",
    })
    target = {
        "candidate_success_rate": candidate_eval["success_rate"],
        "pd_success_rate": pd_eval["success_rate"],
        "gap_to_pd_percentage_points": 100 * gap,
        "target_success_rate": TARGET_SUCCESS,
        "maximum_gap_percentage_points": 100 * MAX_PD_GAP,
        "target_met": candidate_eval["success_rate"] >= TARGET_SUCCESS and gap <= MAX_PD_GAP + 1e-12,
        "scope": "point estimates; Wilson intervals remain in fresh_evaluation.json",
        "next_step": "Stop if target met; otherwise inspect returned results before choosing the second-session recipe.",
    }
    write_json(output / "target_check.json", target)
    print("Target check:", json.dumps(target), flush=True)


def plan():
    return {
        "method": "one conservative PPO recipe, three seeds, shared selected v4 parent; no BC or PD assistance",
        "training_seeds": TRAINING_SEEDS, "additional_steps_per_seed": STEPS,
        "total_new_transitions": STEPS * len(TRAINING_SEEDS),
        "reward": "landing_margin unchanged",
        "changes": "32x125 batch (was 16x125), learning rate 3e-6 (was 1e-5), std .03 bounded [.01,.05] (was .10 [.05,.15])",
        "retained": "annealed learning rate, clip .1, entropy 0, reward scale .01, frozen parent observation statistics",
        "initialization": "parent actor/critic/normalization; fresh optimizer/RNG/environments, reset std",
        "validation_seeds": [VALIDATION_SEEDS[0], VALIDATION_SEEDS[-1]],
        "selection": "800 validation episodes; parent plus 40k snapshots; freeze one candidate before test",
        "fresh_seeds": [FRESH_SEEDS[0], FRESH_SEEDS[-1]],
        "fresh_evaluation": "2000 episodes each for parent, selected candidate and unchanged PD; deduplicate",
        "target": "PPO >=91% and at most 8 percentage points below measured PD; point estimates, not CI lower bounds",
        "physical_scope": "unchanged 2000m nominal starts, full noise/randomization, strict |vy|<1 and |vx|<.5",
        "gpu": "GPU 0; compiled execution and result devices verified; batched evaluation including PD",
        "not_claimed": "guaranteed gain, independent from-scratch seeds, isolated contribution of each config change",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default="inputs/ppo_parent/checkpoint")
    parser.add_argument("--output", default="/kaggle/working/europa_ppo_recovery_v5")
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
            train_arm(parent, output, args.worker, config=configuration(), bind=bindings(),
                      seed=TRAINING_SEEDS[args.worker], checkpoint_steps=SNAPSHOTS,
                      initial_evaluation_key="selection_initial")
    else:
        launch_workers(args.parent, args.output, args.dry_run, worker_module="rl.kaggle_recovery_v5",
                       protocol=plan(), arms=TRAINING_SEEDS, diagnosis_files=("v4_diagnosis.json",))


if __name__ == "__main__":
    main()
