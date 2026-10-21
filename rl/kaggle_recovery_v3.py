"""Kaggle-only conservative PPO refinement from the best v2 checkpoint."""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from rl.kaggle_recovery import write_json
from rl.kaggle_recovery_v2 import choose_best, train_arm
from rl.kaggle_recovery_v2 import configuration as previous_configuration
from rl.kaggle_recovery_v2 import launch as launch_workers

TRAINING_SEEDS = {"seed_2": 2, "seed_3": 3, "seed_4": 4}
STEPS = 160_000
SNAPSHOTS = tuple(range(20_000, STEPS, 20_000))
VALIDATION_SEEDS = tuple(range(4000, 4200))
FRESH_SEEDS = tuple(range(40000, 40500))


def configuration():
    return replace(previous_configuration(), total_steps=STEPS, learning_rate=1e-5)


def bindings():
    from omegaconf import OmegaConf

    from train import lander_bindings
    return lander_bindings(OmegaConf.create({
        "reward": {"mode": "landing_success"},
        "evaluation": {"seed_start": VALIDATION_SEEDS[0], "episodes": len(VALIDATION_SEEDS)},
    }))


def preflight(parent, output):
    from train import evaluate_smoke, load_ppo_initialization
    bind = bindings()
    params, stats, metadata = load_ppo_initialization(
        parent, body_name="europa", vehicle_name="europa_lander", initial_log_std=None,
    )
    replay = evaluate_smoke(bind.env, bind.body_params, bind.vehicle_params, params, stats,
                            seeds=tuple(range(2000, 2100)), policy_label="archived_validation_replay")
    expected = json.loads((parent.parent / "manifest.json").read_text())["evaluation"]["soft_landings"]
    report = {"parent": metadata, "archived_replay": replay, "expected_soft_landings": expected,
              "training_allowed": abs(replay["soft_landings"] - expected) <= 2}
    write_json(output / "preflight.json", report)
    if report["training_allowed"]:
        report["selection_initial"] = bind.evaluate(params, stats)
        report["sampled_initial"] = evaluate_smoke(
            bind.env, bind.body_params, bind.vehicle_params, params, stats,
            seeds=VALIDATION_SEEDS, sample_actions=True, policy_label="parent_sampled_diagnostic",
        )
    report["decision"] = "proceed" if report["training_allowed"] else "stop: parent replay differs"
    write_json(output / "preflight.json", report)


def select_candidate(runs):
    # Run order is fixed to seeds 2,3,4. On equal success and step, the first
    # seed wins; the parent wins any equal-success tie because its step is zero.
    rows = [{**run["selected"], "arm": run["arm"], "training_seed": run["training_seed"]}
            for run in runs]
    selected = choose_best(rows)
    if selected["additional_steps"] == 0:
        selected = {**selected, "arm": "initial_parent", "training_seed": None}
    return selected


def evaluate(parent, output):
    from rl.kaggle_recovery import register_evaluation
    report = json.loads((output / "results.json").read_text())
    selection = {"rule": "validation success; then earliest step; then seed order 2,3,4",
                 "candidate": select_candidate(report["runs"]),
                 "selection_frozen_before_fresh_evaluation": True}
    # The champion is fixed before any fresh results exist. Never promote a
    # different model because it scores higher on this fresh evaluation.
    write_json(output / "selection.json", selection)
    requests = [("initial_parent", str(parent))]
    requests.extend((r["arm"] + "_selected", r["selected"]["checkpoint"]) for r in report["runs"])
    cache, evidence = {}, []
    for label, checkpoint in requests:
        if checkpoint not in cache:
            cache[checkpoint] = register_evaluation(
                output, checkpoint=checkpoint, seeds=FRESH_SEEDS, batched=True,
            )
            cache[checkpoint]["reward_reporting"] = "W3 baseline; actual objective is in training manifests"
        evidence.append({"label": label, "checkpoint": checkpoint, "evaluation": cache[checkpoint]})
        write_json(output / "fresh_evaluation.json", evidence)
    initial = cache[str(parent)]
    selected = cache[selection["candidate"]["checkpoint"]]
    before = {e["seed"]: e["outcome"] == 1 for e in initial["episodes"]}
    after = {e["seed"]: e["outcome"] == 1 for e in selected["episodes"]}
    if before.keys() != after.keys():
        raise ValueError("Paired evaluation seed mismatch")
    write_json(output / "paired_comparison.json", {
        "candidate_checkpoint": selection["candidate"]["checkpoint"], "episodes": len(before),
        "both_landed": sum(before[s] and after[s] for s in before),
        "newly_landed": sum(not before[s] and after[s] for s in before),
        "lost_landings": sum(before[s] and not after[s] for s in before),
        "both_failed": sum(not before[s] and not after[s] for s in before),
        "candidate_minus_parent_success_rate": selected["success_rate"] - initial["success_rate"],
        "scope": "paired evaluation of a validation-selected model; no fresh-set reselection",
    })


def plan():
    return {
        "method": "three PPO fine-tuning seeds from one shared validation-selected v2 parent; no BC",
        "training_seeds": TRAINING_SEEDS, "additional_steps_per_seed": STEPS,
        "total_new_transitions": STEPS * len(TRAINING_SEEDS),
        "reward": "landing_success unchanged", "learning_rate": "1e-5 annealed to zero",
        "other_settings": "16x125, clip .1, entropy 0, std reset .10 bounded [.05,.15], frozen parent obs stats",
        "initialization": "parent actor/critic/normalization; fresh optimizer, RNG and environments",
        "validation_seeds": [VALIDATION_SEEDS[0], VALIDATION_SEEDS[-1]],
        "validation": "parent plus 20k snapshots evaluated after each training call; best success, earliest tie",
        "fresh_seeds": [FRESH_SEEDS[0], FRESH_SEEDS[-1]],
        "fresh_evaluation": "parent and each seed's selected model, only after all selections; deduplicate",
        "selection": "champion fixed by validation before fresh evaluation; never reselect from fresh results",
        "initial_state_and_success": "unchanged 2000m nominal, full noise/randomization, |vy|<1 and |vx|<.5",
        "gpu": "GPU 0 only; compiled executable and result-array verification retained",
        "not_claimed": "independent from-scratch replication, isolation of learning-rate effect, or guaranteed gain",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default="inputs/ppo_parent/checkpoint")
    parser.add_argument("--output", default="/kaggle/working/europa_ppo_recovery_v3")
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
        launch_workers(args.parent, args.output, args.dry_run, worker_module="rl.kaggle_recovery_v3",
                       protocol=plan(), arms=TRAINING_SEEDS, diagnosis_files=("v2_diagnosis.json",))


if __name__ == "__main__":
    main()
