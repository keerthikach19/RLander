"""Kaggle-only paired approach-reward experiment; no local launch."""

import argparse
import json
import os
from pathlib import Path

from rl.kaggle_recovery import register_evaluation, write_json
from rl.kaggle_recovery_v2 import choose_best, train_arm
from rl.kaggle_recovery_v2 import launch as launch_workers
from rl.kaggle_recovery_v3 import SNAPSHOTS, STEPS, configuration

REWARDS = {"control": "landing_success", "margin": "landing_margin"}
TRAINING_SEEDS = (5, 6, 7)
RUNS = {f"{arm}_seed_{seed}": (arm, seed) for arm in REWARDS for seed in TRAINING_SEEDS}
VALIDATION_SEEDS = tuple(range(6000, 6400))
FRESH_SEEDS = tuple(range(60000, 61000))


def bindings(mode):
    from omegaconf import OmegaConf

    from train import lander_bindings
    return lander_bindings(OmegaConf.create({
        "reward": {"mode": mode},
        "evaluation": {"seed_start": VALIDATION_SEEDS[0], "episodes": len(VALIDATION_SEEDS)},
    }))


def preflight(parent, output):
    import numpy as np

    from train import evaluate_smoke, load_ppo_initialization
    bind = bindings("landing_success")
    params, stats, metadata = load_ppo_initialization(
        parent, body_name="europa", vehicle_name="europa_lander", initial_log_std=None,
    )
    replay = evaluate_smoke(bind.env, bind.body_params, bind.vehicle_params, params, stats,
                            seeds=tuple(range(4000, 4200)), policy_label="archived_parent_replay")
    expected = json.loads((parent.parent / "manifest.json").read_text())["evaluation"]["soft_landings"]
    report = {"parent": metadata, "archived_replay": replay, "expected_soft_landings": expected,
              "training_allowed": abs(replay["soft_landings"] - expected) <= 2}
    write_json(output / "preflight.json", report)
    if report["training_allowed"]:
        # Both rewards must produce the same observations and success outcomes.
        # Keep their separate returns for honest objective-specific reporting.
        for arm, mode in REWARDS.items():
            report[f"initial_{arm}"] = bindings(mode).evaluate(params, stats)
        a, b = (report[f"initial_{arm}"]["episodes"] for arm in REWARDS)
        same = (np.array_equal(a["info"]["outcome"], b["info"]["outcome"])
                and np.array_equal(a["steps"], b["steps"])
                and np.allclose(a["final_observation"], b["final_observation"], rtol=1e-5, atol=1e-6)
                and all(np.allclose(a["info"][k], b["info"][k], rtol=1e-5, atol=2e-4)
                        for k in a["info"]))
        report["reward_only_parent_parity"] = bool(same)
        write_json(output / "preflight.json", report)
        if not same:
            raise RuntimeError("Changing reward altered parent evaluation dynamics")
    report["decision"] = "proceed" if report["training_allowed"] else "stop: archived replay differs"
    write_json(output / "preflight.json", report)


def select_candidates(runs):
    chosen = {}
    for arm in REWARDS:
        rows = sorted((r for r in runs if r["arm"].startswith(arm + "_seed_")),
                      key=lambda r: r["training_seed"])
        if [r["training_seed"] for r in rows] != list(TRAINING_SEEDS):
            raise ValueError(f"Missing or duplicate seed in {arm}")
        candidates = [{**r["selected"], "arm": r["arm"], "training_seed": r["training_seed"]}
                      for r in rows]
        chosen[arm] = choose_best(candidates)
    best = max(chosen.values(), key=lambda r: (
        r["evaluation"]["soft_landings"], -r["additional_steps"], -r["training_seed"],
    ))  # insertion order makes control win a remaining exact tie
    if best["additional_steps"] == 0:
        best = {**best, "arm": "initial_parent", "training_seed": None}
    return {"candidate": best, "per_reward": chosen,
            "selection_frozen_before_fresh_evaluation": True,
            "rule": "success count, earliest step, seed order 5/6/7; control wins exact cross-reward ties"}


def paired(initial, other):
    before = {r["seed"]: r["outcome"] == 1 for r in initial["episodes"]}
    after = {r["seed"]: r["outcome"] == 1 for r in other["episodes"]}
    if before.keys() != after.keys():
        raise ValueError("Paired evaluation seeds differ")
    return {
        "episodes": len(before), "both_landed": sum(before[s] and after[s] for s in before),
        "newly_landed": sum(not before[s] and after[s] for s in before),
        "lost_landings": sum(before[s] and not after[s] for s in before),
        "both_failed": sum(not before[s] and not after[s] for s in before),
        "success_rate_difference": other["success_rate"] - initial["success_rate"],
    }


def evaluate(parent, output):
    report = json.loads((output / "results.json").read_text())
    selection = select_candidates(report["runs"])
    write_json(output / "selection.json", selection)
    requests = [("initial_parent", str(parent))]
    requests.extend((arm + "_selected", row["checkpoint"]) for arm, row in selection["per_reward"].items())
    cache, evidence = {}, []
    for label, checkpoint in requests:
        if checkpoint not in cache:
            cache[checkpoint] = register_evaluation(output, checkpoint=checkpoint,
                                                    seeds=FRESH_SEEDS, batched=True)
            cache[checkpoint]["reward_reporting"] = "W3 baseline; use training manifests for actual objective"
        evidence.append({"label": label, "checkpoint": checkpoint, "evaluation": cache[checkpoint]})
        write_json(output / "fresh_evaluation.json", evidence)
    control, margin = (cache[selection["per_reward"][a]["checkpoint"]] for a in REWARDS)
    write_json(output / "paired_comparison.json", {
        "candidate_vs_parent": paired(cache[str(parent)], cache[selection["candidate"]["checkpoint"]]),
        "control_vs_parent": paired(cache[str(parent)], control),
        "margin_vs_parent": paired(cache[str(parent)], margin),
        "margin_vs_control": paired(control, margin),
        "scope": "validation-selected policies; no reselection from fresh evaluation",
    })


def plan():
    return {
        "method": "paired reward comparison from one shared v3 validation-frozen parent",
        "rewards": REWARDS, "training_seeds": TRAINING_SEEDS,
        "additional_steps_per_run": STEPS, "total_new_transitions": STEPS * len(RUNS),
        "matching": "same parent, seeds and PPO settings across rewards; only approach velocity cost changes",
        "settings": "16x125, learning rate 1e-5 annealed, clip .1, entropy 0, std .10 bounded [.05,.15], frozen obs stats",
        "initialization": "actor/critic/obs stats imported; fresh optimizer, RNG, environments, reset std",
        "margin": "below 100m blend toward slower vertical reference and lateral scale .25m/s; full effect below 30m",
        "validation_seeds": [VALIDATION_SEEDS[0], VALIDATION_SEEDS[-1]],
        "selection": "20k snapshots evaluated after training; parent eligible; freeze per-reward and overall candidates",
        "fresh_seeds": [FRESH_SEEDS[0], FRESH_SEEDS[-1]],
        "fresh_evaluation": "parent and each reward's validation-selected candidate; no fresh reselection",
        "unchanged": "terminal score, time/fuel/angle costs, physics, sensors, 2000m starts, strict 1/.5 thresholds",
        "gpu": "GPU 0; verified compiled execution, batched evaluation, streamed phases and timings",
        "not_claimed": "independent from-scratch seeds, isolated vertical/lateral effects, guaranteed gain",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", default="inputs/ppo_parent/checkpoint")
    parser.add_argument("--output", default="/kaggle/working/europa_ppo_recovery_v4")
    parser.add_argument("--worker", choices=("preflight", *RUNS, "evaluate"))
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
            arm, seed = RUNS[args.worker]
            train_arm(parent, output, args.worker, config=configuration(), bind=bindings(REWARDS[arm]),
                      seed=seed, checkpoint_steps=SNAPSHOTS, initial_evaluation_key=f"initial_{arm}")
    else:
        launch_workers(args.parent, args.output, args.dry_run, worker_module="rl.kaggle_recovery_v4",
                       protocol=plan(), arms=RUNS, diagnosis_files=("v3_diagnosis.json",))


if __name__ == "__main__":
    main()
