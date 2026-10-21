"""Build the v4 paired reward notebook from returned v3 evidence; no launch."""

import argparse

from rl.analyze_v3_return import analyze
from rl.prepare_kaggle_recovery_v3 import prepare as build_notebook

INTRODUCTION = """# RLander PPO v4 — test the final-approach reward
**Fresh Kaggle session · GPU accelerator + Internet · Run All.**
Source and the officially selected v3 checkpoint are embedded.

## What v3 established
The parent landed **393/500 (78.6%)** fresh episodes. The candidate selected
before the test—**seed 4 at +20k**, validation 158/200—landed **414/500 (82.8%)**,
Wilson 95% CI 79.2–85.9%. It gained 33 landings and lost 12 versus the parent.
Seed 2 scored 421/500 (84.2%), but was second on validation (157/200).
We retain the preselected seed-4 candidate; we do not switch parents using test scores.
The chosen candidate had 71 vertical-only crashes, 14 lateral-only crashes,
and one timeout. Later checkpoints generally lost validation performance.

## The controlled comparison
Both arms start from that same parent and use paired training seeds **5,6,7**.
Each run receives **160k transitions**, six runs / **960k new transitions total**:

- **control:** the current success-aligned reward.
- **margin:** change only the dense velocity cost near the ground. Above 100m
  it matches control. From 100m to 30m it blends toward a slower descent reference
  and tighter lateral damping; below 30m the full change applies.

For the margin arm, `w = clip((100-h)/70, 0, 1)`,
`vy_target = -min(30, sqrt(0.4² + (1 - 0.75*w)*g*max(h,0)))`,
and the lateral normalization changes from 2 m/s to `2 - 1.75*w` m/s.
At 10m on nominal Europa, the descent reference changes from about -3.65 to
-1.86 m/s. This asks the policy to slow earlier and reduce sideways motion.
These are **reward targets, not success thresholds or a controller override**.
The landing/crash score components and time/fuel/angle costs remain unchanged;
the revised dense velocity cost also applies on the contact step. The paired arms test
this combined approach reward; they do not isolate its vertical and lateral parts.
The changed dense costs are not potential-based and can change optimal behavior.
They may encourage over-braking or hovering; timeout/return diagnostics are retained.

Both arms retain 16×125 rollouts, learning rate 1e-5 annealed to zero, clip .1,
entropy zero, std reset .10 bounded [.05,.15], and frozen parent normalization.
Actor/critic/normalization are imported; optimizer, RNG and environments restart.
No BC or PD action corrections. Nominal 2000m starts, full noise/randomization,
and strict **|vy|<1 / |vx|<.5 m/s** success conditions stay unchanged.

## Selection and evaluation
1. Replay the parent's archived 4000–4199 validation; stop on a discrepancy
   exceeding two landings. Evaluate the parent under both reward configurations
   on new validation seeds and check matching trajectories' terminal observations
   and outcomes. Record each objective's own reward totals.
2. Use **400 validation episodes (6000–6399)**. Save 20k snapshots and evaluate
   after each training call. The parent is eligible in every run.
3. Pick each reward's candidate by validation success, earliest step, then
   seed order 5/6/7. Pick an overall candidate by the same criteria; control wins
   a remaining exact cross-reward tie. Freeze selections before the fresh test.
4. Evaluate **parent + control candidate + margin candidate** on **1000 new seeds
   60000–60999**. Deduplicate identical checkpoints. Never reselect using these
   results. Report paired gains/losses versus parent and margin versus control.
   All six runs retain their selected and final validation results; only each
   reward's chosen policy receives the fresh test.

These are fine-tuning seeds from one shared ancestor, not independent training
from scratch. The experiment tests a hypothesis; improvement is not promised.
The original parent is always preserved.

## Execution and return
GPU 0 only. Compiled executables and completed result arrays are verified;
CPU fallback and disabled JIT are rejected. Evaluations stay batched and
compilation caching remains enabled. CPU compilation/file writing and brief GPU
bursts remain expected. Logs separate compilation from synchronized execution.
No v4 notebook or full training has been run by the author.

**Return `europa_ppo_recovery_v4.zip`**, including on failure or diagnostic stop.
It contains all checkpoints, selection records, fresh results, trajectories,
paired comparisons, timing/device evidence and source/parent provenance.
"""


def prepare(input_zip, output):
    return build_notebook(
        input_zip, output, analyzer=analyze, version="v4", introduction=INTRODUCTION,
        extra_files=("rl/kaggle_recovery_v4.py", "rl/analyze_v3_return.py",
                     "rl/prepare_kaggle_recovery_v4.py"),
        diagnosis_name="v3_diagnosis.json", validation_episodes=400,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_zip")
    parser.add_argument("--output", default="outputs/kaggle_ppo_recovery_v4")
    args = parser.parse_args()
    prepare(args.input_zip, args.output)
