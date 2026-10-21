"""Build a bounded vertical-touchdown PPO notebook; do not execute it."""

import argparse
import json

from rl.analyze_v6_return import analyze
from rl.prepare_kaggle_recovery_v3 import prepare as build_notebook

INTRODUCTION = """# RLander PPO v7 — vertical touchdown precision
**Fresh Kaggle session · GPU accelerator · Internet enabled · Run All.**
Source and the selected V6 PPO checkpoint are embedded. No dataset needed.
If the extraction folder already exists, start a fresh session.

## Starting evidence
The validation-selected V6 policy landed **1799/2000 (89.95%)**, versus PD's
**1979/2000 (98.95%)**. It is seed13 at +160k, selected on **895/1000 validation**
episodes before test evaluation. Its 196 crashes comprised 159 vertical-only,
34 lateral-only and three failing both speed limits, with five further timeouts.
Of the 162 vertical-speed failures, **102 were between 1.0 and 1.2 m/s**.
Median vertical speed was 1.53m/s at 5m altitude in these failed flights.

## One targeted reward change
Keep the V6 approach reward, lateral terms, curriculum and PPO settings.
At ground contact only, add this smooth vertical-speed penalty:

**extra reward = -30 × vy² / (vy² + 0.5²)**.

It is zero at zero vertical speed, -7.94 at .3m/s, -15 at .5m/s,
-24 at 1m/s, and approaches -30 at high speed. It applies to both soft
landings and crashes, in the existing impact-cost component. It gives a
larger incentive to land with speed to spare rather than just below 1m/s.
Non-contact rewards are identical to V6. This is a hypothesis, not a guaranteed gain.
No action correction is applied at inference.

**Success remains exactly |vy|<1 and |vx|<.5 m/s at contact.**
Original noise, randomized physics, observation interface and full-height
evaluation remain unchanged. Lower reward totals are expected because of the
extra cost; reward values from V6 and V7 are not directly comparable.

## Fixed budget, PPO-only
Three refinement seeds **14,15,16 ×640k PPO transitions =1.92m total**.
Same V6 settings: 32×125, LR1e-5 annealed, clip .1, entropy0, reward scale .01,
std reset .03 bounded [.01,.05], frozen parent observation statistics.
Actor/critic are imported; optimizer, RNG and environments restart.

As in V6, collect 1024 flights of this PPO parent, using training seeds
910000–911023, recording live states at 100/40/15/5m. This collection is
additional to the PPO transition budget. Preserve original scales, time,
fuel and physical parameters; regenerate sensor noise at resets.
75% of training resets use the bank, half from failed parent flights and half
from successful flights (use available states if a group is empty).
25% retain original full starts. These are reset probabilities, not step fractions.

No PD demonstrations, imitation losses, expert actions or controller overrides.
All bank states come from the PPO parent itself. This is shared-parent
fine-tuning, not independent learning from scratch. There is no matched control
arm, so this session cannot isolate a causal reward effect from further training.

## Full-start selection and test
1. Replay archived V6 validation9000–9999 (895/1000); stop if it differs by
   more than two landings.
2. Select on **1000 new full-start validation seeds11000–11999**. Save80k
   snapshots; parent eligible. Highest landing count wins, then earliest
   step, then seed14/15/16. Freeze one candidate before the test.
3. Evaluate parent, candidate and unchanged PD on **2000 new full-start test
   seeds90000–91999**, all on GPU. Never reselect using test scores.
4. Report paired gains/losses, Wilson intervals and `target_check.json`.
   Target remains **PPO≥91% and ≤8 percentage points behind PD** on point
   estimates, not on confidence-interval lower bounds.

The curriculum wrapper is used only for training. Training curves include
partial descents and cannot be reported as full-flight success rates.
The old 89.95% checkpoint remains preserved even if this experiment regresses.

## Execution and return
GPU0 only; compiled collection/training/evaluation and output arrays are
checked. Compilation and file writing still use CPU. No sustained GPU
utilization or complete-notebook runtime is promised.
This notebook has been locally checked, **not executed by the author**.

Return **`europa_ppo_recovery_v7.zip`**, including on failure. It contains
checkpoints, bank/provenance, selection, test trajectories, GPU evidence and
the target check. This is the one additional session authorized after V6.
Report its result and stop; no automatic follow-on experiment.
"""


def prepare(input_zip, output):
    path = build_notebook(
        input_zip, output, analyzer=analyze, version="v7", introduction=INTRODUCTION,
        extra_files=("rl/kaggle_recovery_v4.py", "rl/kaggle_recovery_v5.py",
                     "rl/kaggle_recovery_v6.py", "rl/kaggle_recovery_v7.py",
                     "rl/landing_curriculum.py", "rl/analyze_v6_return.py",
                     "rl/prepare_kaggle_recovery_v7.py"),
        diagnosis_name="v6_diagnosis.json", validation_episodes=1000,
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cell = notebook["cells"][3]
    code = "".join(cell["source"]).replace("if process.returncode:", """target_file = output / 'target_check.json'
if target_file.exists():
    print('TARGET:', json.loads(target_file.read_text()))
if process.returncode:""")
    cell["source"] = code.splitlines(keepends=True)
    cell = notebook["cells"][4]
    code = "".join(cell["source"]).replace(
        "Sampled training return", "Training return (mixed starts; V7 reward)",
    ).replace(
        "Training outcomes (10-update weighted window)", "Training outcomes (mixed starts; 10-update window)",
    ).replace(
        "Deterministic validation (1000 seeds)", "Full-start validation (1000 seeds)",
    )
    cell["source"] = code.splitlines(keepends=True)
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_zip")
    parser.add_argument("--output", default="outputs/kaggle_ppo_recovery_v7")
    args = parser.parse_args()
    prepare(args.input_zip, args.output)
