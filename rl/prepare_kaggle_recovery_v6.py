"""Build the last authorized PPO-only Kaggle notebook; never execute it."""

import argparse
import json

from rl.analyze_v5_return import analyze
from rl.prepare_kaggle_recovery_v3 import prepare as build_notebook

INTRODUCTION = """# RLander PPO v6 — final-descent practice, PPO-only
**Fresh Kaggle session · GPU accelerator · Internet enabled · Run All.**
Code and the validation-selected V5 checkpoint are embedded. No dataset needed.
If extraction reports that its folder exists, start a fresh session.

## V5 result and target
V5 landed **1754/2000 (87.7%)**, versus its parent's **1751/2000 (87.55%)**
and PD's **1974/2000 (98.7%)** on matching GPU test episodes.
It gained 37 landings and lost 34: essentially flat, despite more stable training.
V5's selected candidate is seed8 at +200k, validation **706/800**.

This is the **second and final authorized extra session**. Aim for **≥91% PPO**
and **≤8 percentage points behind PD** on a new test. These are point estimates,
not requirements on confidence-interval lower bounds. Improvement is not promised.

## A different training intervention
Use the current PPO policy to collect **1024 full flights**, with separate
training seeds 900000–901023. Store actual simulator states at the first live
crossing of **100, 40, 15 and 5 metres**.

During PPO, **75% of episode resets** start from those states; **25%** still
start from the original full-height distribution. Within bank resets, half
sample parent failed flights and half successful flights. If a group is empty,
use the available states. These are reset probabilities, not percentages of
training timesteps. The bank is fixed for all three refinement runs.

This is **not imitation learning**. There are no PD demonstrations, expert action
labels, behavior-cloning losses, or controller overrides. The parent PPO supplies
starting states only; all learned updates are ordinary on-policy PPO rewards.
The policy still receives only its eight noisy observations.

Preserve each bank state's original observation scales, control timestep,
randomized physics, mass/fuel, elapsed time, and remaining time limit. Regenerate
sensor noise at reset. Do not replace h0 in the vehicle configuration: that would
change the meaning of the policy inputs.

Run **three seeds 11,12,13 ×640k PPO transitions =1.92m total**, plus the separate
1024-episode bank collection. Keep the margin reward, 32×125 rollouts, std .03
bounded [.01,.05], clip .1, entropy0, reward scale .01, and frozen normalization.
Use LR **1e-5**, annealed to zero, to permit more movement than V5's 3e-6.
Actor/critic are imported; optimizer/RNG/environments restart.
This combined curriculum/budget/LR intervention does not isolate each change.

## Prevent an easier test from inflating results
- Replay archived V5 validation **8000–8799** (706/800); stop if it differs by
  more than two landings.
- Select checkpoints using **1000 NEW, ORIGINAL FULL-START episodes,
  9000–9999**. The current parent remains eligible. Snapshots every80k;
  highest landing count wins, then earliest step, then seed11/12/13.
- Freeze ONE candidate before testing.
- Evaluate parent, candidate, and unchanged PD on **2000 new ORIGINAL FULL-START
  seeds, 80000–81999**, all on GPU. Test results cannot change selection.
- Keep original noise/randomization and strict **|vy|<1 / |vx|<.5 m/s** criteria.
  The curriculum reset wrapper is never used for validation or the final test.

Training return/outcome curves include partial descents and must not be reported
as full-flight success rates. Only full-start validation and test rates count.
These are shared-parent refinements, not independent training from scratch.

## Execution and return
GPU0 only; compiled bank collection, PPO and evaluations verify device placement
and synchronize outputs. CPU compilation and file writing still occur. No
sustained GPU utilization or complete-notebook runtime is promised.

The notebook has been locally checked but **not executed by the author**.
Return **`europa_ppo_recovery_v6.zip`**, including on failure. It contains
checkpoints, the hashed curriculum bank, source/parent provenance, full-start
validation/test evidence, paired comparisons and **`target_check.json`**.
After this session, report the result and stop; no automatic further experiment.
"""


def prepare(input_zip, output):
    path = build_notebook(
        input_zip, output, analyzer=analyze, version="v6", introduction=INTRODUCTION,
        extra_files=("rl/kaggle_recovery_v4.py", "rl/kaggle_recovery_v5.py",
                     "rl/kaggle_recovery_v6.py", "rl/landing_curriculum.py",
                     "rl/analyze_v5_return.py", "rl/prepare_kaggle_recovery_v6.py"),
        diagnosis_name="v5_diagnosis.json", validation_episodes=1000,
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
        "Sampled training return", "Training return (mixed starts)",
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
    parser.add_argument("--output", default="outputs/kaggle_ppo_recovery_v6")
    args = parser.parse_args()
    prepare(args.input_zip, args.output)
