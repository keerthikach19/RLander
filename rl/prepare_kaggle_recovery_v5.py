"""Prepare the next bounded Kaggle session, without running the notebook."""

import argparse
import json

from rl.analyze_v4_return import analyze
from rl.prepare_kaggle_recovery_v3 import prepare as build_notebook

INTRODUCTION = """# RLander PPO v5 — precise-control refinement
**Start a fresh Kaggle session. Select a GPU accelerator, enable Internet, then Run All.**
Code and the selected **87.5% v4 policy** are embedded. No dataset upload is needed.
If the workspace already exists, restart with a fresh session rather than rerunning extraction.

## Target and budget
Aim for **PPO ≥91%** and **no more than 8 percentage points below PD**, measured
on the same new test seeds and GPU. These are point-estimate targets; Wilson
95% intervals will also be reported. Success is not guaranteed.

One session, one recipe, three refinement seeds **8,9,10**:
**320k transitions each / 960k total**. No automatic second experiment.
This is PPO-only: no behavior cloning, PD action override, or relaxed thresholds.

## Why this change
V4's validation-selected snapshot landed **875/1000** fresh episodes. Its failures
were 93 vertical-only crashes, 21 lateral-only, 10 failing both speed limits,
and one timeout. The best seed-5 snapshot was early; later training degraded.
Training action std stayed near .10, while inference uses the deterministic mean.
The new hypothesis is that less training noise and smaller, less noisy updates
will refine landing precision while preserving learned behavior.

- Keep the **landing_margin reward unchanged**.
- Reduce initial action std **.10 → .03**, bounded **[.01,.05]**.
- Reduce learning rate **1e-5 → 3e-6**, annealed to zero.
- Increase rollout batch **16×125 → 32×125**.
- Keep clip .1, entropy bonus 0, reward scale .01, and frozen parent obs statistics.

This combines stabilization choices; it does not isolate each one's contribution.
The actor, critic and normalization are imported; optimizer, RNG and environments
restart. These are shared-parent refinements, not independent from-scratch runs.
Lower training noise alone does not change deterministic inference actions.

## Selection and comparison
1. Replay archived parent validation **6000–6399** (339/400); stop if it differs
   by more than two landings. Record sampled-action diagnostics, without changing
   the recipe based on them.
2. Use **800 new validation episodes, 8000–8799**. Save snapshots every 40k,
   validate them after each training call, and include the parent as a candidate.
3. Freeze one overall candidate by validation landings, earliest step, then
   seed order 8/9/10. The parent wins an equal-success tie.
4. Evaluate only the **parent, frozen candidate, and unchanged PD controller**
   on **2000 new seeds, 70000–71999**, all on GPU with the same evaluator.
   Identical checkpoints are evaluated once. Test results never change selection.
5. Report paired gained/lost landings, Wilson intervals, and the target check.
   If the target is missed, return the ZIP before deciding on session two.

Nominal 2000m starts, full sensor noise/randomization, and strict **|vy|<1 /
|vx|<.5 m/s** landing criteria remain unchanged. The old 87.5% checkpoint is preserved.
Previous PD measured 98.6% on the v4 seeds; this session measures it again on
the new seeds instead of assuming the old score transfers.

## Runtime and output
GPU 0 only. Compiled workloads and completed array devices are checked.
CPU compilation, checkpoint serialization and brief GPU bursts still occur;
GPU placement is not a promise of sustained utilization or notebook runtime.
This notebook has been packaged and locally checked, **not executed by the author**.

Download and return **`europa_ppo_recovery_v5.zip`**, including on failure.
It contains checkpoints, source/parent provenance, validation selections,
fresh trajectories, GPU execution evidence, and **`target_check.json`**.
"""


def prepare(input_zip, output):
    path = build_notebook(
        input_zip, output, analyzer=analyze, version="v5", introduction=INTRODUCTION,
        extra_files=("rl/kaggle_recovery_v4.py", "rl/kaggle_recovery_v5.py",
                     "rl/analyze_v4_return.py", "rl/prepare_kaggle_recovery_v5.py"),
        diagnosis_name="v4_diagnosis.json", validation_episodes=800,
    )
    notebook = json.loads(path.read_text(encoding="utf-8"))
    # Surface the target in the output cell as well as its archived JSON file.
    cell = notebook["cells"][3]
    text = "".join(cell["source"])
    text = text.replace("if process.returncode:", """target_file = output / 'target_check.json'
if target_file.exists():
    print('TARGET:', json.loads(target_file.read_text()))
if process.returncode:""")
    cell["source"] = text.splitlines(keepends=True)
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_zip")
    parser.add_argument("--output", default="outputs/kaggle_ppo_recovery_v5")
    args = parser.parse_args()
    prepare(args.input_zip, args.output)
