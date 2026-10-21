# V5: one bounded precision-refinement session

The user raised the acceptance target after the v4 demo was packaged. Preserve
that demo and its 87.5% policy, but allow up to two further user-run Kaggle
sessions. V5 is session one. Do not run the Kaggle notebook locally or remotely
on the user's behalf.

The target is a PPO point estimate of at least **91%**, with a gap of at most
**eight percentage points** to PD on matching test episodes. This clears the
user's 90% minimum and, if PD stays around 98.6%, puts the gap around 7.6 points.
Wilson intervals are reported; 91% is not a requirement on the interval's lower
bound. No achievement is promised before measurement.

## Fixed recipe

Start from v4's validation-selected `margin_seed_5` snapshot at +20k:
339/400 validation, 875/1000 test landings. It is selected by the original
validation record, not by comparing test scores.

Retain `landing_margin`, physics, sensor noise, randomization, success
thresholds, frozen observation statistics, clip .1, entropy zero, and reward
scale .01. Change:

| Setting | V4 | V5 |
| --- | ---: | ---: |
| Training action std at initialization | .10 | .03 |
| Permitted std | [.05, .15] | [.01, .05] |
| Initial learning rate, annealed to zero | 1e-5 | 3e-6 |
| Parallel rollout environments | 16 | 32 |
| Steps per rollout | 125 | 125 |

Run seeds 8, 9, and 10 for 320k transitions each: **960k total**, 80 PPO updates
per run. Snapshot every 40k. These combined changes aim to make updates more
conservative and reduce the mismatch between noisy training and deterministic
inference. They are a hypothesis, not an established cause of the remaining
failures. Lower std alone leaves deterministic action means unchanged.

No behavior cloning, PD assistance, softened landing criteria, curriculum,
additional reward variants, or parameter search is included. The actor and
critic are imported; optimizer, RNG, and environments restart.

## Selection and test

- Replay archived validation 6000–6399; stop if the parent differs by more than
  two landings.
- Select on 800 new validation episodes, seeds 8000–8799. Parent eligible.
  Highest landing count wins; ties use earliest step, then seed 8/9/10.
- Freeze one overall candidate before testing.
- Evaluate parent, candidate, and unchanged PD on 2,000 new seeds 70000–71999,
  all on the same GPU and evaluator. Deduplicate identical checkpoints.
- Record gained/lost landings, Wilson intervals, and `target_check.json`.
  Never change the selection based on these test results.

V4 failures were 93 vertical-only crashes, 21 lateral-only, 10 both, and one
timeout. The selected seed's later checkpoints degraded, motivating smaller
updates rather than simply extending the old recipe.

## Run and return

Notebook: `outputs/kaggle_ppo_recovery_v5/rlander_ppo_recovery_v5.ipynb`.
Start a **fresh Kaggle session**, select a GPU accelerator, enable Internet,
upload the notebook, and Run All. The source and checkpoint are embedded.
Use a fresh session if extraction reports an existing workspace folder.

Only GPU 0 is used. Compiled executable placement and completed output devices
are verified. Compilation and file writing still use CPU; no sustained GPU
utilization or end-to-end runtime is promised.

Return `europa_ppo_recovery_v5.zip`, including if preflight or training fails.
If the target is met, stop training. Otherwise inspect this result before
choosing the already-authorized second session; do not automatically repeat
the same schedule.

Local verification: seven focused tests passed, including an eight-transition
CPU compilation check, validation-first selection, parent deduplication,
matched PD test scope, and embedded checkpoint/source integrity. Ruff passed.
The actual notebook and its experiment have not been executed here.
