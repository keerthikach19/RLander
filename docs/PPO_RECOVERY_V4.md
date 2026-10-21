# V3 findings and the v4 approach-reward experiment

The returned v3 source manifest matches the delivered notebook. Its result
status is complete and the archived parent replay reproduced 82/100.
The new 4000–4199 selection set scored that parent at 149/200; this is a
different seed set, so the change is not itself a policy regression.

## What the fresh test showed

| Policy | Selected additional steps | Validation / 200 | Fresh / 500 |
| --- | ---: | ---: | ---: |
| Parent | 0 | 149 | 393 (78.6%) |
| Seed 2 selected | 60k | 157 | 421 (84.2%) |
| Seed 3 selected | 0 (parent retained) | 149 | 393 (78.6%) |
| **Seed 4: predeclared overall candidate** | **20k** | **158** | **414 (82.8%)** |

The official candidate gained 33 landings and lost 12 relative to its parent
on the same fresh seeds, a net gain of 21/500 or 4.2 percentage points.
Its Wilson 95% success interval is approximately 79.2–85.9%.
Seed 2's higher fresh score is reported descriptively; it does not replace
the candidate selected before testing. The one-episode validation difference
between seeds 2 and 4 is not strong evidence that one is intrinsically better.

The selected seed-4 model had 71 vertical-only crashes, 14 lateral-only
crashes and one timeout. At 10m, median absolute vertical speed was
approximately 0.75 m/s among eventual landings and 2.15 m/s among crashes.
These are observed associations, not proof that one reward formula is the cause.

All three PPO calls were verified on Tesla T4 / cuda:0. Each 160k call took
approximately 1.88s in synchronized execution. Compilation/cache loading took
77.60s, 8.49s and 8.70s respectively. Startup, serialization and other notebook
phases are outside those execution timings.

## Controlled next step

The v4 parent is exactly the frozen seed-4 +20k checkpoint, with 900k training
transitions along its ancestry. The ZIP builder validates the frozen selection
against the run selections before reading fresh scores, then packages only
that parent checkpoint and manifest. The prior ZIP and notebooks are preserved.

Both arms use the same parent, paired RNG seeds 5/6/7, 160k transitions per run,
and the v3 optimizer/exploration settings. There are six runs, 960k new
transitions total. A final descendant has 1.06M ancestral transitions.
These are shared-parent fine-tuning seeds, not independent from-scratch runs.

- **Control:** current `landing_success` reward.
- **Margin:** new `landing_margin` reward. Only the dense velocity cost changes.

For physical height h, let w = clip((100-h)/70, 0, 1).
The vertical reference becomes
`-min(30, sqrt(0.4² + (1 - 0.75*w)*g*max(h,0)))`, and lateral normalization
changes from 2 to `2 - 1.75*w` m/s. The full change applies below 30m and
vanishes above 100m. At 10m on nominal Europa the descent reference changes
from about -3.65 to -1.86 m/s, asking for slower descent earlier.

The same vertical-error normalization formula, cost weights, clipping and dt
integration remain. Landing/crash score components, time/fuel/angle costs and
strict physical success thresholds are unchanged. The revised dense velocity
cost also applies on the contact step. The new dense term is not potential-based;
it may change optimal behavior and could encourage over-braking or hovering.
The control arm and timeout diagnostics are retained to detect that.

The comparison tests the combined approach shaping, not its vertical and
lateral components separately. There is no PD intervention, BC, observation
change, or relaxation of physics/noise/randomization. The 0.25m/s lateral
normalization is a reward scale, not a replacement for the 0.5m/s success limit.

## Selection, evidence and execution

Replay 4000–4199 for parent compatibility, then use new validation seeds
6000–6399 (400 episodes). Record a separate parent reward summary for each arm,
checking that its observations, step counts and outcomes agree between modes.
Save 20k snapshots and evaluate them after each compiled training call.

The parent remains eligible in every run. Select each reward's candidate by
success count, earliest step, then seed order 5/6/7. Apply those criteria to
the overall candidate; control wins any remaining exact cross-reward tie.
Write the selections before fresh testing.

Evaluate parent, control candidate and margin candidate on new seeds
60000–60999 (1000 episodes). Reuse identical checkpoints. Do not promote a
different model after seeing those results. Report paired gained/lost
landings for each candidate versus parent and margin versus control.
Only these three policies receive the fresh test; all six runs retain
snapshot/final validation data.

GPU 0 enforcement, executable/result checks, batched evaluation, persistent
compilation caching and stage-specific timings remain. CPU preparation and
brief GPU execution bursts remain expected; no sustained utilization or
runtime improvement is promised.

Notebook: `outputs/kaggle_ppo_recovery_v4/rlander_ppo_recovery_v4.ipynb`.
It embeds source and checkpoint. Run in a fresh Kaggle GPU session with
Internet enabled and return `europa_ppo_recovery_v4.zip`.
No v4 notebook or full training has been executed locally.

## Completed preparation checks

All **167 local CPU tests passed in 235.81 seconds**, with one existing
Orbax warning in a legacy checkpoint stand-in test. The 11 new v4 cases
cover reward isolation and tapering, unchanged observations/dynamics/outcomes,
paired configuration and seed scopes, an eight-transition PPO integration
check at low altitude, selection tie rules, frozen selection despite reversed
fresh rankings, official-parent extraction and notebook packaging.

Scoped lint and `git diff --check` passed. All five notebook cells are
unexecuted and code cells parse. All 44 embedded source files and 12 selected
parent files match their recorded hashes. These checks establish local
correctness and packaging, not v4 CUDA performance or a landing improvement.
