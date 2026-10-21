# V6 — final authorized session, PPO-only

The user explicitly rejected PD demonstrations. V6 remains PPO-only: no BC,
expert action labels, PD-assisted control, or imitation loss.

## Returned V5 evidence

The returned ZIP completed successfully; its source manifest matches the
delivered notebook. Its validation-frozen candidate was seed 8 at +200,000
transitions, selected with 706/800 validation landings.

| Policy | Landings / test episodes | Success | Wilson 95% interval |
| --- | ---: | ---: | ---: |
| V5 parent | 1751 / 2000 | 87.55% | 86.03–88.93% |
| Selected V5 PPO | 1754 / 2000 | 87.70% | 86.19–89.07% |
| PD baseline | 1974 / 2000 | 98.70% | 98.10–99.11% |

All three used GPU and seeds 70000–71999. The candidate gained 37 landings
and lost 34 versus the parent. It had 246 crashes and no timeouts. The
11-percentage-point gap to PD misses the target. The result does not support
another repetition of V5's small-update schedule.

## Intervention

Collect 1,024 deterministic flights of the selected **PPO parent** on training
seeds 900000–901023. Record the first live crossing of 100, 40, 15, and 5 metres.
Keep the entire simulator state: physical randomization, nondimensional
scales, mass, elapsed time, previous velocity, and remaining episode time.
Regenerate sensor noise when starting a new training episode from a saved state.

For PPO training resets:

- 25% use the original full-start distribution.
- 75% sample the saved-state bank. Within these, half sample states from failed
  parent flights and half from successful flights. If either group is empty,
  its sampling pool falls back to all available states.

These are probabilities per **reset**, not fractions of transitions. The bank
contains states and outcome metadata, not action targets. PPO samples its own
actions and learns from new rewards. This changes the training initial-state
distribution and cannot be described as unchanged full-start training.

Keep the landing-margin reward, 32×125 rollouts, clip .1, entropy bonus zero,
std reset .03 with [.01,.05] bounds, reward scale .01, and frozen normalization.
Use LR1e-5, annealed to zero, rather than V5's 3e-6.

Train seeds 11,12,13 for 640k transitions each: **1.92m PPO transitions total**.
Parent bank collection is additional evaluation work, with its transition
count recorded. Snapshot every80k. This combines curriculum, a larger budget,
and a higher learning rate; it does not isolate their individual effects.

## Evaluation safeguards

Replay V5's archived 8000–8799 validation before training; stop if the count
differs by more than two landings.

Select on **1,000 original full-start episodes**, seeds9000–9999. The parent
remains eligible; ties use earliest step, then training seed11/12/13. Freeze
one overall candidate before testing.

Test parent, candidate, and unchanged PD on **2,000 original full-start
episodes**, seeds80000–81999, on the same GPU. Keep full sensor noise,
domain randomization, and the original contact thresholds:
|vy|<1m/s and |vx|<.5m/s. No saved-state resets enter validation or testing.

Training curves include partial descents and are labelled as such. They are
not evidence of full-flight success. Shared-parent refinements are not
independent training from scratch.

The point-estimate target is ≥91% PPO and ≤8 percentage points behind PD.
Wilson intervals are reported separately. Improvement is not guaranteed.

## Run and return

Upload `outputs/kaggle_ppo_recovery_v6/rlander_ppo_recovery_v6.ipynb` to a fresh
Kaggle session, select GPU, enable Internet, and Run All. Source and parent
checkpoint are embedded. Use a fresh session if the fixed extraction folder
already exists.

Return `europa_ppo_recovery_v6.zip`, including on failure. It contains
`curriculum_bank.npz`, its hash and collection provenance in `preflight.json`,
all checkpoints, full-start validation/test results, GPU execution evidence,
and `target_check.json`.

This is the second and final authorized extra session. Report its result and
stop; do not schedule another experiment automatically.

## Local checks

Fourteen focused V5/V6 tests passed, covering state fidelity, original reset
equivalence, noise regeneration, bank collection/loading, unchanged validation
closure, tiny CPU PPO compatibility, validation-frozen selection, and disjoint
seed scopes. The actual training experiment and Kaggle notebook were not run.
The notebook's cells parse, its source hashes match the local files, and its
parent payload is checked against the returned V5 snapshot.
