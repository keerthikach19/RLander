# V7 — one vertical-touchdown reward change

The user authorized one more Kaggle session after V6, specifically targeting
vertical touchdown speed. Remain PPO-only: no demonstrations, imitation loss,
expert action targets, or PD action assistance.

## Evidence and objective

Use V6's validation-frozen seed13 snapshot at +160k: 895/1000 validation
landings, followed by 1799/2000 (89.95%) on its new full-start test. Its 196
crashes included 159 vertical-only failures and three failing both velocity
limits. Of those 162 vertical failures, 102 had touchdown speed below1.2m/s
but at least1m/s. Their median absolute vertical speed at5m was1.526m/s;
median touchdown speed was1.128m/s.

Add one bounded, smooth cost at **all ground contacts**:

    additional_reward = -30 * vy**2 / (vy**2 + 0.5**2)

| Absolute touchdown vy | Added reward |
| --- | ---: |
| 0m/s | 0 |
| .3m/s | -7.94 |
| .5m/s | -15 |
| 1m/s | -24 |

The cost approaches -30 at high speed. It applies even to successful landings,
so a landing well below the limit earns a better return. Non-contact rewards,
existing lateral terms and time/fuel costs remain unchanged. The extra cost
uses the existing impact-cost component (`crash`), which already contains
continuous impact costs on successful landings.

This is reward mode5, `landing_vertical`, extending mode4 `landing_margin`.
Success still requires ground contact with |vy|<1m/s and |vx|<.5m/s. It does
not change actions at evaluation, dynamics, observations, noise or thresholds.
The effect on learned control remains unproven. Reward totals across V6 and
V7 are not directly comparable.

## Fixed run

Keep V6's curriculum and PPO settings: 75% bank resets /25% full starts;
half bank samples from failed parent flights, half successful; 32×125
rollouts; LR1e-5 annealed; clip .1; entropy0; reward scale .01; std .03
bounded [.01,.05]; frozen normalization.

Collect a new bank from 1,024 flights of the selected PPO parent, using
training seeds910000–911023 and live crossings at100/40/15/5m. Preserve
physical scales, time, fuel and randomized parameters, and regenerate sensor
noise. There are no teacher action labels. Collection transitions are additional
to the PPO budget and are recorded separately.

Train seeds14,15,16 for640k transitions each: **1.92m PPO transitions total**,
with80k snapshots. Parent actor/critic are imported; optimizer/RNG/environments
restart. These are shared-parent refinements. Without a matched control arm,
this experiment does not isolate a causal reward effect from further training.

## Selection and test

- Replay archived V6 validation9000–9999; stop if the result differs by more
  than two landings.
- Select on **1,000 new full-start validation episodes**, seeds11000–11999.
  Parent eligible. Highest landing count, earliest step, then seed14/15/16.
- Freeze one candidate, then test parent, candidate and unchanged PD on
  **2,000 new full-start episodes**, seeds90000–91999, all on GPU.
- Preserve original physical success criteria and noise. Never use curriculum
  starts for validation/test or reselect using test scores.
- Target ≥91% PPO and ≤8 percentage points behind PD on point estimates;
  report Wilson intervals separately.

## Run and return

Notebook: `outputs/kaggle_ppo_recovery_v7/rlander_ppo_recovery_v7.ipynb`.
Use a fresh Kaggle session, select GPU, enable Internet, upload and Run All.
Source and the exact selected parent are embedded. Use a fresh session if the
fixed extraction folder already exists.

Return `europa_ppo_recovery_v7.zip`, including on failure. It records the
checkpoints, source/parent provenance, curriculum bank, full-start selection and
test evidence, GPU device/timing evidence, and `target_check.json`.

No additional experiment follows automatically. The original89.95% checkpoint
and prior notebooks remain preserved. Only GPU0 is used; CPU compilation and
serialization still occur, and sustained GPU utilization is not promised.
The notebook and its experiment have not been executed here.

Local verification: 50 tests passed in55.25s, covering the new contact cost,
unchanged physical outcomes/observations/success thresholds, curriculum
compatibility, validation-frozen selection, analytic physics and environment
contracts. Tiny CPU PPO fixtures were used; the full experiment was not run.
Notebook code cells and embedded source/parent integrity are checked during
final packaging.
