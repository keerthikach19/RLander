# Returned v2 GPU results and v3 refinement

Source: the user's `europa_ppo_recovery_v2_gpu.zip`. Its source manifest matches
the previously delivered GPU-corrected notebook. Analysis only read the archive;
no returned checkpoint was executed locally.

## Observed results

| Policy | Validation (100 episodes) | Fresh success (200 episodes) |
| --- | ---: | ---: |
| Initial v1 parent | 49 | 106 / 200 — 53% |
| Stable-control selected at +40k | 56 | 120 / 200 — 60% |
| Stable-control final at +320k | 28 | 69 / 200 — 34.5% |
| Success-aligned selected at +160k | 82 | 157 / 200 — 78.5% |
| Success-aligned final at +320k | 74 | 150 / 200 — 75% |

Selected success-aligned model's Wilson 95% interval: **72.3%–83.6%**.
On the same fresh seeds, it gained 52 landings over its parent and lost one;
105 episodes landed under both policies and 42 failed under both.
This is evidence about this shared training lineage, not independent replication.

All selected-model episodes reached contact. Of its 43 crashes, 42 failed only
the vertical threshold and one only the lateral threshold. Median absolute vy
at 10m was approximately 0.88 m/s among eventual landings versus 2.11 m/s among
crashes. At touchdown the corresponding medians were 0.63 and 1.29 m/s.
These are group associations, not proof of a particular learning failure.

Both PPO calls and all recorded validation/W3 workloads reported GPU execution
on Tesla T4 / cuda:0, checked against executable shardings and completed arrays.
The two 320k PPO calls took 3.471s and 3.456s in synchronized execution, with
68.470s and 7.653s of compilation/cache loading respectively. These timings
exclude host checkpoint I/O, startup and notebook setup. Snapshot manifests
inherit the timing of the entire training call; they are not separate timing
measurements for each snapshot.

## Next notebook

`outputs/kaggle_ppo_recovery_v3/rlander_ppo_recovery_v3.ipynb`

The parent is chosen by highest existing v2 validation success across arms,
earliest additional step on ties. The builder packages the +160k
success-aligned checkpoint, whose ancestry contains 880k transitions.
Fresh evaluation scores do not enter this selection.

Three new fine-tuning schedules use RNG seeds 2, 3 and 4, 160k transitions each
(480k new transitions total). Each starts from the same parent with a fresh
optimizer/RNG/environment and reset exploration. A descendant trained to its
final checkpoint has 1.04M transitions along its ancestry.

The learning rate is reduced from 3e-5 to 1e-5, annealed to zero. The successful
reward and remaining PPO settings are retained: 16×125, clip .1, entropy zero,
std initialized .10 and bounded [.05,.15], frozen parent normalization.
Physics, sensor noise, domain randomization and strict success limits are unchanged.
There is no BC or PD action correction.

This tests a conservative refinement strategy. It does not isolate learning
rate as a causal variable, because the starting model, training horizon and
selection protocol also differ from v2. It measures variation in refinement
from a shared ancestor, not three independent from-scratch replications.

The archived 2000–2099 validation is used only for a parent replay check.
New checkpoint selection uses 4000–4199 (200 episodes). Snapshots are saved at
20k increments and evaluated after each training call. Each seed's candidate
includes the unchanged parent. Selection prefers highest success, earliest
step, then seed order 2/3/4. The overall candidate is recorded before any fresh
evaluation. A tied parent is labeled `initial_parent`, not a newly trained model.

Only after all selections, the parent and each selected policy are evaluated
on 40000–40499 (500 episodes); duplicate checkpoints are reused. This final set
is separate from previously inspected seeds. A paired parent comparison records
newly gained and lost landings. Fresh results never replace the selected
candidate with another model. Final checkpoints retain validation diagnostics
but do not receive a separate fresh evaluation.

All v2 GPU enforcement, batched evaluation, runtime evidence, log streaming and
partial-result packaging are retained. The old delivered notebooks are preserved.
Return `europa_ppo_recovery_v3.zip` after running a fresh Kaggle GPU session with
Internet enabled.

## Local preparation checks

**25 focused CPU tests passed in 52.37 seconds.** These include the existing
refinement/GPU-parity checks plus eight v3 cases covering validation-only
parent selection despite reversed fresh scores, configuration and seed scopes,
tie rules, launch ordering and GPU enforcement, frozen selection during fresh
evaluation, an eight-transition CPU integration test, and notebook packaging.
No full v3 training or Kaggle execution was performed locally. Improved v3
success or runtime is not claimed in advance.
