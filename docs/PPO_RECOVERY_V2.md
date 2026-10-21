# Returned v1 evidence and v2 refinement

Prepared 2026-10-20 from the user's `europa_ppo_recovery_v1.zip`. The archive is
read as experiment data. The previous notebook and its outputs are preserved.
No Kaggle execution or GPU training is performed during notebook preparation.

## What happened

| Policy | Validation at 720k | Validation at 960k | Final W3, 100 episodes |
|---|---:|---:|---:|
| Reward-only, 128 environments | 0% | 0% | 0% |
| More updates, 16 environments | 49% | 7% | 5% |
| PD | — | — | 100% |

The 720k checkpoint has no v1 W3 result; 49% is validation success only.
The more-updates validation trajectory is 0%, 0%, 49%, 7% at
240k/480k/720k/960k. Saved-vs-live deterministic action errors were
5.96e-8 and 3.58e-7. Observation normalization was restored exactly.

All 95 final more-updates W3 crashes fail only vertical speed. Crash mean |vy|
is 1.839 m/s and mean |vx| is .219 m/s. Across all 100 final trajectories,
median |vy| is 7.618 at 100m, 3.518 at 30m, 2.288 at 10m, and 1.724 at contact.
The contact sample remains the existing terminal RK4 sample, not an interpolated
impact. The large-batch arm still hits at roughly 58 m/s on average.

Mean validation return rises from -4.61 at 720k to -2.76 at 960k while success
falls. This is not proof of a reward optimum at a crashing speed: three 720k
timeouts average -359.18, whereas 960k has no timeouts. At 720k, successful
episodes average +8.92 and crashes +3.74; at 960k, successes average +22.06 and
crashes -4.63. Return is therefore not a reliable checkpoint-selection proxy.
Late approximate KL is small; these logs do not establish an exploding update.

Conclusion: PPO has acquired some landing skill. Final braking and preservation
of that skill are the next targets. Low-noise refinement, fixed normalization,
and explicit success reward are hypotheses to test, not diagnosed root causes.
A curriculum remains a later option; it is not silently enabled in v2.

## v2 protocol

The notebook embeds the parent selected by highest v1 validation success, with
earliest checkpoint breaking ties: more-updates seed 1 at 720k. It includes SHA256
provenance for the original ZIP, source files and selected checkpoint members.
No external dataset upload or repeat of v1 is needed.

Two paired 320k schedules import the same PPO actor, critic and normalization.
Optimizer, environments, RNG schedule and action standard deviation are reset;
this is PPO warm-start fine-tuning, not an exact optimizer resume and not BC.
The parent has 720k transitions of training history, so each final descendant
has 1.04M transitions along its ancestry. New compute totals 640k transitions.

Shared settings: 16 x 125 rollouts, seed 1, learning rate 3e-5 linearly annealed,
clip .1, entropy coefficient zero, reward scale .01, action std initialized .10
and constrained to [.05,.15], observation statistics fixed at the parent.
Stored log-std parameters are projected after optimization so the sampler and
PPO log-probabilities use the same distribution.

- `stable_control`: v1 landing-profile reward unchanged.
- `success_aligned`: dense costs unchanged; contact score becomes half the v1
  continuous score plus 50 for actual soft landing or minus 50 for crash.
  Nominal contact range remains [-100,100]; exact environment success matters.

Both keep nominal 2000m initial altitude, full original domain randomization and
sensor noise, and strict |vy|<1 / |vx|<.5 m/s. No PD action correction is used.
The comparison isolates terminal reward under common stabilization settings;
it does not isolate each optimizer change or estimate independent seed variance.

Before training, deterministic parent replay must be within two landings of its
archived validation count. Original and reduced-noise sampled policies are also
evaluated with matched reset/sensor streams. These comparisons are diagnostics,
not branch-selection criteria. A zero-noise regression test verifies stream
preservation.

Snapshots are saved every 40k; deterministic validation on seeds 2000–2099 is
performed after the compiled training call finishes. The highest
success count wins, earliest on ties. The parent at zero additional steps is
eligible, preventing a worse final validation policy from replacing it by default.
This selection does not guarantee fresh-seed performance; all finals are retained.

Only after BOTH selections, W3 evaluates parent, selected and final policies on
predeclared fresh seeds 30000–30199 (200 episodes per distinct checkpoint).
Identical checkpoints are evaluated once and referenced under each label.
No tuning or checkpoint selection reads those results in this notebook. W3
component totals use baseline YAML, as before; actual rewards are in manifests.

## Run and return

Build: `.venv/Scripts/python.exe -m rl.prepare_kaggle_recovery_v2 <returned-v1.zip>`.
Current corrected notebook:
`outputs/kaggle_ppo_recovery_v2_gpu/rlander_ppo_recovery_v2_gpu.ipynb`.
See [GPU correction and claim audit](PPO_GPU_CORRECTION.md). The original v2
artifact is preserved as historical evidence. Use a fresh Kaggle session, GPU
and Internet, Run All. The corrected version requires and verifies GPU 0.
Return `europa_ppo_recovery_v2_gpu.zip`, including the preserved parent, all saved
checkpoints, training metrics, sampled diagnostics, curves and fresh evaluation
trajectories. This notebook makes no advance claim of improved success.

If promising, the next check is replication across independent PPO training
seeds (or explicitly labeled shared-parent fine-tuning seeds), not a reliability
claim from this single ancestor. If both arms fail, inspect their braking traces
and sampled/mean gap before choosing curriculum or explicitly labeled BC+PPO.

## Original v2 preparation checks (before GPU correction)

All **141 tests passed** in 321.02 seconds, with one existing Orbax sharding
restore warning. This includes ten refinement tests covering PPO warm-start
loading, frozen normalization, bounded exploration, outcome-aligned reward,
parent-preserving selection, orchestration ordering and notebook packaging.
Scoped Ruff and diff checks passed. All 38 embedded source files matched the
working source hashes. The notebook contains no executed cells or outputs.
Tests used CPU; Orbax tests needed Windows loopback access outside the sandbox.
The Kaggle notebook and GPU training were not run during preparation.
