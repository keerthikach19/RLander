# PD behavior cloning and shaped PPO — 2026-10-18

BC-initialized PPO reaches **100/100 soft landings at 500k PPO steps** on the
specified validation set. This is promising validation evidence, not a confirmed
frozen-test or multi-seed result. The method is **PPO fine-tuned from a
behavior-cloned PD controller**, not PPO from scratch. The saved checkpoint reproduces this result. No
three-seed full run or Kaggle run was launched. `env/` is unchanged from the
previous RK4 validation commit, `86cdb9c`.

The [follow-up stability audit](../stability_20261018/README.md) reports three
continuous training seeds, the W3 frozen evaluations, sampled-action sensitivity,
and the unresolved shaping-magnitude finding. Its results qualify the original
single-seed validation result below.

| Policy | PPO steps | Soft landings | Crashes | Timeouts |
|---|---:|---:|---:|---:|
| PD (checkpoint replay comparison) | 0 | 97 | 3 | 0 |
| BC actor before PPO | 0 | 36 | 57 | 7 |
| BC + shaped PPO | 300,000 | 22 | 40 | 38 |
| BC + shaped PPO | 500,000 | 100 | 0 | 0 |

Every row uses 100 episodes on seeds 2000–2099, deterministic policy means,
unchanged Europa randomization, and terminal-info success thresholds of
abs vy < 1 m/s and abs vx < 0.5 m/s. We retain W2's `evaluate_smoke` RNG stream
for comparability with the earlier validation. These reused validation seeds
and one training seed do not establish performance across training seeds or on
an untouched test set. No W3 SQLite evaluation rows are claimed.

BC data collection used 32 vectorized environments, 200,000 actual transitions,
and root PRNG seed 31415. It completed 832 randomized episodes: 820 soft
landings, 12 crashes, and no timeouts. Partial episodes at the collection cutoff
also contribute observation/action pairs. A separate 20,000-transition stream
(root seed 31416) supplies held-out MSE only: 63 soft landings, one crash, no
timeouts, plus partial episodes. Neither validation outcomes nor held-out labels
are used for fitting or checkpoint selection.

The PD teacher receives the noisy `obs` returned by reset/step, together with
nominal config parameters; it never receives clean per-episode state. PPO sees
the same observation interface. Altitude noise is sigma=0.5 m + 0.5% of height,
velocity noise is sigma=0.1 m/s per axis, and acceleration noise is sigma=0.02 g.
Reset randomizes height +/-10%, descent speed +/-20%, initial vx +/-2 m/s,
mass and thrust +/-5%, and gravity +/-2%. Evaluation uses clipped actor means;
it does not sample the learned Gaussian, whose log-std remains trainable.

The reward keeps all six named components and PLAN 5.5 weights (velocity 1,
altitude .5, fuel .1, angle .05, terminals +/-100). PLAN 5.5.1 adds the profile
potential difference to `velocity`, with experimental scale 100 and gamma .995.
The successful run's final training update logs `shaping_to_terminal_ratio`=
1.394605 and `shaping_magnitude_status=above_target`; its deterministic validation
ratio is 1.401677. This exceeds the .2--1 target and remains an explicit audit
finding. No reward rescaling was performed during this validation.

The two-layer 64-unit tanh actor was fitted to PD action labels with unweighted
MSE on raw actor means, 80 fixed epochs, batch size 512, Adam with cosine learning
rate decay from 1e-3 to 1e-4, and fit seed 2718. Epochs shuffle training rows and
drop the last incomplete minibatch. Full training MSE fell from 0.146856 to
0.002973; held-out MSE is 0.002430 (throttle 0.003486, angle 0.001373).

`actor_init.msgpack` is a portable 19,837-byte initialization checkpoint containing
only the three actor MLP layers, observation mean/variance/count, and format/body/
vehicle metadata. PPO restores these normalization statistics (count 200,000),
then updates them normally during PPO. The critic and optimizer are fresh;
log-std starts at -1.5 and is learnable. The checkpoint SHA-256 and fit details
are in [bc_report.json](bc_report.json), and the same hash is recorded in both
PPO run configurations.

Both PPO runs start independently from this same BC checkpoint using training
seed 1; 500k is not a continuation of the 300k run. Other settings match the
previous shaped validation: 16 environments, 125 rollout steps, 8 minibatches,
4 epochs, learning rate 3e-4 annealed over the requested run, gamma .995, lambda
.95, reward scale .01, potential scale 100, and `reward_mode=1`.

The 300k run has mean contact abs vy 1.161 m/s and abs vx 0.056 m/s, excluding
timeouts. Its five plotted episodes (seeds 2000–2004) show two soft landings and
three crashes; those five do not illustrate its 38 other timeout episodes.
The 500k run has mean contact abs vy 0.357 m/s and abs vx 0.160 m/s. All five
plotted episodes land softly in 98–111 seconds, descending to ground after
braking rather than timing out above it. The matched five PD episodes all land
softly in 88–93 seconds. The successful PPO consumes 45.79 kg mean fuel versus
PD's 37.02 kg on the 100-episode comparison.

Each `bc_shaping_*` directory contains altitude and vertical-velocity PNGs,
per-episode summaries, and raw `trajectories.npz` arrays with columns
[time_s, altitude_m, vy_mps, vx_mps]. Traces stop at first termination and PPO
outcomes match the original evaluation. All four PNGs were visually checked.
[results.json](results.json) contains the full run configurations and episode
results. [replay_500000.json](replay_500000.json) contains the independently
restored checkpoint comparison, including PD and control policies.

Reproduce BC in a new output directory:

```powershell
uv run --frozen python -m rl.bc_pretrain --output-dir ckpts/europa_bc_repeat
```

Run the two bounded validations from the committed init (fresh output directory):

```powershell
uv run --frozen python -u -m rl.validate_bc --bc-init rl/validation/bc_20261018/actor_init.msgpack --output-dir ckpts/europa_bc_repeat_validation
```

The new `train.py` option also supports individual runs. With `+bc_init=...`,
the CLI defaults initial log-std to -1.5 unless explicitly overridden:

```powershell
uv run --frozen python train.py +bc_init=rl/validation/bc_20261018/actor_init.msgpack +body=europa +vehicle=europa_lander +seed=1 +ppo.num_envs=16 +ppo.num_steps=125 +ppo.total_steps=300000 +ppo.reward_scale=0.01 +reward.mode=potential_profile +reward.potential_scale=100 +evaluation.seed_start=2000 +evaluation.episodes=100
```

Reproduce the saved 500k trajectory plots (PPO checkpoints remain local):

```powershell
uv run --frozen --with matplotlib python -m rl.plot_trajectories --run-dir ckpts/europa_bc_validation_20261018/checkpoints/ppo-europa-1-20261017T184030-cc37a51a --output-dir rl/validation/bc_20261018/bc_shaping_500000
```

The [Kaggle config](../../configs/kaggle_europa_2m.yaml) is prepared only. It
specifies shaped reward, no BC, seed 1, outputs under `/kaggle/working`, and
128 environments × 125 steps × 125 updates = exactly 2,000,000 PPO steps. It
assumes the repository dependencies and GPU-enabled JAX are available. The
existing smoke gate remains intact (`smoke_success_rate: null`); the BC result
is not evidence that the no-BC policy passes that gate. No override or fictitious
passing score is included. Inspect it without starting training:

```powershell
uv run --frozen python train.py --config-path rl/configs --config-name kaggle_europa_2m --cfg job
```

Verification: `uv run --frozen pytest -q` passed **119 tests, 1 warning** in
393.41 seconds. The warning is the existing Orbax sharding-restore warning in
W3's checkpoint test. New tests check checkpoint round-trip, fresh critic and
log-std, paired normalization, rejection of invalid actor initialization, PD
labels, held-out action learning, and exact Kaggle step count with an enforced
smoke gate. Ruff and `git diff --check` passed. The initial targeted test needed
a tolerance correction for the deliberately tiny nonzero PPO update; the
full suite above includes that correction.
