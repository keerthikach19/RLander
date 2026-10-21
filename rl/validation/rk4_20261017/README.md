# Europa validation after the RK4 merge — 2026-10-17

PPO still does not land. PD succeeds in 97/100 episodes. No full run was
started, and no files under `env/` were edited.

Environment source: `4b2a1eb9dae88c2b8300ad501cc51dd47eb387db` on `w2-rl`,
including RK4 with four substeps and the optional potential-shaping hook.
Execution used JAX 0.11.2 on CPU.

| Policy | Training steps | Soft landings | Crashes | Timeouts | Contact mean abs vy (m/s) | Contact mean abs vx (m/s) |
|---|---:|---:|---:|---:|---:|---:|
| PD | 0 | 97 | 3 | 0 | 0.44 | 0.03 |
| PPO baseline reward | 300,000 | 0 | 100 | 0 | 75.46 | 0.87 |
| PPO baseline reward | 500,000 | 0 | 100 | 0 | 75.46 | 0.87 |
| PPO profile shaping (`reward_mode=1`) | 300,000 | 0 | 100 | 0 | 4.64 | 10.10 |
| PPO profile shaping (`reward_mode=1`) | 500,000 | 0 | 64 | 36 | 1.66 | 4.93 |

Every evaluation uses 100 deterministic mean-policy episodes on validation
seeds 2000–2099, the same set used in the earlier investigation. These are
reused validation seeds, not an untouched final benchmark. Contact statistics
exclude timeouts. Outcomes come from terminal environment info and the existing
Europa success thresholds (abs vy < 1 m/s and abs vx < 0.5 m/s), independently
of reward. This uses the existing W2 `evaluate_smoke` stream for comparability;
it does not populate W3's SQLite evaluation database or use its different RNG
stream. The merged W3 tests are included in the full test result below.

All four PPO runs start fresh with training seed 1, 16 environments, 125 steps
per rollout, 8 minibatches, 4 epochs, learning rate 3e-4 annealed over the
requested run, gamma .995, lambda .95, initial actor mean [-.5, 0], initial
log-std -1, and optimizer reward scale .01. Shaping uses potential scale 100
and gamma .995; baseline uses reward_mode=0. The 500k runs are independent
training runs, not continuations of the 300k checkpoints. Exact configuration,
per-episode outcomes, and checkpoint locations are in [results.json](results.json).

Each policy directory contains `altitude.png` and `vertical_velocity.png`,
with five PPO and five PD episodes (seeds 2000–2004), `summary.json`, and
`trajectories.npz`. NPZ arrays have columns [time_s, altitude_m, vy_mps, vx_mps]
and keys such as `ppo_2000` and `pd_2000`. Traces include the initial state and
stop at first termination. Their PPO outcomes match checkpoint evaluation.

- Baseline PPO: all five plotted episodes crash after about 40–45 seconds at
  74–78 m/s downward. Fuel use across the 100 evaluations is effectively zero.
- Shaping at 300k: all five crash despite substantial braking. Large lateral
  speeds remain, and one trace briefly rises before descending to impact.
- Shaping at 500k: four plotted episodes crash and seed 2004 times out. That
  episode brakes, rises again, and nearly hovers/slowly descends above ground.
  At timeout (222.86 s), altitude is 28.94 m, vy is -0.249 m/s, and vx is
  1.355 m/s. During its final 20 seconds, altitude is 28.94–32.88 m and mean
  abs vy is 0.200 m/s. This is not stationary hover or a successful landing.
  Seed 2002 crashes even with abs vy 0.804 m/s because vx is 3.465 m/s.
- PD: all five plotted episodes land softly in about 88–93 seconds.

The five traces illustrate behavior; the outcome table covers all 100 episodes.
They do not establish that all 36 timeouts follow the same trajectory.

Reproduce the bounded experiment with a new output directory:

```powershell
uv run --frozen python -u -m rl.validate_europa --output-dir ckpts/europa_rk4_repeat
```

Reproduce the final checkpoint's plots (checkpoints are local, not committed):

```powershell
uv run --frozen --with matplotlib python -m rl.plot_trajectories --run-dir ckpts/europa_rk4_validation/checkpoints/ppo-europa-1-20261017T181835-405fdb8e --output-dir rl/validation/rk4_20261017/potential_profile_500000
```

Matplotlib is supplied through uv's temporary environment; project dependencies
and lockfile are unchanged. All eight PNGs, raw plotted traces, and summaries
are committed so the evidence can be viewed without local checkpoints.

Verification: the requested `uv run pytest -q` passed **115 tests, 1 warning**
in 371.86 s. The warning is Orbax's existing checkpoint sharding-restore warning
in `test_checkpoint_stand_in`. After adding the trajectory script and regression
test, `uv run --frozen pytest -q tests/test_rl.py -k trajectory_recording` passed
**1 test** (39 deselected). This checks matching evaluation outcomes, episode
lengths, terminal speeds, no samples after contact, and failure on an insufficient
step budget. Ruff passed for both scripts and `tests/test_rl.py`; `git diff
--check` passed.
