# Europa PPO smoke investigation — 2026-10-16

The all-timeout failure is explained by an unsuitable initial action distribution.
Descent-friendly initialization removes that failure, but PPO soft landing is
still unsolved. No full run was started. All comparisons use Europa's unchanged
config/success thresholds. The original comparisons below use deterministic
evaluation seeds 10000–10099; new reward experiments use validation seeds
2000–2099. Neither repeatedly inspected set is a fresh final benchmark.

| Policy / experiment | Training steps | Soft / crash / timeout | Mean contact vertical speed (m/s) | Mean original return |
|---|---:|---:|---:|---:|
| Original PPO | 50,000 | 0 / 0 / 100 | no contact | -1182.57 |
| PD fallback | 0 | 98 / 2 / 0 | 0.429 | -61.32 |
| Zero thrust | 0 | 0 / 100 / 0 | 75.530 | -234.04 |
| PPO, descent initialization | 50,000 | 0 / 100 / 0 | 40.127 | -250.76 |
| PPO, descent initialization | 200,000 | 0 / 100 / 0 | 29.781 | -243.84 |
| PPO, descent initialization + optimizer reward scale 0.01 | 200,000 | 0 / 100 / 0 | 75.508 | -238.00 |

Descent initialization means actor bias `[-0.5, 0]`, log-std `-1`. Other PPO
settings remain as in the first smoke: 16 envs, 125 rollout steps, 8 minibatches,
4 epochs, lr 3e-4 annealed, gamma .995, lambda .95. Scaling applies uniformly to
the optimizer's rewards/values; environment components and reported returns stay
unchanged. Defaults remain unchanged; these are explicit experimental options.

## Findings

1. The original Gaussian mean maps to 50% throttle; nominal hover needs 33%.
   Broad clipped Gaussian noise also raises expected throttle for a mean near
   nominal hover. The original PPO finishes at mean observed height 6.115 L.
   Lower initial mean/std produces ground-contact experience immediately.
2. The current environment is demonstrably controllable: the observation-only
   PD policy achieves 98/100 soft landings with the same noise/randomization.
   This does not establish physics accuracy or eliminate every possible PPO bug.
3. The dense velocity cost gives weak braking guidance. For monotonic vertical
   descent, sum(|vy|/U) ≈ h0/(U*dt) = 100 because dt=0.01*sqrt(L/g), U=sqrt(gL),
   and L=h0. Measured velocity-component totals are -99.64 for PD and -101.89
   for zero thrust. Slower unsuccessful descents add altitude/fuel cost while
   every crash receives the same -100, independent of impact speed. This creates
   an easy crash local optimum until the sparse +100 soft-landing event is found;
   it does **not** make crashing globally better than the successful PD policy.
4. Raw critic loss is large relative to the 0.2 value clipping scale. Uniform
   reward scaling is available and tested, but did not solve landing: the matched
   experiment learned essentially zero fuel use (0.011 kg mean). Do not present
   this numerical adjustment as a successful landing fix.

## Cross-workstream status

At inspection, main and w1-env point to 4a39c54. W1's committed implementation
explicitly says M1 Euler dynamics; RK4/substeps and analytic tests are still M2
work. `tests/test_analytic.py` contains a passing placeholder. W1 and W3 worktrees
were clean when inspected; w3-eval has no eval/ or storage/ implementation.
W3's missing pipeline blocks final DB/evaluation integration, not PPO learning.

## Approved potential-based experiment

The integrator approved an optional profile reward experiment with the original
reward retained as default, velocity-only changes, a feasibility check, a
preference for potential shaping, and bounded validation. PLAN §5.5.1 records
the scoped ownership exception and numeric parameter extension.

Implemented `potential_profile` **adds** potential shaping to the existing
velocity component. It does not implement the earlier proposed replacement
tracking cost, which would change the objective:

```
vy_target = -min(30 m/s, sqrt((0.4 m/s)^2 + g * max(h, 0)))
Phi(s) = -k * sqrt(vx^2 + (vy - vy_target)^2) / U
r_velocity = original_velocity + gamma * Phi(s_next) - Phi(s)
```

True contact terminals have zero next potential. Truncations retain it because
PPO bootstraps the final observation. Shaping gamma is taken from PPO and checked
before training. The discounted shaping sum telescopes, preserving the original
objective under the usual assumptions; it is not a per-step imitation penalty.
See [Ng, Harada, Russell (1999)](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).
Tests check return telescoping, bootstrap consistency, unchanged terminal
rewards/dynamics/observations/info, JIT operation, and baseline default behavior.

The on-profile deceleration is at most g/2 = 0.671 m/s² at maximum randomized
Europa gravity. Conservative available vertical deceleration, including minimum
thrust, maximum initial mass and full gimbal angle, is 2.106 m/s². Thus the
reference is feasible with fuel available; this does not certify recovery from
arbitrary off-profile states. At the surface the target is -0.4 m/s, not hover.

Three bounded runs used training seed 1, the descent initialization above,
optimizer reward scale 0.01, and 100 deterministic validation seeds 2000–2099:

| Potential scale k | Steps | Soft / crash / timeout | Mean contact abs vy (m/s) | Mean contact abs vx (m/s) |
|---:|---:|---:|---:|---:|
| 20 | 200,000 | 0 / 100 / 0 | 75.685 | 0.873 |
| 100 | 300,000 | 0 / 100 / 0 | 6.582 | 3.010 |
| 100 | 500,000 | 0 / 57 / 43 | 1.513 | 15.363 |

**Validation condition failed: soft-landing rate remains 0%.** The longer run
also develops timeouts, so reduced vertical impact speed cannot be called a
landing solution. Contact-speed averages exclude timeouts. The final run has
mean observed final height 0.0031 L across all episodes; exact timeout trajectories
would need inspection before describing their behavior as stationary hovering.
Both potential scale and run length differ in the first comparison, so it is
not a controlled estimate of the scale's effect. No full run is justified.

Checkpoint replay reproduced the final PPO results exactly. On the same new
validation seeds, PD achieved **99/100** soft landings, one crash and no timeouts
(mean contact |vy| 0.385 m/s, |vx| 0.030 m/s). The detailed comparisons are saved
beside that checkpoint in `diagnostics.json`.

PD proves the task is solvable and the toy PPO acceptance test proves learning
on that toy task; neither proves that the real-env failure is solely a reward
bug. Remaining candidates include optimization/sample efficiency, lateral
control, sparse terminal discovery and the original reward's crash/timeout
tradeoff. Potential shaping deliberately preserves that original objective.

For the Phase B LLM comparison, this profile is a human-designed baseline
candidate. Keep the original baseline ablation and identify extra LLM-generated
terms separately. Do not attribute this profile or these results to Groq.

### W1 handoff (notification not delivered)

W1's Antigravity session was not among accessible Codex chats. The integrator
must relay this handoff; no direct notification is claimed. The scoped change
adds `env/reward_shaping.py`, imports it in `env/lander_env.py`, adds three scalar
reward fields at reset, and changes only the velocity reward expression in step.
Carry those fields unchanged through W1's upcoming RK4 integration. The original
reward remains the default. New tests live in W2's `tests/test_rl.py`; W1's tests
were not edited. PLAN §5.5.1 is the approved addendum for W4 to mirror into the
contracts document. No dependency or lockfile changes were needed.

Verification: `uv run pytest -q` passed **53 tests**, including
`test_ppo_learns_toy_point_mass_descent`, which requires a substantial return
improvement over the exact initial policy on shared evaluation conditions.
The local run uses `UV_FROZEN=1` and `PYTEST_DEBUG_TEMPROOT` pointing to the
workspace's `.pytest_cache/w2-temp` because the OS temp root is inaccessible.

Reproduce the final validation (PowerShell, one command):

```powershell
uv run --frozen python train.py +body=europa +vehicle=europa_lander +seed=1 +ppo.num_envs=16 +ppo.num_steps=125 +ppo.total_steps=500000 '+ppo.initial_action_mean=[-0.5,0.0]' +ppo.initial_log_std=-1.0 +ppo.reward_scale=0.01 +reward.mode=potential_profile +reward.potential_scale=100 +evaluation.seed_start=2000 +evaluation.episodes=100
```

## Local evidence

- Original run and PD/control comparisons: `ckpts/ppo-europa-1-20261015T233529-65668a7a/{manifest,diagnostics}.json`
- Descent 50k: `ckpts/ppo-europa-1-20261015T234724-756a33dc/manifest.json`
- Descent 200k: `ckpts/ppo-europa-1-20261016T053534-4205c283/manifest.json`
- Scaled 200k: `ckpts/ppo-europa-1-20261016T053752-010d6532/manifest.json`
- Potential k=20, 200k: `ckpts/ppo-europa-1-20261016T055605-d7599c22/manifest.json`
- Potential k=100, 300k: `ckpts/ppo-europa-1-20261016T055717-a956b70e/manifest.json`
- Potential k=100, 500k: `ckpts/ppo-europa-1-20261016T055806-b1ac161c/manifest.json`
- One initialization attempt failed before any training; its manifest is marked
  failed with actual_steps=0. It is not included as a completed experiment.

Reproduce baseline diagnostics with:
`uv run python -m rl.diagnostics --run-dir ckpts/ppo-europa-1-20261015T233529-65668a7a`
