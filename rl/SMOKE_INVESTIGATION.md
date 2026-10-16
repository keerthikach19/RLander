# Europa PPO smoke investigation — 2026-10-16

The all-timeout failure is explained by an unsuitable initial action distribution.
Descent-friendly initialization removes that failure, but PPO soft landing is
still unsolved. No full run was started. All reported comparisons use Europa's
unchanged config/success thresholds and deterministic evaluation seeds 10000–10099.
These diagnostic comparisons are not an independent final benchmark after tuning.

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

## Cross-workstream status and proposed next experiment

At inspection, main and w1-env point to 4a39c54. W1's committed implementation
explicitly says M1 Euler dynamics; RK4/substeps and analytic tests are still M2
work. `tests/test_analytic.py` contains a passing placeholder. W1 and W3 worktrees
were clean when inspected; w3-eval has no eval/ or storage/ implementation.
W3's missing pipeline blocks final DB/evaluation integration, not PPO learning.

Proposed W1/integrator change, **not implemented or approved**: replace the
velocity magnitude cost in §5.5 with a smooth descent-profile tracking cost:

```
vy_target = -min(30 m/s, sqrt((0.4 m/s)^2 + g * max(h, 0)))
r_velocity = -w_v * sqrt((vx/U)^2 + ((vy - vy_target)/U)^2)
```

This is the already-tested PD descent profile expressed as a reward target,
not an imitation target for the action. Keep the named component, other weights,
terminal ±100, and frozen success metric unchanged for the first controlled
experiment. It supplies a gradient toward braking before a soft landing has
ever been observed. It needs a contract decision because it changes the locked
§5.5 formula and W1-owned code. It is a hypothesis to test, not a proven fix.
Use separate validation seeds for any further reward/hyperparameter selection;
the repeatedly inspected frozen seeds must not be treated as a fresh final test.

## Local evidence

- Original run and PD/control comparisons: `ckpts/ppo-europa-1-20261015T233529-65668a7a/{manifest,diagnostics}.json`
- Descent 50k: `ckpts/ppo-europa-1-20261015T234724-756a33dc/manifest.json`
- Descent 200k: `ckpts/ppo-europa-1-20261016T053534-4205c283/manifest.json`
- Scaled 200k: `ckpts/ppo-europa-1-20261016T053752-010d6532/manifest.json`
- One initialization attempt failed before any training; its manifest is marked
  failed with actual_steps=0. It is not included as a completed experiment.

Reproduce baseline diagnostics with:
`uv run python -m rl.diagnostics --run-dir ckpts/ppo-europa-1-20261015T233529-65668a7a`
