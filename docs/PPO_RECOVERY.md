# Kaggle PPO recovery v1

User-run experiment, prepared 2026-10-19. No Kaggle execution or GPU training
was performed during preparation. The previous notebook/output is preserved.

## Diagnosis from source and existing output

The previous reward was `potential_profile` with scale 100, not baseline alone.
Its zero-success outcomes do not, by themselves, prove that hovering is optimal.
PPO saves normalization statistics, W3 restores them, deterministic inference
clips the mean action, and GAE bootstraps the final observation at timeouts.
The previous batch was 128 x 125 = 16,000 transitions, so 2M steps gave only
125 PPO updates. Whether that is sufficient remains an experimental question.

W3's printed touchdown averages included timeouts. New output uses contact-only
statistics and separate landed/crashed/timeout summaries. Empty contact sets
report `null`/`n/a`. For SQL compatibility, existing non-null `vy_mean`/`vx_mean`
columns retain their historical all-episode-final meaning, now explicitly
labeled in JSON; corrected statistics are stored in `term_stats._evaluation`.

## New reward (mode 2, opt-in)

`landing_profile` changes the objective, unlike mode 1's potential shaping.
It uses the same PD reference `-min(30, sqrt(.4^2 + g*h))` m/s. The vertical
tracking error is divided by `max(vy_limit, .2*abs(reference))`; lateral velocity
is divided by 2 m/s. Squared errors are capped at 25 and weighted .15/.10 per
second. A .20/second time cost, .02*throttle/second fuel cost, and
.01*normalized_angle/second angle cost are integrated over dt.

At contact, let `e = (vy/vy_limit)^2 + (vx/vx_limit)^2`. The net contact score is
`100/(1+e/4) - 100*e/(25+e)`. This is smooth, bounded, and decreases with impact
speed on either axis. Rational tails retain differences between poor impacts
better than a narrow exponential score. There is no reward-only definition of
success: the environment still requires strict |vy| < 1 and |vx| < .5 m/s.

The six existing component names remain. In this mode `velocity` combines the
two tracking costs, `altitude` records the time cost, `landing` is positive
contact credit, and `crash` is the negative impact cost on every contact.
Use their sum for return; do not infer outcomes from these names.
Baseline and potential-profile reward modes remain available unchanged.

A local CPU PD evaluation on validation seeds 2000–2099 gave 97 soft, 3 crashed,
0 timeout. Its dense-cost / nominal terminal scale was 0.204635 (target .2–1).
This checks the controller and reward scale, not PPO learning. Training has not
been assessed with the new reward.

## Notebook protocol

Build with `.venv/Scripts/python.exe -m rl.prepare_kaggle_recovery`.
The result is `outputs/kaggle_ppo_recovery_v1/rlander_ppo_recovery_v1.ipynb`.
It bundles source, SQL schema and dependency versions with SHA256 hashes.
Upload only the notebook to a fresh Kaggle GPU session; enable Internet.

1. PD through actual W3 on 10000–10099. Below 80%, run noise-off diagnostic
   (domain randomization stays on) and stop before PPO.
2. Evaluate PD, zero-thrust and nominal-hover reward on validation seeds
   2000–2099. Check dense/100 magnitude for PD, not catastrophically failing
   trajectories; stop outside .2–1. This ratio uses the nominal 100-point scale,
   not the variable realized contact score.
3. PPO-only seed 1, 960k transitions, 128 environments x 125 steps, new reward;
   all other PPO settings match the old Kaggle arm. This is a shorter learning
   rate schedule than the old 2M run and is only a bounded pilot.
4. If FINAL validation success is <=10%, start a separate PPO-only seed-1 run
   with 16 environments, same 960k transitions and revised reward. It makes
   480 updates instead of 60. This changes minibatch size, update frequency and
   normalization sampling; do not attribute its result to reward alone.
5. Each arm has snapshots at 240k/480k/720k and a final 960k checkpoint from one
   continuous schedule. Intermediate evaluations use validation seeds only.
   Saved normalization and W3 reloaded actions are checked against memory.
6. After arm decisions, evaluate every attempted final checkpoint with W3 on
   10000–10099. These historically reused seeds are not a pristine holdout.
   W3 reward totals use baseline YAML; revised totals are in training manifests.

Maximum training is 1.92M transitions, across two explicitly separate pilots.
No new full-run waiver, demonstration data, changed initial-state distribution,
or relaxed success thresholds are used. Only GPU 0 is used so the conditional
protocol is the same on one- and two-GPU Kaggle sessions.

The downloadable `europa_ppo_recovery_v1.zip` includes results, per-update JSONL,
CSVs/plots, snapshots, checkpoints with normalization, logs, package/hardware
details, SQLite and trajectories. Failures still package available evidence.
Return that ZIP for the next iteration. No old files are needed to run this
notebook; the old output ZIP would additionally permit checkpoint-specific
replay of the original failures.

## Next decision

If an arm is promising, replicate across seeds 1–3 before claiming reliability.
If both fail, inspect altitude/velocity trajectories and training/eval gaps;
next consider a near-ground curriculum before nominal 2000m fine-tuning, with
original thresholds and full noise restored for final evaluation. BC+PPO is an
explicitly different method with existing evidence in the repository, and PD
remains a separately labeled demo fallback. Curriculum/BC changes are not
silently enabled in this notebook; the next notebook follows returned results.

## Preparation verification

The final regression run passed **131 tests** in 298.99 seconds, with one
existing Orbax sharding-restore warning. It includes the 11 new recovery checks.
Checkpoint tests required running outside the Windows sandbox because its local
socket setup blocked Orbax's asyncio loop; no production checkpoint code was
changed to work around that restriction. The scoped Ruff checks and notebook
code-cell parsing/source-hash checks also passed. No Kaggle notebook or GPU
training was executed during preparation.
