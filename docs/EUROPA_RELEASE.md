# Europa MVP freeze — 20 October 2026

Stop reward and policy tuning here. The current result exceeds the original
MVP policy success target; completing the demo and release now has more value
than another refinement round.

## Frozen result

The returned `europa_ppo_recovery_v4.zip` completed successfully and its source
manifest matches the delivered v4 notebook's source manifest.

The candidate selected **before** fresh evaluation was `margin_seed_5`, snapshot
20,000 additional transitions, with 339/400 successes on validation seeds
6000–6399. The selected snapshot has 920,000 transitions along its training
ancestry; that is not the total compute spent across all experiments.

| Policy | Landed | Crashed | Timeout | Success |
| --- | ---: | ---: | ---: | ---: |
| Parent PPO | 831 | 169 | 0 | 83.1% |
| Selected v4 PPO | 875 | 124 | 1 | 87.5% |
| Existing PD controller | 986 | 14 | 0 | 98.6% |

The test seeds are 60000–60999. The v4 control arm selected its unchanged parent,
so it has the same 831/1000 result. Compared with the parent, the selected policy
gained 45 landings and lost one, a net improvement of 4.4 percentage points.
Its Wilson 95% interval is 85.3–89.4%.

PD was evaluated once locally on CPU, without tuning, on the same seeds and
full sensor noise. PPO results came from the user's Kaggle GPU run. Both use
the W3 batched evaluation path and unchanged physical success thresholds.
Floating-point differences between CPU and GPU are possible. PD's Wilson
interval is 97.7–99.2%.

PPO is 11.1 percentage points below this internal baseline, slightly outside
the suggested 5–10 point tolerance. The reliable PD option is included in the
demo separately; it does not inflate the learned policy's reported success.
Comparisons with another simulator need matched dynamics, observations,
noise, start states, and termination rules. No external parity is claimed.

## What is packaged

- `app/Home.py`: replay-only Streamlit app with flight animation, telemetry,
  policy comparison, and scope/evidence text.
- `app/demo_data/summary.json`: full-evaluation aggregate metrics and selection provenance.
- `app/demo_data/trajectories/`: ten matched replay seeds for each of three policies.
  The seeds include six PPO landings, three crashes, and its timeout.
- `outputs/europa_release/selected/`: the validation-selected snapshot and manifest.
- `outputs/europa_release/checkpoint_sha256.json`: hashes of preserved snapshot files.
- `outputs/europa_release/training_source.zip`: the source used for the v4 notebook.
- `outputs/europa_release/pd_baseline/`: the one local PD evaluation, database,
  and all 1,000 trajectories.

`scripts/prepare_europa_demo.py` packages these already measured results. It
does not train, rerun PPO, or choose a policy using the test scores.
All historical notebooks and returned archives are preserved.

## Local verification

- Streamlit AppTest: passed for all three policy choices and their first/last
  replay selections; both plots rendered without application exceptions.
- All 30 replay files checked for finite data, chronological timestamps, and
  terminal states consistent with their reported outcomes.
- Every packaged checkpoint file matched its recorded SHA-256 digest.
- Ruff passed for the new app, packaging script, and app test.

These checks did not rerun PPO training or the Kaggle notebook. Public browser
deployment remains unverified.

## Claim audit

- The v4 execution logs verify compiled GPU execution and GPU output arrays
  for all six PPO refinements. Each training call executed in approximately
  1.85–1.88 seconds; compile/cache time ranged from 7.8 to 69.9 seconds.
  These are individual-call timings, not complete notebook runtime.
- A GPU target label does not establish high utilization. CPU compilation,
  serialization, and short bursts explain why sparse utilization samples can
  miss the work. No sustained utilization or hardware speedup is claimed.
- Reward tuning is now frozen. PPO's success comes from the environment's
  physical thresholds, not reward magnitude.
- Refinement seeds are shared-parent runs, not independent from-scratch
  replications. The original three from-scratch runs failed and must not be
  represented as three successful training replications.
- The 87.5% score is from one frozen selected policy. Do not substitute the best
  test result from another candidate.
- Timeouts are excluded from touchdown velocity summaries.
- The app is recorded replay, not live policy control. No hosted deployment
  or external simulator benchmark has been verified.

## Short path to release

1. Review the local replay app and the frozen results.
2. Integrate the existing environment, training, evaluation, and app changes;
   run the repository's checks on that integrated revision.
3. Deploy the replay app, verify the public URL, and add a screenshot and URL
   to the README.

No further Kaggle training is required for these steps. Multi-body physics,
LLM reward generation, terrain, and full attitude dynamics remain later scope.
If an old notebook's fixed workspace folder already exists, use a fresh Kaggle
session; there is no need for another notebook revision just for that issue.
