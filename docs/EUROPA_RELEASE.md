# Europa MVP release — V7

Training and reward tuning are frozen. This release uses the existing
validation-selected V7 policy and packages its evidence with the Streamlit
dashboard. No further Kaggle training is required to finish the release.

## Frozen policy and evaluation

The selected model is **V7, seed 15, +80,000 transitions after its parent**.
It achieved 917/1,000 validation landings on seeds 11000–11999, and selection
was frozen before testing. The final 640,000-transition snapshot is not the
release model. The additional-step count does not represent total training
experience or compute across the project.

| Controller | Landed | Crashed | Timeout | Success | Wilson 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected V7 PPO | 1,809 | 184 | 7 | 90.45% | 89.08–91.66% |
| Parent PPO | 1,810 | 186 | 4 | 90.50% | 89.14–91.71% |
| PD baseline | 1,960 | 40 | 0 | 98.00% | 97.29–98.53% |

Each controller was evaluated on Kaggle GPU using seeds 90000–91999,
full-height starts, full sensor noise, and matched simulator settings.
Success requires ground contact with `|vy| < 1.0 m/s` and `|vx| < 0.5 m/s`.
Timeouts are excluded from touchdown-speed summaries.

The observed PPO–PD gap is **7.55 percentage points**, meeting the requested
point-estimate goal of at least 90% and within eight points of PD. The
notebook's stricter 91% flag remains false. Sampling uncertainty does not
establish a population success floor of 90%.

V7 gained 41 landings and lost 42 relative to its parent on the matched test;
there is no demonstrated success improvement. The selected policy has a
0.35% test timeout rate. Late training increased validation timeouts, which
remains an unresolved training behavior rather than a reason to replace the
frozen selection. See [the detailed V7 analysis](PPO_RECOVERY_V7_RESULT.md).

## Dashboard and committed evidence

`app/Home.py` provides five pages:

1. Mission overview with release metrics and an example recorded flight.
2. Flight replay with trajectory, telemetry, and playback controls.
3. Controller comparison on matching recorded seeds.
4. Training and evidence, including validation curves and saved run winners.
5. Live simulation, which generates a new episode and animates the resulting flight.

The current dashboard uses `app/release_data/`:

- `summary.json`: results, validation selection, provenance, and replay hashes.
- `episodes.csv`: outcomes and final-state data for all 6,000 test episodes.
- `trajectories/`: 30 recorded flights, covering ten matched seeds for three controllers.
- `validation.csv`: 188 validation snapshots across the recovery sessions.
- `checkpoint_registry.json`: 22 validation-selected run winners.
- `seed_14_training.csv`, `seed_15_training.csv`, and `seed_16_training.csv`: V7 traces.
- `history.csv`: historical results; different sessions used different test sets.

The replay examples include six selected-PPO landings, three crashes, and one
timeout. They are illustrative rather than a random sample. The older V4
data in `app/demo_data/` remains historical and does not supply current headline
metrics.

## Separate model bundle and live execution

The local `outputs/final_release/` bundle contains:

- `selected/`: the frozen V7 model, manifest, and `sha256.json` file hashes.
- `checkpoints/`: validation-selected models from the historical runs.
- `checkpoint_registry.json` and `freeze.json`: preserved selections and freeze status.
- `evidence/`: original selection, provenance, source, and comparison records.
- `summary.json` and `training_source.zip`: release metrics and V7 training source.

These files are ignored by Git. Back up the bundle separately; a repository
clone contains the dashboard evidence but does not contain the trained model.
To enable live PPO on another machine, restore the complete `selected/`
directory under `outputs/final_release/` and install the locked Python 3.13
environment. The loader verifies file hashes and restores the policy's
observation-normalization statistics. Live PD does not require PPO model files.

Live inference uses the JAX environment through `app/live.py`. Its local CPU
results need not be bitwise identical to the recorded GPU test results.
The live page offers a separate Python physics approximation when JAX is
unavailable and falls back to that mode if inference fails. The approximation
does not execute the PPO checkpoint; its outcomes must not be reported as
fresh evaluations of the frozen policy. Installing JAX alone does not prove
that PPO inference succeeded: the checkpoint and successful execution are
also required.

`scripts/freeze_release.py` created the local package from returned archives
and their original source manifests. It checks selection and trajectory
consistency and does not train. Rebuilding the entire archive requires those
separate inputs; normal dashboard startup does not require rerunning it.

## Verification and CI

The clean-checkout Python 3.13 test run passed **193 tests with one skip**.
The skipped integration test needs the separate frozen model. The earlier
local run with that model present passed all 194 tests. Checks include app
replay and evidence routes, aggregate metrics, physical outcomes, replay
hashes, checkpoint selection, environment behavior, and training utilities.

After narrowing project support to Python 3.13 only, the five notebook
packaging checks passed again. Ruff, lockfile validation, and Git whitespace
checks passed. Dependency versions used by the tested Python 3.13 environment
were preserved.

CI uses Python 3.13 on Ubuntu 24.04 with dependencies installed from `uv.lock`.
The pytest temporary directory now works in a fresh checkout. Updated checkout
and setup-python actions use the Node 24 runtime. CI uses CPU JAX and does not
repeat Kaggle training or the 2,000-seed GPU evaluation. A local test pass is
separate from the GitHub result: check the latest commit's Actions run before
merging.

## Claim boundaries

- Release selection uses validation scores, with the earliest step breaking ties.
  One saved controller is used for every flight; there is no per-flight model switching.
- Later refinements share trained parents. The final +80k snapshot is not an
  independent model trained from scratch in only 80k steps.
- Historical tests use different seed sets and do not establish controlled
  improvements between sessions.
- Recorded GPU execution does not imply sustained high GPU utilization or
  establish a speedup over CPU.
- The PD comparison is an internal baseline, not an external simulator benchmark.
  Its score gap does not establish an out-of-distribution generalization benefit.
- This is a planar point-mass simulator with a flat surface, not a flight-qualified
  controller with full attitude dynamics or terrain avoidance.

## Finish the release

1. Commit and push these documentation updates on `w2-rl`.
2. Open a pull request from `w2-rl` into `main` and review the complete diff.
3. Wait for the PR's Python 3.13 checks to pass, then merge and check the
   resulting `main` run.
4. Start the app from the merged revision and verify replay, comparison,
   evidence, and live execution. Restore the separate model bundle for live
   PPO and check that inference succeeded rather than falling back.
5. If a public demo is required, deploy it separately, verify its URL and
   supported modes, and add the URL to the README. No verified public URL is
   recorded in this release.

Keep the model bundle and original returned archives backed up. Multi-body
support, LLM reward generation, terrain, and full attitude dynamics are later
scope. No further reward experiments are part of this release.
