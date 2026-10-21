# RLander

A planar, thrust-vectored landing simulator with a learned PPO policy and a
hand-designed PD controller. The current MVP models Europa, a flat surface,
fuel consumption, sensor noise, and randomized initial conditions.

## Latest result: V7

The validation-selected V7 checkpoint (seed15, +80k) achieved **1809/2000
landings (90.45%)**, versus **1960/2000 (98.00%)** for PD on matching GPU
test seeds90000–91999. Its gap to PD is **7.55 percentage points**.
This meets the user's original point-estimate goal of at least90% and within
eight points of PD. The notebook's stricter91% flag remains false.

The selected checkpoint had **7/2000 timeouts (0.35%)**. Later checkpoints
regressed to31/1000 validation timeouts; use the selected80k snapshot, not
the final640k checkpoint. The parent scored90.50% on the same test, so V7
does not demonstrate improved success over its parent.
See [V7 results and timeout diagnosis](docs/PPO_RECOVERY_V7_RESULT.md).
No additional training has been started.

## Previous result: V6

The V6 PPO-only curriculum session achieved **1799/2000 landings
(89.95%)**, versus **1979/2000 (98.95%)** for PD on matching GPU test seeds
80000–81999. The observed gap is **9.00 percentage points**. This misses the
target of at least 91% and at most eight points below PD, and is one landing
short of an observed 90%. The user subsequently authorized one V7 session
focused on vertical touchdown speed; its notebook is prepared, not executed here.

The checkpoint was selected on validation before testing: seed13 at +160k,
895/1000 validation landings. Its test Wilson 95% interval is **88.55–91.19%**.
See [V6 result and verification](docs/PPO_RECOVERY_V6_RESULT.md).

## Packaged V4 replay snapshot

The existing replay app below still displays the preserved V4 dataset. It has
not been relabelled with later statistics; the V6/V7 checkpoints and evidence
are packaged separately.

| Policy | Landings / episodes | Success | Wilson 95% interval |
| --- | ---: | ---: | ---: |
| Selected v4 PPO | 875 / 1,000 | 87.5% | 85.3–89.4% |
| Parent PPO | 831 / 1,000 | 83.1% | 80.7–85.3% |
| PD controller | 986 / 1,000 | 98.6% | 97.7–99.2% |

All use evaluation seeds 60000–60999 and the same simulator settings. Success
requires contact with vertical speed below 1.0 m/s and horizontal speed below
0.5 m/s. PPO ran on Kaggle GPU; PD evaluation ran locally on CPU, so this is
not a bitwise cross-device equivalence claim.

The selected PPO checkpoint is `margin_seed_5`, at 20,000 additional transitions.
It was selected using **339/400 validation landings before test evaluation**.
The test result is for that one policy, not an average over independent training
seeds: the refinement runs share a trained parent. No PD assistance contributes
to the PPO score. PPO trails the internal PD baseline by 11.1 percentage points.

## Run the demo

The app replays recorded trajectories and needs no JAX, GPU, or checkpoint:

```sh
python -m pip install -r app/requirements.txt
python -m streamlit run app/Home.py
```

In this workspace, the dependencies are already installed:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/Home.py
```

Choose a policy and episode to replay its flight and inspect altitude,
velocity, throttle, and propellant consumption. The comparison table uses
all 1,000 evaluation episodes. The ten replay seeds deliberately include
PPO landings, crashes, and its timeout; they are illustrative, not a random
sample. Every policy has replays for those same seeds.

## Scope and evidence

- The 87.5% demo snapshot is preserved. V6's separately packaged policy reached
  89.95%. V7's selected checkpoint reached90.45% with a7.55-point gap to PD
  on its fresh test; its report and checkpoint are packaged separately.
- [Evaluation summary](app/demo_data/summary.json) contains metrics and provenance.
- [Release notes](docs/EUROPA_RELEASE.md) explain selection, validation, and remaining release work.
- Training and evaluations are stored as checkpoints, SQLite metadata, and Parquet trajectories.
- The simulator is a planar point-mass model, not a full attitude model, terrain
  avoidance system, or flight-qualified controller. No external benchmark parity is claimed.

The local demo is ready for review. A public deployment and CI verification
on the integrated repository remain outstanding. Multi-body support and LLM
reward generation are later work, outside this MVP.
