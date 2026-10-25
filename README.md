# RLander

A planar, thrust-vectored landing simulator with a learned PPO (Proximal Policy
Optimization) controller and a hand-designed PD (proportional–derivative)
baseline. The Europa MVP includes fuel consumption, sensor noise, randomized
initial conditions, recorded flight replay, and optional live policy inference.

## Gemma 4 Mission Test Lab

The **Gemma Mission Test Lab** (available in the app sidebar) uses **Gemma 4
through the Gemini API** to design controlled landing experiments. You describe
a question in natural language — for example, *"What happens as sideways drift
increases?"* — and Gemma proposes three probe scenarios that vary exactly one
starting-state parameter around a nominal reference. The **frozen PPO** then
flies every case with three matched seeds, and Gemma interprets the measured
outcomes.

Gemma's role is **experiment design and interpretation**, not training. The
policy and its reward function are never changed. Unsupported requests (wind,
terrain obstacles, multiple planets) are rejected explicitly rather than silently
approximated.

A recorded demo is bundled in `app/gemma_demo.json`. It captures a real Gemma
API plan and actual simulator flights and works without an API key. To run your
own experiments, set `GEMINI_API_KEY` in the environment or the Git-ignored `.env`
file. The default model is `gemma-4-26b-a4b-it`.

The first recorded experiment investigated sideways drift (horizontal velocity
`vx`). Gemma selected −1 m/s, 5 m/s, and 12 m/s. At the two larger values (both
outside training starts), PPO crashed on all six flights while PD landed on all
six — a concrete demonstration of a controller limit found through structured
testing.

API reference: [Run Gemma with the Gemini API](https://ai.google.dev/gemma/docs/gemma_cloud_top_level).

## Frozen V7 result

Training is complete. The release uses **seed 15 at +80,000 transitions**,
selected using **917/1,000 validation landings** before the fresh test.
The 80,000 transitions are additional experience after a trained parent;
they are not the total training budget. A transition is one environment step.
One frozen controller is used for every test flight.

| Controller | Landed | Crashed | Timeout | Success | Wilson 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| Selected V7 PPO | 1,809 | 184 | 7 | **90.45%** | 89.08–91.66% |
| Parent PPO | 1,810 | 186 | 4 | 90.50% | 89.14–91.71% |
| PD baseline | 1,960 | 40 | 0 | 98.00% | 97.29–98.53% |

All three used the same 2,000 test seeds (90000–91999), full-height starts,
sensor noise, and physical success rules on Kaggle GPU. Success requires
ground contact with `|vy| < 1.0 m/s` and `|vx| < 0.5 m/s`.
Timeouts are excluded from touchdown-speed summaries.

The selected PPO controller is **7.55 percentage points below PD**, meeting the
requested observed-score goal of at least 90% and within eight points of PD.
The notebook's stricter 91% check remains false. The confidence interval
describes sampling uncertainty; it does not guarantee a population success
rate of at least 90%. V7 did not improve success over its parent on this test.

Later checkpoints had more validation timeouts. The release preserves the
earlier validation-selected checkpoint rather than the final 640,000-step
snapshot. These runs refine shared parents and are not independent training
replications. See the [V7 result and timeout diagnosis](docs/PPO_RECOVERY_V7_RESULT.md).

## Run locally — Python 3.13 only

From the repository root, with Python 3.13 and `uv` installed:

```sh
uv sync --locked --python 3.13
uv run --locked --python 3.13 python -m streamlit run app/Home.py
```

Open the local URL printed by Streamlit. The dashboard contains:

- **Mission overview:** release results and an example flight.
- **Gemma Mission Test Lab:** describe a landing condition; Gemma designs probes, PPO flies them, and Gemma interprets the results. A recorded demo works without an API key.
- **Flight replay:** recorded flights with trajectory and telemetry controls.
- **Compare controllers:** PPO, its parent, and PD on matching test seeds.
- **Training & evidence:** training curves, validation snapshots, and checkpoint registry.
- **Live simulation:** generates a new episode, then animates it with playback controls.

Recorded replay and evidence use the committed `app/release_data/` files.
The ten replay seeds include six PPO landings, three crashes, and one timeout;
all three controllers have recordings for those seeds. These examples are
illustrative. Aggregate results use all 2,000 episodes per controller.

### Live inference and model files

Actual live PPO inference needs JAX and the separately preserved model at
`outputs/final_release/selected/`, including `sha256.json`, `manifest.json`, and
the `checkpoint/` directory. The loader checks the recorded file hashes before
loading the policy. PD inference uses JAX but does not need a PPO checkpoint.

`outputs/` is ignored by Git: cloning the repository does **not** download the
model. Keep the local release bundle backed up; distribute it separately if
others need live PPO inference. On a fresh machine, restore the selected model
at the path above before selecting PPO.

With only the lightweight dependencies in `app/requirements.txt`, recorded
replay and evidence are available without JAX. The live page offers a Python
physics approximation through an explicit opt-in checkbox, disabled by default.
With that option enabled, it can run if actual inference is unavailable or fails.
The approximation has a persistent warning and does not execute the frozen PPO
network. After a flight, execution evidence containing the controller, seed,
device, and checkpoint identity can be downloaded alongside the trajectory.
The episode is calculated before animation; its outcome is revealed only at
the terminal frame. Playback has pause and restart controls. Airborne speed
indicators are not a landing result. Technical execution details remain in
the downloadable evidence after the flight.
Re-running the same seed on the same backend repeats the same flight. Change
the seed to sample another flight. Recorded PPO failures include crash seeds
90040, 90048, and 90053 and timeout seed 90359; CPU results may differ from GPU.

Select **Custom start** to enter horizontal position, altitude, vx and vy in
physical units. Negative vy means descent. The saved controller calculates
actions from the observations of that new flight; no recorded trajectory is
selected by the seed. The seed still controls sensor noise and randomized
vehicle/physical conditions. Initial and requested states are included in the
evidence export. Observation units, fuel and the episode time budget retain
their usual reset definitions. Inputs outside the full-height reset ranges
(altitude 1800-2200 m, vx -2 to +2 m/s, vy -24 to -16 m/s) are exploratory;
the published 90.45% score does not apply to arbitrary starts. Horizontal x
translates the flight over flat terrain; the policy has no position input or
specific landing-pad target. Custom starts require actual inference and cannot
silently become approximation flights.

## Checks

```sh
uv run --locked --python 3.13 ruff check .
uv run --locked --python 3.13 python -m pytest -q
```

GitHub Actions runs these checks on Python 3.13 with the locked dependencies
and CPU JAX. GPU training and the reported test evaluation were performed on
Kaggle; ordinary CI does not repeat them. The actual-checkpoint integration
test skips when the separately distributed model is absent.

## Evidence and release scope

- [Release notes](docs/EUROPA_RELEASE.md): packaged artifacts, verification, and release steps.
- [Release summary](app/release_data/summary.json): metrics, selection, and replay hashes.
- [Checkpoint registry](app/release_data/checkpoint_registry.json): validation-selected run winners.
- [V7 analysis](docs/PPO_RECOVERY_V7_RESULT.md): test counts and timeout diagnosis.

The current dashboard reads V7 data from `app/release_data/`. The older V4
snapshot in `app/demo_data/` is retained as historical evidence.
Checkpoint files and full experiment archives are stored separately from Git.

The model is a planar point mass over a flat surface. Full attitude dynamics,
terrain avoidance, flight qualification, and external simulator benchmark
parity have not been established. The measured PPO–PD gap does not establish
better generalization outside the tested conditions.

The remaining release steps are passing GitHub checks, merging into `main`,
and verifying the app from the merged revision. Public hosting is a separate
step; no deployed URL is recorded here. Further training is not required for
this MVP release.
