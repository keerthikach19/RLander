# RLANDER — Master Build Plan (v2)
**Status:** LOCKED (v2, revised Oct 16 2026) | **Owner:** you (integrator) | **Crew:** Antigravity agents (W1, W3, W4) + OpenAI Astra (W2)
**Phase A:** Europa MVP (resume + other submission) | **Phase B:** multi-body + LLM reward loop (Hacktoberfest) | **Phase C:** depth backlog (no deadline)

---

## Revision log (v1 → v2): what changed and why

| # | Change | Why |
|---|---|---|
| 1 | Renamed Planetfall → **RLander** everywhere (repo `rlander`, `rlander.db`, notebook `rlander-training`) | Project name |
| 2 | Python **3.13** (pin via uv; fallback 3.11). Keep code 3.11-compatible | Kaggle runs 3.13.x; matching avoids "works locally, breaks on Kaggle" |
| 3 | Setup steps rewritten for **Windows PowerShell**; project lives **outside OneDrive** | OneDrive locks `.venv` files ("Access is denied") |
| 4 | Local JAX is **CPU only**; all GPU training on Kaggle (`jax[cuda12]`, 2× T4 seen) | No native-Windows CUDA wheels. P100 reference removed |
| 5 | `cleanrl` removed from deps; vendor `ppo_continuous_action.py` as a **reference only** | It's PyTorch; pulling it in drags unneeded deps. Our actions are continuous |
| 6 | **HF Spaces → Streamlit Community Cloud** (primary demo); HF Hub still used for checkpoints | HF Docker + CPU hardware aren't free; free Spaces are Gradio/Static only |
| 7 | Deployed app is **replay-only** (committed demo data, no JAX) | Free hosts: keep the app light; live rollout only locally |
| 8 | Vehicle contract gained `cd`, `area_m2`, `twr_max`, `min_throttle`; Body lost `rotation_rate`; vacuum bodies use `rho0=0, scale_height=1.0` (never `None`/0) | Drag (Phase B) was impossible without `cd`/`A`; `None` or `H=0` → NaN under `jit` |
| 9 | "Planar 3-DoF" corrected to **planar point-mass with thrust vectoring** | State has no pitch/rate; 3-DoF is a Phase C item |
| 10 | `t_max` 2× free-fall → **4×**; control step fixed in *nondim* time (`dt = 0.01·T`) | 2× would truncate a normal landing (and the PD fallback) before touchdown; fixed nondim `dt` gives equal episode length on all bodies |
| 11 | Sensor noise redefined in **physical units** (was σ=0.01·L = 20 m altitude noise) | 20 m altitude noise makes a 1 m/s soft landing nearly impossible |
| 12 | Added §5.8 **domain randomization** spec | v1's resume bullet claimed it but nothing defined it |
| 13 | PPO hyperparameters rescaled (1024 envs, 2e7 steps) | v1's 16 envs × 5e5 steps ≈ 15 updates: badly undertrained |
| 14 | Reward must return **named components**; added magnitude sanity check | Needed for Reward Lab + LLM feedback; per-step terms could swamp ±100 |
| 15 | Former non-goals → **Phase C backlog** (hazard terrain, 6-DoF, MJX check, GRU/asymmetric policies, multi-round LLM loop + random-search control, failure gallery, 5 seeds) | Time is not a constraint; Phase A/B stay lean as planned |
| 16 | Groq model/limits updated; OpenRouter/HF/Cerebras/GitHub Models notes added | Free tiers changed in 2026 |
| 17 | Added Hacktoberfest **compliance gate** (Day 0) | New format: new project built inside a challenge window; verify before Phase B |
| 18 | Titan watchpoint reversed: likely **too easy** (drag-limited), not "too hard" | With placeholder `Cd·A` terminal speed ≈ 9 m/s |
| 19 | DB: `train_bodies_json`, `jax_version`, `term_stats_json` added | OOD runs train on a body *set*; reproducibility; Reward Lab |

---

## 0. Mission statement

Build RLander: an open-source, multi-body propulsive-landing suite. A JAX environment trains PPO landing policies per planetary body; an open-weight LLM (Groq API) optionally writes reward functions, validated and scored against a hand-shaped baseline. Everything is config-driven and replayed in a Streamlit dashboard.

**One-line pitch:** *"A lander that learns to land on Europa today, Mars tomorrow — same code, new YAML row."*

**Why Europa (defensibility):** top astrobiology target; a lander is a studied concept, not a funded mission; Earth–Europa light time means landing must be fully autonomous. Mars is the validation anchor (real landing data exists); Titan and Enceladus force different regimes (dense atmosphere / near-zero gravity).

---

## 1. Locked decisions (do not re-litigate)

| Decision | Choice |
|---|---|
| Scope | Terminal descent + landing only |
| Dynamics (Phase A) | Planar, point-mass, thrust-vectored (x, y). No attitude dynamics until Phase C |
| Phase A body | Europa only |
| Phase B bodies | Mars, Titan, Enceladus (config rows, zero new env code) |
| Phase A RL | JAX PPO, a port of CleanRL's continuous PPO in a single jitted loop (see §7) |
| Reward Phase A | Hand-shaped only |
| LLM reward loop | Phase B: ONE round, K=3 candidates, hard-capped. Multi-round → Phase C |
| Success metric | Separate from reward, fixed per body, frozen seeds |
| License | MIT |
| Demo target | Streamlit Community Cloud (fallback: README GIF) |
| Seeds | 3 training seeds (Phase A/B); test seeds 10,000–10,099 (frozen) |

**Phase C (backlog, no deadline; never blocks A/B):** see §12.

---

## 2. Tech stack (fixed)

- **Python 3.13** (uv-pinned; 3.11 fallback), `uv` for lockfile/env. JAX (CPU locally, `jax[cuda12]` on Kaggle/Colab), Flax, Optax, Orbax
- CleanRL `ppo_continuous_action.py` vendored in `docs/reference/` as the **algorithm reference** (PyTorch; read it, don't run it locally)
- Hydra for experiment configs; `gymnasium` only for `spaces` interop (the env itself is pure JAX)
- SQLite (metadata) + Parquet (trajectories)
- Streamlit + Plotly dashboard
- **LLM access (Oct 2026, verify before relying):**

| Provider | Role | Notes |
|---|---|---|
| Groq | primary | Free plan ≈ 30 req/min, 1,000/day, 8,000 tokens/min on open-weight models (e.g. gpt-oss-120b, gpt-oss-20b, qwen3.8-27b). Check console.groq.com/limits and log the exact model id |
| OpenRouter | fallback | `:free` models ≈ 20 req/min, 50/day |
| Avoid | — | GitHub Models (shut down Jul 30 2026); Cerebras (card required for free credit); Hugging Face inference ($0.10/month credit) |

  Use the OpenAI-compatible SDK with a configurable `base_url`, `tenacity` retries, and a token-bucket limiter sized to the **tokens/min** cap (keep prompts ≲ 3k tokens)
- HF Hub (checkpoints, model cards). Demo: Streamlit Community Cloud (verify current limits at deploy time)
- GitHub Actions: pytest + ruff on every push (CPU JAX; matrix Python 3.11 + 3.13)

---

## 3. Your setup (Windows / PowerShell)

### 3.1 Status (as of Oct 16 2026)
- [x] Kaggle GPU verified: `CudaDevice` ×2, 4096² matmul ≈ 22.7 ms (healthy). The `dopamine-rl`/`gym` pip warning is harmless
- [x] `.env` / `.env.example` created; `.env` is gitignored
- [x] Project moved out of OneDrive into the home directory
- [ ] `uv python pin 3.13` + `uv add` succeeds locally (fallback: pin 3.11)
- [ ] `docs/contracts.md` written and pushed
- [ ] Keys filled in `.env` (Groq, HF token; OpenRouter optional)
- [ ] Hacktoberfest compliance gate passed (§11 Day 0)

### 3.2 Commands
```powershell
# repo (outside OneDrive, e.g. C:\Users\<you>\rlander)
git clone https://github.com/<you>/rlander.git ; cd rlander

# env
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # then open a NEW terminal
uv init --python 3.13 --name rlander        # delete the generated main.py
uv python pin 3.13                          # fallback: uv python pin 3.11
uv add jax flax optax orbax-checkpoint hydra-core gymnasium pandas pyarrow sqlalchemy streamlit plotly openai tenacity python-dotenv pytest ruff

# vendor the CleanRL reference (continuous-action PPO; do NOT uv add cleanrl)
git clone --depth 1 https://github.com/vwxyzjn/cleanrl.git "$env:TEMP\cleanrl"
New-Item -ItemType Directory -Force docs\reference
Copy-Item "$env:TEMP\cleanrl\cleanrl\ppo_continuous_action.py" docs\reference\cleanrl_ppo_continuous.py

# verify
uv run python --version
uv run python -c "import jax, flax, optax, streamlit; print(jax.devices())"   # CpuDevice on Windows is expected
```
If the venv is held by a process: `deactivate`, close editors, `Stop-Process -Name python -Force`, `Remove-Item -Recurse -Force .venv`. Never keep the project in OneDrive.

### 3.3 Secrets
```powershell
@"
GROQ_API_KEY=
OPENROUTER_API_KEY=
HF_TOKEN=
"@ | Set-Content .env.example
Copy-Item .env.example .env        # fill real values in .env ONLY; no quotes, no spaces around =
notepad .env
Select-String -Path .gitignore -Pattern "^\.env"   # must match
```
Check loading without printing the key: `uv run python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(bool(os.getenv('GROQ_API_KEY')))"`.

### 3.4 Kaggle run recipe (training happens here)
Settings: Accelerator = GPU, Internet = on. Put `GROQ_API_KEY` / `HF_TOKEN` in **Add-ons → Secrets** (never paste keys into cells).
```python
!git clone https://github.com/<you>/rlander.git
%cd rlander
!pip install -q -U "jax[cuda12]"         # pin the same jax version as uv.lock once locked
!pip install -q flax optax orbax-checkpoint hydra-core pandas pyarrow sqlalchemy python-dotenv
import jax; print(jax.devices())          # must show CudaDevice
!python train.py +body=europa +vehicle=europa_lander +seed=1
```
Two GPUs are visible: run two seeds in parallel as separate processes with `CUDA_VISIBLE_DEVICES=0` and `=1`. Download `ckpts/` and results, or push to HF Hub with `HF_TOKEN`. Record `jax.__version__` in every run. Re-check the GPU-hours quota on your profile; Colab (T4) is the backup.

### 3.5 Hosting the demo
- **Primary:** Streamlit Community Cloud from the public GitHub repo (main file `app/Home.py`; light `app/requirements.txt`: streamlit, plotly, pandas, pyarrow; **no JAX**).
- **Alternatives:** Gradio Space (needs a Gradio rewrite) or Static Space (precomputed JSON + Plotly.js, no live sim).
- HF Docker Spaces and CPU Basic are not free: skip. The `HF_TOKEN` is still used for Hub checkpoints.

### 3.6 Agent tooling
Antigravity workspaces ×3 and an Astra session. Give each agent: this PLAN.md + its workstream spec (§9) + the interface contracts (§5).

---

## 4. Architecture

```
configs/ ──▶ env/ (JAX) ──▶ rl/ (PPO) ──▶ eval/ (frozen seeds) ──▶ storage/ ──▶ app/ (Streamlit)
                ▲                │              │
        reward/ (Phase B)       │              └─▶ trajectories.parquet + SQLite
                │               └─▶ checkpoints (Orbax) ─▶ HF Hub
        LLM loop (Groq) ◀───────┘   logs (prompts/candidates/scores)
```

Data flow per run: load body+vehicle config → nondimensionalize → vectorized env (`vmap`, domain randomization) → train PPO → evaluate on frozen seeds with the success metric (NOT the reward) → write run + evals + episodes to SQLite/Parquet → export a small demo snapshot → dashboard reads it.

---

## 5. Interface contracts (write these FIRST into `docs/contracts.md`; agents code against them)

### 5.1 Config pytrees
```python
# Hydra YAML → frozen dict
Body:    name, g (m/s^2), radius_m (optional g(h) only),
         rho0 (kg/m^3; 0.0 for vacuum), scale_height_m (>0 ALWAYS; use 1.0 for vacuum),
         success: {vy_max_mps, vx_max_mps}
Vehicle: name, mass0_kg, dry_mass_kg, twr_max, min_throttle (0.0 in Phase A),
         isp_s, gimbal_limit_rad, cd, area_m2, h0_m, v0_mps (initial descent speed)
         derived: max_thrust_N = twr_max * mass0_kg * body.g
```
Atmosphere: `rho(h) = rho0 * exp(-h / scale_height_m)`; drag `0.5 * rho * |v|^2 * cd * area_m2` opposing velocity. Vacuum bodies are just `rho0 = 0`; never branch on `None`.

### 5.2 Environment
```python
class LanderEnv:  # pure JAX, gymnax-style functional API
    observation_space: (8,)  # see 5.4
    action_space: (2,)       # [throttle in [-1,1] -> mapped to [min_throttle, 1], thrust_angle in [-1,1]]
    reset(key, body_params, vehicle_params) -> obs, state
    step(state, action, key) -> obs, next_state, reward_components, done, info
    # Three clarifications go with it:Parameters live in state. state is a pytree holding x, y, vx, vy, vx_prev, vy_prev, mass, t plus body_params and vehicle_params. reset copies them in, and step carries them forward unchanged. That’s how one compiled step can be vmap’d across different bodies, per the compile-once rule.
    #No rng in state. The caller passes a fresh key to every step, used only for sensor noise. The physics is deterministic given the state. This replaces the old state.rng field, which was redundant.
    #No auto-reset inside the env. done = terminated | truncated, and W2’s training loop handles resetting finished environments, for example with jnp.where over a fresh reset.

    #Two smaller points:

    #reward = sum(reward_components.values()), and W2 does the summing.info["outcome"] is an integer code, because strings can’t go through jit: 0 running, 1 landed, 2 crashed, 3 timeout. info["truncated"] is a bool. The DB maps the codes to text.
```
Dynamics (planar point mass): angle θ = a_angle · gimbal_limit from vertical; `ax = T sinθ/m − Dx/m`, `ay = T cosθ/m − g − Dy/m`; `ṁ = T/(isp·g0)`; thrust is zero once `mass ≤ dry_mass`. Integrate with RK4 (2–4 substeps per control step).

Control step: `dt = 0.01 · T` (nondim; see 5.3), so every body has ≈ equal episode length. Episode ends on **contact** (`y ≤ terrain(x)`, flat 0 in Phase A) = terminated, or at `t_max` = **4 × free-fall time** (`sqrt(2 h0 / g)`; per-body override allowed) = truncated, recorded as outcome `timeout`. (4× ≈ 566 steps at this `dt`.)

### 5.3 Nondimensionalization (fixed)
- Length unit `L = h0_m`; time unit `T = sqrt(L / g)`; velocity unit `U = L / T = sqrt(g L)`
- Observations nondim; actions in [-1, 1]; rewards on nondim quantities; touchdown thresholds in m/s via `info`
- **Compile-once rule:** body and vehicle parameters are a pytree of float32 scalars passed as *data* (so one compiled step serves every body and you can `vmap` across bodies). No Python branching on body name; identical array shapes across bodies

### 5.4 Observation vector (8,), sensor-realistic, no terrain ground truth
`[h/L, vx/U, vy/U, ax/g, ay/g, mass/mass0, fuel_frac, t/t_max]` where `ax, ay` are finite-difference acceleration `(v − v_prev)/dt` in units of local g.

Sensor noise is specified in **physical units** (starting values, tunable), applied per step, then nondimensionalized:
- altitude: σ = 0.5 m + 0.5 % of altitude (lidar-like; small near the ground)
- velocity: σ = 0.1 m/s per axis
- acceleration: σ = 0.02 g
Policy sees noisy obs; clean state goes to `info` for metrics only.

### 5.5 Hand-shaped reward (Phase A baseline)
```
r_t = -w_v * |v|/U - w_h * |h/L| - w_f * throttle - w_theta * |angle|/gimbal_limit
      - 100 * crashed + 100 * landed_soft
landed_soft := |vy_touchdown| < vy_max AND |vx_touchdown| < vx_max   (read from body.success; Europa: 1.0 / 0.5 m/s)
weights: w_v=1.0, w_h=0.5, w_f=0.1, w_theta=0.05
```
- The reward fn **returns a dict of named components** (sum = reward); the Reward Lab and the LLM loop need per-term stats
- **Sanity check (W2):** log cumulative shaping over a typical episode; it should be roughly 0.2–1× the terminal bonus. If it dwarfs ±100 the agent can prefer crashing early or hovering: rescale `w_*`

### 5.5.1 Optional descent-profile potential (approved experiment, Oct 16 2026)

Scoped integrator approval permits W2 to edit `env/` and this plan for this
experiment, with a separate commit and W1 handoff. All other §5.7 ownership
remains unchanged. The §5.5 reward remains the **default** (`baseline`) and the
ablation control; this is not a change to success thresholds or physics.

Select `+reward.mode=potential_profile` in `train.py`; optionally set
`+reward.potential_scale=20` (initial experimental assumption, not a tuned fact).
Only the named `velocity` component changes, by **adding**:

```
v_ref(h) = -min(30 m/s, sqrt((0.4 m/s)^2 + g * max(h, 0)))
Phi(s) = -k * sqrt(vx^2 + (vy - v_ref(h))^2) / U
F(s, s') = gamma * Phi(s') - Phi(s)
```

The baseline speed cost stays in that component. `gamma` must equal PPO's
discount. Contact is absorbing with `Phi=0`; time-limit truncation retains the
final state's potential, consistent with PPO bootstrapping before caller reset.
This preserves the underlying discounted objective under the usual potential
shaping assumptions; it does not guarantee finite-sample PPO convergence.
Undiscounted reward totals across modes are not an invariant comparison.
Altitude/fuel/angle costs, terminal ±100, dynamics, noise, and termination are
unchanged. The nonzero 0.4 m/s reference remains below Europa's fixed threshold.

Optional numeric vehicle parameters carried unchanged in state are
`reward_mode` (float32: 0 baseline, 1 potential_profile), `reward_gamma`
(default 0.995), and `reward_potential_scale` (default 20). Omitting them preserves
baseline behavior. `train.py` maps the string selector to these data fields and
records them in the manifest; no strings or Python body-name branches enter jit.
W4 should mirror this approved addendum into `docs/contracts.md` when integrated.

Feasibility: on the uncapped reference, `dv_ref/dt = g/2`; on its capped segment
it is zero. At maximum randomized Europa gravity this needs 0.671 m/s². Minimum
randomized thrust, maximum initial mass, and full gimbal tilt still provide
about 2.106 m/s² of net vertical acceleration. This checks tracking the profile
with fuel available, not recovery from every possible off-profile state.

Validate on separate seeds (e.g. `+evaluation.seed_start=2000
+evaluation.episodes=100`) in bounded smoke runs. Report soft landings and
timeouts; the §7 full-run gate still applies. The frozen test set already used
in debugging must not be represented as a fresh holdout. For Phase B, treat this
as an explicitly hand-designed baseline candidate alongside §5.5; do not credit
the profile to Groq or silently include it in an LLM candidate's contribution.

### 5.6 Success metric (per body, in `bodies.yaml` under `success:`)
`landed AND |vy| < vy_max AND |vx| < vx_max` (tilt n/a: point mass). Start every body at Europa's values and justify any change in the YAML comments. The metric lives in `eval/metrics.py` and reads only `info`; the reward may know the thresholds, but the metric never reads the reward.

### 5.7 Module ownership (hard rule: agents may ONLY edit their own directories)
| Path | Owner |
|---|---|
| `configs/`, `env/`, `tests/test_analytic.py`, `tests/test_contracts.py` | W1 (Antigravity) |
| `rl/`, `train.py`, `tests/test_rl.py` | W2 (Astra) |
| `eval/`, `storage/`, `scripts/` | W3 (Antigravity) |
| `app/`, `docs/`, `README.md`, deployment, `.github/` | W4 (Antigravity) |

Cross-module needs → file an issue-style note (GitHub issue). You resolve all cross-module edits personally.

### 5.8 Domain randomization (Phase A; per episode in `reset`)
Starting ranges (tunable): `h0 ±10 %`, descent speed `v0 ±20 %`, initial `vx ∈ ±2 m/s`, `mass0 ±5 %`, thrust scale `±5 %`, gravity `±2 %`, plus the sensor noise in 5.4. Required test: two different keys give different initial states and the same key reproduces exactly. Only claim "domain randomization" in the resume if this is implemented.

---

## 6. Body constants (initial values; W1 verifies each against NASA/JPL fact sheets and records the source URL per constant in YAML comments)

| Body | g m/s² | radius km | ρ₀ kg/m³ | scale height km | atmosphere |
|---|---|---|---|---|---|
| Europa | 1.315 | 1560.8 | 0.0 | 1.0 (dummy) | none |
| Mars | 3.71 | 3389.5 | 0.020 | 11.1 | thin CO₂ |
| Titan | 1.352 | 2574.7 | 5.4 | ~20 (verify; effective near-surface value) | thick N₂ |
| Enceladus | 0.113 | 252.1 | 0.0 | 1.0 (dummy) | none |

Phase A vehicle (`europa_lander`): mass0 800 kg, dry 320 kg, h0 2000 m, v0 ≈ 20 m/s downward, `twr_max` 3.0 (thrust = 3·mass0·g_Europa), isp 300 s, gimbal ±15°, `min_throttle` 0.0, `cd` 1.2 and `area_m2` 4.0 (unused in vacuum).
Phase B: one vehicle per body, `twr_max` ∈ [2, 5] chosen so hover is possible with margin. These `cd`/area values are modeling assumptions: state them in the write-up.

**Titan check (do early):** with `cd·A = 4.8 m²` and 800 kg, terminal velocity ≈ 9 m/s with *no thrust*, so Titan may be solvable without control. Treat it as a "passive-descent regime": report it honestly, or shrink `area_m2` so thrust matters.

---

## 7. PPO + training spec

Phase A (port of CleanRL's continuous PPO to JAX, PureJaxRL-style single jitted loop):
- Network: MLP 64-64, tanh, separate actor/critic, orthogonal init, Gaussian policy with state-independent log-std
- Starting hyperparameters (tune after the smoke run): `num_envs` 1024, `num_steps` 128, `total_steps` 2e7 (target: minutes on a T4; verify), γ 0.995, GAE λ 0.95, clip 0.2, 8 minibatches × 4 epochs, lr 3e-4 annealed, `ent_coef` 0.01, `vf_coef` 0.5, `max_grad_norm` 0.5, observation normalization
- 3 training seeds: 1, 2, 3. Record config + seed + `jax.__version__` in DB and JSONL logs
- Vectorization via `jax.vmap` over envs inside one `jit`-compiled train function. Contract = checkpoint + metrics, not the file
- Fallback policy (build regardless, ~20 lines): hand-tuned PD controller that tracks a descent-rate profile. **Verify it lands within `t_max` on every body** (gravity and drag differ a lot)
- Smoke run first: 5e4–1e6 steps must beat the fallback's trend (>10 % soft landings) before the full run

Phase C: validate the JAX PPO against the PyTorch CleanRL reference (on Kaggle, where torch is preinstalled) on a simple benchmark.

---

## 8. Storage schema (SQLite, `storage/schema.sql`)

- `bodies(name, config_json, source_urls_json)`; `vehicles(name, config_json)`
- `runs(id, train_bodies_json, vehicle, policy_type, reward_id, seed, config_hash, jax_version, checkpoint_path, created_at)`
- `evals(run_id, eval_body, n_episodes, success_rate, vy_mean, vy_std, vx_mean, vx_std, fuel_mean, ci95_low, ci95_high)`; CI = Wilson 95 %
- `episodes(run_id, seed, outcome, parquet_path)`
- `reward_candidates(id, round, provider, model, prompt, code, validation_status, score, term_stats_json, created_at)`: Phase B
- Trajectories: `data/trajectories/{run_id}/{seed}.parquet` (t, x, y, vx, vy, mass, throttle, angle, reward)
- **Demo snapshot:** `scripts/export_demo_data.py` writes a small curated subset (best run per body, ~10 episodes each) to `app/demo_data/` (committed). The deployed app reads only that; `data/` and `ckpts/` stay gitignored

---

## 9. Workstream specs (paste each into its agent, verbatim, with §5)

### W1: Config + Environment (Antigravity, Workspace 1)
Deliverables: `configs/bodies.yaml`, `configs/vehicles.yaml`, `configs/experiments/*.yaml`; `env/{dynamics.py, atmosphere.py, sensors.py, terrain.py (Phase B stub), lander_env.py}`; `tests/test_analytic.py`, `tests/test_contracts.py`.
Requirements:
- Analytic tests MUST pass: (a) free fall from h0 with zero thrust reproduces y(t) = h0 − ½gt² within 1 % over 10 s; (b) terminal velocity with atmosphere reproduces v_t = sqrt(2mg/(ρ·Cd·A)) (Phase B bodies); (c) fuel depletes at ṁ = throttle·T_max/(isp·g0); (d) energy check (thrust work + PE + KE consistent)
- Contract tests: obs shape (8,), action shape (2,), `info` keys, reward returns named components; **compile-once** (same jitted step runs Europa and Mars params without recompiling); randomization differs by key and reproduces by key; no NaN for `rho0 = 0`
- Nondim per §5.3, sensor noise per §5.4, flat terrain in Phase A
- Verify all body constants against JPL fact sheets; paste source URLs into YAML comments
- `uv run pytest tests/ -q` must be green before any "done" claim

### W2: RL (Astra)
Deliverables: `rl/{ppo.py, networks.py, vec_env.py, fallback.py}`, `train.py` (Hydra: `+body=europa +vehicle=europa_lander +seed=1`), `tests/test_rl.py`.
Requirements:
- `train.py` writes an Orbax checkpoint to `ckpts/{run_id}/`, appends a row to `runs`, prints an eval summary
- PD fallback in `rl/fallback.py`; test that it lands within `t_max` on every body
- Do not touch `env/`; import only through the §5.2/§5.4 contract
- Validate: smoke run reaches >10 % soft landings before the full run; log reward components and the §5.5 magnitude check
- Must run on Kaggle per §3.4 (parallel seeds on 2 GPUs)

### W3: Eval + Storage (Antigravity, Workspace 2)
Deliverables: `eval/metrics.py`, `eval/run_eval.py`, `storage/{db.py, schema.sql, parquet_io.py}`, `scripts/export_demo_data.py`.
Requirements:
- Frozen test seeds 10,000–10,099 (100 episodes) per run; success metric from `bodies.yaml` `success:`, NEVER the reward
- Wilson 95 % CI on success rate; touchdown stats + fuel; episodes to Parquet, metadata to SQLite
- `run_eval.py --run-id X` reruns deterministically from a checkpoint; supports `--eval-body` different from the training bodies (for OOD)

### W4: App + Docs + Deploy (Antigravity, Workspace 3)
Deliverables: `app/{Home.py, pages/1_Simulator.py, pages/2_Compare.py, pages/3_Results.py}` (+ `pages/4_Reward_Lab.py` in Phase B), `README.md`, `.github/workflows/ci.yml`, `app/requirements.txt`.
Requirements:
- Simulator page: pick body/policy/reward → replay an episode → Plotly subplots (altitude, vertical/horizontal velocity, throttle, fuel vs t) + animated lander replay. **Deployed app is replay-only** from `app/demo_data/`; a live rollout appears only when run locally with JAX installed (`RLANDER_LIVE=1`)
- Compare page: side-by-side runs (PD vs PPO; Phase B: hand vs LLM reward)
- Results page: runs/evals tables with CI; success matrix body × policy
- CI: `uv run pytest` + `uv run ruff check` on push, Python 3.11 and 3.13
- Deploy Phase A to Streamlit Community Cloud; record the live URL in the README header

---

## 10. Phase A: execution plan (gate-based; hours are pacing, not deadlines)

You are the integrator. Agents never merge; you merge. Every agent runs pytest before claiming done.

| Hour | You | Agents (parallel) |
|---|---|---|
| 0–0:45 | §3 setup (mostly done: finish the unchecked boxes) | none |
| 0:45–1 | Write §5 contracts into `docs/contracts.md`, push | Spawn W1, W2, W3 (W4 spawns at hour 4) |
| 1–4 | Review W1 env as it lands; personally verify analytic tests; write `europa_lander` numbers into `vehicles.yaml` if W1 is slow | W1: env + configs + tests. W2: PPO + PD fallback. W3: schema + metrics |
| 4–4:30 | **Gate 1:** `pytest` green; contract conformance (obs 8, action 2, info keys, compile-once) | W4 spawns: Streamlit app + README + CI |
| 4:30–5 | Kick off 3-seed training on Kaggle (2 seeds in parallel, one per GPU) | All continue |
| 5–8 | Review W4 app, draft README bullets, prepare eval runbook | W2 monitors training; W3 dry-runs eval on a random checkpoint |
| 8–10 | **Gate 2:** training done → `run_eval.py` on all 3 seeds × 100 test episodes. If success < 30 % by hour 9: tune reward weights (one change, one retrain) or lead with the PD fallback | W4: screenshots/GIF, `export_demo_data.py` run |
| 10–12 | Deploy to Streamlit Community Cloud; verify live | W4: README polish |
| 12–13 | Fill resume numbers (real only), final commit + push | All: write `docs/agent_notes.md` |
| 13–14 | Buffer | none |

**Phase A Definition of Done:** GitHub repo with CI green; README with GIF/screenshots + live demo link; ≥1 PPO run with ≥50 % success on 100 frozen-seed episodes (or PD fallback ≥80 % + PPO best-effort shown honestly); 3 training seeds logged; resume bullets written. Re-evaluate the 50 % bar after the first full run, since the noise levels in §5.4 set a ceiling.

**Resume bullets (real numbers only; claim domain randomization only if §5.8 is implemented):**
- Built a JAX-based planetary landing simulator (planar, thrust-vectored) with sensor-noised, nondimensionalized observations and domain randomization; trained a PPO agent (JAX port of CleanRL's) to autonomously land on Europa: X % success over 100 frozen-seed test episodes across 3 training seeds
- Implemented analytic validation (free fall, fuel flow, terminal velocity) as CI gates for a physics environment
- Shipped an interactive Streamlit dashboard with animated trajectory replay; deployed a live demo

---

## 11. Phase B: execution plan (Hacktoberfest)

Goal: bodies are rows, and an open-weight LLM wrote a reward that beats or co-matches a human. Cut list stays in §12; do not add scope.

| Day | Deliverable | Owner |
|---|---|---|
| 0 | **Compliance gate:** read dev.to/challenges/hf26 and the Hacktoberfest rules. Confirm (a) which challenge window and deadline you target (as I read it: new windows start Mondays, Oct 19 and Oct 26 remain), (b) the rule that submissions are **new projects built inside the window** vs your existing Phase A repo (consider a fresh repo or scoping the submission to Phase B work), (c) that a hosted open-weight API counts, (d) whether a live demo is required | you |
| 1 | Mars row: constants (verify JPL), terminal-velocity test green, vehicle config, train PPO, ≥40 % success | W1 + W2; you verify |
| 2 | Titan row + same. **Watchpoint:** likely *too easy* (drag-limited, ~9 m/s terminal). Decide: report it as a passive-descent regime, or shrink `area_m2`. If success < 20 % anyway after 2 reward tweaks, set per-body thresholds in config and document it | W1 + W2 |
| 3 | Enceladus row (g = 0.113; verify nondim handles it and the 4× `t_max`). All four bodies pass terminal-velocity/free-fall checks | W1 |
| 4 | OOD: train on {Europa, Mars}, zero-shot eval on Titan and Enceladus. Log the transfer success matrix | W2 + W3 |
| 5 | Results page: body × policy success matrix with CIs; Compare page upgraded | W4 + W3 |
| 6–7 | **LLM reward loop (hard cap):** K=3 candidates from a Groq open-weight model (log exact model id). Pipeline: prompt (`reward/prompts/`) → generate `reward_fn` code → validator (AST banlist: imports beyond math/jax/numpy, file I/O, network; jit dry-run; NaN check; 60 s subprocess timeout with no network) → short train (5e4–1e6 steps, set by measured speed) → score = frozen-seed success → pick best → 1 full train. Reward must return named components. Pace calls to the tokens/min cap. Log EVERY prompt/response/candidate/score to `reward_candidates` | W2 (pipeline) + W3 (storage); you write prompts |
| 8 | Reward Lab page: code diff per candidate, per-term stats, scores | W4 |
| 9 | Buffer / fix-forward. If the LLM loop failed entirely: ship hand-shaped results + full LLM logs (prompts, validator rejections); transparency is the write-up. If time: equal-compute random search over hand-term weights as a control | All |
| 10 | Write-up (article + model card per body), push checkpoints to HF Hub, update the demo, license headers, final CI green | W4; you review |

**Claim discipline:** without the random-search control, say "an open-weight LLM produced a reward comparable to the hand-shaped one", not "LLMs design better rewards".

**Phase B Definition of Done:** 4 bodies each with ≥40 % frozen-seed success (Titan may carry a documented exception); OOD transfer matrix published; 1 LLM-reward round logged end-to-end (success or documented failure); live demo updated; write-up published; compliance gate (Day 0) answered in writing in the README.

---

## 12. Phase C: depth backlog (no deadline; pick in this order; never blocks A/B)

1. **Hazard terrain + landing zone:** procedural heightmap (ridges, slopes, spikes), local height-patch observation, success requires landing in a safe cell. *This is the Europa-specific story: terrain at lander scale is unknown. Until it ships, do not pitch Phase A as hazard avoidance.*
2. **Sensor realism:** radiation-style dropouts/glitches, actuator latency, minimum throttle > 0
3. **Policy comparison:** blind MLP vs GRU (infers gravity/drag from history) vs asymmetric actor-critic (critic sees privileged truth) vs an oracle given the planet parameters; held-out-body zero-shot tests
4. **LLM loop depth:** multi-round with elitism, equal-compute random-search control, small vs large open-weight model, one universal reward vs per-body rewards
5. **Evaluation depth:** 5 seeds + CIs, per-term reward ablations, reward-hacking failure gallery (video + which term was exploited)
6. **Fidelity:** attitude dynamics (planar 3-DoF, then 6-DoF), MJX/MuJoCo sim-to-sim touchdown check with four legs, optional tethered sky-crane stage
7. **Internals:** validate the JAX PPO against the PyTorch CleanRL reference; PPO ablations
8. **UI:** FastAPI + React/Three.js 3D replay (optional)

---

## 13. Risks & kill-switches

| Risk | Signal | Kill-switch |
|---|---|---|
| Training doesn't converge | Hour 9 of Phase A, success < 30 % | PD fallback leads the demo; PPO shown as best effort with training curves |
| Noise ceiling too high | Even a good policy plateaus ~50 % | Lower §5.4 σ values (document it); check obs with the policy given clean state as a diagnostic |
| Titan too easy / reward mis-scaled | Day 2 | See Day 2 watchpoint; per-body thresholds in config |
| Groq quota / tokens-per-min hit | HTTP 429 | `tenacity` backoff + token bucket; loop resumes from `reward_candidates`; OpenRouter fallback; Day 9 transparency fallback |
| Parallel agents collide | Merge conflicts | §5.7 ownership absolute; you merge |
| Free GPU quota gone | Kaggle/Colab out of hours | CPU JAX for evals only; halve `total_steps` |
| JIT breaks on LLM reward code | Validator dry-run fails | Expected; the validator's job. 3 strikes → loop ends, log everything |
| Local env breaks | "Access is denied", wheel missing | Project outside OneDrive; `uv python pin 3.11` fallback |
| Free tiers change | Quota or model gone | Re-check limits at Day 0 and before the loop; log the model id used |
| Hacktoberfest rules mismatch | Day 0 answers | Fresh repo / re-scope the submission; keep Phase A as portfolio-only |

---

## 14. Repo layout (final)

```
rlander/
  PLAN.md                    <- this file
  .python-version  pyproject.toml  uv.lock  .env.example
  configs/   bodies.yaml vehicles.yaml experiments/
  env/       dynamics.py atmosphere.py terrain.py sensors.py lander_env.py
  rl/        ppo.py networks.py vec_env.py fallback.py
  reward/    hand_shaped.py llm_loop.py validator.py prompts/
  eval/      metrics.py run_eval.py ood.py
  storage/   db.py schema.sql parquet_io.py
  scripts/   export_demo_data.py
  app/       Home.py pages/1_Simulator.py 2_Compare.py 3_Results.py 4_Reward_Lab.py
             requirements.txt  demo_data/   (committed, small)
  tests/     test_analytic.py test_contracts.py test_rl.py
  data/      rlander.db trajectories/      (gitignored)
  ckpts/                                   (gitignored)
  docs/      contracts.md agent_notes.md article.md model_cards/ reference/cleanrl_ppo_continuous.py
  .github/workflows/ci.yml
```

## 15. Operating rules for the agent crew

1. Contract-first: code against §5, never against another module's implementation.
2. Every "done" claim = `pytest` green + walkthrough artifact (Antigravity) or explicit summary of files changed (Astra).
3. You, the human, are the only merger. Gates: Gate 1 (hour 4:30) and Gate 2 (hour 8) in Phase A; nightly in Phase B.
4. Real numbers only, everywhere. A 45 % success rate published beats a fake 95 % discovered in an interview.
5. Any agent that needs to touch another workstream's file STOPS and files a note to you instead.
6. Assumed numbers (noise σ, `cd`, area, thresholds, randomization ranges) are labeled as assumptions in YAML comments and in the write-up; verified constants cite their source.
