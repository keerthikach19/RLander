# RLander: Interface Contracts

## 5. Interface contracts (write these FIRST into `docs/contracts.md`; agents code against them)

### 5.1 Config pytrees
```python
# Hydra YAML â†’ frozen dict
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
    def reset(key, body_params, vehicle_params) -> obs, state
    def step(state, action, key) -> obs, reward_components(dict), done, info
    # state: x, y, vx, vy, vx_prev, vy_prev, mass, t, rng
    # info (clean ground truth, eval only): touchdown_v, touchdown_vx, fuel_used, crashed, landed, truncated, outcome
```
Dynamics (planar point mass): angle Î¸ = a_angle Â· gimbal_limit from vertical; `ax = T sinÎ¸/m âˆ’ Dx/m`, `ay = T cosÎ¸/m âˆ’ g âˆ’ Dy/m`; `á¹ = T/(ispÂ·g0)`; thrust is zero once `mass â‰¤ dry_mass`. Integrate with RK4 (2â€“4 substeps per control step).

Control step: `dt = 0.01 Â· T` (nondim; see 5.3), so every body has â‰ˆ equal episode length. Episode ends on **contact** (`y â‰¤ terrain(x)`, flat 0 in Phase A) = terminated, or at `t_max` = **4 Ã— free-fall time** (`sqrt(2 h0 / g)`; per-body override allowed) = truncated, recorded as outcome `timeout`. (4Ã— â‰ˆ 566 steps at this `dt`.)

### 5.3 Nondimensionalization (fixed)
- Length unit `L = h0_m`; time unit `T = sqrt(L / g)`; velocity unit `U = L / T = sqrt(g L)`
- Observations nondim; actions in [-1, 1]; rewards on nondim quantities; touchdown thresholds in m/s via `info`
- **Compile-once rule:** body and vehicle parameters are a pytree of float32 scalars passed as *data* (so one compiled step serves every body and you can `vmap` across bodies). No Python branching on body name; identical array shapes across bodies

### 5.4 Observation vector (8,), sensor-realistic, no terrain ground truth
`[h/L, vx/U, vy/U, ax/g, ay/g, mass/mass0, fuel_frac, t/t_max]` where `ax, ay` are finite-difference acceleration `(v âˆ’ v_prev)/dt` in units of local g.

Sensor noise is specified in **physical units** (starting values, tunable), applied per step, then nondimensionalized:
- altitude: Ïƒ = 0.5 m + 0.5 % of altitude (lidar-like; small near the ground)
- velocity: Ïƒ = 0.1 m/s per axis
- acceleration: Ïƒ = 0.02 g
Policy sees noisy obs; clean state goes to `info` for metrics only.

### 5.5 Hand-shaped reward (Phase A baseline)
```
r_t = -w_v * |v|/U - w_h * |h/L| - w_f * throttle - w_theta * |angle|/gimbal_limit
      - 100 * crashed + 100 * landed_soft
landed_soft := |vy_touchdown| < vy_max AND |vx_touchdown| < vx_max   (read from body.success; Europa: 1.0 / 0.5 m/s)
weights: w_v=1.0, w_h=0.5, w_f=0.1, w_theta=0.05
```
- The reward fn **returns a dict of named components** (sum = reward); the Reward Lab and the LLM loop need per-term stats
- **Sanity check (W2):** log cumulative shaping over a typical episode; it should be roughly 0.2â€“1Ã— the terminal bonus. If it dwarfs Â±100 the agent can prefer crashing early or hovering: rescale `w_*`

### 5.6 Success metric (per body, in `bodies.yaml` under `success:`)
`landed AND |vy| < vy_max AND |vx| < vx_max` (tilt n/a: point mass). Start every body at Europa's values and justify any change in the YAML comments. The metric lives in `eval/metrics.py` and reads only `info`; the reward may know the thresholds, but the metric never reads the reward.

### 5.7 Module ownership (hard rule: agents may ONLY edit their own directories)
| Path | Owner |
|---|---|
| `configs/`, `env/`, `tests/test_analytic.py`, `tests/test_contracts.py` | W1 (Antigravity) |
| `rl/`, `train.py`, `tests/test_rl.py` | W2 (Astra) |
| `eval/`, `storage/`, `scripts/` | W3 (Antigravity) |
| `app/`, `docs/`, `README.md`, deployment, `.github/` | W4 (Antigravity) |

Cross-module needs â†’ file an issue-style note (GitHub issue). You resolve all cross-module edits personally.

### 5.8 Domain randomization (Phase A; per episode in `reset`)
Starting ranges (tunable): `h0 Â±10 %`, descent speed `v0 Â±20 %`, initial `vx âˆˆ Â±2 m/s`, `mass0 Â±5 %`, thrust scale `Â±5 %`, gravity `Â±2 %`, plus the sensor noise in 5.4. Required test: two different keys give different initial states and the same key reproduces exactly. Only claim "domain randomization" in the resume if this is implemented.

---


