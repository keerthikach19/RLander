"""LanderEnv — pure-JAX planetary landing environment.

gymnax-style functional API (§5.2, updated contract):
  reset(key, body_params, vehicle_params) -> obs, state
  step(state, action, key) -> obs, next_state, reward_components, done, info

State is a pytree that carries body_params and vehicle_params (copied in at
reset, carried forward unchanged by step). No rng field — the caller passes
a fresh key to each step, used only for sensor noise. The env does NOT
auto-reset; W2's training loop handles resetting finished environments.

done = terminated | truncated.
info["outcome"] is int32: 0 running, 1 landed, 2 crashed, 3 timeout.
info["truncated"] is a bool.
reward = sum(reward_components.values()); W2 does the summing.

Milestone 1: correct signatures, shapes, info keys. Dummy (Euler) dynamics.
Milestone 2: real RK4 dynamics + analytic test passes.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from env.dynamics import step_dynamics_rk4
from env.reward_shaping import profile_shaping
from env.sensors import add_sensor_noise
from env.terrain import terrain_height


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------
class EnvState(NamedTuple):
    """Full simulation state carried between steps.

    Includes body and vehicle parameters as float32 scalars so that one
    compiled step can serve every body via vmap (compile-once rule §5.3).
    No rng field — the caller passes a fresh key to each step.
    """
    # --- dynamic state ---
    x: jnp.ndarray          # horizontal position [m]
    y: jnp.ndarray          # altitude [m]
    vx: jnp.ndarray         # horizontal velocity [m/s]
    vy: jnp.ndarray         # vertical velocity [m/s]
    vx_prev: jnp.ndarray    # previous vx (for accel obs)
    vy_prev: jnp.ndarray    # previous vy (for accel obs)
    mass: jnp.ndarray       # current mass [kg]
    t: jnp.ndarray          # elapsed time [s]
    # --- nondim / episode constants (set at reset, carried unchanged) ---
    L: jnp.ndarray           # length unit = h0 [m]
    T_unit: jnp.ndarray      # time unit = sqrt(L/g) [s]
    U: jnp.ndarray           # velocity unit = sqrt(g*L) [m/s]
    t_max: jnp.ndarray       # episode time limit [s]
    dt: jnp.ndarray          # control timestep [s]
    max_thrust: jnp.ndarray  # max thrust [N]
    # --- body params (float32 scalars) ---
    g: jnp.ndarray
    rho0: jnp.ndarray
    scale_height_m: jnp.ndarray
    # --- vehicle params (float32 scalars) ---
    mass0: jnp.ndarray
    dry_mass: jnp.ndarray
    isp_s: jnp.ndarray
    gimbal_limit_rad: jnp.ndarray
    min_throttle: jnp.ndarray
    cd: jnp.ndarray
    area_m2: jnp.ndarray
    # --- success thresholds (from body config) ---
    vy_max_mps: jnp.ndarray
    vx_max_mps: jnp.ndarray
    # Optional reward settings; numeric data, never part of the dynamics.
    reward_mode: jnp.ndarray  # 0 baseline (default), 1 potential_profile
    reward_gamma: jnp.ndarray
    reward_potential_scale: jnp.ndarray


# ---------------------------------------------------------------------------
# LanderEnv
# ---------------------------------------------------------------------------
class LanderEnv:
    """Pure-JAX planetary landing environment.

    Observation space: (8,)  — see §5.4
    Action space:      (2,)  — [throttle ∈ [-1,1], thrust_angle ∈ [-1,1]]
    """

    observation_size: int = 8
    action_size: int = 2

    @staticmethod
    def reset(
        key: jax.Array,
        body_params: dict[str, Any],
        vehicle_params: dict[str, Any],
    ) -> tuple[jnp.ndarray, EnvState]:
        """Reset the environment and return initial (obs, state).

        Body and vehicle parameters are copied into the state pytree so that
        step() needs only (state, action, key) — no separate param dicts.

        Args:
            key: PRNG key
            body_params: dict with g, rho0, scale_height_m, radius_m,
                         success.vy_max_mps, success.vx_max_mps
            vehicle_params: dict with mass0_kg, dry_mass_kg, twr_max,
                            min_throttle, isp_s, gimbal_limit_rad,
                            cd, area_m2, h0_m, v0_mps

        Returns:
            obs: shape (8,) initial observation
            state: EnvState namedtuple
        """
        key, k_h0, k_v0, k_vx, k_mass, k_thrust, k_grav, k_noise = jax.random.split(key, 8)

        # Extract params as float32
        g_nom = jnp.float32(body_params["g"])
        rho0 = jnp.float32(body_params["rho0"])
        scale_height_m = jnp.float32(body_params["scale_height_m"])

        h0_nom = jnp.float32(vehicle_params["h0_m"])
        v0_nom = jnp.float32(vehicle_params["v0_mps"])
        mass0_nom = jnp.float32(vehicle_params["mass0_kg"])
        dry_mass = jnp.float32(vehicle_params["dry_mass_kg"])
        twr_max = jnp.float32(vehicle_params["twr_max"])
        min_throttle = jnp.float32(vehicle_params["min_throttle"])
        isp_s = jnp.float32(vehicle_params["isp_s"])
        gimbal_limit_rad = jnp.float32(vehicle_params["gimbal_limit_rad"])
        cd = jnp.float32(vehicle_params["cd"])
        area_m2 = jnp.float32(vehicle_params["area_m2"])

        vy_max_mps = jnp.float32(body_params["success"]["vy_max_mps"])
        vx_max_mps = jnp.float32(body_params["success"]["vx_max_mps"])

        # --- Domain randomization (§5.8) ---
        g = g_nom * (1.0 + 0.02 * jax.random.uniform(k_grav, minval=-1.0, maxval=1.0))
        h0 = h0_nom * (1.0 + 0.10 * jax.random.uniform(k_h0, minval=-1.0, maxval=1.0))
        v0 = v0_nom * (1.0 + 0.20 * jax.random.uniform(k_v0, minval=-1.0, maxval=1.0))
        mass0 = mass0_nom * (1.0 + 0.05 * jax.random.uniform(k_mass, minval=-1.0, maxval=1.0))
        thrust_scale = 1.0 + 0.05 * jax.random.uniform(k_thrust, minval=-1.0, maxval=1.0)
        vx0 = 2.0 * jax.random.uniform(k_vx, minval=-1.0, maxval=1.0)  # ±2 m/s

        # --- Nondim units (§5.3) ---
        L = h0
        T_unit = jnp.sqrt(L / g)
        U = jnp.sqrt(g * L)

        # Episode limits
        t_free_fall = jnp.sqrt(2.0 * h0 / g)
        t_max = 4.0 * t_free_fall
        dt = 0.01 * T_unit  # control step in nondim time

        # Max thrust (with randomized thrust scale)
        max_thrust = twr_max * mass0_nom * g_nom * thrust_scale

        # --- Initial state ---
        x0 = jnp.float32(0.0)
        y0 = h0
        vy0 = -v0  # descending = negative vy

        state = EnvState(
            x=x0, y=y0, vx=vx0, vy=vy0,
            vx_prev=vx0, vy_prev=vy0,
            mass=mass0, t=jnp.float32(0.0),
            L=L, T_unit=T_unit, U=U, t_max=t_max, dt=dt,
            max_thrust=max_thrust,
            g=g, rho0=rho0, scale_height_m=scale_height_m,
            mass0=mass0, dry_mass=dry_mass,
            isp_s=isp_s, gimbal_limit_rad=gimbal_limit_rad,
            min_throttle=min_throttle,
            cd=cd, area_m2=area_m2,
            vy_max_mps=vy_max_mps, vx_max_mps=vx_max_mps,
            reward_mode=jnp.float32(vehicle_params.get("reward_mode", 0.0)),
            reward_gamma=jnp.float32(vehicle_params.get("reward_gamma", 0.995)),
            reward_potential_scale=jnp.float32(
                vehicle_params.get("reward_potential_scale", 20.0)
            ),
        )

        obs = _make_obs(state, k_noise)
        return obs, state

    @staticmethod
    def step(
        state: EnvState,
        action: jnp.ndarray,
        key: jax.Array,
    ) -> tuple[jnp.ndarray, EnvState, dict[str, jnp.ndarray], jnp.ndarray, dict[str, jnp.ndarray]]:
        """Take one control step.

        Args:
            state: current EnvState (carries body/vehicle params)
            action: shape (2,) [throttle ∈ [-1,1], thrust_angle ∈ [-1,1]]
            key: PRNG key (used only for sensor noise; physics is deterministic)

        Returns:
            obs: shape (8,) — noisy observation for the policy
            next_state: EnvState — updated state pytree
            reward_components: dict of named reward terms (W2 sums them)
            done: scalar bool — terminated | truncated
            info: dict of ground-truth metrics for eval
        """
        key, k_noise = jax.random.split(key)

        # --- Map actions ---
        raw_throttle = action[0]
        raw_angle = action[1]

        # throttle: [-1,1] -> [min_throttle, 1]
        throttle = state.min_throttle + (1.0 - state.min_throttle) * (raw_throttle + 1.0) / 2.0
        # thrust angle: [-1,1] -> [-gimbal_limit, gimbal_limit]
        thrust_angle = raw_angle * state.gimbal_limit_rad

        # --- Dynamics step (RK4 with 4 substeps per control step) ---
        x_new, y_new, vx_new, vy_new, mass_new = step_dynamics_rk4(
            state.x, state.y, state.vx, state.vy, state.mass,
            throttle, thrust_angle, state.dt,
            state.g, state.max_thrust, state.isp_s,
            state.rho0, state.scale_height_m, state.cd, state.area_m2,
            state.dry_mass,
        )

        t_new = state.t + state.dt

        # --- Termination ---
        terrain_h = terrain_height(x_new)
        contacted = y_new <= terrain_h
        timed_out = t_new >= state.t_max
        terminated = contacted
        truncated = timed_out & ~contacted

        done = terminated | truncated

        # Clamp y at terrain on contact
        y_new = jnp.where(contacted, terrain_h, y_new)

        # --- Info (ground truth for eval, §5.2) ---
        fuel_used = state.mass0 - mass_new
        touchdown_vy = jnp.abs(vy_new)
        touchdown_vx = jnp.abs(vx_new)
        soft_vy = touchdown_vy < state.vy_max_mps
        soft_vx = touchdown_vx < state.vx_max_mps
        landed_soft = contacted & soft_vy & soft_vx
        crashed = contacted & ~landed_soft

        # outcome: int32 — 0 running, 1 landed, 2 crashed, 3 timeout
        outcome = jnp.where(
            landed_soft, jnp.int32(1),
            jnp.where(crashed, jnp.int32(2),
                       jnp.where(truncated, jnp.int32(3), jnp.int32(0)))
        )

        info = {
            "touchdown_v": touchdown_vy,
            "touchdown_vx": touchdown_vx,
            "fuel_used": fuel_used,
            "crashed": crashed.astype(jnp.float32),
            "landed": landed_soft.astype(jnp.float32),
            "truncated": truncated,  # bool, per contract
            "outcome": outcome,      # int32, per contract
        }

        # --- Reward (§5.5, named components; W2 sums them) ---
        v_mag_nondim = jnp.sqrt(vx_new**2 + vy_new**2) / state.U
        h_nondim = jnp.abs(y_new) / state.L
        angle_norm = jnp.abs(thrust_angle) / (state.gimbal_limit_rad + 1e-8)

        w_v, w_h, w_f, w_theta = 1.0, 0.5, 0.1, 0.05

        r_velocity = -w_v * v_mag_nondim
        r_velocity += jnp.where(
            state.reward_mode == 1.0,
            profile_shaping(state, y_new, vx_new, vy_new, terminated),
            0.0,
        )
        r_altitude = -w_h * h_nondim
        r_fuel = -w_f * throttle
        r_angle = -w_theta * angle_norm
        r_crash = -100.0 * crashed.astype(jnp.float32)
        r_land = 100.0 * landed_soft.astype(jnp.float32)

        reward_components = {
            "velocity": r_velocity,
            "altitude": r_altitude,
            "fuel": r_fuel,
            "angle": r_angle,
            "crash": r_crash,
            "landing": r_land,
        }

        # --- Next state (body/vehicle params carried forward unchanged) ---
        next_state = state._replace(
            x=x_new, y=y_new, vx=vx_new, vy=vy_new,
            vx_prev=state.vx, vy_prev=state.vy,
            mass=mass_new, t=t_new,
        )

        obs = _make_obs(next_state, k_noise)

        return obs, next_state, reward_components, done, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_obs(state: EnvState, key: jax.Array) -> jnp.ndarray:
    """Build the (8,) observation vector from state (§5.4).

    [h/L, vx/U, vy/U, ax/g, ay/g, mass/mass0, fuel_frac, t/t_max]
    """
    h_nondim = state.y / state.L
    vx_nondim = state.vx / state.U
    vy_nondim = state.vy / state.U

    # Finite-difference acceleration in units of g
    dt = state.dt
    # Avoid division by zero on first step
    safe_dt = jnp.maximum(dt, 1e-8)
    ax_g = (state.vx - state.vx_prev) / safe_dt / state.g
    ay_g = (state.vy - state.vy_prev) / safe_dt / state.g

    mass_frac = state.mass / state.mass0
    fuel_frac = (state.mass - state.dry_mass) / (state.mass0 - state.dry_mass + 1e-8)
    t_frac = state.t / (state.t_max + 1e-8)

    obs_clean = jnp.array([
        h_nondim, vx_nondim, vy_nondim,
        ax_g, ay_g,
        mass_frac, fuel_frac, t_frac,
    ])

    # Apply sensor noise (§5.4)
    h_physical = state.y
    obs_noisy = add_sensor_noise(
        obs_clean, key,
        g=state.g, L=state.L, U=state.U,
        t_max=state.t_max, mass0=state.mass0,
        h_physical=h_physical,
    )

    return obs_noisy
