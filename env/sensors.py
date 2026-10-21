"""Sensor noise model for RLander.

Contract (§5.4): Sensor noise in physical units, applied per step,
then nondimensionalized:
  - altitude: σ = 0.5 m + 0.5% of altitude (lidar-like; small near ground)
  - velocity: σ = 0.1 m/s per axis
  - acceleration: σ = 0.02 g
"""

import jax
import jax.numpy as jnp


def add_sensor_noise(
    obs_clean: jnp.ndarray,
    key: jax.Array,
    g: float,
    L: float,
    U: float,
    t_max: float,
    mass0: float,
    h_physical: float,
) -> jnp.ndarray:
    """Apply sensor noise to a clean observation vector.

    obs layout (§5.4): [h/L, vx/U, vy/U, ax/g, ay/g, mass/mass0, fuel_frac, t/t_max]

    Noise is specified in physical units and converted to nondim.

    Args:
        obs_clean: shape (8,) clean nondimensionalized observation
        key: PRNG key
        g: body surface gravity [m/s²]
        L: length unit = h0_m [m]
        U: velocity unit = sqrt(g*L) [m/s]
        t_max: episode time limit [s]
        mass0: initial mass [kg]
        h_physical: current physical altitude [m] (for altitude-dependent noise)

    Returns:
        obs_noisy: shape (8,) observation with sensor noise
    """
    keys = jax.random.split(key, 5)

    # altitude noise: σ = 0.5 m + 0.5% of altitude, nondimensionalized by L
    sigma_h = (0.5 + 0.005 * jnp.abs(h_physical)) / L
    noise_h = sigma_h * jax.random.normal(keys[0], shape=())

    # velocity noise: σ = 0.1 m/s per axis, nondimensionalized by U
    sigma_v = 0.1 / U
    noise_vx = sigma_v * jax.random.normal(keys[1], shape=())
    noise_vy = sigma_v * jax.random.normal(keys[2], shape=())

    # acceleration noise: σ = 0.02 g, already in units of g so nondim
    sigma_a = 0.02
    noise_ax = sigma_a * jax.random.normal(keys[3], shape=())
    noise_ay = sigma_a * jax.random.normal(keys[4], shape=())

    # mass/mass0, fuel_frac, t/t_max: no noise (known precisely)
    noise = jnp.array([noise_h, noise_vx, noise_vy, noise_ax, noise_ay, 0.0, 0.0, 0.0])
    return obs_clean + noise
