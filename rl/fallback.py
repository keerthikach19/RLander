"""Observation-only PD fallback tracking a braking descent-rate profile.

Parameters are numeric mappings from the config contract. This controller is
stateless and supports jit/vmap; it never reads privileged environment state.
Real-environment landing performance must be validated after W1 integration.
"""

import jax.numpy as jnp


def pd_action(obs, body_params, vehicle_params):
    g = body_params["g"]
    vehicle = vehicle_params
    length = vehicle["h0_m"]
    velocity_unit = jnp.sqrt(g * length)
    height = jnp.maximum(obs[..., 0] * length, 0.0)
    vx, vy = obs[..., 1] * velocity_unit, obs[..., 2] * velocity_unit
    mass = jnp.maximum(obs[..., 5] * vehicle["mass0_kg"], vehicle["dry_mass_kg"])
    # Approach touchdown at 0.4 m/s; use at most 50% of g for braking.
    target_vy = -jnp.minimum(30.0, jnp.sqrt(0.4**2 + g * height))
    # The target profile decelerates as altitude falls; feed this forward.
    profile_acc = jnp.where(target_vy > -30.0, 0.5 * g * vy / target_vy, 0.0)
    ax = -0.8 * vx
    ay = profile_acc + 1.2 * (target_vy - vy)
    # Compensate modeled atmospheric drag using sensor estimates.
    density = body_params["rho0"] * jnp.exp(-height / body_params["scale_height_m"])
    drag_factor = 0.5 * density * vehicle["cd"] * vehicle["area_m2"]
    speed = jnp.sqrt(vx * vx + vy * vy)
    fx = mass * ax + drag_factor * speed * vx
    fy = jnp.maximum(mass * (ay + g) + drag_factor * speed * vy, 0.0)
    limit = vehicle["gimbal_limit_rad"]
    angle = jnp.clip(jnp.arctan2(fx, fy), -limit, limit)
    # Match vertical acceleration even when the gimbal saturates.
    thrust = fy / jnp.maximum(jnp.cos(angle), 1e-6)
    max_thrust = vehicle["twr_max"] * vehicle["mass0_kg"] * g
    throttle = jnp.clip(thrust / max_thrust, vehicle["min_throttle"], 1.0)
    throttle_action = (
        2 * (throttle - vehicle["min_throttle"]) / (1 - vehicle["min_throttle"]) - 1
    )
    angle_action = jnp.where(limit > 0, angle / jnp.maximum(limit, 1e-6), 0.0)
    return jnp.stack((throttle_action, angle_action), axis=-1)
