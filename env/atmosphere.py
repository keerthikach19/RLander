"""Atmosphere model for RLander.

Contract (§5.1): rho(h) = rho0 * exp(-h / scale_height_m)
Drag = 0.5 * rho * |v|^2 * cd * area_m2, opposing velocity.
Vacuum bodies use rho0=0; never branch on None.
"""

import jax.numpy as jnp


def air_density(h: jnp.ndarray, rho0: float, scale_height_m: float) -> jnp.ndarray:
    """Compute atmospheric density at altitude h.

    Args:
        h: altitude in meters (can be array)
        rho0: surface density [kg/m³] (0.0 for vacuum)
        scale_height_m: scale height [m] (>0 always; 1.0 for vacuum)

    Returns:
        density [kg/m³]
    """
    return rho0 * jnp.exp(-h / scale_height_m)


def drag_force(
    vx: jnp.ndarray,
    vy: jnp.ndarray,
    h: jnp.ndarray,
    rho0: float,
    scale_height_m: float,
    cd: float,
    area_m2: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute drag force components opposing velocity.

    Args:
        vx, vy: velocity components [m/s]
        h: altitude [m]
        rho0: surface density [kg/m³]
        scale_height_m: scale height [m]
        cd: drag coefficient
        area_m2: reference area [m²]

    Returns:
        (drag_x, drag_y): drag force components [N], opposing velocity
    """
    rho = air_density(h, rho0, scale_height_m)
    v_mag = jnp.sqrt(vx**2 + vy**2 + 1e-12)  # small epsilon for numerical safety
    q = 0.5 * rho * v_mag * cd * area_m2  # dynamic pressure * Cd * A
    # Drag opposes velocity: F_drag = -q * v_hat * |v| = -q * v
    drag_x = q * vx  # magnitude, opposing direction applied by caller
    drag_y = q * vy
    return drag_x, drag_y
