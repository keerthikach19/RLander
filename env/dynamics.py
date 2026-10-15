"""Dynamics module for RLander.

Planar point-mass dynamics with thrust vectoring (§5.2).
Milestone 1: dummy dynamics (Euler integration, basic structure).
Milestone 2: RK4 integration with 2-4 substeps.

Equations of motion (§5.2):
  θ = a_angle · gimbal_limit  (thrust angle from vertical)
  ax = T·sin(θ)/m − Dx/m
  ay = T·cos(θ)/m − g − Dy/m
  ṁ  = T / (isp · g0)
  Thrust is zero once mass ≤ dry_mass.
"""

import jax.numpy as jnp

# Standard gravity for fuel flow calculation
G0 = 9.80665  # m/s², standard gravity


def compute_acceleration(
    x: jnp.ndarray,
    y: jnp.ndarray,
    vx: jnp.ndarray,
    vy: jnp.ndarray,
    mass: jnp.ndarray,
    throttle: jnp.ndarray,
    thrust_angle: jnp.ndarray,
    g: float,
    max_thrust: float,
    isp_s: float,
    rho0: float,
    scale_height_m: float,
    cd: float,
    area_m2: float,
    dry_mass: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute accelerations and mass flow rate.

    Args:
        x, y: position [m]
        vx, vy: velocity [m/s]
        mass: current mass [kg]
        throttle: throttle level [0, 1] (already mapped from action)
        thrust_angle: thrust angle from vertical [rad]
        g: surface gravity [m/s²]
        max_thrust: maximum thrust [N]
        isp_s: specific impulse [s]
        rho0, scale_height_m, cd, area_m2: atmosphere/drag params
        dry_mass: dry mass [kg]

    Returns:
        (ax, ay, mdot): accelerations [m/s²] and mass flow rate [kg/s]
    """
    from env.atmosphere import drag_force

    # Thrust is zero if fuel is depleted
    has_fuel = mass > dry_mass
    T = throttle * max_thrust * has_fuel

    # Thrust components
    thrust_x = T * jnp.sin(thrust_angle)
    thrust_y = T * jnp.cos(thrust_angle)

    # Drag (opposing velocity)
    drag_x, drag_y = drag_force(vx, vy, y, rho0, scale_height_m, cd, area_m2)

    # Accelerations
    ax = thrust_x / mass - drag_x / mass
    ay = thrust_y / mass - g - drag_y / mass

    # Mass flow rate: ṁ = T / (isp · g0)
    mdot = T / (isp_s * G0)

    return ax, ay, mdot


def step_dynamics_euler(
    x: jnp.ndarray,
    y: jnp.ndarray,
    vx: jnp.ndarray,
    vy: jnp.ndarray,
    mass: jnp.ndarray,
    throttle: jnp.ndarray,
    thrust_angle: jnp.ndarray,
    dt: float,
    g: float,
    max_thrust: float,
    isp_s: float,
    rho0: float,
    scale_height_m: float,
    cd: float,
    area_m2: float,
    dry_mass: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single Euler step for Milestone 1 (dummy dynamics).

    Returns:
        (x_new, y_new, vx_new, vy_new, mass_new)
    """
    ax, ay, mdot = compute_acceleration(
        x, y, vx, vy, mass, throttle, thrust_angle,
        g, max_thrust, isp_s, rho0, scale_height_m, cd, area_m2, dry_mass,
    )

    vx_new = vx + ax * dt
    vy_new = vy + ay * dt
    x_new = x + vx * dt
    y_new = y + vy * dt
    mass_new = jnp.maximum(mass - mdot * dt, dry_mass)

    return x_new, y_new, vx_new, vy_new, mass_new
