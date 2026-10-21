"""Dynamics module for RLander.

Planar point-mass dynamics with thrust vectoring (§5.2).
RK4 integration with configurable substeps per control step.

Equations of motion (§5.2):
  θ = a_angle · gimbal_limit  (thrust angle from vertical)
  ax = T·sin(θ)/m − Dx/m
  ay = T·cos(θ)/m − g − Dy/m
  ṁ  = T / (isp · g0)
  Thrust is zero once mass ≤ dry_mass.
"""

import jax
import jax.numpy as jnp

from env.atmosphere import drag_force

# Standard gravity for fuel flow calculation
G0 = 9.80665  # m/s², standard gravity

# Number of RK4 substeps per control step (§5.2: 2–4)
NUM_SUBSTEPS = 4


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


def _derivatives(
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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute state derivatives (dx/dt, dy/dt, dvx/dt, dvy/dt, dm/dt).

    Returns:
        (dx_dt, dy_dt, dvx_dt, dvy_dt, dm_dt)
    """
    ax, ay, mdot = compute_acceleration(
        x, y, vx, vy, mass, throttle, thrust_angle,
        g, max_thrust, isp_s, rho0, scale_height_m, cd, area_m2, dry_mass,
    )
    # dx/dt = vx, dy/dt = vy, dvx/dt = ax, dvy/dt = ay, dm/dt = -mdot
    return vx, vy, ax, ay, -mdot


def _rk4_substep(
    x: jnp.ndarray,
    y: jnp.ndarray,
    vx: jnp.ndarray,
    vy: jnp.ndarray,
    mass: jnp.ndarray,
    throttle: jnp.ndarray,
    thrust_angle: jnp.ndarray,
    h: float,
    g: float,
    max_thrust: float,
    isp_s: float,
    rho0: float,
    scale_height_m: float,
    cd: float,
    area_m2: float,
    dry_mass: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Single RK4 substep with timestep h.

    Throttle and thrust_angle are held constant over each control step
    (zero-order hold), which is physically correct for a control system.

    Returns:
        (x_new, y_new, vx_new, vy_new, mass_new)
    """
    args = (throttle, thrust_angle, g, max_thrust, isp_s,
            rho0, scale_height_m, cd, area_m2, dry_mass)

    # k1
    k1_x, k1_y, k1_vx, k1_vy, k1_m = _derivatives(x, y, vx, vy, mass, *args)

    # k2
    k2_x, k2_y, k2_vx, k2_vy, k2_m = _derivatives(
        x + 0.5 * h * k1_x, y + 0.5 * h * k1_y,
        vx + 0.5 * h * k1_vx, vy + 0.5 * h * k1_vy,
        jnp.maximum(mass + 0.5 * h * k1_m, dry_mass),
        *args,
    )

    # k3
    k3_x, k3_y, k3_vx, k3_vy, k3_m = _derivatives(
        x + 0.5 * h * k2_x, y + 0.5 * h * k2_y,
        vx + 0.5 * h * k2_vx, vy + 0.5 * h * k2_vy,
        jnp.maximum(mass + 0.5 * h * k2_m, dry_mass),
        *args,
    )

    # k4
    k4_x, k4_y, k4_vx, k4_vy, k4_m = _derivatives(
        x + h * k3_x, y + h * k3_y,
        vx + h * k3_vx, vy + h * k3_vy,
        jnp.maximum(mass + h * k3_m, dry_mass),
        *args,
    )

    # RK4 update
    x_new = x + (h / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
    y_new = y + (h / 6.0) * (k1_y + 2.0 * k2_y + 2.0 * k3_y + k4_y)
    vx_new = vx + (h / 6.0) * (k1_vx + 2.0 * k2_vx + 2.0 * k3_vx + k4_vx)
    vy_new = vy + (h / 6.0) * (k1_vy + 2.0 * k2_vy + 2.0 * k3_vy + k4_vy)
    mass_new = mass + (h / 6.0) * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m)
    mass_new = jnp.maximum(mass_new, dry_mass)

    return x_new, y_new, vx_new, vy_new, mass_new


def step_dynamics_rk4(
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
    num_substeps: int = NUM_SUBSTEPS,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Full RK4 integration over one control step with multiple substeps.

    §5.2: 2–4 substeps per control step. Throttle and thrust angle are
    held constant (zero-order hold) over the entire control step.

    Args:
        x, y, vx, vy, mass: current state
        throttle: mapped throttle [0, 1]
        thrust_angle: thrust angle [rad]
        dt: full control timestep [s]
        g, max_thrust, isp_s: propulsion params
        rho0, scale_height_m, cd, area_m2: atmosphere/drag params
        dry_mass: dry mass [kg]
        num_substeps: number of RK4 substeps (default NUM_SUBSTEPS=4)

    Returns:
        (x_new, y_new, vx_new, vy_new, mass_new)
    """
    h = dt / num_substeps  # substep size

    def body_fn(i, state):
        x_i, y_i, vx_i, vy_i, mass_i = state
        return _rk4_substep(
            x_i, y_i, vx_i, vy_i, mass_i,
            throttle, thrust_angle, h,
            g, max_thrust, isp_s,
            rho0, scale_height_m, cd, area_m2, dry_mass,
        )

    x_f, y_f, vx_f, vy_f, mass_f = jax.lax.fori_loop(
        0, num_substeps, body_fn, (x, y, vx, vy, mass)
    )

    return x_f, y_f, vx_f, vy_f, mass_f
