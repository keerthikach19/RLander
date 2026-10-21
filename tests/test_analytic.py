"""Analytic tests for the environment physics (§W1 requirements).

Milestone 2 tests:
  (a) Free fall: y(t) = h0 - 0.5*g*t^2 within 1% over 10 s
  (b) Terminal velocity with atmosphere: v_t = sqrt(2mg/(rho*Cd*A))
  (c) Fuel depletion: mdot = throttle * T_max / (isp * g0)
  (d) Energy check (thrust work + PE + KE consistent)

All tests use the dynamics module directly (not through LanderEnv)
to isolate physics from env logic. Simulations use jax.lax.scan
for performance (avoids per-step Python overhead).
"""

import sys
import os
import math

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.dynamics import step_dynamics_rk4, G0


# ---------------------------------------------------------------------------
# Constants for tests
# ---------------------------------------------------------------------------
EUROPA_G = 1.315       # m/s²
MARS_G = 3.71          # m/s²
MARS_RHO0 = 0.020      # kg/m³
MARS_SCALE_H = 11100.0 # m

MASS0 = 800.0          # kg
DRY_MASS = 320.0       # kg
TWR_MAX = 3.0
ISP_S = 300.0           # s
GIMBAL_RAD = 0.2618     # ~15°
CD = 1.2
AREA_M2 = 4.0
H0 = 2000.0            # m
MAX_THRUST = TWR_MAX * MASS0 * EUROPA_G  # N


# ---------------------------------------------------------------------------
# Helper: jit-compiled multi-step simulation via lax.scan
# ---------------------------------------------------------------------------
@jax.jit
def _simulate_steps(state, params, n_steps):
    """Run n_steps of RK4 dynamics using lax.fori_loop.

    Args:
        state: (x, y, vx, vy, mass) as jnp arrays
        params: dict with throttle, thrust_angle, dt, g, max_thrust, isp_s,
                rho0, scale_height_m, cd, area_m2, dry_mass
        n_steps: number of steps (static)

    Returns:
        final (x, y, vx, vy, mass)
    """
    def body_fn(i, st):
        x, y, vx, vy, mass = st
        return step_dynamics_rk4(
            x, y, vx, vy, mass,
            params["throttle"], params["thrust_angle"], params["dt"],
            params["g"], params["max_thrust"], params["isp_s"],
            params["rho0"], params["scale_height_m"],
            params["cd"], params["area_m2"], params["dry_mass"],
        )

    return jax.lax.fori_loop(0, n_steps, body_fn, state)


def _run_sim(x, y, vx, vy, mass, throttle, thrust_angle, dt,
             g, max_thrust, isp_s, rho0, scale_height_m,
             cd, area_m2, dry_mass, n_steps):
    """Convenience wrapper for _simulate_steps."""
    state = (jnp.float32(x), jnp.float32(y), jnp.float32(vx),
             jnp.float32(vy), jnp.float32(mass))
    params = {
        "throttle": jnp.float32(throttle),
        "thrust_angle": jnp.float32(thrust_angle),
        "dt": jnp.float32(dt),
        "g": jnp.float32(g),
        "max_thrust": jnp.float32(max_thrust),
        "isp_s": jnp.float32(isp_s),
        "rho0": jnp.float32(rho0),
        "scale_height_m": jnp.float32(scale_height_m),
        "cd": jnp.float32(cd),
        "area_m2": jnp.float32(area_m2),
        "dry_mass": jnp.float32(dry_mass),
    }
    return _simulate_steps(state, params, n_steps)


# ---------------------------------------------------------------------------
# (a) Free fall: y(t) = h0 - 0.5*g*t^2 within 1% over 10 s
# ---------------------------------------------------------------------------
def test_free_fall_europa():
    """Free fall with zero thrust on Europa: y(t) ≈ h0 - 0.5·g·t² within 1%.

    RK4 integrator should match the analytic parabola over 10 s.
    """
    g = EUROPA_G
    h0 = H0
    dt = 0.01  # 0.01 s steps
    total_time = 10.0
    n_steps = int(total_time / dt)  # 1000 steps

    x_f, y_f, vx_f, vy_f, mass_f = _run_sim(
        x=0.0, y=h0, vx=0.0, vy=0.0, mass=MASS0,
        throttle=0.0, thrust_angle=0.0, dt=dt,
        g=g, max_thrust=MAX_THRUST, isp_s=ISP_S,
        rho0=0.0, scale_height_m=1.0,
        cd=CD, area_m2=AREA_M2, dry_mass=DRY_MASS,
        n_steps=n_steps,
    )

    y_analytic = h0 - 0.5 * g * total_time**2
    y_sim = float(y_f)
    error = abs(y_sim - y_analytic) / h0
    assert error < 0.01, (
        f"Free fall at t={total_time}s: y_sim={y_sim:.4f}, y_analytic={y_analytic:.4f}, "
        f"rel_error={error:.6f} (>1%)"
    )

    # Also check final velocity: v(10) = -g*t = -13.15 m/s
    vy_analytic = -g * total_time
    vy_sim = float(vy_f)
    vel_error = abs(vy_sim - vy_analytic) / abs(vy_analytic)
    assert vel_error < 0.01, (
        f"Free fall velocity at t={total_time}s: vy_sim={vy_sim:.4f}, "
        f"vy_analytic={vy_analytic:.4f}, rel_error={vel_error:.6f} (>1%)"
    )


def test_free_fall_mars_vacuum():
    """Free fall on Mars (vacuum, to isolate gravity) over 5 s."""
    g = MARS_G
    h0 = H0
    dt = 0.01
    total_time = 5.0
    n_steps = int(total_time / dt)

    x_f, y_f, vx_f, vy_f, mass_f = _run_sim(
        x=0.0, y=h0, vx=0.0, vy=0.0, mass=MASS0,
        throttle=0.0, thrust_angle=0.0, dt=dt,
        g=g, max_thrust=TWR_MAX * MASS0 * g, isp_s=ISP_S,
        rho0=0.0, scale_height_m=1.0,
        cd=CD, area_m2=AREA_M2, dry_mass=DRY_MASS,
        n_steps=n_steps,
    )

    y_analytic = h0 - 0.5 * g * total_time**2
    y_sim = float(y_f)
    error = abs(y_sim - y_analytic) / h0
    assert error < 0.01, (
        f"Mars free fall at t={total_time}s: y_sim={y_sim:.4f}, "
        f"y_analytic={y_analytic:.4f}, rel_error={error:.6f} (>1%)"
    )


# ---------------------------------------------------------------------------
# (b) Terminal velocity with atmosphere: v_t = sqrt(2mg / (rho·Cd·A))
# ---------------------------------------------------------------------------
def test_terminal_velocity_thick_atmosphere():
    """Terminal velocity test with a thick atmosphere (Titan-like params).

    Uses rho0=5.0 kg/m³ with a very large scale height (1e8 m) so density
    is effectively constant throughout the simulation. Starts from rest
    and lets the object accelerate to terminal velocity.

    v_terminal = sqrt(2·m·g / (rho·Cd·A)) at constant density.
    """
    # Thick atmosphere with effectively constant density
    g = 1.352  # Titan surface gravity
    rho0 = 5.0  # thick atmosphere
    scale_height_m = 1e8  # huge: rho ≈ rho0 at all relevant altitudes
    cd = CD
    area = AREA_M2
    mass_val = MASS0
    max_thrust = TWR_MAX * mass_val * g

    # Analytic terminal velocity at constant rho = rho0
    v_terminal = math.sqrt(2.0 * mass_val * g / (rho0 * cd * area))
    # v_terminal ≈ sqrt(2*800*1.352 / (5.0*1.2*4.0)) ≈ sqrt(90.1) ≈ 9.5 m/s

    # Start from rest at high altitude; let gravity accelerate to v_terminal
    h0_test = 100000.0  # 100 km, plenty of room to fall
    dt = 0.5  # large dt OK for RK4
    n_steps = 2000  # 1000 s — much more than needed

    x_f, y_f, vx_f, vy_f, mass_f = _run_sim(
        x=0.0, y=h0_test, vx=0.0, vy=0.0, mass=mass_val,
        throttle=0.0, thrust_angle=0.0, dt=dt,
        g=g, max_thrust=max_thrust, isp_s=ISP_S,
        rho0=rho0, scale_height_m=scale_height_m,
        cd=cd, area_m2=area, dry_mass=DRY_MASS,
        n_steps=n_steps,
    )

    v_sim = abs(float(vy_f))

    # Should converge to v_terminal within 5%
    rel_error = abs(v_sim - v_terminal) / v_terminal
    assert rel_error < 0.05, (
        f"Terminal velocity: v_sim={v_sim:.2f} m/s, v_terminal={v_terminal:.2f} m/s, "
        f"rel_error={rel_error:.4f} (>5%)"
    )


# ---------------------------------------------------------------------------
# (c) Fuel depletion: ṁ = throttle · T_max / (isp · g0)
# ---------------------------------------------------------------------------
def test_fuel_flow_rate():
    """Fuel mass should deplete at the expected rate: ṁ = T / (isp·g0).

    Full throttle for 1 s, check mass consumed vs analytic.
    """
    g = EUROPA_G
    max_thrust = TWR_MAX * MASS0 * g

    dt = 0.1
    n_steps = 10  # 1 s total
    total_time = dt * n_steps

    x_f, y_f, vx_f, vy_f, mass_f = _run_sim(
        x=0.0, y=H0, vx=0.0, vy=0.0, mass=MASS0,
        throttle=1.0, thrust_angle=0.0, dt=dt,
        g=g, max_thrust=max_thrust, isp_s=ISP_S,
        rho0=0.0, scale_height_m=1.0,
        cd=CD, area_m2=AREA_M2, dry_mass=DRY_MASS,
        n_steps=n_steps,
    )

    mass_consumed_sim = MASS0 - float(mass_f)

    # Analytic: mdot = max_thrust / (isp * g0), constant while fuel lasts
    mdot_analytic = max_thrust / (ISP_S * G0)
    mass_consumed_analytic = mdot_analytic * total_time

    rel_error = abs(mass_consumed_sim - mass_consumed_analytic) / mass_consumed_analytic
    assert rel_error < 0.02, (
        f"Fuel flow: consumed_sim={mass_consumed_sim:.4f} kg, "
        f"consumed_analytic={mass_consumed_analytic:.4f} kg, "
        f"rel_error={rel_error:.6f} (>2%)"
    )


def test_fuel_depletes_to_dry_mass():
    """Mass should never go below dry_mass, even with sustained thrust."""
    g = EUROPA_G
    max_thrust = TWR_MAX * MASS0 * g

    dt = 1.0
    n_steps = 500  # enough to exhaust all fuel

    x_f, y_f, vx_f, vy_f, mass_f = _run_sim(
        x=0.0, y=H0, vx=0.0, vy=0.0, mass=MASS0,
        throttle=1.0, thrust_angle=0.0, dt=dt,
        g=g, max_thrust=max_thrust, isp_s=ISP_S,
        rho0=0.0, scale_height_m=1.0,
        cd=CD, area_m2=AREA_M2, dry_mass=DRY_MASS,
        n_steps=n_steps,
    )

    assert float(mass_f) >= DRY_MASS - 1e-3, (
        f"Mass went below dry_mass: mass={float(mass_f)}, dry_mass={DRY_MASS}"
    )


# ---------------------------------------------------------------------------
# (d) Energy check: thrust work + PE + KE consistent
# ---------------------------------------------------------------------------
def test_energy_conservation_free_fall():
    """In free fall (no thrust, no drag), total mechanical energy is conserved.

    E = 0.5·m·v² + m·g·h = constant.
    """
    g = EUROPA_G
    h0 = H0

    dt = 0.01
    n_steps = 500  # 5 s

    x_f, y_f, vx_f, vy_f, mass_f = _run_sim(
        x=0.0, y=h0, vx=0.0, vy=0.0, mass=MASS0,
        throttle=0.0, thrust_angle=0.0, dt=dt,
        g=g, max_thrust=MAX_THRUST, isp_s=ISP_S,
        rho0=0.0, scale_height_m=1.0,
        cd=CD, area_m2=AREA_M2, dry_mass=DRY_MASS,
        n_steps=n_steps,
    )

    E0 = 0.5 * MASS0 * 0.0 + MASS0 * g * h0  # KE=0 at start
    E_final = (0.5 * float(mass_f) * (float(vx_f)**2 + float(vy_f)**2)
               + float(mass_f) * g * float(y_f))

    rel_error = abs(E_final - E0) / abs(E0)
    assert rel_error < 1e-4, (
        f"Energy not conserved in free fall: E0={E0:.2f}, E_final={E_final:.2f}, "
        f"rel_error={rel_error:.8f} (>0.01%)"
    )


def test_thrust_decelerates_descent():
    """With upward thrust during descent, the vehicle should decelerate.

    This is a physics consistency check for the energy budget.
    """
    g = EUROPA_G
    max_thrust = TWR_MAX * MASS0 * g

    dt = 0.01
    n_steps = 200  # 2 s

    x_f, y_f, vx_f, vy_f, mass_f = _run_sim(
        x=0.0, y=H0, vx=0.0, vy=-20.0, mass=MASS0,
        throttle=0.8, thrust_angle=0.0, dt=dt,
        g=g, max_thrust=max_thrust, isp_s=ISP_S,
        rho0=0.0, scale_height_m=1.0,
        cd=CD, area_m2=AREA_M2, dry_mass=DRY_MASS,
        n_steps=n_steps,
    )

    # With twr_max=3.0 and throttle=0.8, thrust ≈ 2.4g > g,
    # so the vehicle should decelerate (vy becomes less negative or positive)
    assert float(vy_f) > -20.0, (
        f"Thrust should decelerate descent: initial vy=-20, final vy={float(vy_f):.4f}"
    )


# ---------------------------------------------------------------------------
# Additional physics checks
# ---------------------------------------------------------------------------
def test_no_nan_rk4_vacuum():
    """RK4 should produce no NaN for vacuum body (rho0=0)."""
    x_f, y_f, vx_f, vy_f, mass_f = _run_sim(
        x=0.0, y=H0, vx=5.0, vy=-20.0, mass=MASS0,
        throttle=0.5, thrust_angle=0.1, dt=0.01,
        g=EUROPA_G, max_thrust=MAX_THRUST, isp_s=ISP_S,
        rho0=0.0, scale_height_m=1.0,
        cd=CD, area_m2=AREA_M2, dry_mass=DRY_MASS,
        n_steps=100,
    )
    vals = jnp.array([x_f, y_f, vx_f, vy_f, mass_f])
    assert not jnp.any(jnp.isnan(vals)), f"NaN in RK4 output: {vals}"


def test_no_nan_rk4_atmosphere():
    """RK4 should produce no NaN with atmosphere (Mars params)."""
    x_f, y_f, vx_f, vy_f, mass_f = _run_sim(
        x=0.0, y=H0, vx=5.0, vy=-20.0, mass=MASS0,
        throttle=0.5, thrust_angle=0.1, dt=0.01,
        g=MARS_G, max_thrust=TWR_MAX * MASS0 * MARS_G, isp_s=ISP_S,
        rho0=MARS_RHO0, scale_height_m=MARS_SCALE_H,
        cd=CD, area_m2=AREA_M2, dry_mass=DRY_MASS,
        n_steps=100,
    )
    vals = jnp.array([x_f, y_f, vx_f, vy_f, mass_f])
    assert not jnp.any(jnp.isnan(vals)), f"NaN in RK4 with atmosphere: {vals}"


def test_horizontal_thrust():
    """Thrust at an angle should produce horizontal acceleration."""
    x_f, y_f, vx_f, vy_f, mass_f = _run_sim(
        x=0.0, y=H0, vx=0.0, vy=0.0, mass=MASS0,
        throttle=1.0, thrust_angle=GIMBAL_RAD, dt=0.1,
        g=EUROPA_G, max_thrust=MAX_THRUST, isp_s=ISP_S,
        rho0=0.0, scale_height_m=1.0,
        cd=CD, area_m2=AREA_M2, dry_mass=DRY_MASS,
        n_steps=10,
    )

    # sin(15°) > 0, so positive thrust_angle -> positive vx
    assert float(vx_f) > 0.0, f"Expected positive vx from angled thrust, got {float(vx_f)}"
    assert float(x_f) > 0.0, f"Expected positive x displacement, got {float(x_f)}"
