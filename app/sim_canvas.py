"""
Pure-Python physics simulation for the live canvas demo.

This is NOT the trained model.  It is a lightweight approximate guidance law
that mimics what a well-trained PPO agent *looks like* when running in the browser
without JAX.  Used as a fallback when the full checkpoint is unavailable.

Physics parameters match the Europa environment constants.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List

# --- Europa / lander constants (match env/lander_env.py defaults) -----------
G_EUROPA = 1.315          # m/s^2
MASS0 = 801.13            # kg
DRY_MASS = 101.13         # kg
ISP = 311.0               # s  (seconds)
G0 = 9.80665              # m/s^2  (standard gravity for Isp)
MAX_THRUST = 5000.0       # N
MIN_THROTTLE = 0.2
VY_SAFE = 1.0             # m/s  safe vertical speed at touchdown
VX_SAFE = 0.5             # m/s  safe horizontal speed at touchdown
DT = 0.3846               # s   (matches env dt)
T_MAX = 140.0             # s   episode timeout


@dataclass
class SimState:
    t: float = 0.0
    x: float = 0.0
    y: float = 1940.0
    vx: float = -1.8
    vy: float = -19.7
    mass: float = MASS0
    throttle: float = 0.0
    angle: float = 0.0       # gimbal angle (radians)
    done: bool = False
    outcome: str = "running"  # running | landed | crashed | timeout

    # history lists for plotting
    ts: List[float] = field(default_factory=list)
    xs: List[float] = field(default_factory=list)
    ys: List[float] = field(default_factory=list)
    vxs: List[float] = field(default_factory=list)
    vys: List[float] = field(default_factory=list)
    masses: List[float] = field(default_factory=list)
    throttles: List[float] = field(default_factory=list)
    angles: List[float] = field(default_factory=list)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _guidance_ppo_approx(state: SimState) -> tuple:
    """
    Two-phase guidance law (mimics trained PPO behaviour):

    Phase 1 - Coast: throttle near-zero while altitude is large and impact
                     is not imminent.  Saves fuel just like the PPO agent.

    Phase 2 - Suicide burn: full authority deceleration once the burn-distance
                            equals remaining altitude.  PPO learned to do
                            exactly this kind of late, aggressive retro-burn.
    """
    y = state.y
    vy = state.vy
    vx = state.vx
    mass = state.mass

    # Available net upward deceleration at full throttle
    a_max = MAX_THRUST / mass - G_EUROPA
    if a_max <= 0.05:
        return 0.0, 0.0   # out of fuel

    v_target = -0.65   # m/s  desired touchdown vertical speed

    # Kinematics: distance needed to brake vy -> v_target at a_max
    # v^2 = u^2 - 2*a*d  ->  d = (vy^2 - v_target^2) / (2*a_max)
    burn_dist = max((vy * vy - v_target * v_target) / (2.0 * a_max), 0.0)
    safety_margin = 30.0   # m buffer so burn starts early enough

    should_burn = (y <= burn_dist + safety_margin)

    # ── Phase 1: Coasting ────────────────────────────────────────────────────
    if not should_burn:
        # Minimal lateral correction during coast if drifting fast
        if abs(vx) > 4.0 and y > 300.0:
            thr_horiz = 0.25
            angle_horiz = -0.25 if vx > 0 else 0.25
            return thr_horiz, angle_horiz
        return 0.0, 0.0

    # ── Phase 2: Retro-burn ───────────────────────────────────────────────────
    speed_excess = abs(vy) - abs(v_target)   # how much faster than target

    if speed_excess > 1.0:
        # Still going fast — burn hard proportional to excess speed
        throttle = _clamp(0.70 + speed_excess / 15.0, MIN_THROTTLE, 1.0)
    elif y < 6.0:
        # Final flare — proportional control to hit v_target exactly
        a_desired = (v_target - vy) * 5.0 + G_EUROPA
        throttle = _clamp(a_desired / (MAX_THRUST / mass), MIN_THROTTLE, 1.0)
    else:
        # Transitioning — blend between full burn and proportional
        a_desired = (v_target - vy) / max(y / max(abs(vy), 0.1), 0.1) + G_EUROPA
        throttle = _clamp(a_desired / (MAX_THRUST / mass), MIN_THROTTLE, 1.0)

    # Gimbal to null horizontal velocity
    F_total = throttle * MAX_THRUST
    if F_total > 50.0 and y > 2.0:
        tti_est = max(-y / vy, 0.2) if vy < 0 else 1.0
        ax_desired = _clamp(-vx / max(tti_est * 0.45, 0.15), -4.0, 4.0)
        sin_angle = _clamp((ax_desired * mass) / F_total, -0.35, 0.35)
        angle = math.asin(sin_angle)
    else:
        angle = 0.0

    return throttle, angle


def step(state: SimState) -> SimState:
    """Advance the simulation by one control timestep DT."""
    if state.done:
        return state

    throttle, angle = _guidance_ppo_approx(state)

    # Small noise to mimic sensor/action stochasticity
    throttle = _clamp(throttle + random.gauss(0, 0.008), 0.0, 1.0)
    angle = _clamp(angle + random.gauss(0, 0.004), -0.35, 0.35)

    # --- Thrust force ---
    F_thrust = throttle * MAX_THRUST
    ax = (F_thrust * math.sin(angle)) / state.mass
    ay = (F_thrust * math.cos(angle)) / state.mass - G_EUROPA

    # --- Fuel burn ---
    mdot = F_thrust / (ISP * G0)
    new_mass = max(state.mass - mdot * DT, DRY_MASS)

    # --- Euler integration ---
    new_vx = state.vx + ax * DT
    new_vy = state.vy + ay * DT
    new_x = state.x + state.vx * DT + 0.5 * ax * DT * DT
    new_y = state.y + state.vy * DT + 0.5 * ay * DT * DT
    new_t = state.t + DT

    # --- Record history ---
    ts = state.ts + [state.t]
    xs = state.xs + [state.x]
    ys = state.ys + [state.y]
    vxs = state.vxs + [state.vx]
    vys = state.vys + [state.vy]
    masses = state.masses + [state.mass]
    throttles = state.throttles + [state.throttle]
    angles = state.angles + [state.angle]

    # --- Termination ---
    done = False
    outcome = "running"

    if new_y <= 0:
        new_y = 0.0
        done = True
        if abs(new_vy) <= VY_SAFE and abs(new_vx) <= VX_SAFE:
            outcome = "landed"
        else:
            outcome = "crashed"
    elif new_t >= T_MAX:
        done = True
        outcome = "timeout"

    return SimState(
        t=new_t, x=new_x, y=max(new_y, 0.0),
        vx=new_vx, vy=new_vy,
        mass=new_mass,
        throttle=throttle, angle=angle,
        done=done, outcome=outcome,
        ts=ts, xs=xs, ys=ys,
        vxs=vxs, vys=vys,
        masses=masses, throttles=throttles, angles=angles,
    )


def init_state(seed: int = 0) -> SimState:
    """Create a starting state with randomisation seeded by `seed`."""
    rng = random.Random(seed)
    y0 = rng.uniform(1700, 2100)
    vx0 = rng.uniform(-3.0, 3.0)
    vy0 = rng.uniform(-22.0, -17.0)
    x0 = rng.uniform(-40.0, 40.0)
    return SimState(y=y0, vx=vx0, vy=vy0, x=x0)
