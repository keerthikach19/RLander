"""Optional potential shaping for the velocity component (PLAN §5.5.1).

All inputs are scalar data, so the same compiled step supports both modes.
This supplements the baseline reward; it does not replace its objective.
"""

import jax.numpy as jnp


def descent_reference(height, gravity):
    """Physical m/s; on-profile braking is at most gravity / 2.

    The 30 m/s cap and 0.4 m/s touchdown target are modeling assumptions,
    shared with the tested PD profile, not changes to success thresholds.
    """
    return -jnp.minimum(30.0, jnp.sqrt(0.4**2 + gravity * jnp.maximum(height, 0)))


def descent_potential(height, vx, vy, gravity, velocity_unit, scale):
    error = jnp.sqrt(vx**2 + (vy - descent_reference(height, gravity)) ** 2)
    return -scale * error / velocity_unit


def profile_shaping(state, height, vx, vy, terminated):
    """gamma Phi(s') - Phi(s), zeroing Phi only at true terminal states.

    Time-limit truncations retain the next potential because PPO bootstraps
    their final observation. Reset states never enter this calculation.
    """
    previous = descent_potential(
        state.y, state.vx, state.vy, state.g, state.U, state.reward_potential_scale
    )
    following = descent_potential(
        height, vx, vy, state.g, state.U, state.reward_potential_scale
    )
    following = jnp.where(terminated, 0.0, following)
    return state.reward_gamma * following - previous


def landing_profile_reward(state, height, vx, vy, throttle, angle_norm, contacted, *, margin=False):
    """Experimental objective in physical units (mode 2), not potential shaping.

    Dense costs are rates integrated over dt. By default the vertical scale
    tightens near the ground to 1 m/s; lateral normalization is 2 m/s.
    margin=True changes the approach reference and lateral normalization for
    mode 4, including its dense cost on contact steps, but not contact scores.
    The rational contact score has a wider useful range than exp(-error),
    which is effectively zero at the observed 5--17 m/s failed touchdowns.
    Success still comes exclusively from the original environment thresholds.
    Keep the six public component names; altitude now records the time cost,
    velocity combines vertical tracking and lateral damping, and crash is a
    continuous impact cost at ALL contacts (including successful ones).
    """
    target = descent_reference(height, state.g)
    lateral_scale = 2.0
    if margin:
        # Full near-ground shaping below 30m; taper to the old objective at
        # 100m. These are reward targets, never altered success thresholds.
        gate = jnp.clip((100.0 - height) / 70.0, 0.0, 1.0)
        braking_fraction = 1.0 - 0.75 * gate
        target = -jnp.minimum(
            30.0, jnp.sqrt(0.4**2 + braking_fraction * state.g * jnp.maximum(height, 0.0))
        )
        lateral_scale = 2.0 - 1.75 * gate
    vertical_scale = jnp.maximum(state.vy_max_mps, 0.2 * jnp.abs(target))
    vertical_error = jnp.square((vy - target) / vertical_scale)
    lateral_error = jnp.square(vx / lateral_scale)
    impact = jnp.square(vy / state.vy_max_mps) + jnp.square(vx / state.vx_max_mps)
    return {
        "velocity": -state.dt * (
            0.15 * jnp.minimum(vertical_error, 25.0)
            + 0.10 * jnp.minimum(lateral_error, 25.0)
        ),
        "altitude": -state.dt * 0.20,
        "fuel": -state.dt * 0.02 * throttle,
        "angle": -state.dt * 0.01 * angle_norm,
        "landing": jnp.where(contacted, 100.0 / (1.0 + impact / 4.0), 0.0),
        "crash": jnp.where(contacted, -100.0 * impact / (25.0 + impact), 0.0),
    }


def success_contact_reward(components, landed_soft, crashed):
    """Mode 3: keep dense costs; mix smooth impact score with true outcome.

    0.5 * continuous contact score + 50 * (success - crash). The contact
    range stays [-100, 100], but crossing the actual landing boundary matters.
    This is intended for refinement after PPO already finds some landings.
    """
    return {**components,
            "landing": 0.5 * components["landing"] + 50.0 * landed_soft,
            "crash": 0.5 * components["crash"] - 50.0 * crashed}


def vertical_touchdown_reward(components, vy, contacted):
    """Mode 5: add a bounded vertical impact cost to the unchanged mode 4 reward.

    Applies to all contacts, including successful ones, to reward safety margin
    below the original 1 m/s success threshold. This is a training objective,
    not a different success criterion or an inference-time action correction.
    At |vy|=.3/.5/1 m/s the extra costs are about 7.94/15/24 points.
    The existing crash component already includes impact costs on soft landings.
    """
    speed_sq = jnp.square(vy)
    penalty = 30.0 * speed_sq / (speed_sq + 0.5**2)
    return {**components, "crash": components["crash"] - jnp.where(contacted, penalty, 0.0)}
