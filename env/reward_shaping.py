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
