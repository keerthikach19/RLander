"""Vectorization and caller-owned reset using only the environment contract."""

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class VecStep:
    obs: object
    state: object
    final_obs: object
    reward_components: object
    done: object
    info: object


def reset_envs(env, keys, body_params, vehicle_params, *, per_env_params=False):
    axis = 0 if per_env_params else None
    return jax.vmap(env.reset, in_axes=(0, axis, axis))(
        keys,
        body_params,
        vehicle_params,
    )


def step_envs(
    env,
    state,
    actions,
    step_keys,
    reset_keys,
    body_params,
    vehicle_params,
    *,
    per_env_params=False,
):
    final_obs, next_state, components, done, info = jax.vmap(env.step)(
        state,
        jnp.clip(actions, -1, 1),
        step_keys,
    )
    reset_obs, reset_state = reset_envs(
        env,
        reset_keys,
        body_params,
        vehicle_params,
        per_env_params=per_env_params,
    )

    def choose(reset, current):
        mask = done.reshape(done.shape + (1,) * (current.ndim - done.ndim))
        return jnp.where(mask, reset, current)

    return VecStep(
        choose(reset_obs, final_obs),
        jax.tree_util.tree_map(choose, reset_state, next_state),
        final_obs,
        components,
        done,
        info,
    )
