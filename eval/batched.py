"""W3 rollouts without Python or device-to-host synchronization per timestep."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from env.lander_env import LanderEnv, _make_obs
from rl.execution import event, execute


@lru_cache(maxsize=16)
def make_rollout(policy, max_steps, custom_start=False):
    """Policy weights are data arguments, so checkpoint changes reuse compilation."""
    def rollout(seeds, body, vehicle, weights, initial_conditions):
        def episode(seed):
            reset_key, sim_key = jax.random.split(jax.random.key(seed))
            obs, state = LanderEnv.reset(reset_key, body, vehicle)
            if custom_start:
                x, y, vx, vy = initial_conditions
                # Keep the training reset's units, vehicle, fuel and time budget.
                # Changing L to the chosen altitude would change what the saved
                # network's inputs mean. Initial acceleration is zero, as at reset.
                state = state._replace(x=x, y=y, vx=vx, vy=vy, vx_prev=vx, vy_prev=vy)
                noise_key = jax.random.split(reset_key, 8)[-1]
                obs = _make_obs(state, noise_key)
            shapes = jax.eval_shape(LanderEnv.step, state, jnp.zeros(2), sim_key)
            zero = lambda s: jnp.zeros(s.shape, s.dtype)
            totals, info = jax.tree_util.tree_map(zero, (shapes[2], shapes[4]))

            def record(state, action, reward):
                return jnp.stack((state.t, state.x, state.y, state.vx, state.vy,
                                  state.mass, action[0], action[1], reward))

            records = jnp.zeros((max_steps + 1, 9), dtype=jnp.float32)
            records = records.at[0].set(record(state, jnp.zeros(2), 0.))

            def step(carry):
                obs, state, key, _, count, totals, _, records = carry
                key, act_key, step_key = jax.random.split(key, 3)
                action = policy(obs, act_key, weights)
                obs, state, components, done, info = LanderEnv.step(state, action, step_key)
                totals = jax.tree_util.tree_map(jnp.add, totals, components)
                records = records.at[count + 1].set(record(state, action, sum(components.values())))
                return obs, state, key, done, count + 1, totals, info, records

            _, state, _, done, count, totals, info, records = jax.lax.while_loop(
                lambda c: ~c[3] & (c[4] < max_steps), step,
                (obs, state, sim_key, jnp.asarray(False), jnp.int32(0), totals, info, records),
            )
            return {"finished": done, "count": count, "totals": totals, "info": info,
                    "records": records, "min_throttle": state.min_throttle,
                    "gimbal_limit": state.gimbal_limit_rad}
        return jax.vmap(episode)(seeds)
    return jax.jit(rollout)


def collect(policy_fn, seeds, body, vehicle, max_steps=1024, *, initial_conditions=None):
    """One compiled batch; transfer once, then serialize trajectories on the CPU."""
    numeric = lambda config: jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype=jnp.float32),
        {k: v for k, v in config.items() if k != "name"},
    )
    policy = getattr(policy_fn, "rollout_policy", None)
    if policy is None:
        policy = lambda obs, key, weights: policy_fn(obs, key)
    weights = getattr(policy_fn, "rollout_weights", ())
    result, evidence = execute(
        make_rollout(policy, max_steps, custom_start=initial_conditions is not None),
        (jnp.asarray(seeds, dtype=jnp.uint32), numeric(body), numeric(vehicle), weights,
         jnp.asarray(initial_conditions if initial_conditions is not None else (0., 0., 0., 0.),
                     dtype=jnp.float32)),
        label="w3_batched_evaluation",
    )
    event("w3_batched_evaluation", "transferring_results_to_host")
    host = jax.device_get(result)
    if not np.asarray(host["finished"]).all():
        raise RuntimeError("Batched W3 evaluation exceeded max_steps before termination")
    event("w3_batched_evaluation", "writing_trajectories_and_database_on_host",
          episodes=len(seeds))
    episodes = []
    columns = ("t", "x", "y", "vx", "vy", "mass", "throttle", "angle", "reward")
    for index in range(len(seeds)):
        count = int(host["count"][index])
        rows = host["records"][index, :count + 1].astype(np.float64)
        minimum = float(host["min_throttle"][index])
        rows[1:, 6] = minimum + (1 - minimum) * (rows[1:, 6] + 1) / 2
        rows[1:, 7] *= float(host["gimbal_limit"][index])
        episodes.append((
            [dict(zip(columns, map(float, row))) for row in rows],
            {k: v[index] for k, v in host["totals"].items()},
            {k: v[index] for k, v in host["info"].items()},
        ))
    return episodes, evidence
