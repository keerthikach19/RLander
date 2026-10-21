"""PPO-only training starts sampled from the parent's own flight states.

The bank contains simulator states, not teacher actions or imitation targets.
Validation/evaluation must use the original LanderEnv, never this reset wrapper.
"""

import jax
import jax.numpy as jnp
import numpy as np

from env.lander_env import EnvState, LanderEnv, _make_obs
from rl.execution import execute
from rl.networks import policy_action

HEIGHTS = (100., 40., 15., 5.)


def make_bank_rollout(max_steps=1024):
    def collect(seeds, body, vehicle, params, stats):
        def episode(seed):
            reset_key, key = jax.random.split(jax.random.key(seed))
            obs, state = LanderEnv.reset(reset_key, body, vehicle)
            saved = jax.tree_util.tree_map(lambda value: jnp.zeros((len(HEIGHTS),), value.dtype), state)
            seen = jnp.zeros(len(HEIGHTS), dtype=bool)

            def step(carry):
                obs, state, key, _, count, seen, saved, _ = carry
                key, _, step_key = jax.random.split(key, 3)
                action = policy_action(params, stats, obs)
                obs, state, _, done, info = LanderEnv.step(state, action, step_key)
                reached = (~seen) & (state.y <= jnp.asarray(HEIGHTS)) & (state.y > 0) & ~done
                saved = jax.tree_util.tree_map(
                    lambda old, value: jnp.where(reached, value, old), saved, state,
                )
                return obs, state, key, done, count + 1, seen | reached, saved, info["outcome"]

            result = jax.lax.while_loop(
                lambda c: ~c[3] & (c[4] < max_steps), step,
                (obs, state, key, jnp.asarray(False), jnp.int32(0), seen, saved, jnp.int32(0)),
            )
            return {"finished": result[3], "steps": result[4], "valid": result[5],
                    "states": result[6], "outcomes": result[7]}
        return jax.vmap(episode)(seeds)
    return jax.jit(collect)


def collect_bank(path, seeds, body, vehicle, params, stats):
    raw, evidence = execute(
        make_bank_rollout(),
        (jnp.asarray(seeds, dtype=jnp.uint32), body, vehicle, params, stats),
        label="ppo_parent_curriculum_collection",
    )
    host = jax.device_get(raw)
    if not host["finished"].all():
        raise RuntimeError("Parent collection exceeded the episode limit")
    mask = host["valid"].reshape(-1)
    if not mask.any():
        raise RuntimeError("Parent never reached a curriculum altitude")
    arrays = {field: np.asarray(value).reshape(-1)[mask]
              for field, value in zip(EnvState._fields, host["states"])}
    arrays["source_seed"] = np.repeat(np.asarray(seeds), len(HEIGHTS))[mask]
    arrays["source_outcome"] = np.repeat(host["outcomes"], len(HEIGHTS))[mask]
    arrays["crossing_height"] = np.tile(np.asarray(HEIGHTS), len(seeds))[mask]
    np.savez_compressed(path, **arrays)
    return {
        "source": "deterministic parent PPO flights; no PD, teacher actions, or BC",
        "seed_range": [int(seeds[0]), int(seeds[-1])],
        "episodes": len(seeds), "collection_transitions": int(host["steps"].sum()),
        "states": int(mask.sum()),
        "states_from_successes": int((arrays["source_outcome"] == 1).sum()),
        "states_from_failures": int((arrays["source_outcome"] != 1).sum()),
        "crossing_heights_m": list(HEIGHTS), "execution": evidence,
    }


def load_bank(path):
    with np.load(path, allow_pickle=False) as data:
        bank = EnvState(**{field: jnp.asarray(data[field]) for field in EnvState._fields})
        outcomes = np.asarray(data["source_outcome"])
    if len(outcomes) == 0 or any(not np.isfinite(np.asarray(value)).all() for value in bank):
        raise ValueError("Invalid curriculum bank")
    if not (np.asarray(bank.y) > 0).all() or not (np.asarray(bank.t) < np.asarray(bank.t_max)).all():
        raise ValueError("Curriculum must start in live, airborne states")
    return bank, outcomes


def mixed_start_env(bank, outcomes, probability=.75):
    """75% of resets use the bank; this is not a fraction of training steps."""
    if not 0 <= probability <= 1:
        raise ValueError("Curriculum reset probability must be in [0, 1]")
    outcomes = np.asarray(outcomes)
    if len(outcomes) != len(bank.y) or len(outcomes) == 0:
        raise ValueError("Bank state/outcome lengths differ or are empty")
    all_indices = np.arange(len(outcomes))
    success_indices = all_indices[outcomes == 1]
    failure_indices = all_indices[outcomes != 1]
    # A degenerate bank can still be tested; available states fill either group.
    success_indices = jnp.asarray(success_indices if len(success_indices) else all_indices)
    failure_indices = jnp.asarray(failure_indices if len(failure_indices) else all_indices)

    class MixedStartEnv:
        observation_size = 8
        action_size = 2
        step = staticmethod(LanderEnv.step)

        @staticmethod
        def reset(key, body, vehicle):
            obs, full_state = LanderEnv.reset(key, body, vehicle)
            if probability == 0:
                return obs, full_state
            keys = jax.random.split(jax.random.fold_in(key, 0xC011), 5)
            success = success_indices[jax.random.randint(keys[0], (), 0, len(success_indices))]
            failure = failure_indices[jax.random.randint(keys[1], (), 0, len(failure_indices))]
            index = jnp.where(jax.random.bernoulli(keys[2]), failure, success)
            state = jax.tree_util.tree_map(lambda value: value[index], bank)
            bank_obs = _make_obs(state, keys[3])
            use_bank = jax.random.bernoulli(keys[4], probability)
            state = jax.tree_util.tree_map(
                lambda partial, full: jnp.where(use_bank, partial, full), state, full_state,
            )
            return jnp.where(use_bank, bank_obs, obs), state

    return MixedStartEnv()
