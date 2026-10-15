"""Separate actor/critic MLPs and an unsquashed diagonal Gaussian policy.

As in the vendored CleanRL reference, log probabilities use the raw sampled
action; only the action sent to the environment is clipped to [-1, 1].
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct


class ActorCritic(nn.Module):
    action_dim: int = 2

    @nn.compact
    def __call__(self, obs):
        actor, critic = obs, obs
        for i in range(2):
            actor = nn.tanh(
                nn.Dense(
                    64,
                    kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
                    name=f"actor_{i}",
                )(actor)
            )
            critic = nn.tanh(
                nn.Dense(
                    64,
                    kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)),
                    name=f"critic_{i}",
                )(critic)
            )
        mean = nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.orthogonal(0.01),
            name="actor_mean",
        )(actor)
        value = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            name="critic_value",
        )(critic)[..., 0]
        log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        return mean, jnp.broadcast_to(log_std, mean.shape), value


def gaussian_log_prob(action, mean, log_std):
    return -0.5 * jnp.sum(
        jnp.square((action - mean) * jnp.exp(-log_std))
        + 2.0 * log_std
        + jnp.log(2.0 * jnp.pi),
        axis=-1,
    )


def gaussian_entropy(log_std):
    return jnp.sum(log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e), axis=-1)


def sample_action(key, mean, log_std):
    action = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
    return action, gaussian_log_prob(action, mean, log_std)


@struct.dataclass
class ObsStats:
    mean: jax.Array
    var: jax.Array
    count: jax.Array

    @classmethod
    def empty(cls, size=8):
        return cls(jnp.zeros(size), jnp.ones(size), jnp.asarray(0.0))


def update_obs_stats(stats, observations):
    """Parallel variance merge, over all leading batch/time dimensions."""
    batch = observations.reshape((-1, observations.shape[-1]))
    count = stats.count + batch.shape[0]
    delta = jnp.mean(batch, axis=0) - stats.mean
    mean = stats.mean + delta * batch.shape[0] / count
    m2 = (
        stats.var * stats.count
        + jnp.var(batch, axis=0) * batch.shape[0]
        + jnp.square(delta) * stats.count * batch.shape[0] / count
    )
    return ObsStats(mean, m2 / count, count)


def normalize_obs(obs, stats):
    return jnp.clip((obs - stats.mean) / jnp.sqrt(stats.var + 1e-8), -10, 10)


def policy_action(params, obs_stats, obs):
    """Deterministic inference from saved parameters and normalization stats."""
    mean, _, _ = ActorCritic().apply(params, normalize_obs(obs, obs_stats))
    return jnp.clip(mean, -1, 1)
