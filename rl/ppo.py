"""Continuous PPO: vmap environments and scan rollouts/updates under one jit.

Algorithm reference: docs/reference/cleanrl_ppo_continuous.py. Time limits
bootstrap from the final observation; GAE never leaks across episode resets.
Observation statistics are frozen during each rollout and its optimization.
"""

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.training.train_state import TrainState

from rl.networks import (
    ActorCritic,
    ObsStats,
    gaussian_entropy,
    gaussian_log_prob,
    normalize_obs,
    replace_actor,
    sample_action,
    update_obs_stats,
)
from rl.vec_env import reset_envs, step_envs


@dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 1024
    num_steps: int = 128
    total_steps: int = 20_000_000
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    num_minibatches: int = 8
    update_epochs: int = 4
    learning_rate: float = 3e-4
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    initial_action_mean: tuple[float, float] = (0.0, 0.0)
    initial_log_std: float = 0.0
    reward_scale: float = 1.0
    freeze_obs_stats: bool = False
    log_std_min: float | None = None
    log_std_max: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "initial_action_mean", tuple(self.initial_action_mean))
        if len(self.initial_action_mean) != 2 or any(
            not math.isfinite(x) or not -1 <= x <= 1 for x in self.initial_action_mean
        ):
            raise ValueError(
                "initial_action_mean must contain two finite values in [-1, 1]"
            )
        if not math.isfinite(self.initial_log_std):
            raise ValueError("initial_log_std must be finite")
        for bound in (self.log_std_min, self.log_std_max):
            if bound is not None and not math.isfinite(bound):
                raise ValueError("log-std bounds must be finite or None")
        if (self.log_std_min is not None and self.initial_log_std < self.log_std_min
                or self.log_std_max is not None and self.initial_log_std > self.log_std_max):
            raise ValueError("initial_log_std must lie within log-std bounds")
        if not math.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError("reward_scale must be finite and positive")
        for name in (
            "num_envs",
            "num_steps",
            "total_steps",
            "num_minibatches",
            "update_epochs",
        ):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size % self.num_minibatches:
            raise ValueError(
                "num_envs * num_steps must be divisible by num_minibatches"
            )
        if self.num_updates < 1:
            raise ValueError("total_steps must cover at least one rollout")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if self.learning_rate <= 0 or self.max_grad_norm <= 0 or self.clip_coef <= 0:
            raise ValueError(
                "learning_rate, max_grad_norm and clip_coef must be positive"
            )
        if self.ent_coef < 0 or self.vf_coef < 0:
            raise ValueError("loss coefficients must be nonnegative")

    @property
    def batch_size(self):
        return self.num_envs * self.num_steps

    @property
    def num_updates(self):
        """Full rollouts only; the logged actual step count makes rounding explicit."""
        return self.total_steps // self.batch_size


@struct.dataclass
class RunnerState:
    train_state: TrainState
    obs_stats: ObsStats
    obs: object
    env_state: object
    key: object
    episode_components: object
    episode_lengths: object


@struct.dataclass
class TrainResult:
    runner: RunnerState
    metrics: object
    checkpoints: tuple = ()


def compute_gae(rewards, values, next_values, done, truncated, gamma, gae_lambda):
    """Use final-state values for timeouts, zero bootstrap for true terminals."""
    terminated = done & ~truncated
    deltas = rewards + gamma * next_values * (~terminated) - values

    def backward(advantage, transition):
        delta, ended = transition
        advantage = delta + gamma * gae_lambda * (~ended) * advantage
        return advantage, advantage

    _, advantages = jax.lax.scan(
        backward,
        jnp.zeros_like(values[-1]),
        (deltas, done),
        reverse=True,
    )
    return advantages, advantages + values


def ppo_loss(params, apply_fn, batch, config):
    mean, log_std, value = apply_fn(params, batch["obs"])
    log_prob = gaussian_log_prob(batch["action"], mean, log_std)
    log_ratio = log_prob - batch["log_prob"]
    ratio = jnp.exp(log_ratio)
    advantage = batch["advantage"]
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    policy_loss = -jnp.minimum(
        ratio * advantage,
        jnp.clip(ratio, 1 - config.clip_coef, 1 + config.clip_coef) * advantage,
    ).mean()
    clipped_value = batch["value"] + jnp.clip(
        value - batch["value"],
        -config.clip_coef,
        config.clip_coef,
    )
    value_loss = (
        0.5
        * jnp.maximum(
            jnp.square(value - batch["return"]),
            jnp.square(clipped_value - batch["return"]),
        ).mean()
    )
    entropy = gaussian_entropy(log_std).mean()
    loss = policy_loss + config.vf_coef * value_loss - config.ent_coef * entropy
    return loss, {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": ((ratio - 1) - log_ratio).mean(),
        "clip_fraction": (jnp.abs(ratio - 1) > config.clip_coef).mean(),
    }


def project_log_std(params, config):
    """Project stored Gaussian parameters so sampling and log-probs stay paired."""
    if config.log_std_min is None and config.log_std_max is None:
        return params
    return {**params, "params": {**params["params"], "log_std": jnp.clip(
        params["params"]["log_std"],
        -jnp.inf if config.log_std_min is None else config.log_std_min,
        jnp.inf if config.log_std_max is None else config.log_std_max,
    )}}


def make_train(
    env,
    config=None,
    *,
    shaping_components=(),
    per_env_params=False,
    checkpoint_steps=(),
):
    """Build train(key, body, vehicle, actor_init=None, obs_stats_init=None, params_init=None).

    Numeric parameter pytrees are data arguments (shared by default, or batched
    on axis 0 with per_env_params=True). Component names are supplied explicitly
    by the caller: the env contract does not prescribe reward dictionary keys.
    Only aggregate per-update diagnostics leave the compiled training loop.
    Optional initialization imports only the actor MLP and paired normalization;
    critic, log-std, optimizer, RNG, and environments are initialized fresh.
    Optional checkpoint_steps retain complete runners on one continuous schedule.
    params_init supports PPO weight warm starts with paired normalization; it
    does not resume optimizer/RNG/environments or the parent's learning schedule.
    """
    config = PPOConfig() if config is None else config
    checkpoint_steps = tuple(checkpoint_steps)
    if checkpoint_steps != tuple(sorted(set(checkpoint_steps))) or any(
        not isinstance(step, int)
        or step <= 0
        or step % config.batch_size
        or step >= config.num_updates * config.batch_size
        for step in checkpoint_steps
    ):
        raise ValueError(
            "checkpoint_steps must be sorted unique rollout boundaries before the final step"
        )
    network = ActorCritic(
        initial_action_mean=config.initial_action_mean,
        initial_log_std=config.initial_log_std,
    )
    updates_per_rollout = config.num_minibatches * config.update_epochs

    def learning_rate(count):
        iteration = count // updates_per_rollout
        return config.learning_rate * (1.0 - iteration / config.num_updates)

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate, eps=1e-5),
    )

    def train(key, body_params, vehicle_params, actor_init=None, obs_stats_init=None, params_init=None):
        if actor_init is not None and params_init is not None:
            raise ValueError("Choose actor-only initialization or full PPO parameters")
        if ((actor_init is None and params_init is None) != (obs_stats_init is None)):
            raise ValueError("Actor init and its observation statistics must be paired")
        if config.freeze_obs_stats and obs_stats_init is None:
            raise ValueError("Frozen normalization requires initialized observation statistics")
        key, init_key, reset_key, shape_key = jax.random.split(key, 4)
        obs, env_state = reset_envs(
            env,
            jax.random.split(reset_key, config.num_envs),
            body_params,
            vehicle_params,
            per_env_params=per_env_params,
        )
        if obs.shape != (config.num_envs, 8):
            raise ValueError("The environment must return observations of shape (8,)")
        params = network.init(init_key, jnp.zeros((8,), dtype=jnp.float32))
        if actor_init is not None:
            params = replace_actor(params, actor_init)
        if params_init is not None:
            if jax.tree_util.tree_structure(params) != jax.tree_util.tree_structure(params_init):
                raise ValueError("PPO warm-start parameter tree does not match network")
            if any(a.shape != b.shape for a, b in zip(
                jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(params_init)
            )):
                raise ValueError("PPO warm-start parameter shapes do not match network")
            params = params_init
        params = project_log_std(params, config)
        train_state = TrainState.create(
            apply_fn=network.apply, params=params, tx=optimizer
        )
        # Discover the reward tree without advancing the environment or consuming a key.
        component_shapes = jax.eval_shape(
            lambda state: jax.vmap(env.step)(
                state,
                jnp.zeros((config.num_envs, 2)),
                jax.random.split(shape_key, config.num_envs),
            )[2],
            env_state,
        )
        if not component_shapes:
            raise ValueError("reward_components must be a nonempty dictionary")
        for name in shaping_components:
            if name not in component_shapes:
                raise ValueError(f"Unknown shaping component: {name}")
        components = jax.tree_util.tree_map(
            lambda shape: jnp.zeros(shape.shape, shape.dtype),
            component_shapes,
        )
        initial = RunnerState(
            train_state,
            (
                update_obs_stats(ObsStats.empty(), obs)
                if obs_stats_init is None
                else obs_stats_init
            ),
            obs,
            env_state,
            key,
            components,
            jnp.zeros(config.num_envs, dtype=jnp.int32),
        )

        def update(runner, iteration):
            def rollout(runner, _):
                key, action_key, step_key, reset_key = jax.random.split(runner.key, 4)
                normalized = normalize_obs(runner.obs, runner.obs_stats)
                mean, log_std, value = network.apply(
                    runner.train_state.params, normalized
                )
                action, log_prob = sample_action(action_key, mean, log_std)
                step = step_envs(
                    env,
                    runner.env_state,
                    action,
                    jax.random.split(step_key, config.num_envs),
                    jax.random.split(reset_key, config.num_envs),
                    body_params,
                    vehicle_params,
                    per_env_params=per_env_params,
                )
                _, _, next_value = network.apply(
                    runner.train_state.params,
                    normalize_obs(step.final_obs, runner.obs_stats),
                )
                totals = jax.tree_util.tree_map(
                    jnp.add,
                    runner.episode_components,
                    step.reward_components,
                )
                lengths = runner.episode_lengths + 1
                completed = jax.tree_util.tree_map(
                    lambda total: jnp.where(step.done, total, 0),
                    totals,
                )
                transition = {
                    "obs": normalized,
                    "action": action,
                    "log_prob": log_prob,
                    "value": value,
                    "next_value": next_value,
                    "reward": sum(step.reward_components.values()),
                    "done": step.done,
                    "truncated": step.info["truncated"],
                    "outcome": step.info["outcome"],
                    "completed_components": completed,
                    "completed_length": jnp.where(step.done, lengths, 0),
                    "raw_next_obs": step.obs,
                }
                runner = runner.replace(
                    obs=step.obs,
                    env_state=step.state,
                    key=key,
                    episode_components=jax.tree_util.tree_map(
                        lambda total: jnp.where(step.done, 0, total),
                        totals,
                    ),
                    episode_lengths=jnp.where(step.done, 0, lengths),
                )
                return runner, transition

            runner, rollout_data = jax.lax.scan(
                rollout, runner, None, length=config.num_steps
            )
            advantages, returns = compute_gae(
                # Uniform optimizer scaling preserves reward preferences. Logs
                # and evaluation retain the original sum of named components.
                rollout_data["reward"] * config.reward_scale,
                rollout_data["value"],
                rollout_data["next_value"],
                rollout_data["done"],
                rollout_data["truncated"],
                config.gamma,
                config.gae_lambda,
            )
            batch = {
                name: rollout_data[name]
                for name in ("obs", "action", "log_prob", "value")
            }
            batch.update(advantage=advantages, **{"return": returns})
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((config.batch_size,) + x.shape[2:]),
                batch,
            )

            def epoch(carry, _):
                train_state, key = carry
                key, shuffle_key = jax.random.split(key)
                indices = jax.random.permutation(shuffle_key, config.batch_size)
                minibatches = jax.tree_util.tree_map(
                    lambda x: x[indices].reshape(
                        (config.num_minibatches, -1) + x.shape[1:]
                    ),
                    batch,
                )

                def minibatch_step(train_state, minibatch):
                    (_, metrics), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
                        train_state.params,
                        network.apply,
                        minibatch,
                        config,
                    )
                    metrics["grad_norm"] = optax.tree.norm(grads)
                    updated = train_state.apply_gradients(grads=grads)
                    updated = updated.replace(params=project_log_std(updated.params, config))
                    return updated, metrics

                train_state, metrics = jax.lax.scan(
                    minibatch_step, train_state, minibatches
                )
                return (train_state, key), metrics

            (train_state, key), loss_metrics = jax.lax.scan(
                epoch,
                (runner.train_state, runner.key),
                None,
                length=config.update_epochs,
            )
            count = rollout_data["done"].sum()
            episode_means = jax.tree_util.tree_map(
                lambda x: x.sum() / jnp.maximum(count, 1),
                rollout_data["completed_components"],
            )
            metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
            metrics.update(
                env_steps=(iteration + 1) * config.batch_size,
                learning_rate=learning_rate(iteration * updates_per_rollout),
                action_std=jnp.exp(train_state.params["params"]["log_std"]),
                episodes=count,
                episode_length=rollout_data["completed_length"].sum()
                / jnp.maximum(count, 1),
                episode_return=sum(episode_means.values()),
                soft_landing_rate=((rollout_data["outcome"] == 1) & rollout_data["done"]).sum()
                / jnp.maximum(count, 1),
                reward_components=episode_means,
                landed=((rollout_data["outcome"] == 1) & rollout_data["done"]).sum(),
                crashed=((rollout_data["outcome"] == 2) & rollout_data["done"]).sum(),
                timeouts=((rollout_data["outcome"] == 3) & rollout_data["done"]).sum(),
                shaping_check_available=jnp.asarray(bool(shaping_components))
                & (count > 0),
                shaping_to_terminal_ratio=jnp.abs(
                    sum(episode_means[name] for name in shaping_components)
                )
                / 100.0,
            )
            runner = runner.replace(
                train_state=train_state,
                key=key,
                obs_stats=runner.obs_stats if config.freeze_obs_stats else update_obs_stats(
                    runner.obs_stats, rollout_data["raw_next_obs"]
                ),
            )
            return runner, metrics

        runner, start = initial, 0
        snapshots, metric_parts = [], []
        boundaries = [step // config.batch_size for step in checkpoint_steps]
        for stop in (*boundaries, config.num_updates):
            runner, metrics = jax.lax.scan(update, runner, jnp.arange(start, stop))
            metric_parts.append(metrics)
            if stop in boundaries:
                snapshots.append(runner)
            start = stop
        metrics = jax.tree_util.tree_map(
            lambda *parts: jnp.concatenate(parts), *metric_parts
        )
        return TrainResult(runner, metrics, tuple(snapshots))

    return jax.jit(train)
