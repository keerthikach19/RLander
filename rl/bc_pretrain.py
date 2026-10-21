"""Fit the actor MLP to PD actions from randomized Europa rollouts.

uv run python -m rl.bc_pretrain --output-dir ckpts/europa_bc
PPO: uv run python train.py +bc_init=ckpts/europa_bc/actor_init.msgpack ...
The init stores actor weights and normalization, never critic/optimizer/log-std.
"""

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from rl.fallback import pd_action
from rl.networks import (
    ACTOR_LAYERS,
    ActorCritic,
    ObsStats,
    normalize_obs,
    replace_actor,
    update_obs_stats,
)
from rl.vec_env import reset_envs, step_envs


def collect_pd(env, body, vehicle, *, steps=200_000, num_envs=32, seed=31415):
    """Every row is a pre-step noisy observation and its PD action label."""
    if steps < 1 or num_envs < 1:
        raise ValueError("steps and num_envs must be positive")
    rollout_steps = math.ceil(steps / num_envs)

    @jax.jit
    def collect(key):
        key, reset_key = jax.random.split(key)
        obs, state = reset_envs(
            env, jax.random.split(reset_key, num_envs), body, vehicle
        )

        def step(carry, _):
            obs, state, key = carry
            action = pd_action(obs, body, vehicle)
            key, step_key, reset_key = jax.random.split(key, 3)
            result = step_envs(
                env,
                state,
                action,
                jax.random.split(step_key, num_envs),
                jax.random.split(reset_key, num_envs),
                body,
                vehicle,
            )
            return (result.obs, result.state, key), (
                obs,
                action,
                result.info["outcome"],
            )

        _, (observations, actions, outcomes) = jax.lax.scan(
            step, (obs, state, key), None, length=rollout_steps
        )
        return observations.reshape(-1, 8), actions.reshape(-1, 2), outcomes

    obs, actions, outcomes = collect(jax.random.key(seed))
    counts = {
        name: int((outcomes == code).sum())
        for code, name in ((1, "soft_landings"), (2, "crashed"), (3, "timeouts"))
    }
    return obs, actions, {"steps": len(obs), "seed": seed, **counts}


def fit_actor(observations, actions, *, epochs=80, batch_size=512, seed=2718):
    """Uniform MSE on raw actor means; no reward or critic fitting."""
    if epochs < 1 or batch_size < 1 or len(observations) < batch_size:
        raise ValueError("Need positive epochs/batch_size and at least one batch")
    stats = update_obs_stats(ObsStats.empty(), observations)
    normalized = normalize_obs(observations, stats)
    network = ActorCritic(initial_log_std=-1.5)
    key, init_key = jax.random.split(jax.random.key(seed))
    fresh = network.init(init_key, jnp.zeros(8))
    actor = {name: fresh["params"][name] for name in ACTOR_LAYERS}
    batches = len(observations) // batch_size
    optimizer = optax.adam(
        optax.cosine_decay_schedule(1e-3, epochs * batches, alpha=0.1)
    )

    def loss(actor, obs, targets):
        mean, _, _ = network.apply(replace_actor(fresh, actor), obs)
        return jnp.mean(jnp.square(mean - targets))

    @jax.jit
    def train(actor, opt_state, key):
        def epoch(carry, _):
            actor, opt_state, key = carry
            key, shuffle = jax.random.split(key)
            indices = jax.random.permutation(shuffle, len(observations))
            indices = indices[: batches * batch_size].reshape(batches, batch_size)

            def batch(carry, indices):
                actor, opt_state = carry
                mse, grads = jax.value_and_grad(loss)(
                    actor, normalized[indices], actions[indices]
                )
                updates, opt_state = optimizer.update(grads, opt_state, actor)
                return (optax.apply_updates(actor, updates), opt_state), mse

            (actor, opt_state), losses = jax.lax.scan(
                batch, (actor, opt_state), indices
            )
            return (actor, opt_state, key), losses.mean()

        (actor, _, _), losses = jax.lax.scan(
            epoch, (actor, opt_state, key), None, length=epochs
        )
        return actor, losses

    initial_mse = float(loss(actor, normalized, actions))
    actor, losses = train(actor, optimizer.init(actor), key)
    return (
        actor,
        stats,
        {
            "initial_train_mse": initial_mse,
            "final_train_mse": float(loss(actor, normalized, actions)),
            "epoch_mse": np.asarray(losses).tolist(),
            "epochs": epochs,
            "batch_size": batch_size,
            "fit_seed": seed,
        },
    )


def save_bc_checkpoint(path, actor, stats):
    payload = {
        "format_version": 1,
        "body": "europa",
        "vehicle": "europa_lander",
        "actor": jax.device_get(actor),
        "obs_stats": serialization.to_state_dict(jax.device_get(stats)),
    }
    # Exclusive creation prevents accidental replacement of a validated init.
    with Path(path).open("xb") as file:
        file.write(serialization.msgpack_serialize(payload))


def load_bc_checkpoint(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    payload = serialization.msgpack_restore(data)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported BC checkpoint format")
    actor = jax.tree_util.tree_map(jnp.asarray, payload["actor"])
    stats = ObsStats(**jax.tree_util.tree_map(jnp.asarray, payload["obs_stats"]))
    template = ActorCritic().init(jax.random.key(0), jnp.zeros(8))
    replace_actor(template, actor)
    if (
        stats.mean.shape != (8,)
        or stats.var.shape != (8,)
        or stats.count.shape != ()
        or float(stats.count) <= 0
        or np.any(np.asarray(stats.var) < 0)
    ):
        raise ValueError("Invalid BC observation statistics")
    if not all(
        np.isfinite(np.asarray(x)).all()
        for x in jax.tree_util.tree_leaves((actor, stats))
    ):
        raise ValueError("Non-finite BC initialization")
    return (
        actor,
        stats,
        {
            "type": "pd_behavior_cloning",
            "path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "body": payload["body"],
            "vehicle": payload["vehicle"],
        },
    )


def pretrain(output_dir, *, steps=200_000, epochs=80):
    from omegaconf import OmegaConf

    from train import _jsonable, evaluate_smoke, lander_bindings

    if steps < 200_000:
        raise ValueError("Europa BC pretraining requires at least 200,000 PD steps")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    bindings = lander_bindings(OmegaConf.create({}))
    args = (bindings.env, bindings.body_params, bindings.vehicle_params)
    print(f"Collecting {steps:,} randomized PD training transitions", flush=True)
    observations, actions, collection = collect_pd(*args, steps=steps)
    # Separate rollout stream, never validation seeds or PPO experience.
    held_obs, held_actions, held_collection = collect_pd(
        *args, steps=20_000, seed=31416
    )
    print(f"Fitting actor for {epochs} epochs", flush=True)
    actor, stats, fit = fit_actor(observations, actions, epochs=epochs)
    fresh = ActorCritic(initial_log_std=-1.5).init(jax.random.key(0), jnp.zeros(8))
    params = replace_actor(fresh, actor)
    mean, _, _ = ActorCritic().apply(params, normalize_obs(held_obs, stats))
    fit["held_out_mse"] = float(jnp.square(mean - held_actions).mean())
    fit["held_out_mse_per_action"] = np.asarray(
        jnp.square(mean - held_actions).mean(axis=0)
    ).tolist()
    checkpoint = output_dir / "actor_init.msgpack"
    save_bc_checkpoint(checkpoint, actor, stats)
    _, _, init_metadata = load_bc_checkpoint(checkpoint)
    evaluation = evaluate_smoke(*args, params, stats, seeds=tuple(range(2000, 2100)))
    report = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "jax_version": jax.__version__,
        "collection": collection,
        "held_out_collection": held_collection,
        "fit": fit,
        "initialization": init_metadata,
        "body_params": bindings.body_params,
        "vehicle_params": bindings.vehicle_params,
        "evaluation": evaluation,
    }
    (output_dir / "report.json").write_text(
        json.dumps(_jsonable(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "fit": fit,
                "evaluation": {
                    k: evaluation[k] for k in ("soft_landings", "crashed", "timeouts")
                },
            }
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="ckpts/europa_bc")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    pretrain(args.output_dir, steps=args.steps, epochs=args.epochs)
