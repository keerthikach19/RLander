"""Hydra training with LanderEnv and a frozen-seed smoke evaluation.

The default adapter loads the named body/vehicle from configs/. W3's database
and full evaluation pipeline remain pending; the smoke evaluator uses only the
documented environment outputs. +adapter=module:function can override bindings.

Exact 50k smoke: +body=europa +vehicle=europa_lander +seed=1
    +ppo.num_envs=16 +ppo.num_steps=125 +ppo.total_steps=50000
"""

import hashlib
import importlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import serialization
from omegaconf import DictConfig, OmegaConf

from rl.networks import policy_action
from rl.ppo import PPOConfig, make_train


@dataclass
class TrainingBindings:
    """Explicit dependency injection until integration contracts are provided.

    record_run receives the §8 runs row as a dict. evaluate receives the policy
    parameters and ObsStats, and returns a JSON-serializable metric summary.
    Frozen-seed evaluation and SQLite ownership remain with W3.
    """

    env: Any
    body_params: Any
    vehicle_params: Any
    shaping_components: tuple[str, ...] = ()
    record_run: Callable | None = None
    evaluate: Callable | None = None
    per_env_params: bool = False
    reward_id: str | None = None
    evaluation_config: dict | None = None


def evaluate_smoke(
    env,
    body_params,
    vehicle_params,
    params,
    obs_stats,
    *,
    seeds=tuple(range(10_000, 10_100)),
    max_steps=1024,
    policy_fn=None,
    policy_label="deterministic_mean",
):
    """Temporary W2 smoke summary until W3's evaluation pipeline is merged.

    Success is determined only from terminal info and body.success (§5.6),
    independently of rewards. Each frozen seed has its own reset/noise stream;
    episodes stop at their first done and never auto-reset during evaluation.
    """
    if not seeds or max_steps < 1:
        raise ValueError("Evaluation needs seeds and a positive max_steps")

    def episode(seed):
        key, reset_key = jax.random.split(jax.random.key(seed))
        obs, state = env.reset(reset_key, body_params, vehicle_params)
        shapes = jax.eval_shape(env.step, state, jnp.zeros(2), key)
        zeros = lambda shape: jnp.zeros(shape.shape, shape.dtype)
        totals = jax.tree_util.tree_map(zeros, shapes[2])
        info = jax.tree_util.tree_map(zeros, shapes[4])

        def step(carry):
            obs, state, key, _, count, totals, _ = carry
            key, step_key = jax.random.split(key)
            action = (
                policy_action(params, obs_stats, obs)
                if policy_fn is None
                else policy_fn(obs)
            )
            obs, state, components, done, info = env.step(state, action, step_key)
            totals = jax.tree_util.tree_map(jnp.add, totals, components)
            return obs, state, key, done, count + 1, totals, info

        final_obs, _, _, done, count, totals, info = jax.lax.while_loop(
            lambda c: ~c[3] & (c[4] < max_steps),
            step,
            (obs, state, key, jnp.asarray(False), jnp.asarray(0), totals, info),
        )
        thresholds = body_params["success"]
        success = (
            done
            & (info["landed"] > 0)
            & (jnp.abs(info["touchdown_v"]) < thresholds["vy_max_mps"])
            & (jnp.abs(info["touchdown_vx"]) < thresholds["vx_max_mps"])
        )
        return {
            "finished": done,
            "steps": count,
            "success": success,
            "info": info,
            "reward_components": totals,
            "final_observation": final_obs,
        }

    episodes = jax.device_get(
        jax.jit(jax.vmap(episode))(jnp.asarray(seeds, dtype=jnp.uint32))
    )
    if not np.asarray(episodes["finished"]).all():
        raise RuntimeError("Smoke evaluation exceeded max_steps before env termination")
    info = episodes["info"]
    contact = np.isin(info["outcome"], [1, 2])

    def contact_mean(values):
        return float(np.mean(np.asarray(values)[contact])) if contact.any() else None

    return {
        "evaluation_type": "smoke",
        "policy": policy_label,
        "seeds": list(seeds),
        "n_episodes": len(seeds),
        "soft_landings": int(episodes["success"].sum()),
        "soft_landing_rate": float(episodes["success"].mean()),
        "landed": int((info["outcome"] == 1).sum()),
        "crashed": int((info["outcome"] == 2).sum()),
        "timeouts": int((info["outcome"] == 3).sum()),
        "touchdown_abs_vy_mean_mps": contact_mean(np.abs(info["touchdown_v"])),
        "touchdown_abs_vx_mean_mps": contact_mean(np.abs(info["touchdown_vx"])),
        "fuel_used_mean_kg": float(np.mean(info["fuel_used"])),
        "episode_steps_mean": float(episodes["steps"].mean()),
        "final_observed_height_over_L_mean": float(
            episodes["final_observation"][:, 0].mean()
        ),
        "reward_components_mean": {
            name: float(values.mean())
            for name, values in episodes["reward_components"].items()
        },
        "episodes": _jsonable(episodes),
    }


def lander_bindings(cfg):
    """Load numeric parameter pytrees and bind the public LanderEnv API."""
    from env.lander_env import LanderEnv

    config_dir = Path(__file__).resolve().parent / "configs"

    def load_params(filename, name):
        row = OmegaConf.to_container(
            OmegaConf.load(config_dir / filename)[name], resolve=True
        )
        # Names are metadata, not leaves of the compile-once numeric pytree.
        row.pop("name", None)
        return jax.tree_util.tree_map(
            lambda value: jnp.asarray(value, dtype=jnp.float32), row
        )

    body = load_params("bodies.yaml", cfg.get("body", "europa"))
    vehicle = load_params("vehicles.yaml", cfg.get("vehicle", "europa_lander"))
    vehicle["max_thrust_N"] = vehicle["twr_max"] * vehicle["mass0_kg"] * body["g"]
    reward_cfg = cfg.get("reward", {})
    if set(reward_cfg) - {"mode", "potential_scale"}:
        raise ValueError("reward accepts only mode and potential_scale; gamma uses PPO")
    mode = reward_cfg.get("mode", "baseline")
    if mode not in ("baseline", "potential_profile"):
        raise ValueError("reward.mode must be baseline or potential_profile")
    scale = float(reward_cfg.get("potential_scale", 20.0))
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("reward.potential_scale must be finite and nonnegative")
    gamma = float(cfg.get("ppo", {}).get("gamma", PPOConfig.gamma))
    if not 0 <= gamma <= 1:
        raise ValueError("PPO gamma must be in [0, 1]")
    vehicle.update(
        reward_mode=jnp.float32(mode == "potential_profile"),
        reward_gamma=jnp.float32(gamma),
        reward_potential_scale=jnp.float32(scale),
    )
    eval_cfg = cfg.get("evaluation", {})
    seed_start = int(eval_cfg.get("seed_start", 10_000))
    episodes = int(eval_cfg.get("episodes", 100))
    if seed_start < 0 or episodes < 1 or seed_start + episodes > 2**32:
        raise ValueError("Evaluation needs a nonempty uint32 seed range")
    seeds = tuple(range(seed_start, seed_start + episodes))
    return TrainingBindings(
        LanderEnv(),
        body,
        vehicle,
        shaping_components=("velocity", "altitude", "fuel", "angle"),
        evaluate=lambda params, stats: evaluate_smoke(
            LanderEnv(), body, vehicle, params, stats, seeds=seeds
        ),
        reward_id=f"hand_shaped_{mode}_v1",
        evaluation_config={"seed_start": seed_start, "episodes": episodes},
    )


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (jax.Array, np.ndarray, np.generic)):
        return np.asarray(value).tolist()
    return value


def run_experiment(
    bindings,
    config,
    *,
    seed=1,
    body_name="europa",
    vehicle_name="europa_lander",
    output_root="ckpts",
    smoke_success_rate=None,
):
    """Train, checkpoint, log diagnostics, and invoke supplied integration hooks.

    Without hooks, the manifest/printed summary explicitly mark DB/evaluation
    pending; a fake-env test is never reported as a planetary landing result.
    Full runs require evidence from a prior smoke evaluation (>10% success).
    """
    if config.total_steps > 1_000_000 and (
        smoke_success_rate is None or not 0.1 < smoke_success_rate <= 1.0
    ):
        raise ValueError(
            "A full run requires a prior smoke_success_rate > 0.10 (and <= 1)"
        )
    # The invariance argument requires the same discount in shaping and PPO.
    # Validate here as callers may build bindings and PPOConfig independently.
    if isinstance(bindings.vehicle_params, dict):
        mode = np.asarray(bindings.vehicle_params.get("reward_mode", 0))
        gamma = np.asarray(bindings.vehicle_params.get("reward_gamma", config.gamma))
        if np.any((mode == 1) & ~np.isclose(gamma, config.gamma, rtol=0, atol=1e-7)):
            raise ValueError("Potential reward_gamma must match PPO gamma")
    run_id = (
        f"ppo-{body_name}-{seed}-{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    )
    if any(c in body_name for c in ("/", "\\", ":")):
        raise ValueError("body_name must be a label, not a path")
    run_dir = Path(output_root).resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    run_config = _jsonable(
        {
            "ppo": asdict(config),
            "seed": seed,
            "body": body_name,
            "vehicle": vehicle_name,
            "body_params": bindings.body_params,
            "vehicle_params": bindings.vehicle_params,
            "shaping_components": bindings.shaping_components,
            "per_env_params": bindings.per_env_params,
            "smoke_success_rate": smoke_success_rate,
            "evaluation": bindings.evaluation_config,
        }
    )
    config_json = json.dumps(run_config, sort_keys=True, allow_nan=False)
    metadata = {
        "id": run_id,
        "train_bodies_json": json.dumps([body_name]),
        "vehicle": vehicle_name,
        "policy_type": "ppo",
        "reward_id": bindings.reward_id,
        "seed": seed,
        "config_hash": hashlib.sha256(config_json.encode()).hexdigest(),
        "jax_version": jax.__version__,
        "checkpoint_path": str(run_dir / "checkpoint"),
        "created_at": datetime.now(UTC).isoformat(),
    }
    manifest = {
        "run": metadata,
        "config": run_config,
        "checkpoint_format_version": 1,
        "status": "training",
        "database_status": "pending",
        "evaluation_status": "pending",
    }

    def write_manifest():
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    write_manifest()
    with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
        log.write(json.dumps({"event": "config", **manifest}, allow_nan=False) + "\n")
        log.flush()
        result = make_train(
            bindings.env,
            config,
            shaping_components=bindings.shaping_components,
            per_env_params=bindings.per_env_params,
        )(jax.random.key(seed), bindings.body_params, bindings.vehicle_params)
        metrics = jax.device_get(result.metrics)
        for index in range(config.num_updates):
            row = _jsonable(jax.tree_util.tree_map(lambda x, i=index: x[i], metrics))
            if row["shaping_check_available"]:
                ratio = row["shaping_to_terminal_ratio"]
                row["shaping_magnitude_status"] = (
                    "below_target"
                    if ratio < 0.2
                    else "above_target"
                    if ratio > 1
                    else "in_range"
                )
            else:
                row["shaping_magnitude_status"] = "unavailable"
            log.write(json.dumps({"event": "update", **row}, allow_nan=False) + "\n")

    runner = result.runner
    payload = {
        "train_state": serialization.to_state_dict(runner.train_state),
        "obs_stats": serialization.to_state_dict(runner.obs_stats),
        "env_state": serialization.to_state_dict(runner.env_state),
        "obs": runner.obs,
        "rng_key_data": jax.random.key_data(runner.key),
        "episode_components": runner.episode_components,
        "episode_lengths": runner.episode_lengths,
    }
    # Host arrays make GPU-trained checkpoints portable to CPU evaluation without
    # restoring device/sharding assignments from the training machine.
    payload = jax.device_get(payload)
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(metadata["checkpoint_path"], payload)
    manifest.update(
        status="checkpoint_saved", actual_steps=config.num_updates * config.batch_size
    )
    write_manifest()
    if bindings.record_run is not None:
        bindings.record_run(metadata)
        manifest["database_status"] = "recorded"
        write_manifest()
    if bindings.evaluate is not None:
        manifest["evaluation"] = _jsonable(
            bindings.evaluate(runner.train_state.params, runner.obs_stats)
        )
        manifest["evaluation_status"] = "complete"
    else:
        manifest["evaluation"] = {"status": "pending W3 evaluation adapter"}
    manifest["status"] = "complete"
    write_manifest()
    print(
        json.dumps(
            {
                "run_id": run_id,
                "checkpoint_path": metadata["checkpoint_path"],
                "actual_steps": manifest["actual_steps"],
                "database_status": manifest["database_status"],
                "evaluation": {
                    k: v for k, v in manifest["evaluation"].items() if k != "episodes"
                },
            },
            allow_nan=False,
        )
    )
    return result, manifest


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg: DictConfig):
    if cfg.get("adapter"):
        module_name, function_name = cfg.adapter.rsplit(":", 1)
        bindings = getattr(importlib.import_module(module_name), function_name)(cfg)
    else:
        bindings = lander_bindings(cfg)
    overrides = OmegaConf.to_container(
        cfg.get("ppo", OmegaConf.create({})), resolve=True
    )
    run_experiment(
        bindings,
        PPOConfig(**overrides),
        seed=int(cfg.get("seed", 1)),
        body_name=cfg.get("body", "europa"),
        vehicle_name=cfg.get("vehicle", "europa_lander"),
        output_root=cfg.get("output_root", "ckpts"),
        smoke_success_rate=cfg.get("smoke_success_rate"),
    )


if __name__ == "__main__":
    main()
