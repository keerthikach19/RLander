"""Hydra training with LanderEnv and a frozen-seed smoke evaluation.

The default adapter loads the named body/vehicle from configs/. W3's database
and full evaluation pipeline remain pending; the smoke evaluator uses only the
documented environment outputs. +adapter=module:function can override bindings.

Exact 50k smoke: +body=europa +vehicle=europa_lander +seed=1
    +ppo.num_envs=16 +ppo.num_steps=125 +ppo.total_steps=50000

Actor-only BC warm start: +bc_init=ckpts/europa_bc/actor_init.msgpack
    imports its normalization, keeps critic/optimizer fresh, and defaults
    ppo.initial_log_std to -1.5 unless explicitly supplied.
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


def make_smoke_rollout(env, body_params, vehicle_params, *, max_steps=1024,
                       policy_fn=None, sample_actions=False):
    def episode(seed, params, obs_stats):
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
            if sample_actions:
                from rl.networks import ActorCritic, normalize_obs, sample_action

                mean, log_std, _ = ActorCritic().apply(params, normalize_obs(obs, obs_stats))
                # Preserve reset/sensor keys across mean and sampled comparisons.
                action_key = jax.random.fold_in(step_key, 0x53414D50)
                action = jnp.clip(sample_action(action_key, mean, log_std)[0], -1, 1)
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

    return jax.jit(jax.vmap(episode, in_axes=(0, None, None)))


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
    sample_actions=False,
    _rollout=None,
):
    """Temporary W2 smoke summary until W3's evaluation pipeline is merged.

    Success is determined only from terminal info and body.success (§5.6),
    independently of rewards. Each frozen seed has its own reset/noise stream;
    episodes stop at their first done and never auto-reset during evaluation.
    """
    if not seeds or max_steps < 1:
        raise ValueError("Evaluation needs seeds and a positive max_steps")
    if sample_actions and policy_fn is not None:
        raise ValueError("Sampled evaluation requires PPO parameters, not a custom policy")

    from rl.execution import execute
    rollout = _rollout or make_smoke_rollout(
        env, body_params, vehicle_params, max_steps=max_steps,
        policy_fn=policy_fn, sample_actions=sample_actions,
    )
    device_episodes, execution = execute(
        rollout, (jnp.asarray(seeds, dtype=jnp.uint32), params, obs_stats),
        label="validation_" + policy_label,
    )
    episodes = jax.device_get(device_episodes)
    if not np.asarray(episodes["finished"]).all():
        raise RuntimeError("Smoke evaluation exceeded max_steps before env termination")
    info = episodes["info"]
    contact = np.isin(info["outcome"], [1, 2])

    def contact_mean(values):
        return float(np.mean(np.asarray(values)[contact])) if contact.any() else None

    return {
        "evaluation_type": "smoke",
        "execution": execution,
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
    modes = {"baseline": 0, "potential_profile": 1, "landing_profile": 2,
             "landing_success": 3, "landing_margin": 4, "landing_vertical": 5}
    if mode not in modes:
        raise ValueError(f"reward.mode must be one of {tuple(modes)}")
    scale = float(reward_cfg.get("potential_scale", 20.0))
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("reward.potential_scale must be finite and nonnegative")
    gamma = float(cfg.get("ppo", {}).get("gamma", PPOConfig.gamma))
    if not 0 <= gamma <= 1:
        raise ValueError("PPO gamma must be in [0, 1]")
    vehicle.update(
        reward_mode=jnp.float32(modes[mode]),
        reward_gamma=jnp.float32(gamma),
        reward_potential_scale=jnp.float32(scale),
    )
    eval_cfg = cfg.get("evaluation", {})
    seed_start = int(eval_cfg.get("seed_start", 10_000))
    episodes = int(eval_cfg.get("episodes", 100))
    if seed_start < 0 or episodes < 1 or seed_start + episodes > 2**32:
        raise ValueError("Evaluation needs a nonempty uint32 seed range")
    seeds = tuple(range(seed_start, seed_start + episodes))
    rollout = make_smoke_rollout(LanderEnv(), body, vehicle)
    return TrainingBindings(
        LanderEnv(),
        body,
        vehicle,
        shaping_components=("velocity", "altitude", "fuel", "angle"),
        evaluate=lambda params, stats: evaluate_smoke(
            LanderEnv(), body, vehicle, params, stats, seeds=seeds, _rollout=rollout
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


PPO_ONLY_ABLATION_WAIVER = "user-approved-2026-10-18-ppo-only-2m"


def check_smoke_gate(
    bindings, config, *, seed, body_name, vehicle_name, bc_init,
    smoke_success_rate, ppo_only_ablation_waiver=None,
):
    """Keep the full-run gate except for the explicitly authorized negative ablation."""
    if ppo_only_ablation_waiver is not None:
        if not (
            ppo_only_ablation_waiver == PPO_ONLY_ABLATION_WAIVER
            and bc_init is None
            and body_name == "europa" and vehicle_name == "europa_lander"
            and seed in (1, 2, 3)
            and config.total_steps == 2_000_000
            and config.num_envs == 128 and config.num_steps == 125
            and bindings.reward_id == "hand_shaped_potential_profile_v1"
            and float(bindings.vehicle_params.get("reward_mode", -1)) == 1
            and float(bindings.vehicle_params.get("reward_potential_scale", -1)) == 100
        ):
            raise ValueError("PPO-only ablation waiver is outside its authorized scope")
        return {
            "status": "waived_for_ppo_only_ablation",
            "authorization": PPO_ONLY_ABLATION_WAIVER,
            "reason": "User explicitly waived >10% gate for the PPO-only 2M ablation, seeds 1-3",
            "prior_no_bc_validation": {"steps": 500_000, "soft": 0, "crash": 64, "timeout": 36},
            "smoke_passed": False,
        }
    if config.total_steps > 1_000_000 and (
        smoke_success_rate is None or not 0.1 < smoke_success_rate <= 1.0
    ):
        raise ValueError("A full run requires a prior smoke_success_rate > 0.10 (and <= 1)")
    return {"status": "passed" if config.total_steps > 1_000_000 else "bounded_run"}


def load_ppo_initialization(checkpoint, *, body_name, vehicle_name, initial_log_std):
    """Import actor, critic and normalization; explicitly reset exploration.

    This is a new fine-tuning schedule, not an exact runner/optimizer resume.
    Host-array checkpoints produced here are portable between CPU and GPU.
    """
    from rl.networks import ObsStats

    checkpoint = Path(checkpoint).resolve()
    parent = json.loads((checkpoint.parent / "manifest.json").read_text(encoding="utf-8"))
    if (parent["config"]["body"], parent["config"]["vehicle"]) != (body_name, vehicle_name):
        raise ValueError("PPO initialization body/vehicle does not match this run")
    with ocp.PyTreeCheckpointer() as reader:
        saved = reader.restore(str(checkpoint))
    params = jax.tree_util.tree_map(jnp.asarray, saved["train_state"]["params"])
    stats = serialization.from_state_dict(ObsStats.empty(), saved["obs_stats"])
    stats = jax.tree_util.tree_map(jnp.asarray, stats)
    if (stats.mean.shape != (8,) or stats.var.shape != (8,) or float(stats.count) <= 0
            or not np.isfinite(float(stats.count))
            or not np.isfinite(np.asarray(stats.mean)).all()
            or not np.isfinite(np.asarray(stats.var)).all()
            or np.any(np.asarray(stats.var) < 0)):
        raise ValueError("Invalid PPO initialization observation statistics")
    if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree_util.tree_leaves(params)):
        raise ValueError("PPO initialization parameters must be finite")
    saved_log_std = np.asarray(params["params"]["log_std"]).tolist()
    if initial_log_std is not None:
        params = {**params, "params": {**params["params"], "log_std": jnp.full_like(
            params["params"]["log_std"], initial_log_std
        )}}
    return params, stats, {
        "method": "ppo_actor_critic_warm_start", "checkpoint": str(checkpoint),
        "parent_run_id": parent["run"]["id"], "parent_steps": parent["actual_steps"],
        "parent_reward_id": parent["run"]["reward_id"],
        "parent_config_hash": parent["run"]["config_hash"],
        "body": body_name, "vehicle": vehicle_name,
        "saved_log_std": saved_log_std, "initial_log_std": initial_log_std,
        "critic_restored": True, "optimizer_restored": False, "environment_restored": False,
    }


def run_experiment(
    bindings,
    config,
    *,
    seed=1,
    body_name="europa",
    vehicle_name="europa_lander",
    output_root="ckpts",
    smoke_success_rate=None,
    bc_init=None,
    ppo_init=None,
    checkpoint_steps=(),
    ppo_only_ablation_waiver=None,
):
    """Train, checkpoint, log diagnostics, and invoke supplied integration hooks.

    Without hooks, the manifest/printed summary explicitly mark DB/evaluation
    pending; a fake-env test is never reported as a planetary landing result.
    Full runs require >10% smoke success, except the recorded PPO-only ablation waiver.
    """
    if ppo_init is not None and (bc_init is not None or ppo_only_ablation_waiver is not None):
        raise ValueError("PPO initialization cannot use BC initialization or the from-scratch waiver")
    smoke_gate = check_smoke_gate(
        bindings, config, seed=seed, body_name=body_name, vehicle_name=vehicle_name,
        bc_init=bc_init, smoke_success_rate=smoke_success_rate,
        ppo_only_ablation_waiver=ppo_only_ablation_waiver,
    )
    actor_init, params_init, obs_stats_init, init_metadata = None, None, None, None
    if bc_init is not None:
        from rl.bc_pretrain import load_bc_checkpoint

        actor_init, obs_stats_init, init_metadata = load_bc_checkpoint(bc_init)
        if (body_name, vehicle_name) != (
            init_metadata["body"],
            init_metadata["vehicle"],
        ):
            raise ValueError("BC checkpoint body/vehicle does not match this run")
    if ppo_init is not None:
        params_init, obs_stats_init, init_metadata = load_ppo_initialization(
            ppo_init, body_name=body_name, vehicle_name=vehicle_name,
            initial_log_std=config.initial_log_std,
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
            "smoke_gate": smoke_gate,
            "evaluation": bindings.evaluation_config,
            "actor_initialization": init_metadata,
            "checkpoint_steps": list(checkpoint_steps),
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
        from rl.execution import event, execute
        train_fn = make_train(
            bindings.env,
            config,
            shaping_components=bindings.shaping_components,
            per_env_params=bindings.per_env_params,
            checkpoint_steps=checkpoint_steps,
        )
        result, execution = execute(train_fn, (
            jax.random.key(seed),
            bindings.body_params,
            bindings.vehicle_params,
            actor_init,
            obs_stats_init,
            params_init,
        ), label="ppo_training")
        execution["environment_steps_per_execution_second"] = (
            config.num_updates * config.batch_size / execution["execution_seconds"]
        )
        manifest["execution"] = execution
        write_manifest()
        event("ppo_training", "writing_metrics_and_checkpoints_on_host")
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

    def save_runner(saved_runner, path):
        payload = {
            "train_state": serialization.to_state_dict(saved_runner.train_state),
            "obs_stats": serialization.to_state_dict(saved_runner.obs_stats),
            "env_state": serialization.to_state_dict(saved_runner.env_state),
            "obs": saved_runner.obs,
            "rng_key_data": jax.random.key_data(saved_runner.key),
            "episode_components": saved_runner.episode_components,
            "episode_lengths": saved_runner.episode_lengths,
        }
        # Host arrays keep checkpoints portable across execution devices.
        with ocp.PyTreeCheckpointer() as checkpointer:
            checkpointer.save(str(path), jax.device_get(payload))

    save_runner(runner, metadata["checkpoint_path"])
    manifest.update(
        status="checkpoint_saved", actual_steps=config.num_updates * config.batch_size
    )
    write_manifest()
    if bindings.record_run is not None:
        bindings.record_run(metadata)
        manifest["database_status"] = "recorded"
        write_manifest()
    if bindings.evaluate is not None:
        event("validation", "evaluating_final_checkpoint")
        manifest["evaluation"] = _jsonable(
            bindings.evaluate(runner.train_state.params, runner.obs_stats)
        )
        manifest["evaluation_status"] = "complete"
    else:
        manifest["evaluation"] = {"status": "pending W3 evaluation adapter"}
    snapshot_paths = {}
    for step, snapshot in zip(checkpoint_steps, result.checkpoints):
        event("validation", "saving_and_evaluating_snapshot", steps=step)
        snapshot_dir = run_dir / "snapshots" / str(step)
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        snapshot_path = snapshot_dir / "checkpoint"
        save_runner(snapshot, snapshot_path)
        snapshot_manifest = {
            **manifest,
            "run": {
                **metadata,
                "id": f"{run_id}-at-{step}",
                "checkpoint_path": str(snapshot_path),
            },
            "parent_run_id": run_id,
            "actual_steps": step,
            "status": "complete",
            "database_status": "pending",
            "evaluation": (
                _jsonable(
                    bindings.evaluate(snapshot.train_state.params, snapshot.obs_stats)
                )
                if bindings.evaluate is not None
                else manifest["evaluation"]
            ),
        }
        (snapshot_dir / "manifest.json").write_text(
            json.dumps(snapshot_manifest, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        snapshot_paths[str(step)] = str(snapshot_dir)
    if snapshot_paths:
        manifest["snapshots"] = snapshot_paths
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
    if cfg.get("bc_init") is not None:
        overrides.setdefault("initial_log_std", -1.5)
    run_experiment(
        bindings,
        PPOConfig(**overrides),
        seed=int(cfg.get("seed", 1)),
        body_name=cfg.get("body", "europa"),
        vehicle_name=cfg.get("vehicle", "europa_lander"),
        output_root=cfg.get("output_root", "ckpts"),
        smoke_success_rate=cfg.get("smoke_success_rate"),
        bc_init=cfg.get("bc_init"),
        ppo_init=cfg.get("ppo_init"),
        checkpoint_steps=tuple(cfg.get("checkpoint_steps", ())),
        ppo_only_ablation_waiver=cfg.get("ppo_only_ablation_waiver"),
    )


if __name__ == "__main__":
    main()
