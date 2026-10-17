"""eval/run_eval.py — Deterministic evaluation runner for RLander (W3, §9).

Contract requirements:
1. Re-run deterministically from a checkpoint on frozen test seeds 10000-10099
   (100 episodes).
2. Success metric reads ONLY info["outcome"] == 1 (landed_soft per §5.2 contract),
   reading thresholds from bodies.yaml `success:`, and NEVER reads reward.
3. Supports --eval-body different from training bodies (for out-of-distribution OOD eval).
4. Writes per-run eval row (Wilson 95% CI, touchdown stats, fuel) to SQLite and
   per-episode trajectories to Parquet files (data/trajectories/{run_id}/{seed}.parquet).
5. Supports PD fallback and random policy as stand-ins for checkpoints in tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import yaml

from eval.metrics import aggregate_episodes, outcome_name, OUTCOME_LANDED
from storage.db import (
    get_body,
    get_run,
    get_vehicle,
    init_db,
    insert_episode,
    insert_eval,
    upsert_body,
    upsert_vehicle,
)
from storage.parquet_io import trajectory_path, write_trajectory

_REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Default body / vehicle configurations (PLAN.md §6 fallbacks)
# ---------------------------------------------------------------------------

DEFAULT_BODIES: dict[str, dict[str, Any]] = {
    "europa": {
        "name": "europa",
        "g": 1.315,
        "radius_m": 1560800.0,
        "rho0": 0.0,
        "scale_height_m": 1.0,
        "success": {"vy_max_mps": 1.0, "vx_max_mps": 0.5},
    },
    "mars": {
        "name": "mars",
        "g": 3.71,
        "radius_m": 3389500.0,
        "rho0": 0.020,
        "scale_height_m": 11100.0,
        "success": {"vy_max_mps": 1.0, "vx_max_mps": 0.5},
    },
    "titan": {
        "name": "titan",
        "g": 1.352,
        "radius_m": 2574700.0,
        "rho0": 5.4,
        "scale_height_m": 20000.0,
        "success": {"vy_max_mps": 1.0, "vx_max_mps": 0.5},
    },
    "enceladus": {
        "name": "enceladus",
        "g": 0.113,
        "radius_m": 252100.0,
        "rho0": 0.0,
        "scale_height_m": 1.0,
        "success": {"vy_max_mps": 1.0, "vx_max_mps": 0.5},
    },
}

DEFAULT_VEHICLES: dict[str, dict[str, Any]] = {
    "europa_lander": {
        "name": "europa_lander",
        "mass0_kg": 800.0,
        "dry_mass_kg": 320.0,
        "twr_max": 3.0,
        "min_throttle": 0.0,
        "isp_s": 300.0,
        "gimbal_limit_rad": 0.2618,
        "cd": 1.2,
        "area_m2": 4.0,
        "h0_m": 2000.0,
        "v0_mps": 20.0,
    }
}


def load_body_config(
    body_name: str,
    conn: sqlite3.Connection | None = None,
    configs_dir: Path | None = None,
) -> dict[str, Any]:
    """Load configuration dictionary for body_name from YAML, DB, or defaults."""
    cdir = configs_dir or (_REPO_ROOT / "configs")
    yaml_path = cdir / "bodies.yaml"
    if yaml_path.exists():
        try:
            with yaml_path.open("r", encoding="utf-8") as f:
                bodies_yaml = yaml.safe_load(f)
                if isinstance(bodies_yaml, dict) and body_name in bodies_yaml:
                    cfg = dict(bodies_yaml[body_name])
                    if "name" not in cfg:
                        cfg["name"] = body_name
                    return cfg
        except Exception:
            pass

    if conn is not None:
        row = get_body(conn, body_name)
        if row is not None:
            return dict(row["config"])

    if body_name in DEFAULT_BODIES:
        return dict(DEFAULT_BODIES[body_name])

    raise KeyError(f"Configuration for body {body_name!r} not found.")


def load_vehicle_config(
    vehicle_name: str,
    conn: sqlite3.Connection | None = None,
    configs_dir: Path | None = None,
) -> dict[str, Any]:
    """Load configuration dictionary for vehicle_name from YAML, DB, or defaults."""
    cdir = configs_dir or (_REPO_ROOT / "configs")
    yaml_path = cdir / "vehicles.yaml"
    if yaml_path.exists():
        try:
            with yaml_path.open("r", encoding="utf-8") as f:
                vehicles_yaml = yaml.safe_load(f)
                if isinstance(vehicles_yaml, dict) and vehicle_name in vehicles_yaml:
                    cfg = dict(vehicles_yaml[vehicle_name])
                    if "name" not in cfg:
                        cfg["name"] = vehicle_name
                    return cfg
        except Exception:
            pass

    if conn is not None:
        row = get_vehicle(conn, vehicle_name)
        if row is not None:
            return dict(row["config"])

    if vehicle_name in DEFAULT_VEHICLES:
        return dict(DEFAULT_VEHICLES[vehicle_name])

    raise KeyError(f"Configuration for vehicle {vehicle_name!r} not found.")


# ---------------------------------------------------------------------------
# Policy builder
# ---------------------------------------------------------------------------

def make_policy_fn(
    policy_type: str,
    checkpoint_path: str | None = None,
    body_params: dict[str, Any] | None = None,
    vehicle_params: dict[str, Any] | None = None,
    custom_policy: Any = None,
) -> Callable[[jnp.ndarray, jax.Array], jnp.ndarray]:
    """Build a deterministic callable (obs, key) -> action of shape (2,) in [-1, 1]."""
    if custom_policy is not None:
        if callable(custom_policy):
            # Check how many arguments it accepts
            import inspect
            sig = inspect.signature(custom_policy)
            num_params = len(sig.parameters)
            if num_params == 1:
                return lambda obs, key: custom_policy(obs)
            elif num_params == 2:
                return lambda obs, key: custom_policy(obs, key)
            elif num_params >= 3:
                return lambda obs, key: custom_policy(obs, body_params, vehicle_params)
            return custom_policy
        raise TypeError(f"custom_policy must be callable, got {type(custom_policy)}")

    if policy_type == "pd_fallback":
        from rl.fallback import pd_action
        clean_body = {
            "g": float(body_params["g"]),
            "rho0": float(body_params["rho0"]),
            "scale_height_m": float(body_params["scale_height_m"]),
        }
        clean_vehicle = {
            "h0_m": float(vehicle_params["h0_m"]),
            "mass0_kg": float(vehicle_params["mass0_kg"]),
            "dry_mass_kg": float(vehicle_params["dry_mass_kg"]),
            "cd": float(vehicle_params["cd"]),
            "area_m2": float(vehicle_params["area_m2"]),
            "gimbal_limit_rad": float(vehicle_params["gimbal_limit_rad"]),
            "twr_max": float(vehicle_params["twr_max"]),
            "min_throttle": float(vehicle_params["min_throttle"]),
        }
        jitted_pd = jax.jit(pd_action)
        return lambda obs, key: jitted_pd(obs, clean_body, clean_vehicle)

    if policy_type == "random":
        return lambda obs, key: jax.random.uniform(key, (2,), minval=-1.0, maxval=1.0)

    if policy_type == "ppo" or checkpoint_path is not None:
        if not checkpoint_path:
            raise ValueError("policy_type is 'ppo' but no checkpoint_path was provided")
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.is_absolute():
            ckpt_p = _REPO_ROOT / ckpt_p
        if not ckpt_p.exists():
            raise FileNotFoundError(f"Checkpoint path not found: {ckpt_p}")

        import orbax.checkpoint as ocp
        from flax import serialization
        from rl.networks import ActorCritic, ObsStats, policy_action

        with ocp.PyTreeCheckpointer() as checkpointer:
            restored = checkpointer.restore(str(ckpt_p))

        if isinstance(restored, dict) and "train_state" in restored and "params" in restored["train_state"]:
            params = restored["train_state"]["params"]
        elif isinstance(restored, dict) and "params" in restored:
            params = restored["params"]
        else:
            params = restored

        if isinstance(restored, dict) and "obs_stats" in restored:
            obs_stats = serialization.from_state_dict(ObsStats.empty(8), restored["obs_stats"])
        else:
            obs_stats = ObsStats.empty(8)

        jitted_policy = jax.jit(lambda o: policy_action(params, obs_stats, o))
        return lambda obs, key: jitted_policy(obs)

    raise ValueError(f"Unknown policy_type: {policy_type!r}")


# ---------------------------------------------------------------------------
# Core Evaluation Function
# ---------------------------------------------------------------------------

def run_evaluation(
    run_id: int,
    *,
    eval_body: str | None = None,
    checkpoint_path: str | None = None,
    policy_override: str | None = None,
    custom_policy: Any = None,
    db_path: str | Path | None = None,
    conn: sqlite3.Connection | None = None,
    seeds: Sequence[int] = range(10000, 10100),
    traj_base_dir: str | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Execute deterministic evaluation on frozen seeds, writing to DB and Parquet.

    Args:
        run_id:           Run ID from the SQLite `runs` table.
        eval_body:        Target celestial body (e.g. "europa", "mars").
                          Defaults to run's first training body.
        checkpoint_path:  Path to Orbax checkpoint (overrides runs.checkpoint_path).
        policy_override:  "pd_fallback" | "random" | "ppo" (overrides runs.policy_type).
        custom_policy:    Optional callable stand-in for policy.
        db_path:          Path to SQLite file (default data/rlander.db).
        conn:             Optional active sqlite3.Connection.
        seeds:            Frozen seeds sequence (default 10000..10099, 100 episodes).
        traj_base_dir:    Base dir for Parquet trajectories (default data/trajectories).
        quiet:            If True, suppress stdout summary.

    Returns:
        Dict of evaluation metrics and run info.
    """
    from env.lander_env import LanderEnv

    close_conn_on_exit = False
    if conn is None:
        conn = init_db(db_path)
        close_conn_on_exit = True

    try:
        run = get_run(conn, run_id)
        if run is None:
            raise ValueError(f"Run id {run_id} not found in database.")

        train_bodies = run.get("train_bodies") or ["europa"]
        target_body = eval_body if eval_body is not None else train_bodies[0]
        vehicle_name = run.get("vehicle", "europa_lander")
        eff_policy_type = policy_override or run.get("policy_type", "ppo")
        eff_ckpt_path = checkpoint_path or run.get("checkpoint_path")

        # Load configs
        body_config = load_body_config(target_body, conn=conn)
        vehicle_config = load_vehicle_config(vehicle_name, conn=conn)

        # Upsert body & vehicle into DB to guarantee foreign keys
        upsert_body(conn, target_body, body_config)
        upsert_vehicle(conn, vehicle_name, vehicle_config)

        # Build policy callable
        policy_fn = make_policy_fn(
            policy_type=eff_policy_type,
            checkpoint_path=eff_ckpt_path,
            body_params=body_config,
            vehicle_params=vehicle_config,
            custom_policy=custom_policy,
        )

        jitted_step = jax.jit(LanderEnv.step)

        infos: list[dict[str, Any]] = []
        episode_records: list[dict[str, Any]] = []
        term_reward_accumulators: dict[str, list[float]] = {}

        for seed in seeds:
            seed_key = jax.random.key(seed)
            reset_key, sim_key = jax.random.split(seed_key)

            obs, state = LanderEnv.reset(reset_key, body_config, vehicle_config)

            # Prepend initial state at t=0
            steps: list[dict[str, float]] = [{
                "t": float(state.t),
                "x": float(state.x),
                "y": float(state.y),
                "vx": float(state.vx),
                "vy": float(state.vy),
                "mass": float(state.mass),
                "throttle": 0.0,
                "angle": 0.0,
                "reward": 0.0,
            }]

            done = False
            last_info: dict[str, Any] | None = None
            ep_component_sums: dict[str, Any] = {}
            step_records = []

            while not done:
                sim_key, act_key, step_key = jax.random.split(sim_key, 3)
                action = policy_fn(obs, act_key)

                obs, next_state, reward_components, done_flag, info = jitted_step(
                    state, action, step_key
                )
                done = bool(done_flag)
                last_info = info

                step_reward = sum(reward_components.values())
                for term_name, val in reward_components.items():
                    ep_component_sums[term_name] = (
                        ep_component_sums.get(term_name, 0.0) + val
                    )

                step_records.append((
                    next_state.t, next_state.x, next_state.y,
                    next_state.vx, next_state.vy, next_state.mass,
                    action[0], action[1], step_reward,
                ))
                state = next_state

            # Convert episode step arrays to host in one batch call
            host_records = jax.device_get(step_records)
            min_thr = float(state.min_throttle)
            gimbal_lim = float(state.gimbal_limit_rad)

            for rec in host_records:
                raw_throttle = float(rec[6])
                raw_angle = float(rec[7])
                phys_throttle = min_thr + (1.0 - min_thr) * (raw_throttle + 1.0) / 2.0
                thrust_angle = raw_angle * gimbal_lim
                steps.append({
                    "t": float(rec[0]),
                    "x": float(rec[1]),
                    "y": float(rec[2]),
                    "vx": float(rec[3]),
                    "vy": float(rec[4]),
                    "mass": float(rec[5]),
                    "throttle": float(phys_throttle),
                    "angle": float(thrust_angle),
                    "reward": float(rec[8]),
                })

            for term_name, total_val in ep_component_sums.items():
                term_reward_accumulators.setdefault(term_name, []).append(
                    float(jax.device_get(total_val))
                )

            assert last_info is not None, f"Episode with seed {seed} produced no steps."
            outcome = int(last_info["outcome"])
            infos.append(last_info)

            # Write trajectory Parquet
            pq_rel = trajectory_path(run_id, seed, base_dir=traj_base_dir)
            write_trajectory(steps, pq_rel)

            insert_episode(
                conn,
                run_id=run_id,
                seed=seed,
                outcome=outcome,
                parquet_path=pq_rel,
            )

            episode_records.append({
                "seed": seed,
                "outcome": outcome,
                "outcome_name": outcome_name(outcome),
                "parquet_path": pq_rel,
            })

        # Compute aggregate eval statistics strictly via info Ground Truth (§5.6)
        agg_stats = aggregate_episodes(infos)

        term_stats = {
            term: {"mean": sum(vals) / len(vals)}
            for term, vals in term_reward_accumulators.items()
        }

        eval_id = insert_eval(
            conn,
            run_id=run_id,
            eval_body=target_body,
            n_episodes=len(infos),
            success_rate=agg_stats["success_rate"],
            vy_mean=agg_stats["vy_mean"],
            vy_std=agg_stats["vy_std"],
            vx_mean=agg_stats["vx_mean"],
            vx_std=agg_stats["vx_std"],
            fuel_mean=agg_stats["fuel_mean"],
            ci95_low=agg_stats["ci95_low"],
            ci95_high=agg_stats["ci95_high"],
            term_stats=term_stats,
        )

        result = {
            "eval_id": eval_id,
            "run_id": run_id,
            "eval_body": target_body,
            "is_ood": target_body not in train_bodies,
            "n_episodes": len(infos),
            "n_success": int(agg_stats["n_success"]),
            "success_rate": agg_stats["success_rate"],
            "ci95_low": agg_stats["ci95_low"],
            "ci95_high": agg_stats["ci95_high"],
            "vy_mean": agg_stats["vy_mean"],
            "vy_std": agg_stats["vy_std"],
            "vx_mean": agg_stats["vx_mean"],
            "vx_std": agg_stats["vx_std"],
            "fuel_mean": agg_stats["fuel_mean"],
            "episodes": episode_records,
            "term_stats": term_stats,
        }

        if not quiet:
            ood_flag = " [OOD]" if result["is_ood"] else ""
            print("=" * 60)
            print(f"RLander Evaluation Summary — Run {run_id} on {target_body}{ood_flag}")
            print(f"Episodes:     {result['n_episodes']} (seeds {seeds[0]}..{seeds[-1]})")
            print(f"Success rate: {result['success_rate']:.1%} (Wilson 95% CI: [{result['ci95_low']:.1%}, {result['ci95_high']:.1%}])")
            print(f"Touchdown vy: {result['vy_mean']:.2f} ± {result['vy_std']:.2f} m/s")
            print(f"Touchdown vx: {result['vx_mean']:.2f} ± {result['vx_std']:.2f} m/s")
            print(f"Fuel used:    {result['fuel_mean']:.1f} kg")
            print(f"Stored eval:  eval_id={eval_id} in SQLite")
            print("=" * 60)

        return result

    finally:
        if close_conn_on_exit:
            conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic evaluation from a checkpoint or policy."
    )
    parser.add_argument(
        "--run-id",
        type=int,
        required=True,
        help="Run ID from the runs table in SQLite.",
    )
    parser.add_argument(
        "--eval-body",
        type=str,
        default=None,
        help="Celestial body to evaluate on (e.g. europa, mars). Defaults to training body.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Override checkpoint path recorded in the run.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        choices=["ppo", "pd_fallback", "random"],
        help="Override policy type to evaluate.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database (defaults to data/rlander.db).",
    )
    parser.add_argument(
        "--seeds-start",
        type=int,
        default=10000,
        help="Starting seed for frozen test set (default: 10000).",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate (default: 100).",
    )
    parser.add_argument(
        "--traj-dir",
        type=str,
        default="data/trajectories",
        help="Base directory for Parquet trajectories (default: data/trajectories).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable summary output.",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    parsed = parse_args(args)
    seeds = range(parsed.seeds_start, parsed.seeds_start + parsed.num_episodes)
    run_evaluation(
        run_id=parsed.run_id,
        eval_body=parsed.eval_body,
        checkpoint_path=parsed.checkpoint_path,
        policy_override=parsed.policy,
        db_path=parsed.db_path,
        seeds=seeds,
        traj_base_dir=parsed.traj_dir,
        quiet=parsed.quiet,
    )


if __name__ == "__main__":
    main()
