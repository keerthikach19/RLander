"""Audit sensor noise, randomization, action sampling and saved shaping logs.

Read-only with respect to environment/checkpoints; evaluates validation seeds only.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from env.lander_env import LanderEnv
from rl.networks import ActorCritic, ObsStats, normalize_obs, sample_action


def audit(run_dir, output):
    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    body, vehicle = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value, dtype=jnp.float32),
        (manifest["config"]["body_params"], manifest["config"]["vehicle_params"]),
    )
    _, state = LanderEnv.reset(jax.random.key(987), body, vehicle)
    observations, states, _, _, _ = jax.jit(
        jax.vmap(lambda key: LanderEnv.step(state, jnp.array([-0.3, 0.0]), key))
    )(jax.random.split(jax.random.key(988), 8192))
    errors = jnp.stack(
        (
            observations[:, 0] * states.L - states.y,
            observations[:, 1] * states.U - states.vx,
            observations[:, 2] * states.U - states.vy,
        ),
        axis=1,
    )
    expected = [float(0.5 + 0.005 * abs(states.y[0])), 0.1, 0.1]
    measured = np.asarray(errors.std(axis=0))
    if not np.allclose(measured, expected, rtol=0.04):
        raise RuntimeError(
            "Measured observation noise disagrees with the sensor contract"
        )
    if any(float(jnp.ptp(value)) != 0 for value in jax.tree_util.tree_leaves(states)):
        raise RuntimeError("Sensor keys unexpectedly changed physical state")
    _, resets = jax.jit(jax.vmap(lambda key: LanderEnv.reset(key, body, vehicle)))(
        jax.random.split(jax.random.key(989), 1024)
    )
    randomization = {
        name: [float(jnp.min(value)), float(jnp.max(value))]
        for name, value in {
            "height_m": resets.y,
            "vy_mps": resets.vy,
            "vx_mps": resets.vx,
            "mass_kg": resets.mass,
            "thrust_N": resets.max_thrust,
            "gravity_mps2": resets.g,
        }.items()
    }
    if any(low == high for low, high in randomization.values()):
        raise RuntimeError("A domain-randomized parameter did not vary")
    with ocp.PyTreeCheckpointer() as checkpointer:
        saved = checkpointer.restore(str(run_dir / "checkpoint"))
    params, stats = saved["train_state"]["params"], ObsStats(**saved["obs_stats"])

    def evaluate(sampled):
        def episode(seed):
            key, reset_key = jax.random.split(jax.random.key(seed))
            obs, state = LanderEnv.reset(reset_key, body, vehicle)

            def step(carry):
                obs, state, key, _, count = carry
                key, step_key = jax.random.split(key)
                mean, log_std, _ = ActorCritic().apply(
                    params, normalize_obs(obs, stats)
                )
                action = (
                    sample_action(jax.random.fold_in(step_key, 917), mean, log_std)[0]
                    if sampled
                    else mean
                )
                obs, state, _, _, info = LanderEnv.step(
                    state, jnp.clip(action, -1, 1), step_key
                )
                return obs, state, key, info["outcome"], count + 1

            final = jax.lax.while_loop(
                lambda carry: (carry[3] == 0) & (carry[4] < 1024),
                step,
                (obs, state, key, jnp.int32(0), jnp.int32(0)),
            )
            return final[3], final[4]

        outcomes, lengths = jax.device_get(
            jax.jit(jax.vmap(episode))(jnp.arange(2000, 2100))
        )
        if np.any(outcomes == 0):
            raise RuntimeError("Audit episode did not finish")
        return {
            "soft_landings": int((outcomes == 1).sum()),
            "crashed": int((outcomes == 2).sum()),
            "timeouts": int((outcomes == 3).sum()),
            "outcomes": outcomes.tolist(),
            "steps": lengths.tolist(),
        }

    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    updates = [
        row
        for row in rows
        if row.get("event") == "update" and row["shaping_check_available"]
    ]
    ratios = np.asarray([row["shaping_to_terminal_ratio"] for row in updates])
    components = manifest["evaluation"]["reward_components_mean"]
    report = {
        "source_run_id": manifest["run"]["id"],
        "validation_seeds": list(range(2000, 2100)),
        "noise_probe": {
            "samples": 8192,
            "axes": ["altitude_m", "vx_mps", "vy_mps"],
            "expected_std": expected,
            "measured_std": measured.tolist(),
            "physics_identical_across_noise_keys": True,
        },
        "domain_randomization_ranges_1024_resets": randomization,
        "final_log_std": np.asarray(params["params"]["log_std"]).tolist(),
        "final_action_std": np.exp(np.asarray(params["params"]["log_std"])).tolist(),
        "deterministic": evaluate(False),
        "sampled": evaluate(True),
        "shaping": {
            "definition": "abs(sum of mean completed-episode velocity+altitude+fuel+angle totals)/100; not the added potential alone",
            "updates_with_completed_episodes": len(updates),
            "above_target_updates": int((ratios > 1).sum()),
            "ratio_min_mean_max": [
                float(ratios.min()),
                float(ratios.mean()),
                float(ratios.max()),
            ],
            "final_training_ratio": float(ratios[-1]),
            "evaluation_ratio": abs(
                sum(components[k] for k in ("velocity", "altitude", "fuel", "angle"))
            )
            / 100,
            "evaluation_components_mean": components,
        },
    }
    if (
        report["deterministic"]["outcomes"]
        != manifest["evaluation"]["episodes"]["info"]["outcome"]
    ):
        raise RuntimeError("Deterministic audit differs from recorded evaluation")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("deterministic", "sampled", "validation_seeds")
            },
            indent=2,
        )
    )
    for mode in ("deterministic", "sampled"):
        print(
            mode,
            {k: v for k, v in report[mode].items() if k not in ("outcomes", "steps")},
        )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit(args.run_dir, args.output)
