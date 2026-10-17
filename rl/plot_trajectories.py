"""Plot five deterministic PPO and five PD episodes from a saved run.

uv run --with matplotlib python -m rl.plot_trajectories --run-dir ckpts/<run>

Uses the manifest's parameters and first five evaluation seeds by default.
Ground-truth altitude/velocity are logged only; policies receive observations.
The reset/noise key sequence matches train.evaluate_smoke exactly.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from env.lander_env import LanderEnv
from rl.fallback import pd_action
from rl.networks import ObsStats, policy_action


def collect_trajectories(body, vehicle, policy, seeds, max_steps=1024):
    """Stop at first done, retaining t=0 and the terminal sample."""

    def sample(state):
        return jnp.array([state.t, state.y, state.vy, state.vx])

    def episode(seed):
        key, reset_key = jax.random.split(jax.random.key(seed))
        obs, state = LanderEnv.reset(reset_key, body, vehicle)
        initial = sample(state)

        def step(carry, _):
            def advance(carry):
                obs, state, key, _, _, count = carry
                key, step_key = jax.random.split(key)
                obs, state, _, done, info = LanderEnv.step(state, policy(obs), step_key)
                return obs, state, key, done, info["outcome"], count + 1

            carry = jax.lax.cond(carry[3], lambda c: c, advance, carry)
            return carry, sample(carry[1])

        final, trace = jax.lax.scan(
            step,
            (
                obs,
                state,
                key,
                jnp.asarray(False),
                jnp.int32(0),
                jnp.int32(0),
            ),
            None,
            length=max_steps,
        )
        return jnp.concatenate([initial[None], trace]), final[4], final[5]

    traces, outcomes, lengths = jax.device_get(
        jax.jit(jax.vmap(episode))(jnp.asarray(seeds, dtype=jnp.uint32))
    )
    if np.any(outcomes == 0):
        raise RuntimeError("Episode exceeded max_steps before termination")
    return [
        {"seed": int(seed), "outcome": int(outcome), "trace": trace[: length + 1]}
        for seed, outcome, trace, length in zip(seeds, outcomes, traces, lengths)
    ]


def plot_run(run_dir, output_dir=None, seed_start=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError("Expected a completed run manifest")
    seeds = (
        manifest["evaluation"]["seeds"][:5]
        if seed_start is None
        else list(range(seed_start, seed_start + 5))
    )
    if len(seeds) != 5:
        raise ValueError("Need five evaluation seeds")
    body, vehicle = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value, dtype=jnp.float32),
        (manifest["config"]["body_params"], manifest["config"]["vehicle_params"]),
    )
    with ocp.PyTreeCheckpointer() as checkpointer:
        saved = checkpointer.restore(str(run_dir / "checkpoint"))
    params, stats = saved["train_state"]["params"], ObsStats(**saved["obs_stats"])
    policies = {
        "PPO": lambda obs: policy_action(params, stats, obs),
        "PD": lambda obs: pd_action(obs, body, vehicle),
    }
    episodes = {
        name: collect_trajectories(body, vehicle, policy, seeds)
        for name, policy in policies.items()
    }
    expected = dict(
        zip(
            manifest["evaluation"]["seeds"],
            manifest["evaluation"]["episodes"]["info"]["outcome"],
        )
    )
    for episode in episodes["PPO"]:
        if (
            episode["seed"] in expected
            and episode["outcome"] != expected[episode["seed"]]
        ):
            raise RuntimeError("Checkpoint replay differs from recorded evaluation")

    output_dir = Path(output_dir) if output_dir else run_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    names = {1: "soft landing", 2: "crash", 3: "timeout"}
    title = (
        f"Europa RK4 | {manifest['run']['reward_id']} | "
        f"{manifest['actual_steps']:,} steps"
    )
    for column, ylabel, filename in (
        (1, "Altitude (m)", "altitude.png"),
        (2, "Vertical velocity (m/s)", "vertical_velocity.png"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
        for ax, (name, rows) in zip(axes, episodes.items()):
            for row in rows:
                trace = row["trace"]
                (line,) = ax.plot(
                    trace[:, 0],
                    trace[:, column],
                    linewidth=1.5,
                    label=f"{row['seed']}: {names[row['outcome']]}",
                )
                ax.scatter(
                    trace[-1, 0],
                    trace[-1, column],
                    s=24,
                    color=line.get_color(),
                    zorder=3,
                )
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set(title=name, xlabel="Time (s)", ylabel=ylabel)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(title, fontsize=11)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=160)
        plt.close(fig)

    summary = {"run_id": manifest["run"]["id"], "policies": {}}
    for name, rows in episodes.items():
        summary["policies"][name] = []
        for row in rows:
            trace = row["trace"]
            tail = trace[trace[:, 0] >= trace[-1, 0] - 20]
            summary["policies"][name].append(
                {
                    "seed": row["seed"],
                    "outcome": names[row["outcome"]],
                    "duration_s": float(trace[-1, 0]),
                    "final_altitude_m": float(trace[-1, 1]),
                    "final_vy_mps": float(trace[-1, 2]),
                    "final_vx_mps": float(trace[-1, 3]),
                    "last_20s_altitude_range_m": [
                        float(tail[:, 1].min()),
                        float(tail[:, 1].max()),
                    ],
                    "last_20s_mean_abs_vy_mps": float(np.abs(tail[:, 2]).mean()),
                }
            )
    np.savez_compressed(
        output_dir / "trajectories.npz",
        **{
            f"{name.lower()}_{row['seed']}": row["trace"]
            for name, rows in episodes.items()
            for row in rows
        },
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--seed-start", type=int)
    args = parser.parse_args()
    plot_run(args.run_dir, args.output_dir, args.seed_start)
