"""W2 contract fakes; none of these environments are production dynamics."""

import json
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# The project installs src/rlander; the agreed rl/ and train.py live at repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl.fallback import pd_action
from rl.networks import (
    ActorCritic,
    ObsStats,
    gaussian_log_prob,
    normalize_obs,
    policy_action,
    update_obs_stats,
)
from rl.ppo import PPOConfig, compute_gae, make_train, ppo_loss
from rl.vec_env import reset_envs, step_envs


def parameters(g=1.315, rho0=0.0, scale_height=1.0):
    """Plan's illustrative values, not verified planetary/config fixtures."""
    body = {
        "g": g,
        "radius_m": 1_560_800.0,
        "rho0": rho0,
        "scale_height_m": scale_height,
        "success": {"vy_max_mps": 1.0, "vx_max_mps": 0.5},
    }
    vehicle = {
        "mass0_kg": 800.0,
        "dry_mass_kg": 320.0,
        "twr_max": 3.0,
        "min_throttle": 0.0,
        "isp_s": 300.0,
        "gimbal_limit_rad": np.pi / 12,
        "cd": 1.2,
        "area_m2": 4.0,
        "h0_m": 2000.0,
        "v0_mps": 20.0,
        "max_thrust_N": 3 * 800 * g,
    }
    return jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, jnp.float32), (body, vehicle)
    )


class FakeEnv:
    """A bounded-action task: learn throttle=0.7, angle=-0.4 in short episodes."""

    observation_space = (8,)
    action_space = (2,)

    def __init__(self, horizon=5, timeout=False):
        self.horizon = horizon
        self.timeout = timeout

    def observe(self, state):
        return jnp.array(
            [
                state["y"],
                state["vx"],
                state["vy"],
                0.0,
                0.0,
                1.0,
                1.0,
                state["t"] / self.horizon,
            ]
        )

    def reset(self, key, body_params, vehicle_params):
        state = {
            "x": jnp.asarray(0.0),
            "y": jax.random.uniform(key, (), minval=0.8, maxval=1.2),
            "vx": jnp.asarray(0.0),
            "vy": jnp.asarray(0.0),
            "vx_prev": jnp.asarray(0.0),
            "vy_prev": jnp.asarray(0.0),
            "mass": vehicle_params["mass0_kg"],
            "t": jnp.asarray(0.0),
            "body_params": body_params,
            "vehicle_params": vehicle_params,
        }
        return self.observe(state), state

    def step(self, state, action, key):
        state = dict(state, t=state["t"] + 1)
        done = state["t"] >= self.horizon
        truncated = done & self.timeout
        info = {
            "touchdown_v": jnp.asarray(0.0),
            "touchdown_vx": jnp.asarray(0.0),
            "fuel_used": jnp.asarray(0.0),
            "crashed": jnp.asarray(False),
            "landed": done & ~truncated,
            "truncated": truncated,
            "outcome": jnp.where(done, 3 if self.timeout else 1, 0).astype(jnp.int32),
        }
        components = {
            "tracking": -jnp.square(action - jnp.array([0.7, -0.4])).sum(),
            "alive": jnp.asarray(-0.25),
        }
        # Exercise fresh sensor-noise keys without feeding randomness into physics.
        obs = self.observe(state).at[3].set(jax.random.normal(key) * 0.01)
        return obs, state, components, done, info


class ToyDescentEnv(FakeEnv):
    """Minimal synthetic point-mass testbed, with clean contract observations.

    Explicit substeps suffice for controller regression tests. This deliberately
    does not stand in for W1's RK4, noise, randomization or analytic test suite.
    """

    def observe(self, state):
        b, v = state["body_params"], state["vehicle_params"]
        unit = jnp.sqrt(b["g"] * v["h0_m"])
        tmax = 4 * jnp.sqrt(2 * v["h0_m"] / b["g"])
        return jnp.array(
            [
                state["y"] / v["h0_m"],
                state["vx"] / unit,
                state["vy"] / unit,
                0.0,
                0.0,
                state["mass"] / v["mass0_kg"],
                (state["mass"] - v["dry_mass_kg"]) / (v["mass0_kg"] - v["dry_mass_kg"]),
                state["t"] / tmax,
            ]
        )

    def reset(self, key, body_params, vehicle_params):
        _, state = super().reset(key, body_params, vehicle_params)
        state.update(
            y=vehicle_params["h0_m"], vy=-vehicle_params["v0_mps"], vx=jnp.asarray(1.0)
        )
        return self.observe(state), state

    def step(self, state, action, key):
        b, v = state["body_params"], state["vehicle_params"]
        dt = 0.01 * jnp.sqrt(v["h0_m"] / b["g"])
        throttle = v["min_throttle"] + (action[0] + 1) * 0.5 * (1 - v["min_throttle"])
        angle = action[1] * v["gimbal_limit_rad"]

        def substep(_, s):
            thrust = jnp.where(
                s["mass"] > v["dry_mass_kg"], throttle * v["max_thrust_N"], 0
            )
            rho = b["rho0"] * jnp.exp(-jnp.maximum(s["y"], 0) / b["scale_height_m"])
            drag = 0.5 * rho * v["cd"] * v["area_m2"] * jnp.hypot(s["vx"], s["vy"])
            ax = (thrust * jnp.sin(angle) - drag * s["vx"]) / s["mass"]
            ay = (thrust * jnp.cos(angle) - drag * s["vy"]) / s["mass"] - b["g"]
            h = dt / 16
            return dict(
                s,
                x=s["x"] + h * s["vx"],
                y=s["y"] + h * s["vy"],
                vx=s["vx"] + h * ax,
                vy=s["vy"] + h * ay,
                mass=jnp.maximum(
                    v["dry_mass_kg"], s["mass"] - h * thrust / (v["isp_s"] * 9.80665)
                ),
            )

        next_state = jax.lax.fori_loop(0, 16, substep, state)
        next_state.update(t=state["t"] + dt, vx_prev=state["vx"], vy_prev=state["vy"])
        landed = next_state["y"] <= 0
        truncated = (~landed) & (
            next_state["t"] >= 4 * jnp.sqrt(2 * v["h0_m"] / b["g"])
        )
        soft = (
            landed & (jnp.abs(next_state["vy"]) < 1) & (jnp.abs(next_state["vx"]) < 0.5)
        )
        info = {
            "touchdown_v": next_state["vy"],
            "touchdown_vx": next_state["vx"],
            "fuel_used": v["mass0_kg"] - next_state["mass"],
            "landed": soft,
            "crashed": landed & ~soft,
            "truncated": truncated,
            "outcome": jnp.where(
                landed, jnp.where(soft, 1, 2), jnp.where(truncated, 3, 0)
            ).astype(jnp.int32),
        }
        return (
            self.observe(next_state),
            next_state,
            {"zero": jnp.asarray(0.0)},
            landed | truncated,
            info,
        )


class LearningDescentEnv(ToyDescentEnv):
    """The same point-mass physics with randomized starts and state-based rewards.

    Rewards measure descent velocity and touchdown, never distance to a target
    action or a PD teacher. This is a small learning regression, not a claim
    about the production reward or planetary soft-landing success.
    """

    def reset(self, key, body_params, vehicle_params):
        _, state = super().reset(key, body_params, vehicle_params)
        height_key, velocity_key = jax.random.split(key)
        state.update(
            y=vehicle_params["h0_m"]
            * jax.random.uniform(
                height_key,
                (),
                minval=0.9,
                maxval=1.1,
            ),
            vx=jax.random.uniform(velocity_key, (), minval=-0.5, maxval=0.5),
        )
        return self.observe(state), state

    def step(self, state, action, key):
        obs, next_state, _, done, info = super().step(state, action, key)
        target_vy = -jnp.minimum(
            2.0,
            jnp.sqrt(
                0.5**2 + state["body_params"]["g"] * jnp.maximum(next_state["y"], 0)
            ),
        )
        components = {
            "descent_tracking": -0.01
            * (
                jnp.square(next_state["vy"] - target_vy)
                + jnp.square(next_state["vx"])
                + 0.1
            ),
            "touchdown": jnp.where(
                done & ~info["truncated"],
                5.0
                - 2.0 * (jnp.square(next_state["vy"]) + jnp.square(next_state["vx"])),
                0.0,
            ),
            "timeout": -5.0 * info["truncated"],
        }
        return obs, next_state, components, done, info


def test_gaussian_and_observation_statistics():
    log_prob = gaussian_log_prob(
        jnp.zeros((2, 2)), jnp.zeros((2, 2)), jnp.zeros((2, 2))
    )
    np.testing.assert_allclose(log_prob, -np.log(2 * np.pi), rtol=1e-6)
    data = jnp.arange(80, dtype=jnp.float32).reshape(10, 8)
    stats = update_obs_stats(update_obs_stats(ObsStats.empty(), data[:3]), data[3:])
    np.testing.assert_allclose(stats.mean, data.mean(0), rtol=1e-6)
    np.testing.assert_allclose(stats.var, data.var(0), rtol=1e-6)
    assert float(stats.count) == 10
    assert np.isfinite(normalize_obs(data, stats)).all()
    network = ActorCritic()
    params = network.init(jax.random.key(1), jnp.zeros(8))
    mean, std, value = network.apply(params, data)
    assert mean.shape == std.shape == (10, 2)
    assert value.shape == (10,)
    np.testing.assert_array_equal(std[0], std[-1])


def test_gae_bootstraps_timeouts_but_never_crosses_resets():
    # First episode times out, second terminates, third continues at the rollout end.
    rewards = jnp.array([[1.0], [2.0], [3.0]])
    values = jnp.array([[0.5], [0.6], [0.7]])
    next_values = jnp.array([[10.0], [20.0], [30.0]])
    done = jnp.array([[True], [True], [False]])
    truncated = jnp.array([[True], [False], [False]])
    adv, returns = compute_gae(rewards, values, next_values, done, truncated, 0.9, 0.95)
    np.testing.assert_allclose(adv[:, 0], [9.5, 1.4, 29.3], rtol=1e-6)
    np.testing.assert_allclose(returns, adv + values)
    adv, _ = compute_gae(
        rewards, values, next_values, jnp.zeros_like(done), truncated, 0.9, 0.95
    )
    np.testing.assert_allclose(
        adv[0, 0], 9.5 + 0.855 * (19.4 + 0.855 * 29.3), rtol=1e-6
    )


def test_ppo_clips_policy_and_value_losses_using_raw_actions():
    actions = jnp.array([[2.0, 0.0], [-2.0, 0.0]])
    batch = {
        "obs": jnp.zeros((2, 8)),
        "action": actions,
        "log_prob": gaussian_log_prob(actions, jnp.zeros((2, 2)), jnp.zeros((2, 2))),
        "advantage": jnp.array([1.0, -1.0]),
        "value": jnp.zeros(2),
        "return": jnp.array([2.0, -2.0]),
    }

    def apply_fn(params, obs):
        return (
            jnp.array([[1.0, 0.0], [1.0, 0.0]]),
            jnp.zeros((2, 2)),
            jnp.array([3.0, -3.0]),
        )

    loss, metrics = ppo_loss({}, apply_fn, batch, PPOConfig(ent_coef=0.0))
    # Likelihood ratios exp(1.5), exp(-2.5) clip to 1.2 and 0.8 in the surrogate.
    np.testing.assert_allclose(metrics["policy_loss"], -0.2, rtol=1e-6)
    np.testing.assert_allclose(metrics["value_loss"], 0.5 * 1.8**2, rtol=1e-6)
    np.testing.assert_allclose(loss, -0.2 + 0.5 * 0.5 * 1.8**2, rtol=1e-6)
    assert float(metrics["clip_fraction"]) == 1.0


def test_vector_reset_preserves_final_obs_and_ongoing_states():
    env = FakeEnv(horizon=3, timeout=True)
    body, vehicle = parameters()
    keys = jax.random.split(jax.random.key(3), 2)
    _, state = reset_envs(env, keys, body, vehicle)
    state["t"] = jnp.array([2.0, 0.0])
    reset_keys = jax.random.split(jax.random.key(5), 2)
    result = jax.jit(
        lambda s: step_envs(
            env,
            s,
            jnp.array([[10.0, -10.0], [0.0, 0.0]]),
            keys,
            reset_keys,
            body,
            vehicle,
        )
    )(state)
    np.testing.assert_array_equal(result.done, [True, False])
    np.testing.assert_allclose(result.state["t"], [0, 1])
    np.testing.assert_allclose(result.final_obs[:, 7], [1, 1 / 3])
    np.testing.assert_allclose(result.obs[:, 7], [0, 1 / 3])
    np.testing.assert_allclose(
        result.reward_components["tracking"][0], -(0.3**2 + 0.6**2), rtol=1e-6
    )
    np.testing.assert_array_equal(result.info["outcome"], [3, 0])
    np.testing.assert_array_equal(result.state["y"][1], state["y"][1])
    expected_reset, _ = env.reset(reset_keys[0], body, vehicle)
    np.testing.assert_array_equal(result.obs[0], expected_reset)


def test_jitted_training_is_reproducible_and_accounts_across_rollouts():
    config = PPOConfig(
        num_envs=4, num_steps=3, total_steps=36, num_minibatches=2, update_epochs=2
    )
    train = make_train(
        FakeEnv(horizon=5), config, shaping_components=("alive", "tracking")
    )
    params = parameters()
    result = train(jax.random.key(10), *params)
    repeated = train(jax.random.key(10), *params)
    for a, b in zip(
        jax.tree_util.tree_leaves(result), jax.tree_util.tree_leaves(repeated)
    ):
        if jax.dtypes.issubdtype(a.dtype, jax.dtypes.prng_key):
            a, b = jax.random.key_data(a), jax.random.key_data(b)
        np.testing.assert_array_equal(a, b)
    assert int(result.runner.train_state.step) == 12
    np.testing.assert_array_equal(result.metrics["episodes"], [0, 4, 0])
    np.testing.assert_allclose(
        result.metrics["reward_components"]["alive"], [0, -1.25, 0]
    )
    np.testing.assert_allclose(result.runner.episode_components["alive"], -1.0)
    np.testing.assert_array_equal(result.runner.episode_lengths, [4] * 4)
    np.testing.assert_array_equal(result.metrics["env_steps"], [12, 24, 36])
    assert np.isfinite(result.metrics["loss"]).all()
    assert np.isfinite(result.metrics["grad_norm"]).all()
    assert (result.metrics["grad_norm"] > 0).all()
    assert bool(result.metrics["shaping_check_available"][1])


def test_ppo_learns_simple_fake_control_task():
    config = PPOConfig(
        num_envs=32,
        num_steps=16,
        total_steps=32_768,
        num_minibatches=4,
        update_epochs=4,
        learning_rate=0.003,
        ent_coef=0.0,
    )
    env = FakeEnv()
    body, vehicle = parameters()
    result = make_train(env, config)(jax.random.key(7), body, vehicle)
    obs, _ = reset_envs(env, jax.random.split(jax.random.key(99), 100), body, vehicle)
    actions = policy_action(
        result.runner.train_state.params, result.runner.obs_stats, obs
    )
    error = np.mean(np.square(np.asarray(actions) - np.array([0.7, -0.4])))
    assert error < 0.04, f"Learned deterministic action MSE: {error}"
    rewards = np.asarray(result.metrics["episode_return"])
    assert rewards[-5:].mean() > rewards[:5].mean() + 1.0


def test_ppo_learns_toy_point_mass_descent():
    """PPO must improve actual trajectory returns, not just an action bandit."""
    env = LearningDescentEnv()
    body, vehicle = parameters()
    vehicle = dict(vehicle, h0_m=jnp.asarray(20.0), v0_mps=jnp.asarray(2.0))
    config = PPOConfig(
        num_envs=32,
        num_steps=64,
        total_steps=131_072,
        num_minibatches=4,
        update_epochs=4,
        learning_rate=0.001,
        ent_coef=0.001,
    )
    training_key = jax.random.key(7)
    result = make_train(
        env,
        config,
        shaping_components=("descent_tracking",),
    )(training_key, body, vehicle)
    # Reproduce the exact pre-optimization weights. Use the SAME normalization
    # for both evaluations, so changing observation statistics alone cannot pass.
    init_key = jax.random.split(training_key, 4)[1]
    untrained_params = ActorCritic().init(init_key, jnp.zeros(8))

    @jax.jit
    def evaluate(params):
        def episode(key):
            key, reset_key = jax.random.split(key)
            obs, state = env.reset(reset_key, body, vehicle)

            def step(carry):
                obs, state, key, total, _, steps = carry
                key, step_key = jax.random.split(key)
                action = policy_action(params, result.runner.obs_stats, obs)
                obs, state, components, done, _ = env.step(state, action, step_key)
                return (
                    obs,
                    state,
                    key,
                    total + sum(components.values()),
                    done,
                    steps + 1,
                )

            _, _, _, total, done, _ = jax.lax.while_loop(
                lambda c: ~c[4] & (c[5] < 567),
                step,
                (obs, state, key, jnp.asarray(0.0), jnp.asarray(False), jnp.asarray(0)),
            )
            return total, done

        return jax.vmap(episode)(jax.vmap(jax.random.key)(jnp.arange(10_000, 10_016)))

    before, before_done = evaluate(untrained_params)
    after, after_done = evaluate(result.runner.train_state.params)
    before, after = np.asarray(before), np.asarray(after)
    assert np.isfinite(result.metrics["loss"]).all()
    assert np.isfinite(before).all() and np.isfinite(after).all()
    assert np.asarray(before_done).all() and np.asarray(after_done).all()
    # A substantial held-out gain is required; an unchanged/random policy fails.
    improvement = float(after.mean() - before.mean())
    required_gain = max(10.0, 0.25 * abs(float(before.mean())))
    print(
        f"Point-mass PPO return: {before.mean():.3f} -> {after.mean():.3f}; gain={improvement:.3f}"
    )
    assert improvement > required_gain, (before.mean(), after.mean(), required_gain)
    assert np.count_nonzero(after > before) >= 12


@pytest.mark.parametrize(
    "g,rho0,scale_height",
    [
        (1.315, 0.0, 1.0),
        (3.71, 0.020, 11100.0),
        (1.352, 5.4, 20000.0),
        (0.113, 0.0, 1.0),
    ],
    ids=["europa-like", "mars-like", "titan-like", "enceladus-like"],
)
def test_pd_lands_in_synthetic_body_regimes(g, rho0, scale_height):
    env = ToyDescentEnv()
    body, vehicle = parameters(g, rho0, scale_height)

    @jax.jit
    def run(body, vehicle):
        key = jax.random.key(3)
        obs, state = env.reset(key, body, vehicle)

        def step(carry):
            obs, state, _, count, key = carry
            key, step_key = jax.random.split(key)
            action = pd_action(obs, body, vehicle)
            obs, state, _, done, _ = env.step(state, action, step_key)
            return obs, state, done, count + 1, key

        return jax.lax.while_loop(
            lambda c: ~c[2] & (c[3] < 567),
            step,
            (obs, state, jnp.asarray(False), jnp.asarray(0), key),
        )

    obs, state, done, steps, _ = run(body, vehicle)
    assert bool(done)
    assert float(state["y"]) <= 0, f"Timeout at height {state['y']} after {steps} steps"
    assert abs(float(state["vy"])) < 1.0
    assert abs(float(state["vx"])) < 0.5
    assert float(obs[7]) < 1.0
    assert float(state["mass"]) > float(vehicle["dry_mass_kg"])


def test_batched_parameters_remain_data_under_jit():
    env = FakeEnv(horizon=1)
    pairs = [parameters(), parameters(3.71, 0.02, 11100)]
    body, vehicle = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *pairs)
    config = PPOConfig(
        num_envs=2, num_steps=2, total_steps=4, num_minibatches=1, update_epochs=1
    )
    train = make_train(env, config, per_env_params=True)
    compiled = train.lower(jax.random.key(1), body, vehicle).compile()
    first = compiled(jax.random.key(1), body, vehicle)
    second = compiled(jax.random.key(1), dict(body, g=body["g"] * 1.02), vehicle)
    np.testing.assert_allclose(first.runner.env_state["body_params"]["g"], body["g"])
    np.testing.assert_allclose(
        second.runner.env_state["body_params"]["g"], body["g"] * 1.02
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_envs": 0},
        {"num_steps": -1},
        {"total_steps": 1},
        {"num_minibatches": 3},
        {"gamma": 2},
        {"learning_rate": 0},
    ],
)
def test_invalid_config_fails_early(kwargs):
    with pytest.raises(ValueError):
        PPOConfig(**kwargs)


def test_checkpoint_logging_and_integration_hooks(tmp_path, capsys):
    import orbax.checkpoint as ocp
    from flax import serialization

    from train import TrainingBindings, run_experiment

    body, vehicle = parameters()
    config = PPOConfig(
        num_envs=4,
        num_steps=4,
        total_steps=32,
        num_minibatches=2,
        update_epochs=1,
    )
    recorded = []

    def record_run(row):
        assert Path(row["checkpoint_path"]).is_dir()
        recorded.append(row)

    def evaluate(params, obs_stats):
        obs, _ = FakeEnv().reset(jax.random.key(10000), body, vehicle)
        action = policy_action(params, obs_stats, obs)
        assert np.isfinite(action).all()
        return {
            "fake_task_action_mse": float(
                jnp.square(action - jnp.array([0.7, -0.4])).mean()
            )
        }

    bindings = TrainingBindings(
        FakeEnv(horizon=4),
        body,
        vehicle,
        shaping_components=("tracking", "alive"),
        record_run=record_run,
        evaluate=evaluate,
    )
    result, manifest = run_experiment(
        bindings,
        config,
        seed=4,
        body_name="synthetic",
        output_root=tmp_path,
    )
    assert len(recorded) == 1
    assert recorded[0]["jax_version"] == jax.__version__
    assert recorded[0]["seed"] == 4
    assert json.loads(recorded[0]["train_bodies_json"]) == ["synthetic"]
    assert manifest["database_status"] == "recorded"
    assert manifest["evaluation_status"] == "complete"
    assert manifest["actual_steps"] == 32
    run_dir = Path(recorded[0]["checkpoint_path"]).parent
    assert json.loads((run_dir / "manifest.json").read_text()) == manifest
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1 + config.num_updates
    assert rows[0]["config"]["seed"] == 4
    assert rows[-1]["env_steps"] == 32
    assert rows[-1]["reward_components"]["alive"] == -1.0
    assert rows[-1]["shaping_magnitude_status"] != "unavailable"
    assert json.loads(capsys.readouterr().out)["evaluation"] == manifest["evaluation"]
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(recorded[0]["checkpoint_path"])
    restored_state = serialization.from_state_dict(
        result.runner.train_state, restored["train_state"]
    )
    restored_stats = serialization.from_state_dict(
        result.runner.obs_stats, restored["obs_stats"]
    )
    np.testing.assert_array_equal(
        policy_action(restored_state.params, restored_stats, result.runner.obs),
        policy_action(
            result.runner.train_state.params, result.runner.obs_stats, result.runner.obs
        ),
    )
    for expected, actual in zip(
        jax.tree_util.tree_leaves(result.runner.train_state.opt_state),
        jax.tree_util.tree_leaves(restored_state.opt_state),
    ):
        np.testing.assert_array_equal(expected, actual)
    np.testing.assert_array_equal(
        restored["rng_key_data"], jax.random.key_data(result.runner.key)
    )


def test_skeleton_reports_pending_integrations(tmp_path, capsys):
    from train import TrainingBindings, run_experiment

    body, vehicle = parameters()
    config = PPOConfig(
        num_envs=2, num_steps=2, total_steps=4, num_minibatches=1, update_epochs=1
    )
    _, manifest = run_experiment(
        TrainingBindings(FakeEnv(), body, vehicle),
        config,
        output_root=tmp_path,
        body_name="synthetic",
    )
    assert manifest["database_status"] == manifest["evaluation_status"] == "pending"
    assert "pending W3" in json.loads(capsys.readouterr().out)["evaluation"]["status"]


def test_full_training_requires_successful_smoke_run(tmp_path):
    from train import TrainingBindings, run_experiment

    body, vehicle = parameters()
    bindings = TrainingBindings(FakeEnv(), body, vehicle)
    for rate in (None, 0.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="prior smoke"):
            run_experiment(
                bindings, PPOConfig(), output_root=tmp_path, smoke_success_rate=rate
            )
    assert not list(tmp_path.iterdir())


def test_hydra_accepts_planned_selectors():
    result = subprocess.run(
        [
            sys.executable,
            "train.py",
            "+body=europa",
            "+vehicle=europa_lander",
            "+seed=1",
            "--cfg",
            "job",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "body: europa" in result.stdout
    assert "vehicle: europa_lander" in result.stdout
    assert "seed: 1" in result.stdout
