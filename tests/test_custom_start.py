"""Custom flights must use entered states, consistent observations and real physics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from app import live
from env.lander_env import LanderEnv, _make_obs
from eval.batched import collect
from eval.run_eval import load_body_config, load_vehicle_config, make_policy_fn


@pytest.mark.parametrize("conditions", [
    {"x": 0, "y": 0, "vx": 0, "vy": -20},
    {"x": 0, "y": 2000, "vx": float("nan"), "vy": -20},
    {"x": 0, "y": float("inf"), "vx": 0, "vy": -20},
    {"x": 0, "y": 2000, "vx": 0},
])
def test_invalid_start_rejected_before_model_loading(monkeypatch, conditions):
    def forbidden(*args):
        raise AssertionError("Invalid inputs must not load a model")
    monkeypatch.setattr(live, "load_controller", forbidden)
    with pytest.raises(ValueError):
        live.simulate("ppo", 42, initial_conditions=conditions)


def test_custom_first_action_and_state_match_independent_environment_step():
    body, vehicle = load_body_config("europa"), load_vehicle_config("europa_lander")
    policy = make_policy_fn("pd_fallback", body_params=body, vehicle_params=vehicle)
    seed = 123
    initial = (137., 173., -1.25, -8.5)
    reset_key, sim_key = jax.random.split(jax.random.key(seed))
    _, original = LanderEnv.reset(reset_key, body, vehicle)
    x, y, vx, vy = map(jnp.float32, initial)
    state = original._replace(x=x, y=y, vx=vx, vy=vy, vx_prev=vx, vy_prev=vy)
    obs = _make_obs(state, jax.random.split(reset_key, 8)[-1])
    _, act_key, step_key = jax.random.split(sim_key, 3)
    action = policy(obs, act_key)
    _, expected, _, _, _ = LanderEnv.step(state, action, step_key)

    episodes, _ = collect(policy, [seed], body, vehicle, initial_conditions=initial)
    rows, _, _ = episodes[0]
    np.testing.assert_array_equal([rows[0][n] for n in ("x", "y", "vx", "vy")], initial)
    np.testing.assert_allclose(
        [rows[1][n] for n in ("t", "x", "y", "vx", "vy", "mass")],
        [getattr(expected, n) for n in ("t", "x", "y", "vx", "vy", "mass")],
        rtol=2e-5, atol=2e-4,
    )
    assert rows[1]["t"] == pytest.approx(float(original.dt))
    assert rows[1]["throttle"] == pytest.approx(float((action[0] + 1) / 2), abs=1e-5)
    assert rows[1]["angle"] == pytest.approx(float(action[1] * state.gimbal_limit_rad), abs=1e-5)


def test_reusing_randomized_start_preserves_the_same_flight():
    body, vehicle = load_body_config("europa"), load_vehicle_config("europa_lander")
    policy = make_policy_fn("pd_fallback", body_params=body, vehicle_params=vehicle)
    baseline, _ = collect(policy, [42], body, vehicle)
    first = baseline[0][0][0]
    same_start = tuple(first[n] for n in ("x", "y", "vx", "vy"))
    custom, _ = collect(policy, [42], body, vehicle, initial_conditions=same_start)
    assert custom[0][2]["outcome"] == baseline[0][2]["outcome"]
    assert len(custom[0][0]) == len(baseline[0][0])
    for name in first:
        np.testing.assert_allclose([r[name] for r in custom[0][0]],
                                   [r[name] for r in baseline[0][0]], rtol=1e-5, atol=1e-4)
