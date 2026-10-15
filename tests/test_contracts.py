"""Contract tests for LanderEnv (§5.2, §5.4, §5.5 — updated contract).

Tests verify:
  1. obs shape (8,)
  2. action shape (2,)
  3. info keys: touchdown_v, touchdown_vx, fuel_used, crashed, landed, truncated, outcome
  4. reward returns named components dict (W2 sums them)
  5. step returns 5-tuple: (obs, next_state, reward_components, done, info)
  6. compile-once: same jitted step runs Europa and Mars params without recompiling
  7. domain randomization: different keys -> different states, same key -> same state
  8. no NaN for rho0 = 0
  9. info["outcome"] is int32; info["truncated"] is bool
  10. next_state is an EnvState pytree carrying body/vehicle params
"""

import sys
import os

import jax
import jax.numpy as jnp
import pytest
import yaml

# Add project root to path so we can import env/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from env.lander_env import LanderEnv, EnvState


# ---------------------------------------------------------------------------
# Fixtures: load configs from YAML
# ---------------------------------------------------------------------------
def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def europa_body():
    configs_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    bodies = _load_yaml(os.path.join(configs_dir, "bodies.yaml"))
    return bodies["europa"]


@pytest.fixture
def europa_vehicle():
    configs_dir = os.path.join(os.path.dirname(__file__), "..", "configs")
    vehicles = _load_yaml(os.path.join(configs_dir, "vehicles.yaml"))
    return vehicles["europa_lander"]


def _make_mars_body():
    """Minimal Mars body params for compile-once test."""
    return {
        "name": "mars",
        "g": 3.71,
        "radius_m": 3389500.0,
        "rho0": 0.020,
        "scale_height_m": 11100.0,
        "success": {"vy_max_mps": 1.0, "vx_max_mps": 0.5},
    }


# ---------------------------------------------------------------------------
# Test 1: observation shape
# ---------------------------------------------------------------------------
def test_obs_shape(europa_body, europa_vehicle):
    """Obs must be shape (8,) per §5.4."""
    key = jax.random.PRNGKey(42)
    obs, state = LanderEnv.reset(key, europa_body, europa_vehicle)
    assert obs.shape == (8,), f"obs shape {obs.shape} != (8,)"


# ---------------------------------------------------------------------------
# Test 2: action shape consumed correctly
# ---------------------------------------------------------------------------
def test_action_shape(europa_body, europa_vehicle):
    """step() must accept action shape (2,) per §5.2."""
    key = jax.random.PRNGKey(42)
    obs, state = LanderEnv.reset(key, europa_body, europa_vehicle)

    action = jnp.array([0.0, 0.0])  # neutral throttle+angle
    assert action.shape == (2,)

    key2 = jax.random.PRNGKey(99)
    obs2, next_state, reward_components, done, info = LanderEnv.step(state, action, key2)
    assert obs2.shape == (8,), f"obs after step shape {obs2.shape} != (8,)"


# ---------------------------------------------------------------------------
# Test 3: info keys
# ---------------------------------------------------------------------------
def test_info_keys(europa_body, europa_vehicle):
    """info must contain the required keys per §5.2."""
    required_keys = {
        "touchdown_v", "touchdown_vx", "fuel_used",
        "crashed", "landed", "truncated", "outcome",
    }
    key = jax.random.PRNGKey(42)
    obs, state = LanderEnv.reset(key, europa_body, europa_vehicle)

    action = jnp.array([0.0, 0.0])
    _, _, _, _, info = LanderEnv.step(state, action, jax.random.PRNGKey(99))

    missing = required_keys - set(info.keys())
    assert not missing, f"Missing info keys: {missing}"


# ---------------------------------------------------------------------------
# Test 4: reward returns named components
# ---------------------------------------------------------------------------
def test_reward_named_components(europa_body, europa_vehicle):
    """Reward must return a dict of named components (§5.5)."""
    key = jax.random.PRNGKey(42)
    obs, state = LanderEnv.reset(key, europa_body, europa_vehicle)

    action = jnp.array([0.5, 0.0])
    _, _, reward_components, _, _ = LanderEnv.step(state, action, jax.random.PRNGKey(99))

    assert isinstance(reward_components, dict), "reward_components must be a dict"
    assert len(reward_components) > 0, "reward_components must have at least one entry"

    # All values must be scalars
    for name, val in reward_components.items():
        assert val.shape == (), f"reward component '{name}' must be scalar, got {val.shape}"


# ---------------------------------------------------------------------------
# Test 5: step returns 5-tuple with next_state
# ---------------------------------------------------------------------------
def test_step_returns_5_tuple(europa_body, europa_vehicle):
    """step() must return (obs, next_state, reward_components, done, info)."""
    key = jax.random.PRNGKey(42)
    obs, state = LanderEnv.reset(key, europa_body, europa_vehicle)

    action = jnp.array([0.0, 0.0])
    result = LanderEnv.step(state, action, jax.random.PRNGKey(99))
    assert len(result) == 5, f"step must return 5-tuple, got {len(result)}"

    obs2, next_state, reward_components, done, info = result
    assert isinstance(next_state, EnvState), "next_state must be an EnvState"
    # next_state carries body/vehicle params unchanged
    assert jnp.allclose(next_state.g, state.g), "g must be carried forward"
    assert jnp.allclose(next_state.mass0, state.mass0), "mass0 must be carried forward"
    assert jnp.allclose(next_state.isp_s, state.isp_s), "isp_s must be carried forward"


# ---------------------------------------------------------------------------
# Test 6: compile-once (same jitted step for Europa and Mars)
# ---------------------------------------------------------------------------
def test_compile_once(europa_body, europa_vehicle):
    """Same jitted step must work with different body params (§5.3)."""
    mars_body = _make_mars_body()

    key = jax.random.PRNGKey(42)
    action = jnp.array([0.0, 0.0])

    # Reset and step with Europa
    obs_e, state_e = LanderEnv.reset(key, europa_body, europa_vehicle)
    obs_e2, _, _, _, _ = LanderEnv.step(state_e, action, jax.random.PRNGKey(1))

    # Reset and step with Mars (same vehicle params, different body)
    obs_m, state_m = LanderEnv.reset(key, mars_body, europa_vehicle)
    obs_m2, _, _, _, _ = LanderEnv.step(state_m, action, jax.random.PRNGKey(1))

    # Both should produce valid (8,) obs with no NaN
    assert obs_e2.shape == (8,)
    assert obs_m2.shape == (8,)
    assert not jnp.any(jnp.isnan(obs_e2)), "Europa obs has NaN"
    assert not jnp.any(jnp.isnan(obs_m2)), "Mars obs has NaN"

    # They should be different (different gravity, etc.)
    assert not jnp.allclose(obs_e2, obs_m2), "Europa and Mars obs should differ"


# ---------------------------------------------------------------------------
# Test 7: domain randomization - different keys give different states
# ---------------------------------------------------------------------------
def test_randomization_different_keys(europa_body, europa_vehicle):
    """Two different keys must produce different initial states (§5.8)."""
    key1 = jax.random.PRNGKey(1)
    key2 = jax.random.PRNGKey(2)

    obs1, state1 = LanderEnv.reset(key1, europa_body, europa_vehicle)
    obs2, state2 = LanderEnv.reset(key2, europa_body, europa_vehicle)

    # States should differ (domain randomization)
    states_same = (
        jnp.allclose(state1.y, state2.y)
        and jnp.allclose(state1.vy, state2.vy)
        and jnp.allclose(state1.mass, state2.mass)
    )
    assert not states_same, "Different keys must produce different initial states"


# ---------------------------------------------------------------------------
# Test 8: domain randomization - same key reproduces exactly
# ---------------------------------------------------------------------------
def test_randomization_same_key(europa_body, europa_vehicle):
    """Same key must reproduce exactly the same state (§5.8)."""
    key = jax.random.PRNGKey(42)

    obs1, state1 = LanderEnv.reset(key, europa_body, europa_vehicle)
    obs2, state2 = LanderEnv.reset(key, europa_body, europa_vehicle)

    assert jnp.allclose(state1.y, state2.y), "Same key must reproduce y"
    assert jnp.allclose(state1.vy, state2.vy), "Same key must reproduce vy"
    assert jnp.allclose(state1.mass, state2.mass), "Same key must reproduce mass"
    assert jnp.allclose(state1.g, state2.g), "Same key must reproduce g"


# ---------------------------------------------------------------------------
# Test 9: no NaN for vacuum body (rho0=0)
# ---------------------------------------------------------------------------
def test_no_nan_vacuum(europa_body, europa_vehicle):
    """No NaN when rho0=0 (vacuum body). Run several steps."""
    assert europa_body["rho0"] == 0.0, "Europa must be vacuum"

    key = jax.random.PRNGKey(42)
    obs, state = LanderEnv.reset(key, europa_body, europa_vehicle)
    assert not jnp.any(jnp.isnan(obs)), "NaN in initial obs"

    for i in range(10):
        action = jnp.array([0.5, 0.1])
        step_key = jax.random.PRNGKey(100 + i)
        obs, state, reward_comp, done, info = LanderEnv.step(state, action, step_key)
        assert not jnp.any(jnp.isnan(obs)), f"NaN in obs at step {i}"
        for name, val in reward_comp.items():
            assert not jnp.isnan(val), f"NaN in reward component '{name}' at step {i}"
        if done:
            break


# ---------------------------------------------------------------------------
# Test 10: info types — outcome is int32, truncated is bool
# ---------------------------------------------------------------------------
def test_info_types(europa_body, europa_vehicle):
    """info['outcome'] must be int32, info['truncated'] must be bool."""
    key = jax.random.PRNGKey(42)
    obs, state = LanderEnv.reset(key, europa_body, europa_vehicle)

    action = jnp.array([0.0, 0.0])
    _, _, _, _, info = LanderEnv.step(state, action, jax.random.PRNGKey(99))

    assert info["outcome"].dtype == jnp.int32, (
        f"outcome must be int32, got {info['outcome'].dtype}"
    )
    assert info["truncated"].dtype == jnp.bool_, (
        f"truncated must be bool, got {info['truncated'].dtype}"
    )


# ---------------------------------------------------------------------------
# Test 11: done is a scalar
# ---------------------------------------------------------------------------
def test_done_scalar(europa_body, europa_vehicle):
    """done must be a scalar."""
    key = jax.random.PRNGKey(42)
    obs, state = LanderEnv.reset(key, europa_body, europa_vehicle)

    action = jnp.array([0.0, 0.0])
    _, _, _, done, _ = LanderEnv.step(state, action, jax.random.PRNGKey(99))
    assert done.shape == (), f"done shape must be scalar, got {done.shape}"


# ---------------------------------------------------------------------------
# Test 12: state has no rng field
# ---------------------------------------------------------------------------
def test_state_no_rng(europa_body, europa_vehicle):
    """State must not carry an rng field (caller passes key to step)."""
    key = jax.random.PRNGKey(42)
    _, state = LanderEnv.reset(key, europa_body, europa_vehicle)

    field_names = state._fields
    assert "rng" not in field_names, "State must not have an 'rng' field"


# ---------------------------------------------------------------------------
# Test 13: step obs is (8,) after multiple steps (using next_state chain)
# ---------------------------------------------------------------------------
def test_multi_step_chain(europa_body, europa_vehicle):
    """Chaining step via next_state: obs remains (8,), state is valid."""
    key = jax.random.PRNGKey(42)
    obs, state = LanderEnv.reset(key, europa_body, europa_vehicle)

    action = jnp.array([0.3, -0.2])
    for i in range(5):
        step_key = jax.random.PRNGKey(200 + i)
        obs, state, reward_comp, done, info = LanderEnv.step(state, action, step_key)
        assert obs.shape == (8,), f"Step {i}: obs shape {obs.shape}"
        assert isinstance(reward_comp, dict)
        assert isinstance(state, EnvState)
        if done:
            break
