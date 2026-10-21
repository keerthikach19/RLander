"""Curriculum state fidelity, PPO compatibility, and full-start evaluation isolation."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from env.lander_env import EnvState, LanderEnv
from rl.kaggle_recovery_v6 import (
    BANK_SEEDS,
    FRESH_SEEDS,
    SNAPSHOTS,
    STEPS,
    TRAINING_SEEDS,
    VALIDATION_SEEDS,
    bindings,
    configuration,
    evaluate,
    select_candidate,
    training_bindings,
)
from rl.landing_curriculum import collect_bank, load_bank, mixed_start_env


def fixture_bank():
    bind = bindings()
    _, state = bind.env.reset(jax.random.key(15), bind.body_params, bind.vehicle_params)
    state = state._replace(y=jnp.float32(10.), t=jnp.float32(90.),
                           vy=jnp.float32(-1.5), vy_prev=jnp.float32(-1.6),
                           mass=state.mass - 40.)
    return jax.tree_util.tree_map(lambda value: value[None], state), bind


def write_bank(path):
    bank, _ = fixture_bank()
    np.savez_compressed(path, **{name: np.asarray(value) for name, value in zip(EnvState._fields, bank)},
                        source_outcome=np.array([2]))
    return bank


def test_reset_preserves_physics_scales_time_and_noisy_observation_contract():
    bank, bind = fixture_bank()
    full = mixed_start_env(bank, [2], probability=0)
    key = jax.random.key(17)
    actual = full.reset(key, bind.body_params, bind.vehicle_params)
    expected = bind.env.reset(key, bind.body_params, bind.vehicle_params)
    for a, b in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_array_equal(a, b)
    partial = mixed_start_env(bank, [2], probability=1.)
    obs, state = partial.reset(key, bind.body_params, bind.vehicle_params)
    for value, saved in zip(state, bank):
        np.testing.assert_array_equal(value, saved[0])
    assert float(state.L) > 1800 and float(state.y) == 10.
    assert float(state.sensor_noise_scale) == 1. and float(state.t) == 90.
    assert partial.step is LanderEnv.step
    different_noise, _ = partial.reset(jax.random.key(18), bind.body_params, bind.vehicle_params)
    assert obs.shape == (8,) and not np.array_equal(obs, different_noise)


def test_bank_loading_rejects_dead_state(tmp_path):
    path = tmp_path / "bank.npz"
    bank = write_bank(path)
    loaded, outcomes = load_bank(path)
    for saved, value in zip(bank, loaded):
        np.testing.assert_array_equal(saved, value)
    assert outcomes.tolist() == [2]
    np.savez(path, **{name: np.asarray(value) for name, value in zip(EnvState._fields, bank._replace(
        y=jnp.array([0.])) )}, source_outcome=np.array([2]))
    with pytest.raises(ValueError, match="live, airborne"):
        load_bank(path)


def test_collection_contains_actual_parent_states_and_no_action_targets(tmp_path):
    from rl.networks import ActorCritic, ObsStats
    bind = bindings()
    params = ActorCritic().init(jax.random.key(1), jnp.zeros(8))
    params["params"]["actor_mean"]["kernel"] = jnp.zeros_like(params["params"]["actor_mean"]["kernel"])
    params["params"]["actor_mean"]["bias"] = jnp.array([-1., 0.])
    path = tmp_path / "collected.npz"
    report = collect_bank(path, (900001, 900002), bind.body_params, bind.vehicle_params,
                          params, ObsStats.empty())
    bank, outcomes = load_bank(path)
    assert report["episodes"] == 2 and report["states"] == len(bank.y) > 0
    assert (np.asarray(bank.L) > 1800).all()
    assert (np.asarray(bank.t) > 0).all() and (np.asarray(bank.y) <= 100).all()
    assert (outcomes == 2).all()
    with np.load(path) as data:
        assert "actions" not in data and "targets" not in data
        assert set(data["source_seed"]) <= {900001, 900002}


def test_training_wrapper_cannot_replace_validation_env(tmp_path, monkeypatch):
    import rl.kaggle_recovery_v6 as worker
    path = tmp_path / "curriculum_bank.npz"
    write_bank(path)
    (tmp_path / "preflight.json").write_text(json.dumps({
        "curriculum_bank": {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}))
    base = bindings()
    monkeypatch.setattr(worker, "bindings", lambda: base)
    curriculum = training_bindings(tmp_path)
    assert curriculum.evaluate is base.evaluate
    assert curriculum.evaluation_config == {"seed_start": 9000, "episodes": 1000}
    assert curriculum.env is not base.env
    assert isinstance(base.env, LanderEnv)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash changed"):
        training_bindings(tmp_path)


def test_curriculum_works_with_eight_transition_ppo_cpu_check():
    from rl.ppo import make_train
    bank, bind = fixture_bank()
    env = mixed_start_env(bank, [2], probability=.75)
    cfg = replace(configuration(), num_envs=2, num_steps=4, total_steps=8,
                  update_epochs=1, num_minibatches=2, freeze_obs_stats=False)
    result = make_train(env, cfg)(jax.random.key(21), bind.body_params, bind.vehicle_params)
    assert np.isfinite(np.asarray(result.metrics["loss"])).all()
    assert int(result.runner.train_state.step) == 2


def rows():
    result = []
    for arm, seed in TRAINING_SEEDS.items():
        candidate = {"additional_steps": 80000, "checkpoint": arm,
                     "evaluation": {"soft_landings": 920}}
        result.append({"arm": arm, "training_seed": seed, "selected": candidate,
                       "validation": [candidate]})
    return result


def test_selection_and_seed_budget_constraints():
    assert select_candidate(rows()[::-1])["arm"] == "seed_11"
    with pytest.raises(ValueError, match="Missing or duplicate"):
        select_candidate(rows()[:-1])
    assert STEPS * len(TRAINING_SEEDS) == 1920000
    cfg = configuration()
    assert cfg.num_updates == 160 and all(s % cfg.batch_size == 0 for s in SNAPSHOTS)
    old = set(range(2000, 2100)) | set(range(4000, 4200)) | set(range(6000, 6400))
    old |= set(range(8000, 8800)) | set(range(10000, 10100))
    old |= set(range(30000, 30200)) | set(range(40000, 40500))
    old |= set(range(60000, 61000)) | set(range(70000, 72000))
    assert len(VALIDATION_SEEDS) == 1000 and len(FRESH_SEEDS) == 2000 and len(BANK_SEEDS) == 1024
    assert set(VALIDATION_SEEDS).isdisjoint(old)
    assert set(FRESH_SEEDS).isdisjoint(old | set(VALIDATION_SEEDS))
    assert set(BANK_SEEDS).isdisjoint(old | set(VALIDATION_SEEDS) | set(FRESH_SEEDS))


def test_full_test_cannot_select_another_policy(tmp_path, monkeypatch):
    import rl.kaggle_recovery_v6 as worker
    (tmp_path / "results.json").write_text(json.dumps({"runs": rows()}))
    calls = []

    def fake(output, *, checkpoint, seeds, batched):
        assert json.loads((output / "selection.json").read_text())["candidate"]["arm"] == "seed_11"
        assert seeds == FRESH_SEEDS and batched
        calls.append(checkpoint)
        landed = 99 if checkpoint is None else 95 if checkpoint == "parent" else 91
        return {"success_rate": landed / 100, "episodes": [
            {"seed": 80000 + i, "outcome": 1 if i < landed else 2} for i in range(100)
        ]}

    monkeypatch.setattr(worker, "register_evaluation", fake)
    evaluate(Path("parent"), tmp_path)
    assert calls == ["parent", "seed_11", None]
    assert json.loads((tmp_path / "selection.json").read_text())["candidate"]["arm"] == "seed_11"
    assert json.loads((tmp_path / "target_check.json").read_text())["target_met"]
