"""Vertical contact reward isolation and full-start selection safeguards."""

import json
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from env.reward_shaping import vertical_touchdown_reward
from rl.kaggle_recovery_v6 import configuration as previous_configuration
from rl.kaggle_recovery_v7 import (
    BANK_SEEDS,
    FRESH_SEEDS,
    TRAINING_SEEDS,
    VALIDATION_SEEDS,
    bindings,
    configuration,
    evaluate,
    select_candidate,
)


def test_vertical_penalty_is_contact_only_bounded_and_monotone():
    base = {k: jnp.float32(v) for k, v in {
        "velocity": -1., "altitude": -.2, "fuel": -.01,
        "angle": -.02, "landing": 80., "crash": -5.,
    }.items()}
    penalties = []
    for speed in (0., .3, .5, 1., 1.2, 10., 100.):
        noncontact = vertical_touchdown_reward(base, -speed, False)
        for key, value in base.items():
            np.testing.assert_array_equal(noncontact[key], value)
        contact = vertical_touchdown_reward(base, -speed, True)
        for key, value in base.items():
            if key != "crash":
                np.testing.assert_array_equal(contact[key], value)
        penalty = float(base["crash"] - contact["crash"])
        assert 0 <= penalty <= 30
        np.testing.assert_allclose(penalty, 30*speed**2/(speed**2+.25), atol=3e-6)
        penalties.append(penalty)
        np.testing.assert_array_equal(contact["crash"],
                                      vertical_touchdown_reward(base, speed, True)["crash"])
    assert all(a < b for a, b in pairwise(penalties))


@pytest.mark.parametrize("height,vy,vx", [
    (5., -1.2, .1), (.001, -.4, .1), (.001, -1.2, .1), (.001, -.4, .7),
])
def test_mode5_changes_only_contact_cost_not_dynamics_noise_or_success(height, vy, vx):
    bind = bindings()
    _, state = bind.env.reset(jax.random.key(0), bind.body_params, bind.vehicle_params)
    state = state._replace(y=jnp.float32(height), vy=jnp.float32(vy), vx=jnp.float32(vx))
    args = (jnp.array([-.4, 0.]), jax.random.key(99))
    old = bind.env.step(state._replace(reward_mode=jnp.float32(4)), *args)
    new = bind.env.step(state._replace(reward_mode=jnp.float32(5)), *args)
    np.testing.assert_array_equal(old[0], new[0])
    for field in state._fields:
        if field != "reward_mode":
            np.testing.assert_array_equal(getattr(old[1], field), getattr(new[1], field))
    assert bool(old[3]) == bool(new[3])
    for key in old[4]:
        np.testing.assert_array_equal(old[4][key], new[4][key])
    contacted = int(new[4]["outcome"]) in (1, 2)
    expected = vertical_touchdown_reward(old[2], new[1].vy, contacted)
    for key in expected:
        np.testing.assert_allclose(new[2][key], expected[key], atol=1e-6)
    if contacted:
        assert bool(new[4]["landed"]) == (abs(float(new[1].vy)) < 1 and abs(float(new[1].vx)) < .5)


def test_unchanged_ppo_settings_and_unused_seed_scopes():
    assert configuration() == previous_configuration()
    assert list(TRAINING_SEEDS.values()) == [14, 15, 16]
    used = set(range(2000, 2100)) | set(range(4000, 4200)) | set(range(6000, 6400))
    used |= set(range(8000, 10100)) | set(range(30000, 30200)) | set(range(40000, 40500))
    used |= set(range(60000, 61000)) | set(range(70000, 72000)) | set(range(80000, 82000))
    used |= set(range(900000, 901024))
    assert len(VALIDATION_SEEDS) == 1000 and set(VALIDATION_SEEDS).isdisjoint(used)
    assert len(FRESH_SEEDS) == 2000 and set(FRESH_SEEDS).isdisjoint(used | set(VALIDATION_SEEDS))
    assert len(BANK_SEEDS) == 1024 and set(BANK_SEEDS).isdisjoint(used | set(FRESH_SEEDS))


def test_new_reward_compiles_in_eight_transition_ppo_curriculum_fixture():
    from rl.landing_curriculum import mixed_start_env
    from rl.ppo import make_train
    bind = bindings()
    _, state = bind.env.reset(jax.random.key(4), bind.body_params, bind.vehicle_params)
    state = state._replace(y=jnp.float32(.001), vy=jnp.float32(-.4), vy_prev=jnp.float32(-.4))
    bank = jax.tree_util.tree_map(lambda value: value[None], state)
    env = mixed_start_env(bank, [1], probability=1.)
    cfg = replace(configuration(), num_envs=2, num_steps=4, total_steps=8,
                  update_epochs=1, num_minibatches=2, freeze_obs_stats=False)
    result = make_train(env, cfg)(jax.random.key(15), bind.body_params, bind.vehicle_params)
    assert np.isfinite(np.asarray(result.metrics["loss"])).all()
    assert int(result.runner.train_state.step) == 2
    assert float(state.reward_mode) == 5.


def test_test_scores_cannot_reselect_v7_checkpoint(tmp_path, monkeypatch):
    import rl.kaggle_recovery_v7 as worker
    rows = []
    for arm, seed in TRAINING_SEEDS.items():
        candidate = {"additional_steps": 80000, "checkpoint": arm,
                     "evaluation": {"soft_landings": 920}}
        rows.append({"arm": arm, "training_seed": seed, "selected": candidate,
                     "validation": [candidate]})
    assert select_candidate(rows[::-1])["arm"] == "seed_14"
    with pytest.raises(ValueError, match="Missing or duplicate"):
        select_candidate(rows[:-1])
    (tmp_path / "results.json").write_text(json.dumps({"runs": rows}))
    calls = []

    def fake(output, *, checkpoint, seeds, batched):
        assert json.loads((output / "selection.json").read_text())["candidate"]["arm"] == "seed_14"
        assert seeds == FRESH_SEEDS and batched
        calls.append(checkpoint)
        landed = 99 if checkpoint is None else 95 if checkpoint == "parent" else 91
        return {"success_rate": landed/100, "episodes": [
            {"seed": 90000+i, "outcome": 1 if i < landed else 2} for i in range(100)
        ]}

    monkeypatch.setattr(worker, "register_evaluation", fake)
    evaluate(Path("parent"), tmp_path)
    assert calls == ["parent", "seed_14", None]
    assert json.loads((tmp_path / "selection.json").read_text())["candidate"]["arm"] == "seed_14"
    assert json.loads((tmp_path / "target_check.json").read_text())["target_met"]
