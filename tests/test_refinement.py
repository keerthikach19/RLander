"""PPO refinement regression tests; no Kaggle execution or full training."""

import ast
import base64
import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from flax import serialization
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.reward_shaping import landing_profile_reward, success_contact_reward
from rl.kaggle_recovery_v2 import FRESH_SEEDS, choose_best, configuration
from rl.networks import ActorCritic, ObsStats, update_obs_stats
from rl.ppo import PPOConfig, make_train, project_log_std
from train import (
    evaluate_smoke,
    lander_bindings,
    load_ppo_initialization,
    run_experiment,
)


@pytest.fixture
def parent(tmp_path):
    params = ActorCritic(initial_log_std=-.9).init(jax.random.key(2), jnp.zeros(8))
    stats = update_obs_stats(ObsStats.empty(), jax.random.normal(jax.random.key(3), (8, 8)))
    checkpoint = tmp_path / "parent" / "checkpoint"
    checkpoint.parent.mkdir()
    manifest = {"config": {"body": "europa", "vehicle": "europa_lander"}, "actual_steps": 720000,
                "run": {"id": "parent-at-720000", "reward_id": "hand_shaped_landing_profile_v1", "config_hash": "example"}}
    (checkpoint.parent / "manifest.json").write_text(json.dumps(manifest))
    with ocp.PyTreeCheckpointer() as writer:
        writer.save(str(checkpoint), jax.device_get({"train_state": {"params": params},
                                                   "obs_stats": serialization.to_state_dict(stats)}))
    return checkpoint, params, stats


def test_warm_start_loads_actor_critic_stats_and_explicitly_resets_noise(parent):
    checkpoint, params, stats = parent
    loaded, paired, metadata = load_ppo_initialization(
        checkpoint, body_name="europa", vehicle_name="europa_lander", initial_log_std=math.log(.1)
    )
    for name in params["params"]:
        if name != "log_std":
            for a, b in zip(jax.tree_util.tree_leaves(loaded["params"][name]),
                            jax.tree_util.tree_leaves(params["params"][name])):
                np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(np.exp(loaded["params"]["log_std"]), .1, rtol=1e-6)
    for a, b in zip(jax.tree_util.tree_leaves(paired), jax.tree_util.tree_leaves(stats)):
        np.testing.assert_array_equal(a, b)
    assert metadata["critic_restored"] and not metadata["optimizer_restored"]
    assert metadata["parent_steps"] == 720000
    unchanged, _, _ = load_ppo_initialization(checkpoint, body_name="europa",
                                            vehicle_name="europa_lander", initial_log_std=None)
    np.testing.assert_array_equal(unchanged["params"]["log_std"], params["params"]["log_std"])
    with pytest.raises(ValueError, match="body/vehicle"):
        load_ppo_initialization(checkpoint, body_name="mars", vehicle_name="europa_lander", initial_log_std=-2.)


def test_bounded_real_environment_warm_start_freezes_normalization(parent, tmp_path):
    checkpoint, _, stats = parent
    bind = lander_bindings(OmegaConf.create({"reward": {"mode": "landing_success"}}))
    bind.evaluate = None  # Unit integration: eight transitions, no full rollout evaluation.
    cfg = PPOConfig(num_envs=2, num_steps=4, total_steps=8, num_minibatches=2, update_epochs=1,
                    initial_log_std=math.log(.1), log_std_min=math.log(.05), log_std_max=math.log(.15),
                    freeze_obs_stats=True, reward_scale=.01)
    result, manifest = run_experiment(bind, cfg, ppo_init=checkpoint, output_root=tmp_path / "runs")
    for a, b in zip(jax.tree_util.tree_leaves(result.runner.obs_stats), jax.tree_util.tree_leaves(stats)):
        np.testing.assert_array_equal(a, b)
    assert np.isfinite(result.metrics["loss"]).all()
    assert np.all(result.metrics["action_std"] >= .05) and np.all(result.metrics["action_std"] <= .15)
    assert int(result.runner.train_state.step) == 2
    assert manifest["config"]["actor_initialization"]["method"] == "ppo_actor_critic_warm_start"
    assert manifest["run"]["reward_id"] == "hand_shaped_landing_success_v1"


def test_missing_frozen_normalization_and_conflicting_initializers_fail():
    bind = lander_bindings(OmegaConf.create({}))
    cfg = PPOConfig(num_envs=2, num_steps=4, total_steps=8, num_minibatches=2, freeze_obs_stats=True)
    with pytest.raises(ValueError, match="Frozen normalization"):
        make_train(bind.env, cfg)(jax.random.key(0), bind.body_params, bind.vehicle_params)
    with pytest.raises(ValueError, match="cannot use BC"):
        run_experiment(bind, cfg, ppo_init="unused", bc_init="unused")


def test_log_std_projection_is_stored_and_bounds_are_validated():
    cfg = configuration()
    params = {"params": {"log_std": jnp.array([-10., 10.]), "actor_mean": jnp.ones(2)}}
    constrained = project_log_std(params, cfg)
    np.testing.assert_allclose(np.exp(constrained["params"]["log_std"]), [.05, .15], rtol=1e-6)
    np.testing.assert_array_equal(constrained["params"]["actor_mean"], params["params"]["actor_mean"])
    for kwargs in ({"log_std_min": 1.}, {"log_std_max": -1.}, {"log_std_min": float("nan")}):
        with pytest.raises(ValueError, match="log.std"):
            PPOConfig(**kwargs)


def test_success_reward_keeps_dense_costs_and_contact_scale():
    bind = lander_bindings(OmegaConf.create({}))
    _, state = bind.env.reset(jax.random.key(0), bind.body_params, bind.vehicle_params)
    def score(vy):
        continuous = landing_profile_reward(state, 0., .1, vy, .4, .0, True)
        revised = success_contact_reward(continuous, abs(vy) < 1., abs(vy) >= 1.)
        for key in ("velocity", "altitude", "fuel", "angle"):
            np.testing.assert_array_equal(continuous[key], revised[key])
        return float(revised["landing"] + revised["crash"])
    assert 0 < score(-.999) <= 100
    assert -100 <= score(-1.001) < 0
    assert score(-.999) - score(-1.001) > 99
    assert score(-3.) > score(-10.)
    running = landing_profile_reward(state, 50., 0., -2., .4, .0, False)
    assert success_contact_reward(running, False, False)["landing"] == 0
    # Actual environment outcomes/observations do not depend on the reward mode.
    for vy in (-.4, -3.):
        start = state._replace(y=jnp.float32(.001), vy=jnp.float32(vy), vx=jnp.float32(.1))
        old = bind.env.step(start._replace(reward_mode=2.), jnp.zeros(2), jax.random.key(4))
        new = bind.env.step(start._replace(reward_mode=3.), jnp.zeros(2), jax.random.key(4))
        np.testing.assert_array_equal(old[0], new[0])
        for key in old[4]:
            np.testing.assert_array_equal(old[4][key], new[4][key])


def test_best_checkpoint_includes_parent_and_breaks_ties_early():
    def candidate(step, soft):
        return {"additional_steps": step, "evaluation": {"soft_landings": soft}}
    initial = candidate(0, 49)
    assert choose_best([initial, candidate(40000, 20), candidate(320000, 7)]) is initial
    assert choose_best([initial, candidate(40000, 49)]) is initial
    assert choose_best([initial, candidate(40000, 60), candidate(80000, 60)])["additional_steps"] == 40000
    assert len(FRESH_SEEDS) == 200 and min(FRESH_SEEDS) == 30000
    assert set(FRESH_SEEDS).isdisjoint(range(10000, 10100))


@pytest.mark.parametrize("allowed", [False, True])
def test_v2_launcher_preserves_parent_and_evaluates_after_both_selections(tmp_path, monkeypatch, allowed):
    from rl import kaggle_recovery_v2 as recovery
    parent = tmp_path / "input" / "checkpoint"
    parent.mkdir(parents=True)
    (parent / "fixture").write_text("parent data")
    (parent.parent / "manifest.json").write_text("{}")
    output = tmp_path / "output"
    original_is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda p: True if str(p).replace("\\", "/") == "/kaggle/working"
                        else original_is_dir(p))
    monkeypatch.setattr(recovery.subprocess, "check_output", lambda *a, **k: "mock packages/hardware")
    stages = []

    class FakeProcess:
        def __init__(self, command, **kwargs):
            stage = command[command.index("--worker") + 1]
            stages.append(stage)
            assert Path(command[command.index("--parent") + 1]) == output / "parent" / "checkpoint"
            if stage == "preflight":
                recovery.write_json(output / "preflight.json", {"training_allowed": allowed})
            elif stage in recovery.ARMS:
                recovery.write_json(output / f"{stage}.json", {"arm": stage, "selected": "mock"})
            else:
                report = json.loads((output / "results.json").read_text())
                assert [r["arm"] for r in report["runs"]] == list(recovery.ARMS)
                recovery.write_json(output / "fresh_evaluation.json", [])

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(recovery.subprocess, "Popen", FakeProcess)
    recovery.launch(parent, output)
    assert stages == (["preflight", *recovery.ARMS, "evaluate"] if allowed else ["preflight"])
    assert (output / "parent" / "checkpoint" / "fixture").read_text() == "parent data"
    assert (parent / "fixture").read_text() == "parent data"


def test_sampled_comparison_preserves_sensor_stream_when_noise_is_zero():
    from tests.test_rl import FakeEnv, parameters
    body, vehicle = parameters()
    params = ActorCritic(initial_log_std=-100.).init(jax.random.key(0), jnp.zeros(8))
    stats = ObsStats.empty()
    kwargs = {"seeds": (2000, 2001), "max_steps": 10}
    a = evaluate_smoke(FakeEnv(horizon=3), body, vehicle, params, stats, **kwargs)
    b = evaluate_smoke(FakeEnv(horizon=3), body, vehicle, params, stats, sample_actions=True, **kwargs)
    assert a["episodes"] == b["episodes"]


def test_notebook_embeds_validation_parent_without_reading_fresh_results(tmp_path):
    from rl.prepare_kaggle_recovery_v2 import prepare
    def validation(step, outcomes):
        n = len(outcomes)
        return {"steps": step, "evaluation": {"soft_landings": outcomes.count(1),
            "crashed": outcomes.count(2), "timeouts": 0,
            "episodes": {"info": {"outcome": outcomes, "touchdown_v": [.4, 2.],
                                  "touchdown_vx": [.1] * n}, "reward_components": {"velocity": [-1.] * n}}}}
    prefix = "more_updates/example/snapshots/720000"
    final = {"arm": "more_updates", "success_rate": .0, "run_id": 3, "by_outcome": {},
             "episodes": [{"seed": 10000, "outcome": 2}]}
    buffer = io.BytesIO()
    pq.write_table(pa.table({"y": [2000., 0.], "vy": [-20., -2.], "vx": [1., .2]}), buffer)
    archive_path = tmp_path / "v1.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("results.json", json.dumps({"status": "complete", "runs": [], "frozen": [final],
                                                    "preflight": {"pd_w3": {"success_rate": 1.}}}))
        archive.writestr("more_updates.json", json.dumps({"run_dir": "/old/more_updates/example",
            "actual_steps": 960000, "validation": [validation(720000, [1, 2]), validation(960000, [2, 2])]}))
        archive.writestr(prefix + "/manifest.json", json.dumps({"run": {"id": "selected"}}))
        archive.writestr(prefix + "/checkpoint/fixture", b"parent fixture")
        archive.writestr("trajectories/3/10000.parquet", buffer.getvalue())
    path = prepare(archive_path, tmp_path / "notebook")
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    tree = ast.parse("".join(notebook["cells"][2]["source"]))
    encoded = {node.targets[0].id: node.value.args[0].value for node in tree.body
               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
               and node.targets[0].id in ("source_payload", "parent_payload")}
    provenance = json.loads((path.parent / "parent_provenance.json").read_text())
    payload = base64.b64decode(encoded["parent_payload"])
    assert hashlib.sha256(payload).hexdigest() == provenance["parent_zip_sha256"]
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert archive.read("checkpoint/fixture") == b"parent fixture"
    assert provenance["selection"]["steps"] == 720000
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded["source_payload"]))) as archive:
        for name, digest in json.loads(archive.read("source_manifest.json"))["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
