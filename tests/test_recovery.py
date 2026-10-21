"""Recovery regressions: reward preferences, honest metrics, portable packaging."""

import ast
import base64
import hashlib
import io
import itertools
import json
import sys
import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env.lander_env import LanderEnv
from env.reward_shaping import descent_reference, landing_profile_reward
from eval.metrics import aggregate_episodes
from rl.kaggle_recovery import ARMS, SNAPSHOTS, STEPS, plan
from train import lander_bindings


@pytest.fixture
def setup():
    bind = lander_bindings(OmegaConf.create({"reward": {"mode": "landing_profile"}}))
    obs, state = bind.env.reset(jax.random.key(1), bind.body_params, bind.vehicle_params)
    return bind, obs, state


def test_contact_reward_prefers_gentler_impacts_on_both_axes(setup):
    _, _, state = setup
    def contact(vx, vy):
        reward = landing_profile_reward(state, 0., vx, vy, .4, 0., True)
        return float(reward["landing"] + reward["crash"])
    values = [contact(0., -v) for v in (0., .4, .99, 1.01, 3., 7.7, 16.)]
    assert all(a > b for a, b in itertools.pairwise(values))
    assert contact(.49, -.4) > contact(.51, -.4) > contact(5.7, -.4) > contact(16.8, -.4)
    assert abs(contact(0., -.999) - contact(0., -1.001)) < .2


def test_profile_tracking_and_lateral_damping_have_signal(setup):
    _, _, state = setup
    for height in (.0, 5., 50., 500.):
        target = descent_reference(height, state.g)
        def dense(vx, vy, height=height):
            return sum(landing_profile_reward(state, height, vx, vy, .4, 0., False).values())
        assert dense(0., target) > dense(0., 0.)
        assert dense(0., target) > dense(2., target) > dense(16., target)
        assert dense(0., target) < 0  # waiting is never free, even on profile


def test_reward_mode_changes_neither_physics_observations_nor_success(setup):
    bind, _, state = setup
    for y, vx, vy in ((.01, .1, -.4), (.01, 2., -5.), (50., 0., 0.)):
        start = state._replace(y=jnp.float32(y), vx=jnp.float32(vx), vy=jnp.float32(vy))
        action = jnp.array([-.3, .1])
        old = LanderEnv.step(start._replace(reward_mode=jnp.float32(1)), action, jax.random.key(4))
        new = LanderEnv.step(start, action, jax.random.key(4))
        np.testing.assert_array_equal(old[0], new[0])
        assert bool(old[3]) == bool(new[3])
        for name in old[4]:
            np.testing.assert_array_equal(old[4][name], new[4][name])
        for name in ("x", "y", "vx", "vy", "mass"):
            np.testing.assert_array_equal(getattr(old[1], name), getattr(new[1], name))
    assert float(bind.body_params["success"]["vy_max_mps"]) == 1.0
    assert float(bind.body_params["success"]["vx_max_mps"]) == .5


def test_noise_off_is_diagnostic_and_keeps_physics(setup):
    bind, _, state = setup
    vehicle = dict(bind.vehicle_params, sensor_noise_scale=0.)
    clean, quiet_state = bind.env.reset(jax.random.key(1), bind.body_params, vehicle)
    np.testing.assert_allclose(clean[:3], [state.y / state.L, state.vx / state.U, state.vy / state.U])
    a = bind.env.step(state, jnp.zeros(2), jax.random.key(10))
    b = bind.env.step(quiet_state, jnp.zeros(2), jax.random.key(10))
    for name in ("y", "vx", "vy", "mass"):
        np.testing.assert_array_equal(getattr(a[1], name), getattr(b[1], name))


def test_timeout_velocities_are_not_touchdowns():
    infos = [{"outcome": o, "touchdown_v": v, "touchdown_vx": v / 2, "fuel_used": 10.}
             for o, v in ((1, .4), (2, 8.), (3, 1000.))]
    result = aggregate_episodes(infos)
    assert result["touchdown"]["n_episodes"] == 2
    assert result["touchdown"]["vy_mean"] == pytest.approx(4.2)
    assert result["by_outcome"]["crashed"]["vy_mean"] == 8.
    assert result["by_outcome"]["timeout"]["vy_mean"] == 1000.
    assert result["success_rate"] == pytest.approx(1 / 3)
    empty = aggregate_episodes(infos[-1:])
    assert empty["touchdown"]["n_episodes"] == 0
    assert empty["touchdown"]["vy_mean"] is None
    json.dumps(empty, allow_nan=False)


def test_pilots_are_bounded_with_exact_snapshot_boundaries():
    assert STEPS <= 1_000_000
    for num_envs in ARMS.values():
        assert STEPS % (num_envs * 125) == 0
        assert all(s % (num_envs * 125) == 0 for s in SNAPSHOTS)
    assert [a["updates"] for a in plan()["arms"]] == [60, 480]


def test_revised_reward_runs_under_compiled_ppo(setup):
    from rl.ppo import PPOConfig, make_train
    bind, _, _ = setup
    config = PPOConfig(num_envs=2, num_steps=4, total_steps=8, num_minibatches=2,
                       update_epochs=1, reward_scale=.01)
    result = make_train(bind.env, config, shaping_components=bind.shaping_components)(
        jax.random.key(9), bind.body_params, bind.vehicle_params
    )
    assert np.isfinite(result.metrics["loss"]).all()
    assert float(result.runner.obs_stats.count) > 0


@pytest.mark.parametrize("allowed,rate,expected", [
    (False, .0, ["preflight"]),
    (True, .11, ["preflight", "reward_only", "evaluate"]),
    (True, .10, ["preflight", "reward_only", "more_updates", "evaluate"]),
])
def test_launcher_gates_and_fallback_without_starting_processes(
    tmp_path, monkeypatch, allowed, rate, expected
):
    from rl import kaggle_recovery as recovery
    original_is_dir = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda p: True if str(p).replace("\\", "/") == "/kaggle/working"
                        else original_is_dir(p))
    monkeypatch.setattr(recovery.subprocess, "check_output", lambda *a, **k: "mock hardware/packages")
    stages = []
    output = tmp_path / "experiment"

    class FakeProcess:
        def __init__(self, command, **kwargs):
            stage = command[command.index("--worker") + 1]
            stages.append(stage)
            if stage == "preflight":
                recovery.write_json(output / "preflight.json", {"training_allowed": allowed})
            elif stage in ARMS:
                recovery.write_json(output / f"{stage}.json", {
                    "arm": stage, "final_validation_rate": rate, "checkpoint": "mock",
                })
            else:
                report = json.loads((output / "results.json").read_text())
                assert len(report["runs"]) == (2 if rate <= .1 else 1)
                recovery.write_json(output / "frozen.json", [{"arm": r["arm"]} for r in report["runs"]])

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(recovery.subprocess, "Popen", FakeProcess)
    recovery.launch(output)
    assert stages == expected
    report = json.loads((output / "results.json").read_text())
    assert report["status"] == ("complete" if allowed else "diagnostic_stop")


def test_recovery_notebook_embeds_current_sources_without_execution(tmp_path):
    from rl.prepare_kaggle_recovery import prepare
    path = prepare(tmp_path)
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None and cell["outputs"] == []
    tree = ast.parse("".join(notebook["cells"][2]["source"]))
    encoded = next(node.value.args[0].value for node in tree.body
                   if isinstance(node, ast.Assign) and node.targets[0].id == "payload")
    payload = base64.b64decode(encoded)
    assert payload == (tmp_path / "source.zip").read_bytes()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        manifest = json.loads(archive.read("source_manifest.json"))
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        for required in ("rl/kaggle_recovery.py", "storage/schema.sql", "env/reward_shaping.py"):
            assert required in archive.namelist()
    assert all(".env" != Path(name).name for name in manifest["files"])
