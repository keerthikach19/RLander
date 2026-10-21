"""Reward isolation, frozen selection and notebook packaging for v4."""

import ast
import base64
import hashlib
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from env.reward_shaping import (
    descent_reference,
    landing_profile_reward,
    success_contact_reward,
)
from rl.analyze_v3_return import analyze
from rl.kaggle_recovery_v3 import configuration as previous_configuration
from rl.kaggle_recovery_v4 import (
    FRESH_SEEDS,
    RUNS,
    STEPS,
    TRAINING_SEEDS,
    VALIDATION_SEEDS,
    bindings,
    configuration,
    evaluate,
    select_candidates,
)


def test_margin_changes_only_near_ground_velocity_cost():
    bind = bindings("landing_margin")
    _, state = bind.env.reset(jax.random.key(0), bind.body_params, bind.vehicle_params)
    def reward(h, vx, vy, margin):
        return success_contact_reward(
            landing_profile_reward(state, h, vx, vy, .4, .1, True, margin=margin), True, False,
        )
    for height in (0., 10., 30., 65., 100., 500.):
        old, new = reward(height, .4, -2., False), reward(height, .4, -2., True)
        for key in old:
            if key != "velocity":
                np.testing.assert_array_equal(old[key], new[key])
        if height >= 100.:
            np.testing.assert_allclose(old["velocity"], new["velocity"], rtol=1e-6)
    fast = float(descent_reference(10., state.g))
    slow = -float(jnp.sqrt(.4**2 + .25 * state.g * 10.))
    assert reward(10., 0., slow, True)["velocity"] > reward(10., 0., fast, True)["velocity"]
    assert reward(10., 0., fast, False)["velocity"] > reward(10., 0., slow, False)["velocity"]
    old_delta = reward(10., .4, slow, False)["velocity"] - reward(10., 0., slow, False)["velocity"]
    new_delta = reward(10., .4, slow, True)["velocity"] - reward(10., 0., slow, True)["velocity"]
    assert new_delta < old_delta < 0
    # Continuous taper at 30m and 100m; no hidden reward jump.
    for height in (30., 100.):
        assert abs(float(reward(height-1e-4, .4, -2., True)["velocity"]
                         - reward(height+1e-4, .4, -2., True)["velocity"])) < 1e-4


@pytest.mark.parametrize("height,vy,vx", [(20., -3., .2), (.001, -.4, .1), (.001, -2., .1), (.001, -.4, .7)])
def test_margin_does_not_change_environment_observations_or_outcomes(height, vy, vx):
    bind = bindings("landing_success")
    _, state = bind.env.reset(jax.random.key(7), bind.body_params, bind.vehicle_params)
    start = state._replace(y=jnp.float32(height), vy=jnp.float32(vy), vx=jnp.float32(vx))
    action, key = jnp.array([-.2, 0.]), jax.random.key(9)
    a = bind.env.step(start._replace(reward_mode=jnp.float32(3)), action, key)
    b = bind.env.step(start._replace(reward_mode=jnp.float32(4)), action, key)
    np.testing.assert_array_equal(a[0], b[0])
    assert bool(a[3]) == bool(b[3])
    for key in a[4]:
        np.testing.assert_array_equal(a[4][key], b[4][key])
    for field in state._fields:
        if field != "reward_mode":
            np.testing.assert_array_equal(getattr(a[1], field), getattr(b[1], field))


def test_matched_configs_and_nonoverlapping_seed_sets():
    assert configuration() == previous_configuration()
    assert len(RUNS) == 6 and STEPS * len(RUNS) == 960000
    assert {seed for arm, seed in RUNS.values() if arm == "control"} == set(TRAINING_SEEDS)
    assert {seed for arm, seed in RUNS.values() if arm == "margin"} == set(TRAINING_SEEDS)
    old = set(range(2000, 2100)) | set(range(4000, 4200)) | set(range(10000, 10100))
    old |= set(range(30000, 30200)) | set(range(40000, 40500))
    assert len(VALIDATION_SEEDS) == 400 and len(FRESH_SEEDS) == 1000
    assert set(VALIDATION_SEEDS).isdisjoint(old)
    assert set(FRESH_SEEDS).isdisjoint(old | set(VALIDATION_SEEDS))


def test_margin_compiles_with_ppo_updates_in_eight_transition_cpu_check():
    from rl.ppo import make_train
    bind = bindings("landing_margin")
    # Low-altitude unit fixture exercises the modified reward immediately.
    vehicle = {**bind.vehicle_params, "h0_m": jnp.float32(10.), "v0_mps": jnp.float32(1.)}
    cfg = replace(configuration(), num_envs=2, num_steps=4, total_steps=8,
                  update_epochs=1, num_minibatches=2, freeze_obs_stats=False)
    result = make_train(bind.env, cfg)(jax.random.key(11), bind.body_params, vehicle)
    assert np.isfinite(np.asarray(result.metrics["loss"])).all()
    assert int(result.runner.train_state.step) == 2


def candidates():
    return [{"arm": name, "training_seed": seed, "selected": {
        "checkpoint": name, "additional_steps": 20000,
        "evaluation": {"soft_landings": 300}}} for name, (_, seed) in RUNS.items()]


def test_cross_reward_ties_use_step_then_seed_then_control():
    rows = candidates()
    assert select_candidates(rows)["candidate"]["arm"] == "control_seed_5"
    rows[0]["selected"]["evaluation"]["soft_landings"] = 299
    assert select_candidates(rows)["candidate"]["arm"] == "margin_seed_5"
    rows[0]["selected"].update(additional_steps=0, checkpoint="parent")
    rows[0]["selected"]["evaluation"]["soft_landings"] = 300
    assert select_candidates(rows)["candidate"]["arm"] == "initial_parent"
    with pytest.raises(ValueError, match="Missing or duplicate"):
        select_candidates(rows[:-1])


def test_fresh_test_cannot_change_selected_reward(tmp_path, monkeypatch):
    import rl.kaggle_recovery_v4 as worker
    rows = candidates()
    (tmp_path / "results.json").write_text(json.dumps({"runs": rows}))
    calls = []
    def fake(output, *, checkpoint, seeds, batched):
        assert json.loads((output / "selection.json").read_text())["candidate"]["arm"] == "control_seed_5"
        assert seeds == FRESH_SEEDS and batched
        calls.append(checkpoint)
        values = [1, 1] if checkpoint.startswith("margin") else [1, 2]
        return {"success_rate": values.count(1)/2,
                "episodes": [{"seed": 60000+i, "outcome": v} for i, v in enumerate(values)]}
    monkeypatch.setattr(worker, "register_evaluation", fake)
    evaluate(Path("parent"), tmp_path)
    assert calls == ["parent", "control_seed_5", "margin_seed_5"]
    assert json.loads((tmp_path / "selection.json").read_text())["candidate"]["arm"] == "control_seed_5"
    assert json.loads((tmp_path / "paired_comparison.json").read_text())["margin_vs_control"]["newly_landed"] == 1


@pytest.fixture
def returned_v3(tmp_path):
    path = tmp_path / "v3.zip"
    prefix = "seed_4/run/snapshots/20000"
    checkpoint = "/kaggle/working/result/" + prefix + "/checkpoint"
    def candidate(steps, soft, ckpt):
        return {"additional_steps": steps, "checkpoint": ckpt, "evaluation": {
            "seeds": list(range(4000, 4200)), "soft_landings": soft, "n_episodes": 200}}
    runs = []
    for seed, steps, soft in ((2, 60000, 157), (3, 0, 149), (4, 20000, 158)):
        row = candidate(steps, soft, checkpoint if seed == 4 else f"model_{seed}")
        runs.append({"arm": f"seed_{seed}", "training_seed": seed, "selected": row, "validation": [row]})
    from rl.kaggle_recovery_v3 import select_candidate
    selected = select_candidate(runs)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(prefix + "/manifest.json", json.dumps({
            "evaluation": selected["evaluation"], "run": {"id": "official_parent"}}))
        z.writestr(prefix + "/checkpoint/fixture", b"official seed 4")
        z.writestr("parent_provenance.json", json.dumps({"selection": {"ancestral_steps": 880000}}))
        fresh = []
        for label, ckpt, landed, run_id in (("initial_parent", "parent", 393, 1),
                                           ("seed_2_selected", "model_2", 421, 2),
                                           ("seed_4_selected", checkpoint, 414, 3)):
            episodes = [{"seed": s, "outcome": 1 if s-40000 < landed else 2} for s in range(40000, 40500)]
            fresh.append({"label": label, "checkpoint": ckpt, "evaluation": {
                "run_id": run_id, "n_success": landed, "n_episodes": 500, "success_rate": landed/500,
                "ci95_low": .7, "ci95_high": .9, "by_outcome": {}, "episodes": episodes}})
            if run_id == 3:
                buffers = {}
                for outcome, vy in ((1, -.5), (2, -2.)):
                    buf = io.BytesIO()
                    pq.write_table(pa.table({"y": [2000., 10., 0.], "vy": [-20., -3., vy],
                                            "vx": [1., .2, .2]}), buf)
                    buffers[outcome] = buf.getvalue()
                for ep in episodes:
                    z.writestr(f"trajectories/3/{ep['seed']}.parquet", buffers[ep["outcome"]])
        z.writestr("results.json", json.dumps({"status": "complete", "runs": runs,
            "selection": {"candidate": selected, "selection_frozen_before_fresh_evaluation": True},
            "fresh_evaluation": fresh}))
    return path


def test_analysis_honors_official_parent_not_higher_fresh_score(returned_v3):
    report, payload = analyze(returned_v3)
    assert report["selected_parent"]["arm"] == "seed_4"
    assert report["selected_parent"]["ancestral_steps"] == 900000
    assert report["selected_fresh_failures"]["vertical_only"] == 86
    assert not report["gpu_training_verified"]
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        assert z.read("checkpoint/fixture") == b"official seed 4"


def test_v4_notebook_embeds_exact_source_parent_and_correct_protocol(returned_v3, tmp_path):
    from rl.prepare_kaggle_recovery_v4 import prepare
    path = prepare(returned_v3, tmp_path / "bundle")
    notebook = json.loads(path.read_text())
    assert path.name == "rlander_ppo_recovery_v4.ipynb"
    code = ""
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            text = "".join(cell["source"])
            ast.parse(text)
            code += text
            assert cell["outputs"] == [] and cell["execution_count"] is None
    assert "rl.kaggle_recovery_v4" in code and "rl.kaggle_recovery_v3" not in code
    assert "/kaggle/working/europa_ppo_recovery_v4" in code
    assert "validation (400 seeds)" in code and "v3_diagnosis.json" in code
    tree = ast.parse("".join(notebook["cells"][2]["source"]))
    payloads = {n.targets[0].id: base64.b64decode(n.value.args[0].value) for n in tree.body
                if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ("source_payload", "parent_payload")}
    with zipfile.ZipFile(io.BytesIO(payloads["source_payload"])) as z:
        for name, digest in json.loads(z.read("source_manifest.json"))["files"].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == digest
    with zipfile.ZipFile(io.BytesIO(payloads["parent_payload"])) as z:
        assert z.read("checkpoint/fixture") == b"official seed 4"
