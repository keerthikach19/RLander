"""Bounded PPO configuration and validation-first selection for v5."""

import ast
import base64
import hashlib
import io
import json
import math
import zipfile
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rl.kaggle_recovery_v5 import (
    FRESH_SEEDS,
    SNAPSHOTS,
    STEPS,
    TRAINING_SEEDS,
    VALIDATION_SEEDS,
    bindings,
    configuration,
    evaluate,
    select_candidate,
)


def test_budget_reward_and_new_seed_scopes():
    cfg = configuration()
    assert cfg.num_envs == 32 and cfg.num_steps == 125 and cfg.num_updates == 80
    assert STEPS * len(TRAINING_SEEDS) == 960000
    assert all(step % cfg.batch_size == 0 for step in SNAPSHOTS)
    assert cfg.learning_rate == 3e-6 and math.isclose(math.exp(cfg.initial_log_std), .03)
    assert cfg.log_std_min < cfg.initial_log_std < cfg.log_std_max
    assert cfg.freeze_obs_stats and cfg.ent_coef == 0 and cfg.clip_coef == .1
    used = set(range(2000, 2100)) | set(range(4000, 4200)) | set(range(6000, 6400))
    used |= set(range(10000, 10100)) | set(range(30000, 30200))
    used |= set(range(40000, 40500)) | set(range(60000, 61000))
    assert set(VALIDATION_SEEDS).isdisjoint(used) and len(VALIDATION_SEEDS) == 800
    assert set(FRESH_SEEDS).isdisjoint(used | set(VALIDATION_SEEDS)) and len(FRESH_SEEDS) == 2000
    bind = bindings()
    _, state = bind.env.reset(jax.random.key(0), bind.body_params, bind.vehicle_params)
    assert float(state.reward_mode) == 4.


def test_eight_transition_cpu_compile_with_new_exploration_bounds():
    from rl.ppo import make_train
    bind = bindings()
    cfg = replace(configuration(), num_envs=2, num_steps=4, total_steps=8,
                  update_epochs=1, num_minibatches=2, freeze_obs_stats=False)
    result = make_train(bind.env, cfg)(jax.random.key(8), bind.body_params, bind.vehicle_params)
    assert np.isfinite(np.asarray(result.metrics["loss"])).all()
    log_std = np.asarray(result.runner.train_state.params["params"]["log_std"])
    assert (log_std >= cfg.log_std_min - 1e-6).all()
    assert (log_std <= cfg.log_std_max + 1e-6).all()


def runs():
    rows = []
    for arm, seed in TRAINING_SEEDS.items():
        candidate = {"additional_steps": 40000, "checkpoint": arm,
                     "evaluation": {"soft_landings": 720}}
        rows.append({"arm": arm, "training_seed": seed, "selected": candidate,
                     "validation": [candidate]})
    return rows


def test_selection_is_order_independent_and_preserves_parent_ties():
    rows = runs()
    assert select_candidate(rows[::-1])["arm"] == "seed_8"
    rows[1]["selected"].update(additional_steps=0, checkpoint="parent")
    assert select_candidate(rows)["arm"] == "initial_parent"
    with pytest.raises(ValueError, match="Missing or duplicate"):
        select_candidate(rows[:-1])
    rows[0]["selected"] = {**rows[0]["selected"], "checkpoint": "unvalidated"}
    with pytest.raises(ValueError, match="disagrees with validation"):
        select_candidate(rows)


@pytest.mark.parametrize("keep_parent", [False, True])
def test_test_scores_cannot_reselect_and_pd_uses_same_scope(tmp_path, monkeypatch, keep_parent):
    import rl.kaggle_recovery_v5 as worker
    rows = runs()
    if keep_parent:
        for row in rows:
            row["selected"].update(checkpoint="parent", additional_steps=0)
    (tmp_path / "results.json").write_text(json.dumps({"runs": rows}))
    calls = []

    def fake(output, *, checkpoint, seeds, batched):
        frozen = json.loads((output / "selection.json").read_text())["candidate"]
        assert frozen["checkpoint"] == ("parent" if keep_parent else "seed_8")
        assert seeds == FRESH_SEEDS and batched
        calls.append(checkpoint)
        landed = 99 if checkpoint is None else 94 if checkpoint == "parent" else 91
        return {"success_rate": landed / 100, "episodes": [
            {"seed": 70000 + i, "outcome": 1 if i < landed else 2} for i in range(100)
        ]}

    monkeypatch.setattr(worker, "register_evaluation", fake)
    evaluate(Path("parent"), tmp_path)
    assert calls == (["parent", None] if keep_parent else ["parent", "seed_8", None])
    frozen = json.loads((tmp_path / "selection.json").read_text())
    assert frozen["candidate"]["checkpoint"] == ("parent" if keep_parent else "seed_8")
    target = json.loads((tmp_path / "target_check.json").read_text())
    assert target["target_met"]
    assert target["candidate_success_rate"] == (.94 if keep_parent else .91)


@pytest.fixture
def returned_v4(tmp_path):
    from rl.kaggle_recovery_v4 import RUNS, select_candidates
    path = tmp_path / "v4.zip"
    prefix = "margin_seed_5/run/snapshots/20000"
    checkpoint = "/kaggle/working/result/" + prefix + "/checkpoint"
    rows = []
    for arm, (_, seed) in RUNS.items():
        candidate = {"checkpoint": checkpoint if arm == "margin_seed_5" else arm,
                     "additional_steps": 20000,
                     "evaluation": {"seeds": list(range(6000, 6400)), "n_episodes": 400,
                                    "soft_landings": 339 if arm == "margin_seed_5" else 316}}
        rows.append({"arm": arm, "training_seed": seed, "validation": [candidate], "selected": candidate})
    selection = select_candidates(rows)
    episodes = [{"seed": seed, "outcome": 1 if seed < 60875 else 2}
                for seed in range(60000, 61000)]
    evaluation = {"n_success": 875, "n_episodes": 1000, "success_rate": .875,
                  "ci95_low": .85, "ci95_high": .9, "run_id": 2, "episodes": episodes}
    result = {"status": "complete", "runs": rows, "selection": selection,
              "fresh_evaluation": [{"checkpoint": checkpoint, "evaluation": evaluation}]}
    buf = io.BytesIO()
    pq.write_table(pa.table({"vy": [-1.2], "vx": [.1]}), buf)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("results.json", json.dumps(result))
        archive.writestr("parent_provenance.json", json.dumps({"selection": {"ancestral_steps": 900000}}))
        archive.writestr(prefix + "/manifest.json", json.dumps({
            "evaluation": selection["candidate"]["evaluation"], "run": {"id": "v4_selected"}}))
        archive.writestr(prefix + "/checkpoint/fixture", b"selected v4 checkpoint")
        for ep in episodes:
            if ep["outcome"] == 2:
                archive.writestr(f"trajectories/2/{ep['seed']}.parquet", buf.getvalue())
    return path


def test_notebook_embeds_selected_parent_and_all_runtime_dependencies(returned_v4, tmp_path):
    from rl.prepare_kaggle_recovery_v5 import prepare
    path = prepare(returned_v4, tmp_path / "bundle")
    notebook = json.loads(path.read_text())
    code = ""
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            text = "".join(cell["source"])
            ast.parse(text)
            code += text
            assert cell["outputs"] == [] and cell["execution_count"] is None
    assert "rl.kaggle_recovery_v5" in code and "rl.kaggle_recovery_v3" not in code
    assert "/kaggle/working/europa_ppo_recovery_v5" in code
    assert "target_check.json" in code and "validation (800 seeds)" in code
    tree = ast.parse("".join(notebook["cells"][2]["source"]))
    payloads = {node.targets[0].id: base64.b64decode(node.value.args[0].value)
                for node in tree.body if isinstance(node, ast.Assign)
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("source_payload", "parent_payload")}
    with zipfile.ZipFile(io.BytesIO(payloads["source_payload"])) as archive:
        for name, digest in json.loads(archive.read("source_manifest.json"))["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        for name in ("rl/kaggle_recovery_v5.py", "rl/kaggle_recovery_v4.py", "eval/batched.py",
                     "rl/execution.py"):
            assert name in archive.namelist()
    with zipfile.ZipFile(io.BytesIO(payloads["parent_payload"])) as archive:
        assert archive.read("checkpoint/fixture") == b"selected v4 checkpoint"
    diagnosis = json.loads((path.parent / "v4_diagnosis.json").read_text())
    assert diagnosis["selected_parent"]["ancestral_steps"] == 920000
    assert diagnosis["selected_fresh_failures"]["vertical_only"] == 125


def test_analyzer_rejects_changed_frozen_selection(returned_v4, tmp_path):
    from rl.analyze_v4_return import analyze
    target = tmp_path / "tampered.zip"
    with zipfile.ZipFile(returned_v4) as source, zipfile.ZipFile(target, "w") as out:
        for name in source.namelist():
            data = source.read(name)
            if name == "results.json":
                result = json.loads(data)
                result["selection"]["candidate"]["checkpoint"] = "test_cherry_pick"
                data = json.dumps(result).encode()
            out.writestr(name, data)
    with pytest.raises(ValueError, match="Frozen selection"):
        analyze(target)
