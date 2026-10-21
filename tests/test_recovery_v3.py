"""Evidence selection, notebook packaging and bounded CPU integration for v3."""

import ast
import base64
import hashlib
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rl.analyze_refinement_return import analyze
from rl.kaggle_recovery_v2 import configuration as v2_config
from rl.kaggle_recovery_v2 import launch, train_arm
from rl.kaggle_recovery_v3 import (
    FRESH_SEEDS,
    SNAPSHOTS,
    STEPS,
    TRAINING_SEEDS,
    VALIDATION_SEEDS,
    bindings,
    configuration,
    evaluate,
    plan,
    select_candidate,
)
from tests.test_refinement import parent as parent  # noqa: PLC0414 -- pytest fixture


@pytest.fixture
def returned_zip(tmp_path):
    archive_path = tmp_path / "returned.zip"
    runs, fresh = [], []
    with zipfile.ZipFile(archive_path, "w") as archive:
        for index, (arm, soft, fresh_soft) in enumerate(
                [("stable_control", 56, 2), ("success_aligned", 82, 0)], start=1):
            prefix = f"{arm}/run/snapshots/160000"
            checkpoint = f"/kaggle/working/result/{prefix}/checkpoint"
            evaluation = {"seeds": list(range(2000, 2100)), "soft_landings": soft,
                          "soft_landing_rate": soft / 100, "n_episodes": 100}
            candidate = {"additional_steps": 160000, "checkpoint": checkpoint, "evaluation": evaluation}
            runs.append({"arm": arm, "validation": [candidate], "selected": candidate,
                         "max_action_reload_error": 0.})
            manifest = {"run": {"id": arm}, "actual_steps": 160000, "evaluation": evaluation,
                        "config": {"actor_initialization": {"parent_steps": 720000}}}
            archive.writestr(prefix + "/manifest.json", json.dumps(manifest))
            archive.writestr(prefix + "/checkpoint/fixture", b"parent " + arm.encode())
            episodes = [{"seed": seed, "outcome": 1 if fresh_soft else 2} for seed in (30000, 30001)]
            fresh.append({"label": arm + "_selected", "checkpoint": checkpoint, "evaluation": {
                "run_id": index, "n_episodes": 2, "n_success": fresh_soft,
                "success_rate": fresh_soft / 2, "ci95_low": 0., "ci95_high": 1.,
                "by_outcome": {}, "episodes": episodes}})
            for ep in episodes:
                table = pa.table({"y": [2000., 10., 0.], "vy": [-20., -3., -.5 if fresh_soft else -2.],
                                  "vx": [1., .2, .2]})
                buffer = io.BytesIO()
                pq.write_table(table, buffer)
                archive.writestr(f"trajectories/{index}/{ep['seed']}.parquet", buffer.getvalue())
        archive.writestr("results.json", json.dumps(
            {"status": "complete", "runs": runs, "fresh_evaluation": fresh}))
    return archive_path


def test_parent_selection_uses_validation_despite_reversed_fresh_scores(returned_zip):
    report, payload = analyze(returned_zip)
    assert report["selected_parent"]["arm"] == "success_aligned"
    assert report["selected_parent"]["ancestral_steps"] == 880000
    assert report["selected_fresh_failures"]["vertical_only"] == 2
    assert not report["gpu_workloads_verified"]  # no fabricated execution evidence
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        assert archive.read("checkpoint/fixture") == b"parent success_aligned"


def test_v3_config_and_seed_scopes():
    previous, current = v2_config(), configuration()
    assert current == replace(previous, total_steps=160000, learning_rate=1e-5)
    assert STEPS * len(TRAINING_SEEDS) == 480000
    assert SNAPSHOTS == tuple(range(20000, 160000, 20000))
    assert all(step % current.batch_size == 0 for step in SNAPSHOTS)
    old_sets = set(range(2000, 2100)) | set(range(10000, 10100)) | set(range(30000, 30200))
    assert len(VALIDATION_SEEDS) == 200 and len(FRESH_SEEDS) == 500
    assert set(VALIDATION_SEEDS).isdisjoint(old_sets)
    assert set(FRESH_SEEDS).isdisjoint(old_sets | set(VALIDATION_SEEDS))


def test_selection_preserves_parent_and_uses_predeclared_ties():
    def row(seed, step, soft):
        return {"arm": f"seed_{seed}", "training_seed": seed, "selected": {
            "additional_steps": step, "checkpoint": "parent" if not step else f"model_{seed}",
            "evaluation": {"soft_landings": soft}}}
    chosen = select_candidate([row(2, 20000, 150), row(3, 0, 150), row(4, 40000, 149)])
    assert chosen["arm"] == "initial_parent" and chosen["training_seed"] is None
    chosen = select_candidate([row(2, 20000, 155), row(3, 20000, 155), row(4, 40000, 155)])
    assert chosen["training_seed"] == 2


@pytest.mark.parametrize("allowed", [True, False])
def test_v3_launch_order_device_enforcement_and_preserved_parent(tmp_path, monkeypatch, allowed):
    import rl.kaggle_recovery_v2 as worker
    parent = tmp_path / "input" / "checkpoint"
    parent.mkdir(parents=True)
    (parent / "fixture").write_text("original")
    (parent.parent / "manifest.json").write_text("{}")
    output = tmp_path / "output"
    original = Path.is_dir
    monkeypatch.setattr(Path, "is_dir", lambda p: True if str(p).replace("\\", "/") == "/kaggle/working"
                        else original(p))
    monkeypatch.setattr(worker.subprocess, "check_output", lambda *a, **k: "mock")
    stages = []

    class Process:
        def __init__(self, command, **kwargs):
            assert command[command.index("-m") + 1] == "rl.kaggle_recovery_v3"
            env = kwargs["env"]
            assert env["RLANDER_REQUIRE_GPU"] == "1" and env["JAX_PLATFORMS"] == "cuda"
            assert env["CUDA_VISIBLE_DEVICES"] == "0" and env["JAX_DISABLE_JIT"] == "false"
            stage = command[command.index("--worker") + 1]
            stages.append(stage)
            if stage == "preflight":
                worker.write_json(output / "preflight.json", {"training_allowed": allowed})
            elif stage in TRAINING_SEEDS:
                worker.write_json(output / f"{stage}.json", {"arm": stage})
            else:
                report = json.loads((output / "results.json").read_text())
                assert len(report["runs"]) == 3
                worker.write_json(output / "fresh_evaluation.json", [])
                worker.write_json(output / "selection.json", {"frozen": True})

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(worker.subprocess, "Popen", Process)
    launch(parent, output, worker_module="rl.kaggle_recovery_v3", protocol=plan(), arms=TRAINING_SEEDS)
    assert stages == (["preflight", *TRAINING_SEEDS, "evaluate"] if allowed else ["preflight"])
    assert (output / "parent/checkpoint/fixture").read_text() == "original"


def test_fresh_evaluation_cannot_reselect_candidate_and_deduplicates(tmp_path, monkeypatch):
    from rl import kaggle_recovery
    parent = tmp_path / "parent/checkpoint"
    rows = []
    for seed, checkpoint, successes in ((2, "winner", 170), (3, "other", 160), (4, str(parent), 150)):
        rows.append({"arm": f"seed_{seed}", "training_seed": seed, "selected": {
            "checkpoint": checkpoint, "additional_steps": 0 if seed == 4 else 20000,
            "evaluation": {"soft_landings": successes}}})
    (tmp_path / "results.json").write_text(json.dumps({"runs": rows}))
    calls = []

    def fake_evaluation(output, *, checkpoint, seeds, batched):
        assert json.loads((output / "selection.json").read_text())["candidate"]["checkpoint"] == "winner"
        assert seeds == FRESH_SEEDS and batched
        calls.append(checkpoint)
        outcomes = [1, 2] if checkpoint == str(parent) else [2, 1] if checkpoint == "winner" else [1, 1]
        return {"success_rate": outcomes.count(1) / 2,
                "episodes": [{"seed": 40000 + i, "outcome": o} for i, o in enumerate(outcomes)]}
    monkeypatch.setattr(kaggle_recovery, "register_evaluation", fake_evaluation)
    evaluate(parent, tmp_path)
    assert calls == [str(parent), "winner", "other"]
    assert json.loads((tmp_path / "selection.json").read_text())["candidate"]["checkpoint"] == "winner"
    paired = json.loads((tmp_path / "paired_comparison.json").read_text())
    assert paired["newly_landed"] == paired["lost_landings"] == 1
    assert paired["candidate_minus_parent_success_rate"] == 0


def test_shared_training_helper_honors_v3_overrides_in_bounded_cpu_run(parent, tmp_path):
    checkpoint, _, _ = parent
    bind = bindings()
    evaluation = {"soft_landings": 1, "soft_landing_rate": .5}
    bind.evaluate = lambda params, stats: evaluation
    (tmp_path / "preflight.json").write_text(json.dumps({"selection_initial": evaluation}))
    cfg = replace(configuration(), num_envs=2, num_steps=4, total_steps=8,
                  num_minibatches=2, update_epochs=1)
    train_arm(checkpoint, tmp_path, "seed_2", config=cfg, bind=bind, seed=2,
              checkpoint_steps=(), initial_evaluation_key="selection_initial")
    row = json.loads((tmp_path / "seed_2.json").read_text())
    assert row["training_seed"] == 2 and row["final"]["additional_steps"] == 8
    assert row["selected"]["additional_steps"] == 0
    assert row["max_action_reload_error"] <= 1e-5
    manifest = json.loads((Path(row["final"]["checkpoint"]).parent / "manifest.json").read_text())
    assert manifest["config"]["seed"] == 2
    assert manifest["config"]["ppo"]["learning_rate"] == 1e-5
    assert manifest["run"]["reward_id"] == "hand_shaped_landing_success_v1"


def test_v3_notebook_is_unexecuted_and_embeds_verified_parent_and_source(returned_zip, tmp_path):
    from rl.prepare_kaggle_recovery_v3 import prepare
    path = prepare(returned_zip, tmp_path / "bundle")
    notebook = json.loads(path.read_text())
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    code = ast.parse("".join(notebook["cells"][2]["source"]))
    payloads = {n.targets[0].id: base64.b64decode(n.value.args[0].value)
                for n in code.body if isinstance(n, ast.Assign)
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in ("source_payload", "parent_payload")}
    with zipfile.ZipFile(io.BytesIO(payloads["source_payload"])) as archive:
        manifest = json.loads(archive.read("source_manifest.json"))
        assert {"rl/kaggle_recovery_v3.py", "rl/execution.py", "eval/batched.py"} <= set(archive.namelist())
        for name, digest in manifest["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
    with zipfile.ZipFile(io.BytesIO(payloads["parent_payload"])) as archive:
        assert archive.read("checkpoint/fixture") == b"parent success_aligned"
