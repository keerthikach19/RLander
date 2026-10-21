"""CPU correctness tests; these do not assert that CUDA has been tested."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from eval.batched import collect, make_rollout
from eval.run_eval import (
    load_body_config,
    load_vehicle_config,
    make_policy_fn,
    run_evaluation,
)
from rl.execution import execute, require_gpu
from storage.db import init_db, insert_run, upsert_body, upsert_vehicle
from storage.parquet_io import read_trajectory
from tests.test_refinement import parent as parent  # noqa: PLC0414 -- expose pytest fixture


def test_required_gpu_rejects_cpu_before_workload(monkeypatch):
    if jax.default_backend() != "cpu":
        pytest.skip("CPU rejection test")
    monkeypatch.setenv("RLANDER_REQUIRE_GPU", "1")
    with pytest.raises(RuntimeError, match="GPU execution required"):
        require_gpu()
    with pytest.raises(RuntimeError, match="GPU execution required"):
        execute(jax.jit(lambda x: x + 1), (jnp.ones(3),), label="must_not_run")


def test_execution_reports_actual_compiled_device_and_completed_output(tmp_path, monkeypatch):
    monkeypatch.delenv("RLANDER_REQUIRE_GPU", raising=False)
    path = tmp_path / "execution.jsonl"
    monkeypatch.setenv("RLANDER_DEVICE_LOG", str(path))
    output, evidence = execute(jax.jit(lambda x: x @ x), (jnp.eye(4),), label="test")
    np.testing.assert_array_equal(output, np.eye(4))
    assert evidence["gpu_verified"] == (jax.default_backend() == "gpu")
    assert evidence["compiled_devices"] == [str(jax.devices()[0])]
    assert evidence["execution_seconds"] > 0
    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [e["phase"] for e in events] == [
        "compiling_or_loading_cached_executable", "executing", "completed"]


@pytest.mark.parametrize("policy_type", ["pd_fallback", "random", "ppo", "hover"])
def test_batched_w3_matches_serial_rng_terminal_metrics_and_trajectories(tmp_path, parent, policy_type):
    checkpoint, _, _ = parent
    body, vehicle = load_body_config("europa"), load_vehicle_config("europa_lander")
    # Arrest vertical speed and maintain altitude: exercises timeout handling.
    def hover(obs, key):
        vy = obs[2] * np.sqrt(body["g"] * vehicle["h0_m"])
        mass = obs[5] * vehicle["mass0_kg"]
        thrust = vehicle["twr_max"] * vehicle["mass0_kg"] * body["g"]
        return jnp.array([jnp.clip(2 * mass * (body["g"] - .5 * vy) / thrust - 1, -1, 1), 0.])
    kwargs = {"custom_policy": hover} if policy_type == "hover" else {}
    if policy_type == "ppo":
        kwargs["checkpoint_path"] = str(checkpoint)
    conn = init_db(tmp_path / "eval.sqlite")
    try:
        upsert_body(conn, "europa", body)
        upsert_vehicle(conn, "europa_lander", vehicle)
        results = []
        for batched in (False, True):
            run = insert_run(conn, train_bodies=["europa"], vehicle="europa_lander",
                             policy_type=policy_type if policy_type != "hover" else "random",
                             seed=1, jax_version=jax.__version__)
            results.append(run_evaluation(
                run, conn=conn, seeds=(10000, 10001), batched=batched,
                traj_base_dir=str(tmp_path / "trajectories"), quiet=True, **kwargs,
            ))
        serial, batch = results
        for key in ("n_success", "success_rate", "n_episodes"):
            assert serial[key] == batch[key]
        for key in ("vy_mean", "vx_mean", "fuel_mean"):
            np.testing.assert_allclose(serial[key], batch[key], rtol=2e-4, atol=2e-3)
        for old, new in zip(serial["episodes"], batch["episodes"]):
            assert old["seed"] == new["seed"] and old["outcome"] == new["outcome"]
            before, after = read_trajectory(old["parquet_path"]), read_trajectory(new["parquet_path"])
            assert len(before) == len(after)  # no extra steps after first termination
            for column in before[0]:
                np.testing.assert_allclose(
                    [r[column] for r in before], [r[column] for r in after],
                    rtol=3e-4, atol=3e-3, err_msg=column,
                )
        if policy_type == "hover":
            assert all(e["outcome"] == 3 for e in batch["episodes"])
            assert batch["touchdown"]["n_episodes"] == 0
        assert batch["rollout_mode"] == "batched_compiled"
        assert batch["execution"]["output_array_count"] > 0
    finally:
        conn.close()


def test_batched_guard_and_reuse_across_checkpoint_weights():
    policy = make_policy_fn("random")
    body, vehicle = load_body_config("europa"), load_vehicle_config("europa_lander")
    with pytest.raises(RuntimeError, match="exceeded max_steps"):
        collect(policy, (10000, 10001), body, vehicle, max_steps=1)
    assert make_rollout(policy.rollout_policy, 1024) is make_rollout(policy.rollout_policy, 1024)
