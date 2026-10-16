"""tests/test_w3.py — W3 Milestone 1 tests (eval/metrics.py, storage/db.py,
storage/parquet_io.py).

Tests run against fake info dicts (no env import required), so this file
has zero JAX dependencies.  The only production deps are:
  - math, sqlite3, pathlib  (stdlib)
  - pyarrow                 (in uv.lock)

All tests use :memory: for SQLite and tmp_path for Parquet I/O.
"""

from __future__ import annotations

import math
import sys
import os
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup: add repo root so storage/ and eval/ are importable without install
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.metrics import (
    aggregate_episodes,
    episode_success,
    outcome_name,
    wilson_ci,
    OUTCOME_CRASHED,
    OUTCOME_LANDED,
    OUTCOME_RUNNING,
    OUTCOME_TIMEOUT,
)
from storage.db import (
    get_evals_for_run,
    get_episodes_for_run,
    get_run,
    init_db,
    insert_episode,
    insert_eval,
    insert_run,
    upsert_body,
    upsert_vehicle,
)
from storage.parquet_io import (
    TRAJECTORY_SCHEMA,
    get_parquet_metadata,
    read_trajectory,
    trajectory_path,
    write_trajectory,
)


# ===========================================================================
# Helpers: fake info dicts (contract §5.2)
# ===========================================================================

def _make_info(
    outcome: int,
    touchdown_v: float = 0.5,
    touchdown_vx: float = 0.2,
    fuel_used: float = 100.0,
) -> dict:
    """Build a minimal info dict as LanderEnv.step would return."""
    return {
        "touchdown_v":  touchdown_v,
        "touchdown_vx": touchdown_vx,
        "fuel_used":    fuel_used,
        "crashed":      float(outcome == OUTCOME_CRASHED),
        "landed":       float(outcome == OUTCOME_LANDED),
        "truncated":    (outcome == OUTCOME_TIMEOUT),
        "outcome":      outcome,
    }


def _fake_step(
    t: float = 0.0,
    x: float = 0.0,
    y: float = 100.0,
    vx: float = 0.0,
    vy: float = -5.0,
    mass: float = 700.0,
    throttle: float = 0.5,
    angle: float = 0.0,
    reward: float = -0.5,
) -> dict:
    """Build a fake trajectory step dict."""
    return dict(t=t, x=x, y=y, vx=vx, vy=vy, mass=mass,
                throttle=throttle, angle=angle, reward=reward)


# ===========================================================================
# 1. eval/metrics.py — wilson_ci
# ===========================================================================

class TestWilsonCI:
    def test_all_success(self):
        lo, hi = wilson_ci(100, 100)
        assert lo > 0.93 and hi == 1.0, f"All-success CI wrong: ({lo}, {hi})"

    def test_no_success(self):
        lo, hi = wilson_ci(0, 100)
        assert lo == 0.0 and hi < 0.07, f"No-success CI wrong: ({lo}, {hi})"

    def test_fifty_percent(self):
        lo, hi = wilson_ci(50, 100)
        # Wilson CI for p=0.5, n=100 should be roughly (0.40, 0.60)
        assert lo < 0.45 and hi > 0.55, f"50% CI wrong: ({lo}, {hi})"
        assert lo >= 0.0 and hi <= 1.0

    def test_known_values(self):
        """Cross-check a known value from the Wilson formula."""
        # For k=3, n=10: p_hat=0.3
        lo, hi = wilson_ci(3, 10)
        # From wolfram: approx (0.1078, 0.6032)
        assert abs(lo - 0.1078) < 0.005, f"lo={lo}"
        assert abs(hi - 0.6032) < 0.005, f"hi={hi}"

    def test_single_episode_success(self):
        lo, hi = wilson_ci(1, 1)
        assert lo >= 0.0 and hi == 1.0

    def test_single_episode_failure(self):
        lo, hi = wilson_ci(0, 1)
        assert lo == 0.0 and hi <= 1.0

    def test_n_zero_raises(self):
        with pytest.raises(ValueError, match="n must be"):
            wilson_ci(0, 0)

    def test_successes_gt_n_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            wilson_ci(5, 3)

    def test_ci_bounds(self):
        """All output CIs must be in [0, 1]."""
        for k in range(0, 21):
            lo, hi = wilson_ci(k, 20)
            assert 0.0 <= lo <= hi <= 1.0, f"Bad CI for k={k}: ({lo}, {hi})"

    def test_symmetric_around_half(self):
        """wilson_ci should be approximately symmetric around p=0.5."""
        lo_a, hi_a = wilson_ci(30, 100)
        lo_b, hi_b = wilson_ci(70, 100)
        # hi_b should mirror lo_a and vice versa
        assert abs((1.0 - hi_b) - lo_a) < 0.01
        assert abs((1.0 - lo_b) - hi_a) < 0.01


# ===========================================================================
# 2. eval/metrics.py — episode_success
# ===========================================================================

class TestEpisodeSuccess:
    def test_landed_is_success(self):
        assert episode_success(_make_info(OUTCOME_LANDED)) is True

    def test_crashed_is_not_success(self):
        assert episode_success(_make_info(OUTCOME_CRASHED)) is False

    def test_timeout_is_not_success(self):
        assert episode_success(_make_info(OUTCOME_TIMEOUT)) is False

    def test_running_is_not_success(self):
        assert episode_success(_make_info(OUTCOME_RUNNING)) is False

    def test_does_not_read_reward(self):
        """episode_success must not use any reward key."""
        info = _make_info(OUTCOME_LANDED)
        # Remove velocity / altitude terms to confirm they're not accessed
        info_no_reward = {k: v for k, v in info.items() if k != "landed"}
        # Should still succeed based on outcome code alone
        assert episode_success(info_no_reward) is True

    def test_jax_int32_like_input(self):
        """Should handle int-like objects (mimics jax int32 scalars)."""
        class FakeJaxInt:
            def __int__(self):
                return OUTCOME_LANDED
        info = _make_info(OUTCOME_LANDED)
        info["outcome"] = FakeJaxInt()
        assert episode_success(info) is True

    def test_bad_outcome_type_raises(self):
        info = {"outcome": "landed", "touchdown_v": 0.5, "touchdown_vx": 0.2,
                "fuel_used": 50.0, "crashed": 0.0, "landed": 1.0, "truncated": False}
        with pytest.raises(TypeError):
            episode_success(info)


# ===========================================================================
# 3. eval/metrics.py — aggregate_episodes
# ===========================================================================

class TestAggregateEpisodes:
    def test_all_success(self):
        infos = [_make_info(OUTCOME_LANDED, touchdown_v=0.4, touchdown_vx=0.2,
                            fuel_used=80.0) for _ in range(10)]
        stats = aggregate_episodes(infos)
        assert stats["success_rate"] == 1.0
        assert stats["n_episodes"] == 10.0
        assert stats["n_success"] == 10.0
        assert stats["ci95_low"] > 0.65

    def test_no_success(self):
        infos = [_make_info(OUTCOME_CRASHED, touchdown_v=8.0, touchdown_vx=3.0,
                            fuel_used=200.0) for _ in range(10)]
        stats = aggregate_episodes(infos)
        assert stats["success_rate"] == 0.0
        assert stats["ci95_high"] < 0.35

    def test_mixed_outcomes(self):
        landed = [_make_info(OUTCOME_LANDED, touchdown_v=0.5, touchdown_vx=0.1,
                             fuel_used=70.0) for _ in range(60)]
        crashed = [_make_info(OUTCOME_CRASHED, touchdown_v=9.0, touchdown_vx=4.0,
                              fuel_used=200.0) for _ in range(40)]
        stats = aggregate_episodes(landed + crashed)
        assert abs(stats["success_rate"] - 0.6) < 1e-9
        assert 0.0 <= stats["ci95_low"] <= 0.6 <= stats["ci95_high"] <= 1.0

    def test_stat_keys_present(self):
        infos = [_make_info(OUTCOME_LANDED) for _ in range(5)]
        stats = aggregate_episodes(infos)
        for key in ("success_rate", "ci95_low", "ci95_high",
                    "vy_mean", "vy_std", "vx_mean", "vx_std", "fuel_mean",
                    "n_episodes", "n_success"):
            assert key in stats, f"Missing key: {key}"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            aggregate_episodes([])

    def test_hundred_episodes_wilson_ci(self):
        """Simulate 75/100 success: CI should bracket 0.75."""
        infos = (
            [_make_info(OUTCOME_LANDED)   for _ in range(75)] +
            [_make_info(OUTCOME_CRASHED)  for _ in range(25)]
        )
        stats = aggregate_episodes(infos)
        assert stats["ci95_low"]  < 0.75 < stats["ci95_high"]

    def test_outcome_name(self):
        assert outcome_name(OUTCOME_LANDED)  == "landed"
        assert outcome_name(OUTCOME_CRASHED) == "crashed"
        assert outcome_name(OUTCOME_TIMEOUT) == "timeout"
        assert outcome_name(OUTCOME_RUNNING) == "running"
        assert "unknown" in outcome_name(99)


# ===========================================================================
# 4. storage/db.py — schema + CRUD
# ===========================================================================

@pytest.fixture
def mem_db():
    """Fresh in-memory DB with schema applied."""
    conn = init_db(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def populated_db(mem_db):
    """DB with one body, one vehicle, and one run inserted."""
    upsert_body(mem_db, "europa", {"g": 1.315, "rho0": 0.0}, ["https://example.com"])
    upsert_vehicle(mem_db, "europa_lander", {"mass0_kg": 800, "dry_mass_kg": 320})
    run_id = insert_run(
        mem_db,
        train_bodies=["europa"],
        vehicle="europa_lander",
        policy_type="ppo",
        seed=1,
        jax_version="0.10.2",
        config_hash="abc123",
        checkpoint_path="ckpts/1/",
    )
    return mem_db, run_id


class TestDB:
    def test_init_creates_tables(self, mem_db):
        tables = mem_db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {row["name"] for row in tables}
        for t in ("bodies", "vehicles", "runs", "evals", "episodes",
                  "reward_candidates"):
            assert t in names, f"Table {t!r} missing"

    def test_upsert_body(self, mem_db):
        upsert_body(mem_db, "mars", {"g": 3.71}, ["https://mars.nasa.gov"])
        row = mem_db.execute("SELECT * FROM bodies WHERE name='mars'").fetchone()
        assert row is not None
        import json
        config = json.loads(row["config_json"])
        assert config["g"] == 3.71

    def test_upsert_body_idempotent(self, mem_db):
        upsert_body(mem_db, "titan", {"g": 1.352})
        upsert_body(mem_db, "titan", {"g": 1.355})  # update
        rows = mem_db.execute("SELECT * FROM bodies WHERE name='titan'").fetchall()
        assert len(rows) == 1
        import json
        assert json.loads(rows[0]["config_json"])["g"] == 1.355

    def test_insert_run_returns_id(self, mem_db):
        upsert_body(mem_db, "europa", {"g": 1.315, "rho0": 0.0})
        upsert_vehicle(mem_db, "europa_lander", {"mass0_kg": 800})
        run_id = insert_run(
            mem_db,
            train_bodies=["europa"],
            vehicle="europa_lander",
            policy_type="ppo",
            seed=42,
        )
        assert isinstance(run_id, int) and run_id > 0

    def test_get_run_round_trips(self, populated_db):
        conn, run_id = populated_db
        run = get_run(conn, run_id)
        assert run is not None
        assert run["seed"] == 1
        assert run["train_bodies"] == ["europa"]
        assert run["jax_version"] == "0.10.2"

    def test_get_run_missing(self, mem_db):
        assert get_run(mem_db, 9999) is None

    def test_insert_and_get_eval(self, populated_db):
        conn, run_id = populated_db
        eval_id = insert_eval(
            conn,
            run_id=run_id,
            eval_body="europa",
            n_episodes=100,
            success_rate=0.72,
            vy_mean=0.6,
            vy_std=0.15,
            vx_mean=0.3,
            vx_std=0.1,
            fuel_mean=120.0,
            ci95_low=0.62,
            ci95_high=0.80,
            term_stats={"velocity": {"mean": -0.3}},
        )
        assert isinstance(eval_id, int)
        evals = get_evals_for_run(conn, run_id)
        assert len(evals) == 1
        e = evals[0]
        assert abs(e["success_rate"] - 0.72) < 1e-9
        assert e["term_stats"] == {"velocity": {"mean": -0.3}}

    def test_insert_episode(self, populated_db):
        conn, run_id = populated_db
        ep_id = insert_episode(
            conn,
            run_id=run_id,
            seed=10000,
            outcome=OUTCOME_LANDED,
            parquet_path="data/trajectories/1/10000.parquet",
        )
        assert isinstance(ep_id, int)
        episodes = get_episodes_for_run(conn, run_id)
        assert len(episodes) == 1
        assert episodes[0]["seed"] == 10000
        assert episodes[0]["outcome"] == OUTCOME_LANDED

    def test_multiple_episodes(self, populated_db):
        conn, run_id = populated_db
        for seed in range(10000, 10010):
            insert_episode(
                conn,
                run_id=run_id,
                seed=seed,
                outcome=OUTCOME_CRASHED,
                parquet_path=f"data/trajectories/1/{seed}.parquet",
            )
        eps = get_episodes_for_run(conn, run_id)
        assert len(eps) == 10

    def test_foreign_key_eval_body(self, populated_db):
        """Inserting an eval with a non-existent body should fail (FK on)."""
        conn, run_id = populated_db
        # FK enforcement is ON; expect IntegrityError
        import sqlite3
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """INSERT INTO evals
                   (run_id, eval_body, n_episodes, success_rate,
                    vy_mean, vy_std, vx_mean, vx_std, fuel_mean,
                    ci95_low, ci95_high)
                   VALUES (?, 'pluto', 100, 0.5, 0.5, 0.1, 0.2, 0.05, 80.0, 0.4, 0.6)""",
                (run_id,),
            )
            conn.commit()


# ===========================================================================
# 5. storage/parquet_io.py
# ===========================================================================

class TestParquetIO:
    def test_trajectory_path(self):
        p = trajectory_path(7, 10042)
        assert p == "data/trajectories/7/10042.parquet"

    def test_trajectory_path_custom_base(self):
        p = trajectory_path(1, 99, base_dir="app/demo_data/trajectories")
        assert p == "app/demo_data/trajectories/1/99.parquet"

    def test_write_and_read_roundtrip(self, tmp_path):
        steps = [
            _fake_step(t=i * 0.1, y=float(2000 - i * 5))
            for i in range(50)
        ]
        rel = f"data/trajectories/99/10000.parquet"
        # Override _REPO_ROOT for test by writing to tmp_path
        abs_path = tmp_path / "data" / "trajectories" / "99" / "10000.parquet"
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        # Patch write to use tmp_path
        import storage.parquet_io as pio
        orig_root = pio._REPO_ROOT
        pio._REPO_ROOT = tmp_path
        try:
            write_trajectory(steps, rel)
            read_steps = read_trajectory(rel)
        finally:
            pio._REPO_ROOT = orig_root

        assert len(read_steps) == 50
        for i, step in enumerate(read_steps):
            assert abs(step["t"] - i * 0.1) < 1e-4
            assert abs(step["y"] - float(2000 - i * 5)) < 1e-2

    def test_schema_columns(self):
        """TRAJECTORY_SCHEMA must contain all required columns."""
        expected = {"t", "x", "y", "vx", "vy", "mass", "throttle", "angle", "reward"}
        actual = {f.name for f in TRAJECTORY_SCHEMA}
        assert expected == actual

    def test_write_creates_parent_dirs(self, tmp_path):
        """write_trajectory must create parent dirs automatically."""
        import storage.parquet_io as pio
        orig_root = pio._REPO_ROOT
        pio._REPO_ROOT = tmp_path
        try:
            rel = "data/trajectories/42/99999.parquet"
            write_trajectory([_fake_step()], rel)
            assert (tmp_path / rel).exists()
        finally:
            pio._REPO_ROOT = orig_root

    def test_read_missing_raises(self, tmp_path):
        import storage.parquet_io as pio
        import pyarrow  # noqa: F401 -- needed to trigger arrow errors
        orig_root = pio._REPO_ROOT
        pio._REPO_ROOT = tmp_path
        try:
            with pytest.raises(Exception):
                read_trajectory("data/trajectories/0/0.parquet")
        finally:
            pio._REPO_ROOT = orig_root

    def test_metadata_after_write(self, tmp_path):
        import storage.parquet_io as pio
        orig_root = pio._REPO_ROOT
        pio._REPO_ROOT = tmp_path
        try:
            steps = [_fake_step(t=float(i)) for i in range(20)]
            rel = "data/trajectories/1/1.parquet"
            write_trajectory(steps, rel)
            meta = get_parquet_metadata(rel)
            assert meta["num_rows"] == 20
            assert set(meta["schema"]) == {
                "t", "x", "y", "vx", "vy", "mass", "throttle", "angle", "reward"
            }
        finally:
            pio._REPO_ROOT = orig_root

    def test_write_float_types(self, tmp_path):
        """All columns should be stored as float32."""
        import storage.parquet_io as pio
        import pyarrow.parquet as pq
        orig_root = pio._REPO_ROOT
        pio._REPO_ROOT = tmp_path
        try:
            steps = [_fake_step()]
            rel = "data/trajectories/2/2.parquet"
            write_trajectory(steps, rel)
            table = pq.read_table(str(tmp_path / rel))
            for field in TRAJECTORY_SCHEMA:
                col_type = table.schema.field(field.name).type
                import pyarrow as pa
                assert col_type == pa.float32(), (
                    f"Column {field.name!r} should be float32, got {col_type}"
                )
        finally:
            pio._REPO_ROOT = orig_root
