"""storage/db.py — SQLite database helpers for RLander (W3, §8).

Design rules:
- Uses SQLAlchemy Core (not ORM) with a plain sqlite:// URL.
- Schema is authoritative in schema.sql; this module only opens the DB,
  applies the DDL (idempotent IF NOT EXISTS), and provides thin insert/
  query helpers that the rest of W3 (run_eval.py, parquet_io.py) call.
- All public functions accept and return plain Python dicts / lists so
  that the rest of the codebase never imports SQLAlchemy directly.
- Path default follows PLAN.md §8: data/rlander.db (gitignored).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_DEFAULT_DB_PATH = _REPO_ROOT / "data" / "rlander.db"


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Return a sqlite3 connection with WAL mode and FK enforcement.

    Args:
        db_path: Path to the SQLite file.  Defaults to data/rlander.db.
                 Pass ":memory:" for an in-memory DB (tests).
    """
    path = str(db_path) if db_path is not None else str(_DEFAULT_DB_PATH)
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Open the DB and apply schema.sql (idempotent CREATE IF NOT EXISTS).

    Returns the open connection; caller is responsible for closing it.
    """
    conn = _connect(db_path)
    ddl = _SCHEMA_PATH.read_text(encoding="utf-8")
    # Split on semicolons so we can execute each statement individually.
    # sqlite3 module does not support executescript within a transaction.
    conn.executescript(ddl)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# bodies / vehicles
# ---------------------------------------------------------------------------

def upsert_body(
    conn: sqlite3.Connection,
    name: str,
    config: dict[str, Any],
    source_urls: list[str] | None = None,
) -> None:
    """Insert or replace a body config row."""
    conn.execute(
        """
        INSERT INTO bodies (name, config_json, source_urls_json)
        VALUES (?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            config_json      = excluded.config_json,
            source_urls_json = excluded.source_urls_json
        """,
        (name, json.dumps(config), json.dumps(source_urls or [])),
    )
    conn.commit()


def upsert_vehicle(
    conn: sqlite3.Connection,
    name: str,
    config: dict[str, Any],
) -> None:
    """Insert or replace a vehicle config row."""
    conn.execute(
        """
        INSERT INTO vehicles (name, config_json)
        VALUES (?, ?)
        ON CONFLICT(name) DO UPDATE SET config_json = excluded.config_json
        """,
        (name, json.dumps(config)),
    )
    conn.commit()


def get_body(conn: sqlite3.Connection, name: str) -> dict[str, Any] | None:
    """Return the body config dict, or None if not found."""
    row = conn.execute("SELECT * FROM bodies WHERE name = ?", (name,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["config"] = json.loads(d.pop("config_json"))
    d["source_urls"] = json.loads(d.pop("source_urls_json"))
    return d


def get_all_bodies(conn: sqlite3.Connection) -> list[str]:
    """Return all body names registered in the DB."""
    rows = conn.execute("SELECT name FROM bodies ORDER BY name").fetchall()
    return [row["name"] for row in rows]


def get_vehicle(conn: sqlite3.Connection, name: str) -> dict[str, Any] | None:
    """Return the vehicle config dict, or None if not found."""
    row = conn.execute("SELECT * FROM vehicles WHERE name = ?", (name,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["config"] = json.loads(d.pop("config_json"))
    return d


def get_all_vehicles(conn: sqlite3.Connection) -> list[str]:
    """Return all vehicle names registered in the DB."""
    rows = conn.execute("SELECT name FROM vehicles ORDER BY name").fetchall()
    return [row["name"] for row in rows]


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------

def insert_run(
    conn: sqlite3.Connection,
    *,
    train_bodies: list[str],
    vehicle: str,
    policy_type: str,
    seed: int,
    config_hash: str | None = None,
    jax_version: str | None = None,
    checkpoint_path: str | None = None,
    reward_id: int | None = None,
) -> int:
    """Insert a new run row and return its auto-generated id."""
    cur = conn.execute(
        """
        INSERT INTO runs
            (train_bodies_json, vehicle, policy_type, reward_id,
             seed, config_hash, jax_version, checkpoint_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            json.dumps(train_bodies),
            vehicle,
            policy_type,
            reward_id,
            seed,
            config_hash,
            jax_version,
            checkpoint_path,
        ),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def get_run(conn: sqlite3.Connection, run_id: int) -> dict[str, Any] | None:
    """Return the run row as a dict, or None if not found."""
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["train_bodies"] = json.loads(d.pop("train_bodies_json"))
    return d


def get_runs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all run rows as a list of dicts."""
    rows = conn.execute("SELECT * FROM runs ORDER BY id ASC").fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["train_bodies"] = json.loads(d.pop("train_bodies_json"))
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# evals
# ---------------------------------------------------------------------------

def insert_eval(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    eval_body: str,
    n_episodes: int,
    success_rate: float,
    vy_mean: float,
    vy_std: float,
    vx_mean: float,
    vx_std: float,
    fuel_mean: float,
    ci95_low: float,
    ci95_high: float,
    term_stats: dict[str, Any] | None = None,
) -> int:
    """Insert an eval result row and return its id."""
    cur = conn.execute(
        """
        INSERT INTO evals
            (run_id, eval_body, n_episodes, success_rate,
             vy_mean, vy_std, vx_mean, vx_std, fuel_mean,
             ci95_low, ci95_high, term_stats_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            eval_body,
            n_episodes,
            success_rate,
            vy_mean,
            vy_std,
            vx_mean,
            vx_std,
            fuel_mean,
            ci95_low,
            ci95_high,
            json.dumps(term_stats or {}),
        ),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def get_evals_for_run(
    conn: sqlite3.Connection, run_id: int
) -> list[dict[str, Any]]:
    """Return all eval rows for a given run_id."""
    rows = conn.execute(
        "SELECT * FROM evals WHERE run_id = ?", (run_id,)
    ).fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["term_stats"] = json.loads(d.pop("term_stats_json"))
        result.append(d)
    return result


def get_all_evals(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Return all eval rows."""
    rows = conn.execute("SELECT * FROM evals ORDER BY id ASC").fetchall()
    result = []
    for row in rows:
        d = dict(row)
        d["term_stats"] = json.loads(d.pop("term_stats_json"))
        result.append(d)
    return result


def get_best_run_for_body(
    conn: sqlite3.Connection, body_name: str
) -> dict[str, Any] | None:
    """Return the eval and run dict with highest success_rate on body_name."""
    row = conn.execute(
        """
        SELECT e.*, r.vehicle, r.policy_type, r.seed as run_seed,
               r.train_bodies_json, r.checkpoint_path, r.config_hash, r.jax_version
        FROM evals e
        JOIN runs r ON e.run_id = r.id
        WHERE e.eval_body = ?
        ORDER BY e.success_rate DESC, e.vy_mean ASC, e.fuel_mean DESC, e.id DESC
        LIMIT 1
        """,
        (body_name,),
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["term_stats"] = json.loads(d.pop("term_stats_json"))
    d["train_bodies"] = json.loads(d.pop("train_bodies_json"))
    return d


# ---------------------------------------------------------------------------
# episodes
# ---------------------------------------------------------------------------

def insert_episode(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    seed: int,
    outcome: int,
    parquet_path: str,
) -> int:
    """Insert an episode row and return its id."""
    cur = conn.execute(
        """
        INSERT INTO episodes (run_id, seed, outcome, parquet_path)
        VALUES (?, ?, ?, ?)
        """,
        (run_id, seed, outcome, parquet_path),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def get_episodes_for_run(
    conn: sqlite3.Connection, run_id: int
) -> list[dict[str, Any]]:
    """Return all episode rows for a run."""
    rows = conn.execute(
        "SELECT * FROM episodes WHERE run_id = ?", (run_id,)
    ).fetchall()
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# reward_candidates (Phase B stubs — present so schema FK is never broken)
# ---------------------------------------------------------------------------

def insert_reward_candidate(
    conn: sqlite3.Connection,
    *,
    round_: int,
    provider: str,
    model: str,
    prompt: str,
    code: str,
    validation_status: str = "pending",
    score: float | None = None,
    term_stats: dict[str, Any] | None = None,
) -> int:
    """Insert a reward candidate row and return its id."""
    cur = conn.execute(
        """
        INSERT INTO reward_candidates
            (round, provider, model, prompt, code,
             validation_status, score, term_stats_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            round_,
            provider,
            model,
            prompt,
            code,
            validation_status,
            score,
            json.dumps(term_stats or {}),
        ),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]
