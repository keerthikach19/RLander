"""storage/parquet_io.py — Parquet trajectory I/O for RLander (W3, §8).

Schema per PLAN.md §8:
    Trajectory file: data/trajectories/{run_id}/{seed}.parquet
    Columns: t, x, y, vx, vy, mass, throttle, angle, reward
    (reward = sum of reward_components, i.e. the scalar W2 computes)

Design:
- Uses pyarrow directly (no pandas dependency in the write path, though
  pandas is used for read convenience).
- Paths are always returned/expected as strings relative to the repo root,
  matching the `parquet_path` field in the `episodes` table.
- The caller (run_eval.py) is responsible for building the trajectory list
  and calling write_trajectory; this module only handles I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

TRAJECTORY_SCHEMA = pa.schema([
    pa.field("t",        pa.float32()),
    pa.field("x",        pa.float32()),
    pa.field("y",        pa.float32()),
    pa.field("vx",       pa.float32()),
    pa.field("vy",       pa.float32()),
    pa.field("mass",     pa.float32()),
    pa.field("throttle", pa.float32()),
    pa.field("angle",    pa.float32()),   # thrust angle in radians
    pa.field("reward",   pa.float32()),   # scalar sum of reward_components
])

_COLUMNS = [f.name for f in TRAJECTORY_SCHEMA]

_REPO_ROOT = Path(__file__).parent.parent
_DEFAULT_TRAJ_DIR = "data/trajectories"  # relative to repo root


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def trajectory_path(run_id: int, seed: int, base_dir: str | None = None) -> str:
    """Return the canonical relative path for a trajectory parquet file.

    Args:
        run_id:   The integer run id (from the `runs` table).
        seed:     The episode seed.
        base_dir: Base directory, relative to repo root.  Defaults to
                  ``data/trajectories``.

    Returns:
        Relative path string: ``{base_dir}/{run_id}/{seed}.parquet``
    """
    b = base_dir or _DEFAULT_TRAJ_DIR
    return f"{b}/{run_id}/{seed}.parquet"


def _abs_path(rel_path: str) -> Path:
    return _REPO_ROOT / rel_path


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_trajectory(
    steps: Sequence[dict[str, float]],
    rel_path: str,
) -> None:
    """Write a list of step dicts to a Parquet file.

    Args:
        steps:    Sequence of dicts, each with keys matching TRAJECTORY_SCHEMA
                  (t, x, y, vx, vy, mass, throttle, angle, reward).
                  Extra keys are silently ignored; missing keys raise KeyError.
        rel_path: Relative path (from repo root) where the file is written.
                  Parent directories are created automatically.
    """
    abs_path = _abs_path(rel_path)
    abs_path.parent.mkdir(parents=True, exist_ok=True)

    columns: dict[str, list[float]] = {col: [] for col in _COLUMNS}
    for step in steps:
        for col in _COLUMNS:
            columns[col].append(float(step[col]))

    arrays = [pa.array(columns[col], type=TRAJECTORY_SCHEMA.field(col).type)
              for col in _COLUMNS]
    table = pa.Table.from_arrays(arrays, schema=TRAJECTORY_SCHEMA)
    pq.write_table(table, str(abs_path), compression="snappy")


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def read_trajectory(rel_path: str) -> list[dict[str, float]]:
    """Read a Parquet trajectory file and return a list of step dicts.

    Args:
        rel_path: Relative path (from repo root).

    Returns:
        List of dicts with the trajectory columns as float values.
    """
    abs_path = _abs_path(rel_path)
    table = pq.read_table(str(abs_path), schema=TRAJECTORY_SCHEMA)
    result: list[dict[str, float]] = []
    col_arrays = {col: table.column(col).to_pylist() for col in _COLUMNS}
    n = len(table)
    for i in range(n):
        result.append({col: float(col_arrays[col][i]) for col in _COLUMNS})
    return result


def read_trajectory_pandas(rel_path: str):  # type: ignore[return]
    """Read a trajectory as a pandas DataFrame (requires pandas).

    Args:
        rel_path: Relative path (from repo root).

    Returns:
        pd.DataFrame with TRAJECTORY_SCHEMA columns.
    """
    try:
        import pandas as pd  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "pandas is required for read_trajectory_pandas"
        ) from exc
    abs_path = _abs_path(rel_path)
    return pd.read_parquet(str(abs_path))


# ---------------------------------------------------------------------------
# Metadata helpers (thin wrappers for common Parquet metadata patterns)
# ---------------------------------------------------------------------------

def get_parquet_metadata(rel_path: str) -> dict[str, Any]:
    """Return basic metadata for a Parquet file (num_rows, schema, etc.)."""
    abs_path = _abs_path(rel_path)
    meta = pq.read_metadata(str(abs_path))
    return {
        "num_rows":       meta.num_rows,
        "num_row_groups": meta.num_row_groups,
        "serialized_size": meta.serialized_size,
        "schema":         [f.name for f in TRAJECTORY_SCHEMA],
    }
