"""scripts/export_demo_data.py — Export curated demo snapshot for Streamlit dashboard (W3, §8).

PLAN.md §8:
    "scripts/export_demo_data.py writes a small curated subset (best run per body,
    ~10 episodes each) to app/demo_data/ (committed). The deployed app reads only
    that; data/ and ckpts/ stay gitignored"

Features:
- Identifies the highest-performing run per celestial body based on success_rate
  (with lowest touchdown velocity vy as tie-breaker).
- Selects ~10 curated episodes per body (prioritizing successful landings to showcase
  the policy, while keeping diverse outcomes).
- Copies Parquet trajectories into app/demo_data/trajectories/{run_id}/{seed}.parquet.
- Populates an isolated, standalone app/demo_data/rlander.db SQLite database.
- Writes an app/demo_data/manifest.json summary for lightweight frontend consumption.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Sequence

from storage.db import (
    get_best_run_for_body,
    get_body,
    get_run,
    get_vehicle,
    init_db,
    upsert_body,
    upsert_vehicle,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SOURCE_DB = _REPO_ROOT / "data" / "rlander.db"
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "app" / "demo_data"


def export_demo_data(
    source_db_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    num_episodes_per_body: int = 10,
    bodies: Sequence[str] | None = None,
    quiet: bool = False,
) -> dict[str, Any]:
    """Export the best run per body and ~10 episodes each to output_dir.

    Args:
        source_db_path:        Path to main SQLite DB (default data/rlander.db).
        output_dir:            Target folder (default app/demo_data).
        num_episodes_per_body: Max episodes per best run to export (default 10).
        bodies:                Optional filter of body names to export.
        quiet:                 If True, suppress stdout printing.

    Returns:
        Summary dict containing export metadata.
    """
    src_db = Path(source_db_path) if source_db_path else _DEFAULT_SOURCE_DB
    out_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR

    if not src_db.exists():
        if not quiet:
            print(f"Source database {src_db} does not exist. Nothing to export.")
        return {"status": "error", "message": f"Database not found: {src_db}"}

    src_conn = sqlite3.connect(str(src_db))
    src_conn.row_factory = sqlite3.Row

    out_dir.mkdir(parents=True, exist_ok=True)
    demo_db_path = out_dir / "rlander.db"
    demo_conn = init_db(demo_db_path)

    try:
        # Determine target bodies
        if bodies is not None and len(bodies) > 0:
            target_bodies = list(bodies)
        else:
            rows = src_conn.execute("SELECT DISTINCT eval_body FROM evals").fetchall()
            target_bodies = [r["eval_body"] for r in rows]

        if not target_bodies:
            if not quiet:
                print("No evaluations found in source database.")
            return {"status": "empty", "exported_bodies": {}}

        manifest: dict[str, Any] = {
            "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "source_db": str(src_db),
            "bodies": {},
        }

        for body_name in target_bodies:
            best_eval = get_best_run_for_body(src_conn, body_name)
            if best_eval is None:
                continue

            run_id = best_eval["run_id"]
            run_row = get_run(src_conn, run_id)
            if run_row is None:
                continue

            # Copy body configuration
            body_data = get_body(src_conn, body_name)
            if body_data:
                upsert_body(
                    demo_conn,
                    body_name,
                    body_data["config"],
                    source_urls=body_data["source_urls"],
                )

            # Copy vehicle configuration
            vehicle_name = run_row["vehicle"]
            vehicle_data = get_vehicle(src_conn, vehicle_name)
            if vehicle_data:
                upsert_vehicle(demo_conn, vehicle_name, vehicle_data["config"])

            # Copy run row (re-insert into demo DB with same ID)
            demo_conn.execute(
                """
                INSERT OR REPLACE INTO runs
                    (id, train_bodies_json, vehicle, policy_type, reward_id,
                     seed, config_hash, jax_version, checkpoint_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_row["id"],
                    json.dumps(run_row["train_bodies"]),
                    run_row["vehicle"],
                    run_row["policy_type"],
                    run_row["reward_id"],
                    run_row["seed"],
                    run_row["config_hash"],
                    run_row["jax_version"],
                    run_row["checkpoint_path"],
                    run_row["created_at"],
                ),
            )
            demo_conn.commit()

            # Copy eval row
            demo_conn.execute(
                """
                INSERT OR REPLACE INTO evals
                    (id, run_id, eval_body, n_episodes, success_rate,
                     vy_mean, vy_std, vx_mean, vx_std, fuel_mean,
                     ci95_low, ci95_high, term_stats_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    best_eval["id"],
                    best_eval["run_id"],
                    best_eval["eval_body"],
                    best_eval["n_episodes"],
                    best_eval["success_rate"],
                    best_eval["vy_mean"],
                    best_eval["vy_std"],
                    best_eval["vx_mean"],
                    best_eval["vx_std"],
                    best_eval["fuel_mean"],
                    best_eval["ci95_low"],
                    best_eval["ci95_high"],
                    json.dumps(best_eval["term_stats"]),
                ),
            )
            demo_conn.commit()

            # Select ~10 curated episodes (prefer successful landings first, then others)
            ep_rows = src_conn.execute(
                """
                SELECT * FROM episodes
                WHERE run_id = ?
                ORDER BY (CASE WHEN outcome = 1 THEN 0 ELSE 1 END), seed ASC
                LIMIT ?
                """,
                (run_id, num_episodes_per_body),
            ).fetchall()

            exported_episodes = []
            dest_traj_base = out_dir / "trajectories" / str(run_id)
            dest_traj_base.mkdir(parents=True, exist_ok=True)

            src_base = src_db.parent.parent
            for ep in ep_rows:
                seed = ep["seed"]
                outcome = ep["outcome"]
                src_parquet_rel = ep["parquet_path"]
                cand_abs = Path(src_parquet_rel)
                cand_src = src_base / src_parquet_rel
                cand_repo = _REPO_ROOT / src_parquet_rel

                if cand_abs.is_absolute() and cand_abs.exists():
                    src_parquet_path = cand_abs
                elif cand_src.exists():
                    src_parquet_path = cand_src
                elif cand_repo.exists():
                    src_parquet_path = cand_repo
                else:
                    src_parquet_path = cand_repo

                dest_file_rel = f"app/demo_data/trajectories/{run_id}/{seed}.parquet"
                dest_file_abs = dest_traj_base / f"{seed}.parquet"

                if src_parquet_path.exists():
                    shutil.copy2(src_parquet_path, dest_file_abs)

                demo_conn.execute(
                    """
                    INSERT OR REPLACE INTO episodes (id, run_id, seed, outcome, parquet_path)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (ep["id"], run_id, seed, outcome, dest_file_rel),
                )

                exported_episodes.append({
                    "seed": seed,
                    "outcome": outcome,
                    "parquet_path": dest_file_rel,
                })

            demo_conn.commit()

            manifest["bodies"][body_name] = {
                "run_id": run_id,
                "vehicle": vehicle_name,
                "policy_type": run_row["policy_type"],
                "success_rate": best_eval["success_rate"],
                "ci95_low": best_eval["ci95_low"],
                "ci95_high": best_eval["ci95_high"],
                "vy_mean": best_eval["vy_mean"],
                "vx_mean": best_eval["vx_mean"],
                "fuel_mean": best_eval["fuel_mean"],
                "num_episodes": len(exported_episodes),
                "episodes": exported_episodes,
            }

        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if not quiet:
            print("=" * 60)
            print(f"Exported Demo Data to {out_dir}")
            print(f"Demo SQLite DB:   {demo_db_path}")
            print(f"Demo Manifest:    {manifest_path}")
            for bname, meta in manifest["bodies"].items():
                print(
                    f"  Body: {bname:<10} Run: #{meta['run_id']} ({meta['policy_type']}) "
                    f"Success: {meta['success_rate']:.1%} [{meta['ci95_low']:.1%}, {meta['ci95_high']:.1%}] "
                    f"({meta['num_episodes']} episodes)"
                )
            print("=" * 60)

        return manifest

    finally:
        src_conn.close()
        demo_conn.close()


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export curated demo data (best run per body) to app/demo_data/."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(_DEFAULT_SOURCE_DB),
        help=f"Source database path (default: {_DEFAULT_SOURCE_DB})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(_DEFAULT_OUTPUT_DIR),
        help=f"Target output directory (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of curated episodes per body (default: 10)",
    )
    parser.add_argument(
        "--bodies",
        type=str,
        default=None,
        help="Comma-separated list of bodies to export (default: all with evaluations)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress human-readable summary output",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> None:
    parsed = parse_args(args)
    bodies_list = [b.strip() for b in parsed.bodies.split(",")] if parsed.bodies else None
    export_demo_data(
        source_db_path=parsed.db_path,
        output_dir=parsed.output_dir,
        num_episodes_per_body=parsed.num_episodes,
        bodies=bodies_list,
        quiet=parsed.quiet,
    )


if __name__ == "__main__":
    main()
