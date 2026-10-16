-- RLander SQLite schema (W3, PLAN.md §8)
-- All DDL changes go through the integrator (human).
-- Agents: never ALTER these tables; add a migration script instead.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- -------------------------------------------------------------------------
-- bodies: one row per celestial body; config is the full YAML section.
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS bodies (
    name             TEXT PRIMARY KEY,
    config_json      TEXT NOT NULL,          -- full body config dict as JSON
    source_urls_json TEXT NOT NULL DEFAULT '[]'  -- list of NASA/JPL source URLs
);

-- -------------------------------------------------------------------------
-- vehicles: one row per vehicle configuration.
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vehicles (
    name        TEXT PRIMARY KEY,
    config_json TEXT NOT NULL               -- full vehicle config dict as JSON
);

-- -------------------------------------------------------------------------
-- runs: one row per training run.
--   train_bodies_json: JSON list of body names used during training
--   policy_type: "ppo" | "pd_fallback" | "llm_reward"
--   reward_id: NULL for hand-shaped; FK to reward_candidates.id for Phase B
--   config_hash: sha256 of the Hydra config used
--   jax_version: recorded for reproducibility (PLAN.md rev-log #2)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    train_bodies_json TEXT NOT NULL DEFAULT '["europa"]',
    vehicle          TEXT NOT NULL REFERENCES vehicles(name),
    policy_type      TEXT NOT NULL DEFAULT 'ppo',
    reward_id        INTEGER REFERENCES reward_candidates(id),
    seed             INTEGER NOT NULL,
    config_hash      TEXT,
    jax_version      TEXT,
    checkpoint_path  TEXT,
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- -------------------------------------------------------------------------
-- evals: evaluation results per run × eval_body.
--   Wilson 95% CI on success_rate stored as ci95_low / ci95_high.
--   n_episodes = 100 (frozen seeds 10_000–10_099 per §9.W3).
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    eval_body    TEXT    NOT NULL REFERENCES bodies(name),
    n_episodes   INTEGER NOT NULL DEFAULT 100,
    success_rate REAL    NOT NULL,
    vy_mean      REAL    NOT NULL,
    vy_std       REAL    NOT NULL,
    vx_mean      REAL    NOT NULL,
    vx_std       REAL    NOT NULL,
    fuel_mean    REAL    NOT NULL,
    ci95_low     REAL    NOT NULL,
    ci95_high    REAL    NOT NULL,
    -- term_stats_json: per-reward-term mean/std over the episodes (Phase B / Reward Lab)
    term_stats_json TEXT NOT NULL DEFAULT '{}'
);

-- -------------------------------------------------------------------------
-- episodes: one row per evaluated episode; links to the Parquet trajectory.
--   outcome codes: 0 running, 1 landed, 2 crashed, 3 timeout (§5.2)
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS episodes (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    seed         INTEGER NOT NULL,
    outcome      INTEGER NOT NULL,           -- 0/1/2/3 per contract
    parquet_path TEXT    NOT NULL            -- relative to repo root
);

-- -------------------------------------------------------------------------
-- reward_candidates: Phase B LLM reward loop (§8, §11 Day 6-7).
-- -------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reward_candidates (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    round             INTEGER NOT NULL DEFAULT 1,
    provider          TEXT    NOT NULL,      -- "groq" | "openrouter"
    model             TEXT    NOT NULL,      -- exact model id logged
    prompt            TEXT    NOT NULL,
    code              TEXT    NOT NULL,
    validation_status TEXT    NOT NULL DEFAULT 'pending',  -- pending/passed/failed
    score             REAL,                  -- frozen-seed success rate after training
    term_stats_json   TEXT    NOT NULL DEFAULT '{}',
    created_at        TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- -------------------------------------------------------------------------
-- Indexes for common query patterns
-- -------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_evals_run_id   ON evals(run_id);
CREATE INDEX IF NOT EXISTS idx_episodes_run_id ON episodes(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_seed       ON runs(seed);
