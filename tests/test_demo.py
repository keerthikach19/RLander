"""Exercise real release data and the app's interactive routes."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "app/release_data"


def page(name):
    app = AppTest.from_file(str(ROOT / "app/Home.py"), default_timeout=30).run()
    assert not app.exception
    app.switch_page(f"pages/{name}.py").run()
    assert not app.exception
    return app


def test_demo_policy_and_episode_controls():
    app = page("1_Simulator")
    for policy in ("ppo", "parent", "pd"):
        app.selectbox[0].select(policy).run()
        assert not app.exception
        for index in range(len(app.selectbox[2].options)):
            app.selectbox[2].select_index(index).run()
            assert not app.exception
        assert len(app.get("plotly_chart")) == 2
    app.selectbox[1].select("timeout").run()
    assert not app.exception
    assert app.info  # PD has no timeout example; empty filter is handled.
    app.selectbox[0].select("ppo").run()
    assert not app.exception
    assert app.metric[0].value == "Timeout"


def test_comparison_and_history_routes():
    app = page("2_Compare")
    assert len(app.get("plotly_chart")) == 4
    app.selectbox[1].select("parent").run()
    app.selectbox[0].select_index(9).run()
    app.slider[0].set_value(100.).run()
    assert not app.exception
    app = page("3_Results")
    for version in app.selectbox[0].options:
        app.selectbox[0].select(version).run()
        assert not app.exception
    app.selectbox[0].select("v7").run()
    for arm in ("seed_14", "seed_15", "seed_16"):
        app.selectbox[1].select(arm).run()
        assert not app.exception


def test_release_data_matches_physics_and_metrics():
    summary = json.loads((DATA / "summary.json").read_text())
    episodes = pd.read_csv(DATA / "episodes.csv")
    assert summary["selection"]["frozen_before_test"]
    assert summary["selection"]["additional_steps"] == 80000
    for key, policy in summary["policies"].items():
        rows = episodes[episodes.policy == key]
        assert sorted(rows.seed) == list(range(90000, 92000))
        assert (rows.outcome == "landed").sum() == policy["n_success"]
        assert policy["success_rate"] == policy["n_success"] / len(rows)
        contact = rows.final_height_m <= 0
        safe = (rows.final_vy_mps.abs() < 1) & (rows.final_vx_mps.abs() < .5)
        assert ((contact & safe) == (rows.outcome == "landed")).all()
        assert ((~contact) == (rows.outcome == "timeout")).all()
        assert policy["touchdown"]["n_episodes"] == contact.sum()
        for replay in policy["replays"]:
            payload = (DATA / replay["file"]).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == replay["sha256"]
            frame = pd.read_parquet(DATA / replay["file"])
            assert np.isfinite(frame.to_numpy()).all()
            assert (frame.t.diff().iloc[1:] > 0).all()
            assert rows[rows.seed == replay["seed"]].outcome.item() == replay["outcome"]


def test_registry_selection_uses_validation_only():
    registry = json.loads((DATA / "checkpoint_registry.json").read_text())
    validation = pd.read_csv(DATA / "validation.csv")
    for entry in registry:
        rows = validation[(validation.version == entry["version"]) & (validation.arm == entry["arm"])]
        winner = rows.sort_values(["landed", "additional_steps"], ascending=[False, True]).iloc[0]
        assert entry["additional_steps"] == winner.additional_steps
        assert entry["validation_successes"] == winner.landed
