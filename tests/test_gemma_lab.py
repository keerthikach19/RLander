"""Mission planning must drive bounded, genuine simulations and grounded reports."""

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app import gemma_lab as lab
from app import live

ROOT = Path(__file__).resolve().parents[1]


def plan():
    return copy.deepcopy(lab.PLAN_SCHEMA)


def test_accepts_a_json_fence_but_not_prose_or_duplicate_keys():
    assert lab.validate_plan('```json\n' + json.dumps(plan()) + '\n```') == plan()
    with pytest.raises(ValueError):
        lab.model_json('My answer is: ' + json.dumps(plan()))
    with pytest.raises(ValueError):
        lab.model_json('{"status":"ready","status":"unsupported"}')


@pytest.mark.parametrize("change", [
    lambda p: p.update(parameter="wind"),
    lambda p: p["scenarios"][0].update(value=True),
    lambda p: p["scenarios"][0].update(value=10000),
    lambda p: p["scenarios"][0].update(value=float("nan")),
    lambda p: p["scenarios"][0].update(value=0),
    lambda p: p["scenarios"][0].update(value=6),
    lambda p: p.update(scenarios=p["scenarios"][:2]),
    lambda p: p.update(code="print('not executable')"),
])
def test_rejects_unbounded_or_uncontrolled_plans(change):
    value = plan()
    change(value)
    with pytest.raises(ValueError):
        lab.validate_plan(json.dumps(value))


def test_plan_only_varies_one_parameter_and_keeps_reference():
    cases = lab.scenarios_for(plan())
    assert cases[0]["initial"] == lab.REFERENCE
    assert not cases[0]["outside_training_starts"]
    for case in cases[1:]:
        assert case["initial"]["y"] == 2000 and case["initial"]["vy"] == -20
        assert case["outside_training_starts"]
    with pytest.raises(ValueError, match="unsupported"):
        lab.scenarios_for({"status": "unsupported", "reason": "No wind model"})


def fake_flight(controller, seed, *, initial_conditions):
    landed = initial_conditions["vx"] < 6
    vy = -.4 if landed else -1.2
    frame = pd.DataFrame({"t": [0., 1.], "x": [0., 1.], "y": [2000., 0.],
                          "vx": [initial_conditions["vx"], .1], "vy": [-20., vy],
                          "mass": [800., 799.], "throttle": [.3, .4], "angle": [0., .01]})
    return frame, {"controller": controller, "seed": seed,
                   "outcome": "landed" if landed else "crashed", "backend": "cpu",
                   "device": "cpu:0", "initial_state": frame.iloc[0].to_dict(),
                   "checkpoint_manifest_sha256": "fixture" if controller == "ppo" else None}


def test_executes_all_cases_on_matched_seeds_and_scores_actual_flights(monkeypatch):
    calls = []

    def simulate(controller, seed, *, initial_conditions):
        calls.append((controller, seed, initial_conditions.copy()))
        return fake_flight(controller, seed, initial_conditions=initial_conditions)

    monkeypatch.setattr(live, "simulate", simulate)
    report = lab.run_plan(plan(), 17)
    assert len(calls) == 24
    assert {s for _, s, _ in calls} == {17, 18, 19}
    assert all(r["tested"] == 3 for r in report["summary"])
    assert report["summary"][0]["landed"] == 3
    assert report["summary"][-1]["crashed"] == 3
    assert report["flights"][-1]["touchdown_abs_vy_mps"] == 1.2
    assert report["flights"][-1]["fuel_used_kg"] == 1
    altered = copy.deepcopy(report)
    altered["flights"].pop()
    with pytest.raises(ValueError, match="Incomplete"):
        lab.summarize(altered)


def test_missing_runtime_never_becomes_a_fake_or_approximate_flight(monkeypatch):
    def fail(*args, **kwargs):
        raise FileNotFoundError("Missing checkpoint")
    monkeypatch.setattr(live, "simulate", fail)
    with pytest.raises(FileNotFoundError):
        lab.run_plan(plan(), 42)


def interpretation():
    return {"headline": "Some probes failed", "findings": [
        {"case_id": "probe_3", "controller": "ppo", "observation": "none_landed",
         "hypothesis": "The higher initial drift may be difficult to correct."},
    ], "next_question": "Try intermediate drift values."}


def test_interpretation_cannot_invent_results_or_controllers(monkeypatch):
    monkeypatch.setattr(live, "simulate", fake_flight)
    report = lab.run_plan(plan(), 17, compare_pd=False)
    item = interpretation()
    lab.validate_interpretation(item, report["summary"])
    item["findings"][0]["observation"] = "all_landed"
    with pytest.raises(ValueError, match="contradicts"):
        lab.validate_interpretation(item, report["summary"])
    item["findings"][0]["controller"] = "pd"
    with pytest.raises(ValueError, match="not run"):
        lab.validate_interpretation(item, report["summary"])


def test_live_page_runs_design_simulation_interpretation_and_export(monkeypatch):
    monkeypatch.setattr(lab, "api_key", lambda: "fixture-not-a-real-key")
    monkeypatch.setattr(lab, "request_plan", lambda question, key:
                        (plan(), {"returned_model": "fixture-gemma", "prompt": question}))
    monkeypatch.setattr(live, "simulate", fake_flight)
    monkeypatch.setattr(lab, "interpret", lambda report, key: (interpretation(), {}))
    app = AppTest.from_file(str(ROOT / "app/Home.py"), default_timeout=30).run()
    app.switch_page("pages/5_Gemma.py").run()
    assert not app.exception
    next(b for b in app.button if b.label == "Design and run mission tests").click().run()
    assert not app.exception
    assert app.session_state["gemma_mission_report"]["origin"] == "live"
    assert len(app.session_state["gemma_mission_report"]["flights"]) == 24
    assert app.get("download_button") and app.get("plotly_chart")


def test_rejected_interpretation_keeps_measured_results(monkeypatch):
    monkeypatch.setattr(lab, "api_key", lambda: "fixture-not-a-real-key")
    monkeypatch.setattr(lab, "request_plan", lambda question, key:
                        (plan(), {"returned_model": "fixture-gemma"}))
    monkeypatch.setattr(live, "simulate", fake_flight)
    def fail(*args):
        raise ValueError("Contradicts measured data")
    monkeypatch.setattr(lab, "interpret", fail)
    app = AppTest.from_file(str(ROOT / "app/Home.py"), default_timeout=30).run()
    app.switch_page("pages/5_Gemma.py").run()
    next(b for b in app.button if b.label == "Design and run mission tests").click().run()
    assert not app.exception
    assert app.warning and app.get("download_button")
    assert app.session_state["gemma_mission_report"]["interpretation"] is None
