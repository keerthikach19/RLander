"""Actual inference failures cannot silently become approximation flights."""

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from app import live, sim_canvas

ROOT = Path(__file__).resolve().parents[1]


def live_page():
    app = AppTest.from_file(str(ROOT / "app/Home.py"), default_timeout=30).run()
    return app.switch_page("pages/4_Local.py").run()


def run_button(app):
    return next(button for button in app.button if "Run Simulation" in button.label)


def test_actual_inference_error_stops_without_approximation(monkeypatch):
    def fail(*args):
        raise FileNotFoundError("test checkpoint missing")

    def forbidden(*args):
        raise AssertionError("Fallback must require opt-in")

    monkeypatch.setattr(live, "simulate", fail)
    monkeypatch.setattr(sim_canvas, "init_state", forbidden)
    app = live_page()
    assert not app.checkbox[0].value
    run_button(app).click().run()
    assert not app.exception
    assert any("test checkpoint missing" in error.value for error in app.error)
    assert "live_traj" not in app.session_state


def test_opted_in_fallback_retains_visible_identity(monkeypatch):
    def fail(*args):
        raise FileNotFoundError("test checkpoint missing")

    def short_flight(seed):
        return sim_canvas.SimState(
            done=True, outcome="landed", t=1, y=0, vy=-0.5, vx=0,
            ts=[0], xs=[0], ys=[1], vxs=[0], vys=[-0.5],
            masses=[801.13], throttles=[0], angles=[0],
        )

    monkeypatch.setattr(live, "simulate", fail)
    monkeypatch.setattr(sim_canvas, "init_state", short_flight)
    app = live_page()
    app.checkbox[0].check().run()
    run_button(app).click().run()
    assert not app.exception
    assert app.session_state["live_meta"]["controller"] == "physics-approx"
    assert any("Physics approximation" in warning.value for warning in app.warning)
    assert not any("inference completed" in success.value for success in app.success)


def test_result_only_appears_at_terminal_frame_and_no_end_button(monkeypatch):
    def crash(controller, seed):
        frame = pd.DataFrame({
            "t": [0, 1], "x": [0, 0], "y": [1, 0], "vx": [0, 0],
            "vy": [-1.029, -1.029], "mass": [800, 799],
            "throttle": [0, 0.3], "angle": [0, 0],
        })
        return frame, {"controller": controller, "seed": seed, "outcome": "crashed",
                       "backend": "cpu", "device": "cpu:0", "dry_mass_kg": 320}

    monkeypatch.setattr(live, "simulate", crash)
    app = live_page()
    app.number_input[0].set_value(90040).run()
    run_button(app).click().run()
    assert not app.exception
    assert any("Crash" in error.value for error in app.error)
    assert not any("End" in button.label for button in app.button)
    next(button for button in app.button if "Reset" in button.label).click().run()
    assert not app.exception
    assert not app.error and not app.success
    assert not app.download_button
    assert not app.json
    assert not any("cpu:0" in caption.value for caption in app.caption)
    app.session_state["live_fidx"] = 1
    app.run()
    assert any("Crash" in error.value for error in app.error)
