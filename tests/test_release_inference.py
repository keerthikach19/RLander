"""Optional integration check using the actual locally preserved release model."""

from pathlib import Path

import numpy as np
import pytest

from app.live import load_controller, simulate

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not (ROOT / "outputs/final_release/selected/sha256.json").exists(),
    reason="Release checkpoints are distributed separately from the lightweight app",
)


def test_frozen_checkpoint_and_local_rollouts():
    policy, _, _ = load_controller("ppo")
    params, stats = policy.rollout_weights
    assert float(stats.count) > 0
    assert np.isfinite(np.asarray(stats.mean)).all()
    assert np.isfinite(np.asarray(stats.var)).all()
    assert params
    observed = {}
    for controller, seed in (("ppo", 90000), ("ppo", 90040), ("ppo", 90359), ("pd", 90000)):
        frame, result = simulate(controller, seed)
        assert np.isfinite(frame.to_numpy()).all()
        assert (frame.t.diff().iloc[1:] > 0).all()
        assert frame.throttle.between(0, 1).all()
        final = frame.iloc[-1]
        physical = ("landed" if abs(final.vy) < 1 and abs(final.vx) < .5 else "crashed") if final.y <= 0 else "timeout"
        assert result["outcome"] == physical
        observed[controller, seed] = result["outcome"]
        assert result["dry_mass_kg"] == 320
        if controller == "ppo":
            assert result["model_run_id"] == "ppo-europa-15-20261020T131659-d56d5360-at-80000"
            assert len(result["checkpoint_manifest_sha256"]) == 64
    assert observed["ppo", 90040] == "crashed"
    assert observed["ppo", 90359] == "timeout"
    first, _ = simulate("ppo", 90000)
    second, _ = simulate("ppo", 90000)
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())


def test_frozen_ppo_responds_to_entered_starting_conditions():
    conditions = {"x": 137., "y": 1900., "vx": -1.25, "vy": -17.5}
    first, evidence = simulate("ppo", 42, initial_conditions=conditions)
    other, other_evidence = simulate("ppo", 42, initial_conditions={**conditions, "vy": -23.})
    for name, value in conditions.items():
        assert first.iloc[0][name] == value
        assert evidence["initial_state"][name] == value
    assert evidence["requested_initial_conditions"] == conditions
    assert evidence["start_mode"] == "custom"
    assert evidence["model_run_id"] == other_evidence["model_run_id"]
    assert evidence["checkpoint_manifest_sha256"] == other_evidence["checkpoint_manifest_sha256"]
    # Same seed and weights, changed physical input: the network recomputes commands.
    assert not np.allclose(first.iloc[1][["throttle", "angle"]].to_numpy(),
                           other.iloc[1][["throttle", "angle"]].to_numpy())
    for frame, result in ((first, evidence), (other, other_evidence)):
        assert np.isfinite(frame.to_numpy()).all()
        final = frame.iloc[-1]
        physical = ("landed" if abs(final.vy) < 1 and abs(final.vx) < .5 else "crashed") if final.y <= 0 else "timeout"
        assert result["outcome"] == physical
