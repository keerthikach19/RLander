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
    for controller, seed in (("ppo", 90000), ("ppo", 90359), ("pd", 90000)):
        frame, result = simulate(controller, seed)
        assert np.isfinite(frame.to_numpy()).all()
        assert (frame.t.diff().iloc[1:] > 0).all()
        assert frame.throttle.between(0, 1).all()
        final = frame.iloc[-1]
        physical = ("landed" if abs(final.vy) < 1 and abs(final.vx) < .5 else "crashed") if final.y <= 0 else "timeout"
        assert result["outcome"] == physical
    first, _ = simulate("ppo", 90000)
    second, _ = simulate("ppo", 90000)
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())
