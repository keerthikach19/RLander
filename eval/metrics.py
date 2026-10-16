"""eval/metrics.py — Episode success metric and Wilson 95% CI (W3, §5.6, §9).

Contract rules (§5.6):
- The success metric reads ONLY from `info` (ground-truth dict returned by
  LanderEnv.step); it NEVER reads the reward or reward components.
- success = (outcome == 1)  i.e. landed_soft per §5.2 outcome codes.
- Thresholds vy_max_mps / vx_max_mps live in bodies.yaml under `success:`;
  they are baked into the env state and reflected in info["outcome"]=1 when
  the landing is soft.  We trust the env's outcome code directly.

Wilson score interval (§8):
    CI = Wilson 95% on a Bernoulli proportion.
    Formula: (p̂ + z²/(2n) ± z*sqrt(p̂(1-p̂)/n + z²/(4n²))) / (1 + z²/n)
    where z = 1.96 (two-sided 95%).
"""

from __future__ import annotations

import math
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Outcome code constants (mirrors §5.2 contract)
# ---------------------------------------------------------------------------

OUTCOME_RUNNING = 0
OUTCOME_LANDED  = 1   # soft landing within thresholds
OUTCOME_CRASHED = 2
OUTCOME_TIMEOUT = 3

_OUTCOME_NAMES = {
    OUTCOME_RUNNING: "running",
    OUTCOME_LANDED:  "landed",
    OUTCOME_CRASHED: "crashed",
    OUTCOME_TIMEOUT: "timeout",
}


def outcome_name(code: int) -> str:
    """Human-readable name for an outcome code."""
    return _OUTCOME_NAMES.get(code, f"unknown({code})")


# ---------------------------------------------------------------------------
# Wilson 95% confidence interval
# ---------------------------------------------------------------------------

_Z95 = 1.959963985  # z_{0.975}


def wilson_ci(successes: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score 95% CI for a Bernoulli proportion.

    Args:
        successes: Number of successful episodes.
        n:         Total episodes (denominator).
        z:         z-score for the desired confidence level (default 1.96 ≈ 95%).

    Returns:
        (ci_low, ci_high) both in [0, 1].

    Raises:
        ValueError: if n <= 0.

    Reference: Wilson (1927), "Probable Inference, the Law of Succession,
    and Statistical Inference", JASA 22:105, p.209-212.
    """
    if n <= 0:
        raise ValueError(f"wilson_ci: n must be > 0, got {n}")
    if successes < 0 or successes > n:
        raise ValueError(
            f"wilson_ci: successes={successes} must be in [0, n={n}]"
        )

    p_hat = successes / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt(
        p_hat * (1.0 - p_hat) / n + z2 / (4 * n * n)
    )
    lo = 0.0 if successes == 0 else max(0.0, centre - margin)
    hi = 1.0 if successes == n else min(1.0, centre + margin)
    return lo, hi


# ---------------------------------------------------------------------------
# Per-episode success check
# ---------------------------------------------------------------------------

def episode_success(info: dict[str, Any]) -> bool:
    """Return True if the episode ended in a soft landing.

    Reads only info["outcome"] (int32 from the env contract §5.2).
    Does NOT read reward components.

    Args:
        info: dict returned by LanderEnv.step (last step of the episode).
    """
    outcome = info["outcome"]
    # Support both JAX arrays and plain Python ints
    try:
        outcome_int = int(outcome)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"info['outcome'] must be int-like, got {type(outcome)}"
        ) from exc
    return outcome_int == OUTCOME_LANDED


# ---------------------------------------------------------------------------
# Aggregate eval stats over a batch of episodes
# ---------------------------------------------------------------------------

def aggregate_episodes(
    infos: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Compute aggregate eval statistics over a list of episode-final infos.

    Each element of `infos` is the `info` dict from the *last* step of one
    episode (outcome != 0).

    Returns a dict with:
        success_rate, ci95_low, ci95_high,
        vy_mean, vy_std, vx_mean, vx_std, fuel_mean,
        n_episodes, n_success.
    """
    n = len(infos)
    if n == 0:
        raise ValueError("aggregate_episodes: infos must be non-empty")

    successes = 0
    vy_vals: list[float] = []
    vx_vals: list[float] = []
    fuel_vals: list[float] = []

    for info in infos:
        if episode_success(info):
            successes += 1
        # touchdown_v is |vy| at termination (always present, §5.2)
        vy_vals.append(float(info["touchdown_v"]))
        vx_vals.append(float(info["touchdown_vx"]))
        fuel_vals.append(float(info["fuel_used"]))

    success_rate = successes / n
    ci_lo, ci_hi = wilson_ci(successes, n)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    def _std(xs: list[float]) -> float:
        m = _mean(xs)
        variance = sum((x - m) ** 2 for x in xs) / len(xs)
        return math.sqrt(variance)

    return {
        "n_episodes":   float(n),
        "n_success":    float(successes),
        "success_rate": success_rate,
        "ci95_low":     ci_lo,
        "ci95_high":    ci_hi,
        "vy_mean":      _mean(vy_vals),
        "vy_std":       _std(vy_vals),
        "vx_mean":      _mean(vx_vals),
        "vx_std":       _std(vx_vals),
        "fuel_mean":    _mean(fuel_vals),
    }
