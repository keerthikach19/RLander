"""Analytic tests for the environment physics (§W1 requirements).

Milestone 1: placeholder — real analytic tests come in Milestone 2.
Milestone 2 tests:
  (a) Free fall: y(t) = h0 - 0.5*g*t^2 within 1% over 10s
  (b) Terminal velocity with atmosphere
  (c) Fuel depletion: mdot = throttle * T_max / (isp * g0)
  (d) Energy check
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_placeholder():
    """Placeholder for Milestone 2 analytic tests."""
    pass
