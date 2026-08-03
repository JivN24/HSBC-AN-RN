"""Step 6 tests: the validation harness itself. Uses small n_trials/
n_bootstrap for test speed -- the notebook runs the full-scale version
(larger grids, more trials) for the actual power/size report."""

from __future__ import annotations

import numpy as np
import pytest

from evc.validate import power_curve, recovery_check, size_check


def test_recovery_check_computes_bias_and_rmse() -> None:
    rc = recovery_check("test_estimator", {"A": 1.2, "B": 0.9}, {"A": 1.0, "B": 1.0})
    assert rc.loc["A", "bias"] == pytest.approx(0.2)
    assert rc.loc["B", "bias"] == pytest.approx(-0.1)
    assert rc.attrs["pooled_rmse"] > 0


def test_power_increases_with_effect_size() -> None:
    """Power should be monotonically higher at a larger true omega,
    at a fixed sample size -- the qualitative check that doesn't
    require many trials to be informative (unlike an exact power
    number, direction is stable even with a handful of trials)."""
    rng = np.random.default_rng(0)
    pc = power_curve(n_grid=[1500], omega_grid=[1.0, 2.5], n_trials=5, rng=rng, n_bootstrap=50)
    power_null = pc.loc[pc["omega"] == 1.0, "power"].iloc[0]
    power_large_effect = pc.loc[pc["omega"] == 2.5, "power"].iloc[0]
    assert power_large_effect >= power_null


def test_size_check_returns_valid_rate() -> None:
    rate = size_check(n_trials=5, rng=np.random.default_rng(1), n_steps=1500, n_bootstrap=50)
    assert 0.0 <= rate <= 1.0
