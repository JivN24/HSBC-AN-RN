"""Step 5.3/5.4 tests: Granger causality functions against a known
synthetic causal relationship (y[t] driven by x[t-2], x exogenous)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evc.causality.granger import (
    conditional_granger,
    linear_granger,
    nonlinear_granger_oos,
    toda_yamamoto,
)


@pytest.fixture(scope="module")
def lagged_causal_series():
    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.5 * y[t - 1] + 0.8 * x[t - 2] + 0.1 * rng.standard_normal()
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(y, index=idx, name="y"), pd.Series(x, index=idx, name="x")


def test_linear_granger_detects_true_direction_at_true_lag(lagged_causal_series) -> None:
    y, x = lagged_causal_series
    forward = linear_granger(y, x, max_lag=3)
    assert forward.loc[2, "pvalue"] < 0.01  # true lag is 2
    assert forward.loc[1, "pvalue"] > 0.05  # lag 1 has no true relationship


def test_linear_granger_reverse_direction_is_null_placebo(lagged_causal_series) -> None:
    """x is exogenous noise -- the reverse-direction placebo (y -> x)
    should NOT be significant. This is the CLAUDE.md constraint 1
    check: a real look-ahead bug would make this direction spuriously
    significant."""
    y, x = lagged_causal_series
    reverse = linear_granger(x, y, max_lag=3)
    assert (reverse["pvalue"] > 0.05).all()


def test_toda_yamamoto_detects_true_direction(lagged_causal_series) -> None:
    y, x = lagged_causal_series
    result = toda_yamamoto(y, x, max_lag=3)
    assert result.loc[0, "pvalue"] < 0.01


def test_conditional_granger_survives_irrelevant_control(lagged_causal_series) -> None:
    y, x = lagged_causal_series
    rng = np.random.default_rng(1)
    controls = pd.DataFrame({"z": rng.standard_normal(len(y))}, index=y.index)
    result = conditional_granger(y, x, controls, max_lag=3)
    assert result.loc[0, "pvalue"] < 0.01


def test_nonlinear_granger_oos_detects_gain(lagged_causal_series) -> None:
    y, x = lagged_causal_series
    controls = pd.DataFrame(index=y.index)
    controls["z"] = np.random.default_rng(2).standard_normal(len(y))
    result = nonlinear_granger_oos(y, x, controls, max_lag=3, n_folds=4, n_permutations=20, rng=np.random.default_rng(1))
    assert result.loc[0, "gain"] > 0.3
    assert result.loc[0, "pvalue"] < 0.1
