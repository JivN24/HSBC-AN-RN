"""Step 5.5/5.6 tests: KSG mutual information vs the analytic Gaussian
MI benchmark (explicit engineering requirement), and transfer entropy
direction detection with both the continuous KSG and discrete/
symbolic estimators."""

from __future__ import annotations

import numpy as np
import pytest

from evc.causality.infotheory import (
    ksg_mutual_information,
    symbolic_transfer_entropy,
    tdmi_curve,
    transfer_entropy,
)


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
def test_ksg_mi_matches_analytic_gaussian_benchmark(rho: float) -> None:
    rng = np.random.default_rng(0)
    n = 3000
    cov = [[1, rho], [rho, 1]]
    samples = rng.multivariate_normal([0, 0], cov, size=n)
    x, y = samples[:, 0], samples[:, 1]
    mi_hat = ksg_mutual_information(x, y, k=4)
    mi_true = -0.5 * np.log(1 - rho**2)
    assert abs(mi_hat - mi_true) < 0.02


def test_tdmi_curve_peaks_at_true_lag() -> None:
    """y[t] depends on x[t-2] -- TDMI(x_t; y_{t+lag}) should peak at
    lag=2 (matching the linear_granger test's true-lag setup)."""
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.8 * x[t - 2] + 0.3 * rng.standard_normal()

    curve = tdmi_curve(x, y, max_lag=5, k=4)
    lags = np.arange(-5, 6)
    peak_lag = lags[np.nanargmax(curve)]
    assert peak_lag == 2


def test_transfer_entropy_detects_true_direction() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + 0.6 * x[t - 1] + 0.2 * rng.standard_normal()

    te_forward = transfer_entropy(x, y, history_len=1, k=4)
    te_reverse = transfer_entropy(y, x, history_len=1, k=4)
    assert te_forward > 10 * te_reverse


def test_symbolic_te_agrees_on_direction_with_continuous_ksg() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.standard_normal(n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.7 * y[t - 1] + 0.6 * x[t - 1] + 0.2 * rng.standard_normal()

    ste_forward = symbolic_transfer_entropy(x, y, history_len=1, n_bins=4)
    ste_reverse = symbolic_transfer_entropy(y, x, history_len=1, n_bins=4)
    assert ste_forward > 5 * ste_reverse
