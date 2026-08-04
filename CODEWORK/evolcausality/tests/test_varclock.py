"""Step 5.2 tests: the variance-clock regression against evc.synthetic's
known ground truth."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evc.causality.varclock import build_regression_panel, naive_same_day_estimator, variance_clock_regression
from evc.config import TENOR_CALENDAR_DAYS, SyntheticConfig
from evc.synthetic import simulate_paths, synthesize_iv_surface


@pytest.fixture(scope="module")
def synthetic_fixture():
    cfg = SyntheticConfig(n_steps=6000)
    rng = np.random.default_rng(cfg.seed)
    paths = simulate_paths(cfg, rng)
    iv = synthesize_iv_surface(paths, cfg, TENOR_CALENDAR_DAYS)
    events = pd.concat(
        [
            pd.DataFrame({"event_type": et, "snapshot_date": paths.index[paths[f"is_{et}"]]})
            for et in cfg.event_omega
        ],
        ignore_index=True,
    )
    return cfg, paths, iv, events


def test_variance_clock_recovers_correct_ranking(synthetic_fixture) -> None:
    cfg, paths, iv, events = synthetic_fixture
    panel = build_regression_panel(iv, events, TENOR_CALENDAR_DAYS)
    result = variance_clock_regression(panel, n_bootstrap=100, rng=np.random.default_rng(1))

    true_ranking = sorted(cfg.event_omega, key=cfg.event_omega.get, reverse=True)
    estimated_ranking = result["omega_hat"].sort_values(ascending=False).index.tolist()
    # exact rank order isn't guaranteed (documented additive-vs-
    # multiplicative bias in the module docstring), but the top and
    # bottom should be right: FOMC highest, JP_CPI lowest.
    assert estimated_ranking[0] == true_ranking[0] == "FOMC"
    assert estimated_ranking[-1] == true_ranking[-1] == "JP_CPI"


def test_variance_clock_omega_all_above_one_and_cis_exclude_one(synthetic_fixture) -> None:
    """Every configured event type has omega_true > 1 -- the estimator
    should clearly reject omega=1 for all of them (bootstrap CI entirely
    above 1), i.e. detect that events matter at all, even given the
    documented upward bias in the point estimate itself."""
    cfg, paths, iv, events = synthetic_fixture
    panel = build_regression_panel(iv, events, TENOR_CALENDAR_DAYS)
    result = variance_clock_regression(panel, n_bootstrap=100, rng=np.random.default_rng(1))
    assert (result["ci_lo"] > 1.0).all()


def test_naive_estimator_is_biased_low_relative_to_variance_clock(synthetic_fixture) -> None:
    """CLAUDE.md's documented expectation: the naive same-day estimator
    understates omega relative to the (already upward-biased, see
    varclock module docstring) variance-clock estimate, since a single
    day's realised move misses the anticipation priced in beforehand."""
    cfg, paths, iv, events = synthetic_fixture
    logvar_1m = 2 * np.log(iv["iv_1m"])
    dlogvar_1m = logvar_1m.diff()
    naive = naive_same_day_estimator(iv, events, dlogvar_1m)

    panel = build_regression_panel(iv, events, TENOR_CALENDAR_DAYS)
    vc = variance_clock_regression(panel, n_bootstrap=50, rng=np.random.default_rng(1))

    comparison = naive.join(vc[["omega_hat"]])
    assert (comparison["omega_naive"] < comparison["omega_hat"]).all()
