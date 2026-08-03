"""Step 1 tests: the synthetic simulator must be trustworthy before any
causality estimator is checked against it in Step 6.

Anticipation/crush is tested in ISOLATION (all event weights but one
fixed at 1.0) rather than on the full five-event-type production
calendar, because with five overlapping event types firing every ~5
business days on average, any single type's marginal effect on a
21-business-day rolling window is confounded by the others -- exactly
the kind of confound evc.causality.eventstudy and varclock exist to
control for in Step 5. Step 1 only needs to prove the *mechanism*
works; disentangling overlapping event types on the real calendar is
Step 5's job.
"""

from __future__ import annotations

import numpy as np
import pytest

from evc.config import SyntheticConfig, TENOR_CALENDAR_DAYS
from evc.synthetic import simulate_paths, synthesize_iv_surface


def _isolated_fomc_config(**overrides) -> SyntheticConfig:
    return SyntheticConfig(
        event_omega={"FOMC": 2.10, "NFP": 1.0, "US_CPI": 1.0, "BOJ": 1.0, "JP_CPI": 1.0},
        **overrides,
    )


def test_reproducible_with_seed() -> None:
    cfg = SyntheticConfig()
    p1 = simulate_paths(cfg, np.random.default_rng(cfg.seed))
    p2 = simulate_paths(cfg, np.random.default_rng(cfg.seed))
    pd_testing = pytest.importorskip("pandas.testing")
    pd_testing.assert_frame_equal(p1, p2)


def test_variance_stays_positive_and_finite() -> None:
    cfg = SyntheticConfig()
    paths = simulate_paths(cfg, np.random.default_rng(cfg.seed))
    assert (paths["v"] > 0).all()
    assert np.isfinite(paths["S"]).all()
    assert np.isfinite(paths["sigma"]).all()


def test_overnight_vol_spikes_on_event_days() -> None:
    """Constraint 3: overnight vol on an event day should sit well above
    its unconditional mean -- it's the cleanest instrument for omega."""
    cfg = _isolated_fomc_config()
    rng = np.random.default_rng(cfg.seed)
    paths = simulate_paths(cfg, rng)
    iv = synthesize_iv_surface(paths, cfg, TENOR_CALENDAR_DAYS)

    event_mask = paths["is_FOMC"].to_numpy()
    on_event = iv["iv_on"][event_mask].mean()
    on_overall = iv["iv_on"].mean()
    assert on_event > on_overall * 1.15


def test_1m_vol_shows_anticipation_and_crush() -> None:
    """The rolling-window anticipation signature: iv_1m should be
    elevated throughout the D_T-day window before a known future event
    (a step-up when the event enters the constant-maturity window, not
    necessarily a smooth ramp -- see module docstring), spike further
    right at realisation, then decay back down afterwards."""
    cfg = _isolated_fomc_config()
    rng = np.random.default_rng(cfg.seed)
    paths = simulate_paths(cfg, rng)
    iv = synthesize_iv_surface(paths, cfg, TENOR_CALENDAR_DAYS)

    event_idx = np.where(paths["is_FOMC"].to_numpy())[0]
    event_idx = event_idx[(event_idx > 40) & (event_idx < len(paths) - 40)]
    assert len(event_idx) > 10

    def window_mean(lo: int, hi: int) -> float:
        return float(np.mean([iv["iv_1m"].iloc[i + lo : i + hi + 1].mean() for i in event_idx]))

    far_before = window_mean(-35, -25)  # event not yet in the 21-day window
    just_before = window_mean(-15, -1)  # event inside the window: anticipation
    at_event = window_mean(0, 0)
    long_after = window_mean(25, 35)  # event long gone from window: crush complete

    assert just_before > far_before, "anticipation: no rise once event enters the window"
    assert at_event > just_before, "no additional spike at realisation"
    assert at_event > long_after, "no crush after the event"
