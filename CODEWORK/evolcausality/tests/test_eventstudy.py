"""Step 5.1 test: event study on the isolated-FOMC synthetic series
(known ground truth from Step 1) -- checks the spike-and-crush at/after
realisation is detected against both nulls, and documents (rather than
hides) that pre-event anticipation is not reliably detected by this
module's simple baseline (see module docstring)."""

from __future__ import annotations

import numpy as np

from evc.causality.eventstudy import event_study
from evc.config import TENOR_CALENDAR_DAYS, SyntheticConfig
from evc.synthetic import simulate_paths, synthesize_iv_surface


def test_event_study_detects_spike_and_crush_against_both_nulls() -> None:
    cfg = SyntheticConfig(event_omega={"FOMC": 2.10, "NFP": 1.0, "US_CPI": 1.0, "BOJ": 1.0, "JP_CPI": 1.0})
    rng = np.random.default_rng(cfg.seed)
    paths = simulate_paths(cfg, rng)
    iv = synthesize_iv_surface(paths, cfg, TENOR_CALENDAR_DAYS)
    logvar_1m = 2 * np.log(iv["iv_1m"])
    event_dates = paths.index[paths["is_FOMC"]]

    res = event_study(logvar_1m, event_dates, window=10, n_surrogates=200, rng=np.random.default_rng(1))

    assert res.loc[0, "response"] > res.loc[0, "bootstrap_hi95"]
    assert res.loc[0, "response"] > res.loc[0, "placebo_hi95"]
    # response should be decaying (crush) by lag +8..+10, back within
    # (or close to) the null band
    assert res.loc[10, "response"] < res.loc[0, "response"]
