"""Step 4 tests: feature construction on the real data pipeline
end-to-end (data.py -> events.py -> features.py)."""

from __future__ import annotations

import pandas as pd

from evc.data import load_iv_panel
from evc.events import align_to_snapshot, load_all_events
from evc.features import build_event_features, build_nuisance_controls, build_vol_features


def test_vol_features_on_real_panel() -> None:
    panel = load_iv_panel("csv")
    feats = build_vol_features(panel)
    assert "logvar_1m" in feats
    assert "slope_on_1m" in feats
    assert "vov" in feats
    # log-variance should be finite wherever iv_1m itself is present
    valid = panel["iv_1m"].notna()
    assert feats.loc[valid, "logvar_1m"].notna().all()


def test_event_features_on_real_calendar() -> None:
    panel = load_iv_panel("csv")
    events = load_all_events()
    events = events[events["scheduled"]]
    aligned = align_to_snapshot(events, panel.index)
    feats = build_event_features(aligned, panel.index)

    assert "is_FOMC" in feats
    assert "d2n_FOMC" in feats
    assert "inwin21_FOMC" in feats
    # every FOMC snapshot date should show is_FOMC True on that exact row
    fomc_dates = aligned.loc[aligned["event_type"] == "FOMC", "snapshot_date"]
    assert feats.loc[fomc_dates, "is_FOMC"].all()
    # d2n should be 0 exactly on event days
    assert (feats.loc[fomc_dates, "d2n_FOMC"] == 0).all()


def test_nuisance_controls_flag_first_friday_collinearity_note() -> None:
    """NFP is almost always the first Friday -- day-of-week alone
    predicts NFP's dow value in the overwhelming majority of cases,
    confirming the collinearity the module docstring warns about (not
    something to fix, just to be aware of when both appear in the same
    regression). Not literally 100%: real exceptions exist (July 4th
    Independence Day shifts the release to Thursday most years; the
    Oct 2013 and 2025 government shutdowns delayed a handful of
    releases to other weekdays) -- checked directly against the data,
    15 of 354 releases, all explainable, none a data bug."""
    panel = load_iv_panel("csv")
    events = load_all_events()
    aligned = align_to_snapshot(events[events["scheduled"]], panel.index)
    nfp_dates = aligned.loc[aligned["event_type"] == "NFP", "snapshot_date"]
    controls = build_nuisance_controls(panel.index)
    frac_friday = (controls.loc[nfp_dates, "dow"] == 4).mean()
    assert frac_friday > 0.9
