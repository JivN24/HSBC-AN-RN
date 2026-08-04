"""Engineering requirement (CLAUDE.md): no look-ahead, ever. Plants a
future spike in a synthetic vol panel and asserts build_vol_features
does not move before it -- this is the test that would catch e.g. an
accidental centered rolling window or a .shift(-1) typo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evc.features import build_event_features, build_vol_features


def _flat_panel(n: int = 200) -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=n)
    iv_1m = np.full(n, 0.10)
    spot = np.full(n, 100.0)
    return pd.DataFrame({"iv_1m": iv_1m, "iv_3m": iv_1m, "spot": spot}, index=index)


def test_vol_features_do_not_move_before_a_future_spike() -> None:
    panel = _flat_panel()
    spike_pos = 150
    panel.iloc[spike_pos, panel.columns.get_loc("iv_1m")] = 0.50

    feats = build_vol_features(panel)

    before = feats.iloc[: spike_pos - 1]  # strictly before the spike row
    # nothing derived from iv_1m should differ from the flat-panel
    # baseline before the spike happens
    baseline = build_vol_features(_flat_panel())
    pd.testing.assert_frame_equal(before[["logvar_1m", "dlogvar_1m"]], baseline.iloc[: spike_pos - 1][["logvar_1m", "dlogvar_1m"]])


def test_vol_features_react_only_at_or_after_the_spike() -> None:
    panel = _flat_panel()
    spike_pos = 150
    panel.iloc[spike_pos, panel.columns.get_loc("iv_1m")] = 0.50
    feats = build_vol_features(panel)

    # the spike itself should show up in logvar_1m at spike_pos, and in
    # dlogvar_1m (the day-over-day change) at spike_pos -- not earlier.
    assert feats["logvar_1m"].iloc[spike_pos] > feats["logvar_1m"].iloc[spike_pos - 1]
    assert feats["dlogvar_1m"].iloc[spike_pos] > 0
    assert feats["dlogvar_1m"].iloc[spike_pos - 1] == 0


def test_event_features_do_not_encode_unannounced_future_events() -> None:
    """The event calendar is legitimately known in advance (constraint
    2), but only events that are ACTUALLY in the `events` table -- an
    index date past the last known event must not silently invent
    d2n/inwin values from nothing (they should saturate at the cap,
    not from any real future information the loader didn't provide)."""
    index = pd.bdate_range("2020-01-01", periods=50)
    events = pd.DataFrame(
        {
            "event_type": ["FOMC"],
            "snapshot_date": [index[10]],
        }
    )
    feats = build_event_features(events, index)
    # after the only known event, d2n saturates at the cap rather than
    # referencing any event beyond what was actually supplied
    assert (feats["d2n_FOMC"].iloc[11:] == feats["d2n_FOMC"].iloc[-1]).all()
    assert feats["is_FOMC"].sum() == 1
    assert feats["is_FOMC"].iloc[10]
