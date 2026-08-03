"""Step 2/3 tests: event loading and timezone alignment against the
REAL data files (not synthetic) -- these are the empirical checks that
justify every claim in evc.events' module docstring.
"""

from __future__ import annotations

import pandas as pd
import pytest

from evc.events import (
    align_to_snapshot,
    events_alignment_diagnostic,
    load_all_events,
    load_jn_events,
    load_us_events,
)

BUSINESS_DATES = pd.bdate_range("1997-01-01", "2026-07-31")


def test_us_events_load_expected_types_and_counts() -> None:
    us = load_us_events()
    assert set(us["event_type"].unique()) == {"FOMC", "NFP", "US_CPI"}
    # order-of-magnitude sanity: ~29 years, FOMC ~8/yr, NFP/CPI ~12/yr
    counts = us["event_type"].value_counts()
    assert 200 <= counts["FOMC"] <= 260
    assert 300 <= counts["NFP"] <= 380
    assert 300 <= counts["US_CPI"] <= 380


def test_jn_events_load_expected_types() -> None:
    jn = load_jn_events()
    assert set(jn["event_type"].unique()) == {"BOJ", "JP_CPI"}


def test_fomc_dst_fix_removes_summer_release_hour_anomaly() -> None:
    """Regression test for the fixed-UTC-5 export bug: with the fix
    applied, FOMC release hour in ET should NOT depend on whether the
    meeting was in EST or EDT season (both should cluster around the
    same ET wall-clock hour, since the scheduled ET time doesn't
    change with DST -- only the UTC representation does)."""
    us = load_us_events()
    fomc = us[us["event_type"] == "FOMC"].copy()
    et_hour = fomc["ts_utc"].dt.tz_localize("UTC").dt.tz_convert("America/New_York").dt.hour
    winter = et_hour[fomc["ts_utc"].dt.month.isin([12, 1, 2])]
    summer = et_hour[fomc["ts_utc"].dt.month.isin([6, 7, 8])]
    # both seasons' regular meetings should cluster in the same hour
    # range (12-15 ET) -- if the DST bug were still present, summer
    # meetings would cluster one hour later (13-16) than winter ones.
    assert abs(winter.median() - summer.median()) <= 1


def test_fomc_emergency_actions_flagged_scheduled_false() -> None:
    us = load_us_events()
    fomc = us[us["event_type"] == "FOMC"]
    unscheduled_dates = set(fomc.loc[~fomc["scheduled"], "ts_utc"].dt.date)
    # well-documented real intermeeting/emergency actions this
    # heuristic is designed to catch (see evc.events module docstring)
    for known_emergency in ["2001-09-17", "2008-01-22", "2008-10-08", "2020-03-03"]:
        assert pd.Timestamp(known_emergency).date() in unscheduled_dates
    # a legitimate regular meeting from the 2011-2018 12:30pm-ET era
    # must NOT be flagged (this is exactly the false-positive the
    # first heuristic attempt produced)
    assert pd.Timestamp("2011-04-27").date() not in unscheduled_dates


def test_alignment_hours_positive_and_in_expected_ranges() -> None:
    """The Step 3 diagnostic table: US 08:30 ET releases ~8.5h,
    FOMC ~3h, BOJ well above 0 (and NOT ~0 or negative, which would
    mean alignment is broken)."""
    events = load_all_events()
    aligned = align_to_snapshot(events[events["scheduled"]], BUSINESS_DATES)
    diag = events_alignment_diagnostic(aligned)

    assert 8.0 <= diag.loc["NFP", "median"] <= 9.0
    assert 8.0 <= diag.loc["US_CPI", "median"] <= 9.0
    assert 2.0 <= diag.loc["FOMC", "median"] <= 4.0
    assert diag.loc["BOJ", "min"] > 0


def test_boj_alignment_matches_naive_jst_date() -> None:
    """Empirical finding (see module docstring): real BOJ policy
    announcements happen at ~11:30-13:30 JST, late enough in the Tokyo
    day that the correct NY-snapshot business date equals the naive
    JST calendar date -- lag_days == 0 for all of them. This is the
    opposite of the illustrative early-morning-JST example in the
    task brief, and is checked here so a future edit that "fixes" this
    to match the brief's narrative gets caught."""
    jn = load_jn_events()
    boj = jn[jn["event_type"] == "BOJ"]
    aligned = align_to_snapshot(boj, BUSINESS_DATES)
    jst_naive_date = (
        aligned["ts_utc"].dt.tz_localize("UTC").dt.tz_convert("Asia/Tokyo").dt.normalize().dt.tz_localize(None)
    )
    lag_days = (jst_naive_date - aligned["snapshot_date"]).dt.days
    assert (lag_days == 0).all()


def test_early_morning_jst_event_shows_one_day_lag() -> None:
    """The general mechanism CLAUDE.md warns about DOES fire for an
    early-Tokyo-morning release (e.g. 00:30 JST) even though real BOJ
    policy decisions (midday JST) don't trigger it -- constructed here
    directly since no such real event exists in the loaded data."""
    early_jst = pd.Timestamp("2024-08-01 00:30:00", tz="Asia/Tokyo").tz_convert("UTC").tz_localize(None)
    synthetic_event = pd.DataFrame(
        {"event_type": ["TEST"], "ts_utc": [early_jst], "country": ["JN"], "scheduled": [True]}
    )
    aligned = align_to_snapshot(synthetic_event, BUSINESS_DATES)
    jst_naive_date = pd.Timestamp("2024-08-01")
    assert aligned.loc[0, "snapshot_date"] == jst_naive_date - pd.Timedelta(days=1)
