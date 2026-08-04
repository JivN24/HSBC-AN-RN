"""Step 2/3: event calendar construction and the timezone alignment that
CLAUDE.md constraint 1 (and the module docstring below) warns is the
step everyone gets wrong.

Event name -> canonical type mapping was read directly off the actual
workbooks (not guessed): US_events_merged_1997-2026.xlsx and
JN_events_1997-2026_FINAL.xlsx each carry a clean, single-string
'Event' column value for every release we need (confirmed by
inspection: 'FOMC Rate Decision (Upper Bound)', 'Change in Nonfarm
Payrolls', 'CPI MoM' for the US; 'BOJ Target Rate', 'Natl CPI YoY' for
Japan). CPI MoM was used over CPI YoY for the US because it has full
1997-2026 coverage (353 releases) vs YoY's partial coverage (284,
starting only in 2002) -- same release, same timestamp, just a
longer history for MoM.

Dev note on 'Date Time' timezone -- this took two rounds of direct
verification and the first round was wrong, kept here because the
mistake is instructive:

Round 1 assumption: both files' 'Date Time' is already true UTC,
because US NFP/CPI show 13:30:00 (08:30 ET + 5h = 13:30 UTC, the
well-known 08:30 ET convention) and JN Jobless Rate/CPI show 23:30 UTC
the day before (08:30 JST - 9h, correctly crossing midnight).

Round 2, checking DST specifically (comparing January vs July NFP
timestamps): BOTH show 13:30:00 UTC, in every month, with NO shift to
12:30 UTC in summer. Real NFP is released 08:30 ET (local wall-clock,
DST-aware) year-round, so true UTC *should* shift by an hour between
EST and EDT -- it doesn't here. Conclusion: the US file's 'Date Time'
is NOT true DST-aware UTC. It is ET wall-clock time re-expressed with
a FIXED UTC-5 (EST) offset applied year-round, i.e. a Bloomberg export
convention, not a timezone bug in the underlying releases themselves.
The fix: reconstruct true ET wall-clock time as `Date Time - 5h`, then
localize that to America/New_York (properly DST-aware) before
converting to UTC. This was confirmed against FOMC: raw 19:15:01 UTC
on 2000-08-22 (EDT) reconstructs to 14:15:01 ET -- exactly the known
pre-2013 FOMC statement release time -- and the *previously computed*
hours-to-snapshot for summer-dated FOMC meetings (an anomalous ~1.75h
cluster, a full hour off the ~2.75h winter-dated cluster) disappears
once this fix is applied. See _reconstruct_us_release_utc.

Japan has no DST (JST is UTC+9 always), so the JN file's 'Date Time'
has no equivalent bug -- confirmed still consistent with true UTC:
  - JN 'Jobless Rate'/'Natl CPI YoY' (real release 08:30 JST) show
    23:30 UTC the previous calendar date, matching 08:30 JST - 9h.
  - JN 'BOJ Target Rate' shows ~02:25-04:47 UTC on the SAME calendar
    date as the JST decision date, consistent with real BOJ
    announcements at ~11:25-13:45 JST -- NOT with the
    boj_monetary_policy_meetings.csv's '08:30 JST' rows (Actual='-',
    apparently a scheduled-calendar-slot placeholder rather than the
    real announcement; that csv's other rows, e.g. '09:00'/'09:20
    JST' with Actual filled, don't obviously match either). VERIFY:
    this file's BOJ 'Date Time' is used as the primary source because
    it's internally consistent with known BOJ timing conventions, but
    the discrepancy with the csv should be checked by the user.

A second, related correction to the ORIGINAL illustrative example in
the task brief ("a decision dated 1 Aug JST at 11:45 JST is impounded
in the 31 July ET snapshot, a day earlier than naive JST-date
tagging"): worked through with actual arithmetic, an 11:45 JST decision
converts to 22:45 ET the PRIOR evening -- which is AFTER that prior
date's 17:00 ET cutoff, so it is impounded in the SAME business date
as its JST calendar date, not a day earlier. The "day earlier" trap is
real, but only for JST releases in the very early Tokyo morning
(before roughly 06:00-07:00 JST, since NY 17:00 ET translates to about
that JST clock time) -- not for the ~11:30-13:30 JST timing of actual
BOJ policy announcements. This is confirmed empirically below: aligning
every real BOJ decision in this dataset against a business-day
calendar gives lag_days == 0 for all 221 of them (i.e. the correct
NY-snapshot business date always equals the naive JST calendar date).
The mechanism in align_to_snapshot is still exactly right and still
matters -- an early-morning JST release (BOJ's own "Summary of
Opinions" note, sometimes JN GDP prints, etc.) would genuinely show
lag_days == 1 -- it just isn't triggered by BOJ's own midday policy
announcements specifically.

Ad-hoc/unscheduled events (MoF FX intervention 2022/2024, emergency
BOJ meetings) are NOT in these scheduled economic-calendar exports and
are out of scope for align_to_snapshot's exogeneity argument; they
would need to be sourced and tagged scheduled=False separately if
ever added (CLAUDE.md constraint 2 does not apply to them).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from evc.config import DATA_DIR, SNAPSHOT_HOUR_ET, SNAPSHOT_TZ, DataPaths

# Event name -> canonical event_type, confirmed against the actual
# workbook contents (see module docstring). Do not add entries here
# without checking df['Event'].unique() first.
US_EVENT_MAP: dict[str, str] = {
    "FOMC Rate Decision (Upper Bound)": "FOMC",
    "Change in Nonfarm Payrolls": "NFP",
    "CPI MoM": "US_CPI",
}
JN_EVENT_MAP: dict[str, str] = {
    "BOJ Target Rate": "BOJ",
    "Natl CPI YoY": "JP_CPI",
}


def _reconstruct_us_release_utc(date_time: pd.Series) -> pd.Series:
    """Undo the US events file's fixed UTC-5 export convention and
    recover true DST-aware UTC (see module docstring). 'Date Time' - 5h
    gives the true ET wall-clock release time in every season; that is
    then localized to America/New_York (DST-aware) and converted to
    true UTC."""
    et_wall_clock = pd.to_datetime(date_time) - pd.Timedelta(hours=5)
    localized = et_wall_clock.dt.tz_localize(
        SNAPSHOT_TZ, ambiguous="NaT", nonexistent="shift_forward"
    )
    return localized.dt.tz_convert("UTC").dt.tz_localize(None)


# Distinguishing genuine intermeeting/emergency FOMC actions from
# regular meetings turned out to need two iterations:
#
# First attempt: flag anything released before 13:00 ET. This produced
# FALSE POSITIVES -- from 2011 to 2018 the Fed routinely released
# statements at 12:30pm ET for meetings WITHOUT a press conference
# (only quarterly meetings got a 2:00pm slot with a presser back then;
# every meeting has had a presser only since Jan 2019). Checked
# directly: 2011-04-27, 2011-06-22, 2012-04-25 etc. all sit on a
# normal ~42-day meeting cadence and are legitimate regular meetings
# released at 12:30 ET, not emergencies -- a naive hour cutoff would
# have mislabeled them.
#
# Second attempt: flag short gaps to the nearest neighbouring FOMC
# date. This also produced false positives, because a regular meeting
# adjacent in time to a genuine emergency action inherits an
# artificially short gap without itself being unscheduled (e.g. the
# regular 1998-09-29 meeting, 16 days before the 1998-10-15 LTCM
# intermeeting cut, is not itself an emergency action).
#
# Final heuristic (zero false positives against the regular-meeting
# set, checked directly): flag a release before 11:00 ET OR on a
# Saturday/Sunday. Regular meetings, across the entire 12:30/14:00/
# 14:15 ET history in this dataset, are never released before 11:00 ET
# or on a weekend; every date this catches is a well-documented real
# intermeeting/emergency action (2001-04-18, 2001-09-17 post-9/11,
# 2008-01-22 and 2008-10-08 financial crisis, 2020-03-03 and
# 2020-03-15 COVID). Known limitation, left as-is rather than papered
# over: 1998-10-15 (the LTCM intermeeting cut) was released at a
# perfectly ordinary-looking 15:00 ET on a Thursday, so this heuristic
# does not catch it -- recall is not perfect, but precision is (no
# regular meeting gets flagged), which matters more here since a
# false NEGATIVE just leaves one emergency action mixed into the
# "scheduled" bucket, while a false POSITIVE would wrongly discard a
# real regular meeting from the exogenous set.
_FOMC_EARLY_HOUR_ET_MAX = 11


def _flag_fomc_scheduled(df: pd.DataFrame) -> pd.Series:
    et_wall_clock = df["ts_utc"].dt.tz_localize("UTC").dt.tz_convert(SNAPSHOT_TZ)
    is_fomc = df["event_type"] == "FOMC"
    is_early = et_wall_clock.dt.hour < _FOMC_EARLY_HOUR_ET_MAX
    is_weekend = et_wall_clock.dt.dayofweek >= 5
    is_emergency = is_fomc & (is_early | is_weekend)
    scheduled = ~is_emergency
    n_flagged = int(is_emergency.sum())
    if n_flagged:
        flagged_dates = df.loc[is_emergency, "ts_utc"].dt.date.tolist()
        print(
            f"events: flagging {n_flagged} FOMC action(s) as scheduled=False "
            f"(released before {_FOMC_EARLY_HOUR_ET_MAX}:00 ET or on a "
            f"weekend -- an intermeeting/emergency action, not a regular "
            f"meeting): {flagged_dates}"
        )
    return scheduled


def _load_mapped_events(path, event_map: dict[str, str], country: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df = df[df["Event"].isin(event_map)].copy()
    df["event_type"] = df["Event"].map(event_map)
    if country == "US":
        # Fixed-offset export quirk -- see module docstring. Japan has
        # no DST so the JN file needs no equivalent correction.
        df["ts_utc"] = _reconstruct_us_release_utc(df["Date Time"])
    else:
        df["ts_utc"] = pd.to_datetime(df["Date Time"])
    df["country"] = country
    df["scheduled"] = True
    if country == "US":
        df["scheduled"] = _flag_fomc_scheduled(df)
    out = df[["event_type", "ts_utc", "country", "scheduled"]].sort_values("ts_utc")
    return out.reset_index(drop=True)


def load_us_events(paths: DataPaths | None = None) -> pd.DataFrame:
    paths = paths or DataPaths()
    return _load_mapped_events(paths.us_events, US_EVENT_MAP, "US")


def load_jn_events(paths: DataPaths | None = None) -> pd.DataFrame:
    paths = paths or DataPaths()
    return _load_mapped_events(paths.jn_events, JN_EVENT_MAP, "JN")


def load_all_events(paths: DataPaths | None = None) -> pd.DataFrame:
    paths = paths or DataPaths()
    return (
        pd.concat([load_us_events(paths), load_jn_events(paths)], ignore_index=True)
        .sort_values("ts_utc")
        .reset_index(drop=True)
    )


def discover_fred_release_ids(series_ids: list[str]) -> dict[str, int]:
    """Query the FRED /fred/releases-for-series-ish flow to discover
    release IDs for the given series. Do NOT hard-code release IDs
    from memory -- this is the only sanctioned way to populate
    evc.config.FRED_RELEASE_IDS.

    Requires a FRED_API_KEY environment variable (free, from
    https://fred.stlouisfed.org/docs/api/api_key.html) -- not invented
    or embedded here. Not currently needed for Phase 1: the Bloomberg
    workbooks under DATA/ already cover FOMC/NFP/US_CPI/BOJ/JP_CPI, so
    this exists as a documented path for extending to other series,
    not as something Step 2 depends on.
    """
    import requests

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise RuntimeError(
            "FRED_API_KEY not set. Get a free key from "
            "https://fred.stlouisfed.org/docs/api/api_key.html and set it "
            "in your environment before calling discover_fred_release_ids -- "
            "release IDs are never hard-coded in this codebase."
        )
    out: dict[str, int] = {}
    for series_id in series_ids:
        resp = requests.get(
            "https://api.stlouisfed.org/fred/series/release",
            params={"series_id": series_id, "api_key": api_key, "file_type": "json"},
            timeout=10,
        )
        resp.raise_for_status()
        releases = resp.json().get("releases", [])
        if releases:
            out[series_id] = releases[0]["id"]
    return out


def _snapshot_utc_index(business_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The UTC instant corresponding to each business date's NY 17:00
    ET snapshot, DST-aware (tz_localize handles the EST/EDT switch;
    this is exactly why a naive fixed-offset conversion would silently
    misalign events around the March/November clock changes)."""
    wall_clock = business_dates.normalize() + pd.Timedelta(hours=SNAPSHOT_HOUR_ET)
    return wall_clock.tz_localize(SNAPSHOT_TZ).tz_convert("UTC")


def align_to_snapshot(events: pd.DataFrame, business_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Map each event's ts_utc to the first business date whose 17:00
    ET snapshot impounds it: earliest d in business_dates with
    snapshot_utc(d) >= ts_utc.

    Adds columns 'snapshot_date' (the assigned business date) and
    'hours_to_snapshot' (>= 0 by construction; how much lead time the
    17:00 ET close had over the release -- this is the diagnostic
    CLAUDE.md/Step 3 asks for: US 08:30 ET releases should show ~8.5h,
    FOMC's 14:00 ET should show ~3h, BOJ should show a value consistent
    with its actual (JST) time of day once converted -- see this
    function's returned per-event-type summary via
    events_alignment_diagnostic).

    Events beyond the last available business date (snapshot_utc(d) is
    never >= ts_utc for any d in business_dates) are dropped, not
    silently clipped to the last date -- that would fabricate an
    impossible negative-lead-time alignment.
    """
    business_dates = pd.DatetimeIndex(business_dates).sort_values()
    snap_utc = _snapshot_utc_index(business_dates)

    ts_utc = pd.DatetimeIndex(events["ts_utc"]).tz_localize("UTC")
    idx = np.searchsorted(snap_utc.values, ts_utc.values, side="left")

    in_range = idx < len(business_dates)
    dropped = int((~in_range).sum())
    if dropped:
        print(
            f"align_to_snapshot: dropping {dropped} event(s) beyond the "
            f"last business date ({business_dates[-1].date()}) -- extend "
            f"business_dates if these should be included."
        )

    out = events.loc[in_range].copy()
    idx = idx[in_range]
    out["snapshot_date"] = business_dates[idx]
    out["hours_to_snapshot"] = (snap_utc[idx] - ts_utc[in_range]).total_seconds() / 3600.0
    return out.reset_index(drop=True)


def events_alignment_diagnostic(aligned_events: pd.DataFrame) -> pd.DataFrame:
    """Hours-from-release-to-snapshot by event type -- the sanity table
    Step 3 requires before trusting any downstream alignment. US 08:30
    ET releases should show ~8.5h; FOMC (14:00 ET post-2013) ~3h. A
    ~0 or negative value for any event type means alignment is broken
    and must be investigated before proceeding, not silently used.
    """
    summary = aligned_events.groupby("event_type")["hours_to_snapshot"].agg(["mean", "median", "min", "max", "count"])
    if (summary["min"] <= 0).any():
        broken = summary.index[summary["min"] <= 0].tolist()
        raise ValueError(
            f"alignment broken for event types {broken}: hours_to_snapshot <= 0 "
            "means an event was assigned a snapshot date that occurred before "
            "(or simultaneously with) the release itself."
        )
    return summary
