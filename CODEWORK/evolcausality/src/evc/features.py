"""Step 4: vol and event feature construction.

Vol features are computed on log-variance throughout (2*log(iv)), per
CLAUDE.md's core-math section: the paper's event weight omega is an
additive shift in log-variance, not vol levels, so every downstream
regression/estimator should see logvar, not iv.

Event features implement CLAUDE.md constraint 1 directly: d2n_X (days
to the NEXT event) is what lets any estimator see anticipation --
without it, only dsince_X would be available and a Granger-style test
could never detect "IV rises before the event" at all, since nothing
in a dsince-only feature set varies before the event happens.

No look-ahead: build_vol_features uses only .diff()/.rolling() (both
strictly backward-looking by construction). build_event_features uses
only the PRE-ANNOUNCED event calendar (event dates known in advance,
CLAUDE.md constraint 2's exogeneity argument) -- this is legitimate
forward information, not a look-ahead leak, and is fundamentally
different from e.g. a centered rolling window over REALISED vol data
would be. tests/test_no_lookahead.py plants a future spike in a
synthetic vol series and asserts build_vol_features features don't
move before it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evc.config import D2N_CAP_DAYS, INWIN_LONG, INWIN_SHORT, RV_WINDOWS


def build_vol_features(panel: pd.DataFrame) -> pd.DataFrame:
    """logvar_{tenor}, dlogvar_{tenor}, slope_1m_3m, slope_on_1m, curv,
    rv_{5,10,21,63}, ivrv, vov.

    slope_on_1m is called out in the spec as the purest event-premium
    signal: the ON tenor is (constraint 3) almost entirely event risk
    on an event day, while 1M averages that single day's risk over ~21
    days, so their logvar difference isolates the concentrated event
    premium from the diffuse term-structure level.

    curv is a discrete second difference around the 1M point using its
    maturity-adjacent tenors (2W and 2M) -- the standard finite-
    difference curvature/"butterfly" construction: curv > 0 means the
    term structure is convex (short and long tenors both above the 1M
    midpoint), which is the typical shape when a nearby event elevates
    tenors that span it more than the 1M tenor itself.

    vov (paper's nu_i, vol-of-vol) is the rolling std of dlogvar_1m --
    deliberately NOT centered, so it only ever reflects the PAST
    INWIN_LONG days of realised vol-of-vol, never a future one.
    """
    out = pd.DataFrame(index=panel.index)

    iv_cols = [c for c in panel.columns if c.startswith("iv_")]
    for col in iv_cols:
        tenor = col[len("iv_") :]
        out[f"logvar_{tenor}"] = 2 * np.log(panel[col])
        out[f"dlogvar_{tenor}"] = out[f"logvar_{tenor}"].diff()

    if "logvar_1m" in out and "logvar_3m" in out:
        out["slope_1m_3m"] = out["logvar_1m"] - out["logvar_3m"]
    if "logvar_on" in out and "logvar_1m" in out:
        out["slope_on_1m"] = out["logvar_on"] - out["logvar_1m"]
    if {"logvar_2w", "logvar_1m", "logvar_2m"} <= set(out.columns):
        out["curv"] = out["logvar_2w"] + out["logvar_2m"] - 2 * out["logvar_1m"]

    if "spot" in panel.columns:
        log_ret = np.log(panel["spot"]).diff()
        for window in RV_WINDOWS:
            out[f"rv_{window}"] = log_ret.rolling(window, min_periods=max(2, window // 2)).std() * np.sqrt(252)
        if "rv_21" in out and "logvar_1m" in out:
            # ivrv: variance risk premium proxy, log(IV_1M) vs log(RV
            # over a matched ~1-month/21-business-day horizon). Real
            # spot data has occasional exactly-zero-return windows
            # (stale/repeated quotes, mostly in illiquid early
            # history) that make rv_21 degenerately 0 -- log(0) is
            # -inf, not a meaningful variance risk premium, so those
            # rows are NaN rather than -inf.
            rv_21_safe = out["rv_21"].where(out["rv_21"] > 0)
            out["ivrv"] = 0.5 * out["logvar_1m"] - np.log(rv_21_safe)

    if "dlogvar_1m" in out:
        out["vov"] = out["dlogvar_1m"].rolling(INWIN_LONG, min_periods=max(2, INWIN_LONG // 2)).std()

    return out


def _event_type_features(
    event_dates: pd.DatetimeIndex, index: pd.DatetimeIndex, cap_days: int, inwin_short: int, inwin_long: int
) -> pd.DataFrame:
    """Vectorised, position-based (not calendar-day-based, since the
    index is itself a business-day grid and calendar-day arithmetic
    would overcount weekends) event feature block for one event type.
    """
    n = len(index)
    idx_pos = np.arange(n)

    event_positions = np.searchsorted(index.values, np.asarray(sorted(event_dates)))
    event_positions = event_positions[(event_positions >= 0) & (event_positions < n)]
    event_positions = np.unique(event_positions)

    is_event = np.zeros(n, dtype=bool)
    is_event[event_positions] = True

    # days to next event >= t (0 if today is one)
    next_search = np.searchsorted(event_positions, idx_pos, side="left")
    has_next = next_search < len(event_positions)
    next_pos = np.where(has_next, event_positions[np.clip(next_search, 0, len(event_positions) - 1)], -1)
    d2n = np.where(has_next, next_pos - idx_pos, cap_days)
    d2n = np.clip(d2n, 0, cap_days)

    # days since last event <= t (0 if today is one)
    since_search = np.searchsorted(event_positions, idx_pos, side="right") - 1
    has_prev = since_search >= 0
    prev_pos = np.where(has_prev, event_positions[np.clip(since_search, 0, len(event_positions) - 1)], -1)
    dsince = np.where(has_prev, idx_pos - prev_pos, cap_days)
    dsince = np.clip(dsince, 0, cap_days)

    def count_strictly_after_within(window: int) -> np.ndarray:
        # count of event_positions in (i, i+window] -- today (position
        # i) never contributes: its event, if any, is already in this
        # snapshot's contemporaneous is_X, not "remaining" exposure.
        hi = np.searchsorted(event_positions, idx_pos + window, side="right")
        lo = np.searchsorted(event_positions, idx_pos, side="right")
        return hi - lo

    return pd.DataFrame(
        {
            "is": is_event,
            "d2n": d2n,
            "dsince": dsince,
            "pre": d2n <= min(cap_days, 5),
            "post": dsince <= min(cap_days, 5),
            f"inwin{inwin_short}": count_strictly_after_within(inwin_short),
            f"inwin{inwin_long}": count_strictly_after_within(inwin_long),
        },
        index=index,
    )


def build_event_features(events: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """One feature block per event_type in `events` (expects a
    'snapshot_date' column, i.e. the output of evc.events.align_to_snapshot,
    and an 'event_type' column). See module docstring for why d2n_X is
    required, not optional, and why this is not a look-ahead leak.
    """
    index = pd.DatetimeIndex(index)
    out = pd.DataFrame(index=index)
    for event_type, sub in events.groupby("event_type"):
        block = _event_type_features(
            pd.DatetimeIndex(sub["snapshot_date"]), index, D2N_CAP_DAYS, INWIN_SHORT, INWIN_LONG
        )
        block = block.add_suffix(f"_{event_type}")
        out = out.join(block)
    return out


def build_nuisance_controls(index: pd.DatetimeIndex) -> pd.DataFrame:
    """day-of-week, month-end, and the 1M option expiry roll (3rd
    Wednesday of the month) -- confounds that have to be controlled
    for before attributing any vol movement to a specific event type.

    Note (documented, not resolved here): NFP is *always* the first
    Friday of the month, so a "day-of-week == Friday" control and an
    "NFP effect" are not separately identified from day-of-week alone
    without either exploiting NFP's own date jitter (there isn't much)
    or a very long sample; any regression using both should flag this.
    """
    index = pd.DatetimeIndex(index)
    out = pd.DataFrame(index=index)
    out["dow"] = index.dayofweek
    out["is_month_end"] = index.is_month_end.astype(int)

    # 3rd Wednesday of the month: the standard FX 1M option expiry
    # convention.
    wed_of_month = (index.dayofweek == 2) & (index.day <= 21) & (index.day >= 15)
    out["is_1m_expiry_roll"] = wed_of_month.astype(int)
    return out
