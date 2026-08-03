"""Step 2: canonical data loading.

Produces a single DataFrame indexed by business date (tz-naive,
representing the NY 17:00 snapshot -- evc.config.SNAPSHOT_HOUR_ET),
columns: iv_on, iv_1w, iv_2w, iv_1m, iv_2m, iv_3m, iv_6m, iv_1y,
rr25_1m, bf25_1m, spot. Vol stored in decimals (0.0925 not 9.25).

Three modes behind one interface:
  - "synthetic": wraps evc.synthetic output into the canonical schema.
  - "csv": the real Bloomberg export workbooks under DATA/ (named
    "csv" per the original interface spec, even though the actual
    files are .xlsx -- see DataPaths). Tickers confirmed by direct
    inspection: USDJPYVON/V1W/V2W/V1M/V2M/V3M/V6M/V1Y BGN Curncy in
    ATM_Volatility_-_all_currencies_diff_tenors.xlsx (sheet '1D' is
    the overnight/VON tenor); USDJPY25R1M BGN Curncy in the 25R
    workbook; USDJPY25B1M BGN Curncy in the 25B workbook; USDJPY spot
    from daily_mid.csv. Example BQL pull for when on a terminal:

        import bql
        bq = bql.Service()
        tickers = ["USDJPYVON Curncy", "USDJPYV1W Curncy", "USDJPYV2W Curncy",
                   "USDJPYV1M Curncy", "USDJPYV2M Curncy", "USDJPYV3M Curncy",
                   "USDJPYV6M Curncy", "USDJPYV1Y Curncy",
                   "USDJPY25R1M Curncy", "USDJPY25B1M Curncy", "USDJPY Curncy"]
        req = bql.Request(tickers, {"px": bq.data.px_last(dates=bq.func.range("1997-01-01", "2026-07-31"))})
        res = bq.execute(req)

  - "proxy": realised vol computed from spot only. Prints a loud
    warning: realised vol is backward-looking, so the anticipation
    structure central to CLAUDE.md constraint 1 is structurally
    absent -- any causality result in this mode is a pipeline
    smoke-test, not a finding.

Rescaling to decimals: the ATM/25R/25B workbooks store vol in
PERCENTAGE POINTS (confirmed: USDJPY 1M ATM vol values are ~7-25,
clearly points not decimals). A blind "divide by 100 if median > 1"
auto-detector works fine for ATM levels (always >> 1) but is UNSAFE
for the skew (25R/25B) quotes, which are often already < 1 point in
magnitude (e.g. -0.6, 0.85) -- auto-detection would wrongly treat
those as "already decimal". Skew quotes use the SAME Bloomberg vol-
point convention as ATM (confirmed: same 'BGN Curncy' field family),
so they are rescaled with an explicit flag instead of relying on
auto-detection. See rescale_to_decimal.

Forward-fill gaps of at most evc.config.MAX_FORWARD_FILL_DAYS; a
longer gap is a data problem, not a holiday, and raises rather than
being silently filled (filling it fabricates zero vol-changes that
would bias every autocorrelation/MI estimate downward).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from evc.config import (
    CCY_PAIR,
    MAX_FORWARD_FILL_DAYS,
    TENOR_SHEET_MAP,
    VOL_PERCENT_THRESHOLD,
    DataPaths,
)

Mode = Literal["synthetic", "csv", "proxy"]

CANONICAL_IV_COLUMNS = [f"iv_{tenor}" for tenor in TENOR_SHEET_MAP]


def rescale_to_decimal(series: pd.Series, assume_percent: bool | None = None) -> pd.Series:
    """Convert a vol/skew quote series from percentage points to
    decimals.

    If assume_percent is None, auto-detect via median(abs(x)) >
    evc.config.VOL_PERCENT_THRESHOLD. This works for ATM vol levels
    (always well above 1 in point terms) but is unreliable for skew
    quotes that can be < 1 point in magnitude -- callers handling
    25R/25B data should pass assume_percent=True explicitly rather
    than rely on auto-detection (see module docstring).
    """
    if assume_percent is None:
        assume_percent = series.abs().median() > VOL_PERCENT_THRESHOLD
    return series / 100.0 if assume_percent else series


def _read_ccy_pair_sheet(path, sheet: str, ccy_pair: str) -> pd.Series:
    """Read one (Date, ccy_pair) column pair out of the wide multi-
    currency workbook layout shared by ATM_Volatility/25R/25B: each
    currency has its own Date column immediately to its left, row 0
    holds the Bloomberg ticker string, and real data starts at row 2
    (row 1 is blank)."""
    df = pd.read_excel(path, sheet_name=sheet, header=0)
    cols = df.columns.tolist()
    idx = cols.index(ccy_pair)
    date_col = cols[idx - 1]
    sub = df[[date_col, ccy_pair]].copy()
    sub.columns = ["date", "value"]
    sub = sub.iloc[2:].dropna(subset=["date"])
    sub["date"] = pd.to_datetime(sub["date"])
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    return sub.set_index("date")["value"].sort_index()


def _load_csv_mode(paths: DataPaths) -> pd.DataFrame:
    columns = {}
    for tenor, sheet in TENOR_SHEET_MAP.items():
        series = _read_ccy_pair_sheet(paths.atm_vol, sheet, CCY_PAIR)
        # A literal 0% implied vol is not economically meaningful --
        # it's a bad/placeholder print, not a real quote (confirmed:
        # iv_2w has 12 exact zeros, concentrated in its sparsest early
        # history, vs 0 for every other tenor). Treat as missing so it
        # doesn't propagate as -inf through log(iv) downstream.
        n_zero = int((series == 0).sum())
        if n_zero:
            print(f"evc.data: iv_{tenor} has {n_zero} exact-zero quote(s), treated as missing (not a real 0% vol)")
        series = series.replace(0.0, np.nan)
        columns[f"iv_{tenor}"] = rescale_to_decimal(series, assume_percent=True)

    rr = _read_ccy_pair_sheet(paths.risk_reversal_25d, "1M", CCY_PAIR)
    columns["rr25_1m"] = rescale_to_decimal(rr, assume_percent=True)
    bf = _read_ccy_pair_sheet(paths.butterfly_25d, "1M", CCY_PAIR)
    columns["bf25_1m"] = rescale_to_decimal(bf, assume_percent=True)

    spot_df = pd.read_csv(paths.spot, usecols=["Date", CCY_PAIR])
    spot_df["Date"] = pd.to_datetime(spot_df["Date"])
    columns["spot"] = spot_df.set_index("Date")[CCY_PAIR].sort_index()

    panel = pd.DataFrame(columns)
    return panel.sort_index()


def _load_proxy_mode(paths: DataPaths) -> pd.DataFrame:
    print(
        "evc.data: PROXY MODE -- filling iv_* with backward-looking realised "
        "vol from spot returns. The anticipation structure in CLAUDE.md "
        "constraint 1 is structurally absent from realised vol; any "
        "causality test run on this panel is a pipeline smoke-test, not a "
        "finding. Do not report proxy-mode causality results as evidence."
    )
    spot_df = pd.read_csv(paths.spot, usecols=["Date", CCY_PAIR])
    spot_df["Date"] = pd.to_datetime(spot_df["Date"])
    spot = spot_df.set_index("Date")[CCY_PAIR].sort_index()
    log_ret = np.log(spot).diff()

    # rough trading-day tenor lengths for the realised-vol window
    tenor_days = {"on": 1, "1w": 5, "2w": 10, "1m": 21, "2m": 42, "3m": 63, "6m": 126, "1y": 252}
    columns = {}
    for tenor, window in tenor_days.items():
        if window == 1:
            # a rolling std needs >= 2 points; single-day "overnight"
            # realised vol proxy is just that day's absolute return.
            columns[f"iv_{tenor}"] = log_ret.abs() * np.sqrt(252)
        else:
            columns[f"iv_{tenor}"] = log_ret.rolling(window, min_periods=max(2, window // 2)).std() * np.sqrt(252)
    columns["rr25_1m"] = np.nan
    columns["bf25_1m"] = np.nan
    columns["spot"] = spot
    return pd.DataFrame(columns).sort_index()


def _forward_fill_limited(panel: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill gaps of at most MAX_FORWARD_FILL_DAYS, then report
    any that remain -- but only within each column's own [first valid,
    last valid] range. Different instruments in this panel start (and
    sometimes end) on different dates by nature of the source data
    (e.g. rr25_1m/bf25_1m only exist from Oct 2003, spot ends a few
    weeks before the vol sheets in the latest export) -- counting those
    leading/trailing NaNs as "gaps" would flag thousands of dates that
    are not gaps at all, drowning out the genuine interior gaps this
    check exists to catch.
    """
    filled = panel.ffill(limit=MAX_FORWARD_FILL_DAYS)
    for col in filled.columns:
        series = filled[col]
        fvi, lvi = series.first_valid_index(), series.last_valid_index()
        if fvi is None:
            continue
        interior = series.loc[fvi:lvi]
        n_missing = int(interior.isna().sum())
        if n_missing:
            print(
                f"evc.data: {col} has {n_missing} interior date(s) still "
                f"missing after a {MAX_FORWARD_FILL_DAYS}-day forward-fill "
                f"(within its own active range {fvi.date()}..{lvi.date()}) -- "
                "likely a gap longer than a normal holiday; inspect before "
                "using this column for anything beyond a warned pipeline run."
            )
    return filled


def load_iv_panel(mode: Mode, paths: DataPaths | None = None) -> pd.DataFrame:
    """Load and return the canonical IV/spot panel for the given mode."""
    paths = paths or DataPaths()
    if mode == "csv":
        panel = _load_csv_mode(paths)
    elif mode == "proxy":
        panel = _load_proxy_mode(paths)
    elif mode == "synthetic":
        from evc.config import SyntheticConfig, TENOR_CALENDAR_DAYS
        from evc.synthetic import simulate_paths, synthesize_iv_surface

        cfg = SyntheticConfig()
        rng = np.random.default_rng(cfg.seed)
        paths_df = simulate_paths(cfg, rng)
        iv = synthesize_iv_surface(paths_df, cfg, TENOR_CALENDAR_DAYS)
        panel = iv.copy()
        panel["rr25_1m"] = np.nan
        panel["bf25_1m"] = np.nan
        panel["spot"] = paths_df["S"]
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    return _forward_fill_limited(panel)
