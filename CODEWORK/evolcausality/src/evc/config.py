"""All magic numbers for the project live here, and nowhere else.

Every constant is either (a) taken directly from the paper
(docs/SimpleEventVolModel.pdf), (b) measured from the data files under
DATA/, or (c) marked VERIFY because it is a real-world convention (a
release time, a release ID) that changes over history and that the
author must confirm before it is trusted. Nothing here is invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Repo root is the parent of this package's grandparent (src/evc/config.py
# -> src/evc -> src -> evolcausality). DATA/ lives one level above that,
# alongside evolcausality/, CODEWORK/, READINGS/.
REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "DATA"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


@dataclass(frozen=True)
class DataPaths:
    """Locations of the real-data files this project reads.

    These are the files the user pointed at directly. Schemas were
    confirmed by inspection (see evc.data docstrings), not assumed:
      - *_events_*.xlsx: single sheet 'CC_events_YYYY-YYYY', columns
        ['Date Time', 'Country Code', 'Event', 'Period', 'Survey',
         'Actual', 'Prior', 'Revised', 'Relevance', 'Ticker'].
        'Date Time' for JN events is already in a UTC-like release
        timestamp per Bloomberg convention -- this must be verified
        against the BOJ csv below, not assumed (see Step 3 note).
      - ATM_Volatility_-_all_currencies_diff_tenors.xlsx: one sheet per
        tenor ('1D','1W','2W','1M','2M','3M','6M','1Y'; '1D' is the
        overnight/VON tenor), each sheet wide with repeated
        Date/CCYPAIR column pairs per currency pair, row 0 holding the
        Bloomberg ticker string (e.g. 'USDJPYVON BGN Curncy'), values
        in vol POINTS (percent), not decimals.
      - 25R / 25B files: same wide layout, tenors {1M,3M,6M,1Y} for
        risk reversal, {1W,1M,3M,6M} for butterfly. No 2W skew data.
      - boj_monetary_policy_meetings_2015_2026.csv: columns include
        'Date (dd/mm/yyyy)', 'Time (JST)', 'Time (UTC)' -- this file
        gives us the ground truth for JST->UTC alignment; the JN
        events file's 'Date Time' should be cross-checked against it
        rather than trusted blindly.
    """

    us_events: Path = DATA_DIR / "US_events_merged_1997-2026.xlsx"
    jn_events: Path = DATA_DIR / "JN_events_1997-2026_FINAL.xlsx"
    boj_meetings: Path = DATA_DIR / "boj_monetary_policy_meetings_2015_2026.csv"
    atm_vol: Path = DATA_DIR / "ATM_Volatility_-_all_currencies_diff_tenors.xlsx"
    risk_reversal_25d: Path = DATA_DIR / "25R - all currencies diff tenors.xlsx"
    butterfly_25d: Path = DATA_DIR / "25B - all currencies diff tenors.xlsx"
    # Clean Date/USDJPY columns, no header/ticker-row cruft (confirmed
    # by inspection) -- used over the messy multi-header
    # "Currency Daily BidAskLast.xlsx" workbook for spot.
    spot: Path = DATA_DIR / "daily_mid.csv"


# ---------------------------------------------------------------------------
# Tenor structure
# ---------------------------------------------------------------------------

# Canonical schema tenor names -> ATM vol sheet name in the xlsx.
# '1D' in the workbook is the overnight tenor (VON ticker) -- relabelled
# 'on' here because "1 day" is ambiguous (calendar day vs business day)
# while "overnight" is the specific FX-market instrument constraint 3
# in the task spec calls for.
TENOR_SHEET_MAP: dict[str, str] = {
    "on": "1D",
    "1w": "1W",
    "2w": "2W",
    "1m": "1M",
    "2m": "2M",
    "3m": "3M",
    "6m": "6M",
    "1y": "1Y",
}

# Business-day tenor lengths used to build D_T in the variance-clock
# regression. These are approximate market conventions (act/360-style
# FX option tenor ~ calendar days * 5/7), NOT derived from the data --
# VERIFY against actual expiry calendars before treating D_T as exact
# rather than a normalising approximation.
TENOR_CALENDAR_DAYS: dict[str, int] = {
    "on": 1,
    "1w": 7,
    "2w": 14,
    "1m": 30,
    "2m": 60,
    "3m": 90,
    "6m": 180,
    "1y": 365,
}

CCY_PAIR = "USDJPY"

# ---------------------------------------------------------------------------
# Snapshot / timezone convention
# ---------------------------------------------------------------------------

# The canonical daily snapshot this project works on: NY 17:00 ET, the
# FX market's global end-of-day convention (start of the next settlement
# day). All event-to-snapshot alignment (Step 3) maps a release
# timestamp to the first business date whose 17:00 ET snapshot is >= it.
SNAPSHOT_HOUR_ET = 17
SNAPSHOT_TZ = "America/New_York"

# ---------------------------------------------------------------------------
# Event release-time constants -- VERIFY before trusting
# ---------------------------------------------------------------------------

# FOMC statement release time changed from 14:15 ET to 14:00 ET during
# 2013 (fewer post-meeting press conferences skipped -> more consistent
# early timing). VERIFY the exact changeover meeting before using this
# to align pre-2014 FOMC dates.
FOMC_RELEASE_TIME_ET_PRE_2013 = "14:15"
FOMC_RELEASE_TIME_ET_POST_2013 = "14:00"
FOMC_TIME_CHANGE_YEAR = 2013  # VERIFY

# NFP (US employment situation) is always released 08:30 ET, always the
# first Friday of the month (with rare BLS holiday shifts) -- this
# collinearity with day-of-week/first-Friday effects is called out
# explicitly in evc.features.
NFP_RELEASE_TIME_ET = "08:30"

# US CPI: release time has been consistently 08:30 ET historically.
US_CPI_RELEASE_TIME_ET = "08:30"

# BOJ decision announcement time is NOT fixed -- it depends on how long
# the Policy Board meeting runs and has historically varied roughly
# 11:30-13:00 JST. The boj_monetary_policy_meetings csv gives actual
# per-meeting times where available and should be preferred over this
# fallback. VERIFY.
BOJ_RELEASE_TIME_JST_FALLBACK = "11:30-13:00"  # VERIFY, use csv times instead

# ---------------------------------------------------------------------------
# FRED release discovery -- populate only after running the discovery
# helper in evc.events (do NOT hard-code release IDs from memory)
# ---------------------------------------------------------------------------

FRED_RELEASE_IDS: dict[str, int] = {
    # e.g. "CPI": <id>, "NFP": <id>, "FOMC": <id>
    # Populate by running evc.events.discover_fred_release_ids() and
    # inspecting the output -- do not fill this in by guessing.
}

# ---------------------------------------------------------------------------
# Synthetic model parameters (paper eqs 1-2, n=2, Heston-style mean
# reversion -- see evc.synthetic docstring for why plain constant alpha
# is replaced)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticConfig:
    n_steps: int = 3000
    dt: float = 1.0 / 252.0  # one business day, in years
    s0: float = 100.0

    # Heston-style mean reversion for variance: dv = kappa*(theta^2 - v)dt + xi*sqrt(v)*dW
    #
    # kappa=50 (half-life ~3.5 business days) is much faster than a
    # textbook equity-Heston kappa (typically 1-5, half-life months).
    # This is required, not cosmetic: eq (2) applies omega
    # MULTIPLICATIVELY to the entire next state (not just that day's
    # shock), so a run of overlapping event types firing every ~5
    # business days on average (5 event streams with 21-46 day periods)
    # compounds geometrically unless mean reversion pulls the elevated
    # level back down before the next event arrives. kappa=3 was tried
    # first and blew up to v ~ 1e45 over 3000 steps (see evc.synthetic
    # module docstring / dev notes) -- kappa=50 keeps sigma bounded and
    # is also more realistic: FX event-vol premium empirically decays
    # over days, not months.
    kappa: float = 50.0  # speed of mean reversion (annualised)
    theta: float = 0.10  # long-run vol level (so theta^2 is long-run variance)
    xi: float = 0.35  # vol-of-vol
    sigma0: float = 0.10  # initial vol level, set to theta at simulation start

    rho: float = -0.4  # spot/vol correlation (leverage effect), constant here

    # Per-event-type ground-truth weights (paper's omega), KNOWN because
    # this is the synthetic data generating process, used to check
    # recovery in Step 6. These are deliberately spread across a wide
    # range so power/recovery curves are informative at both ends.
    event_omega: dict[str, float] = field(
        default_factory=lambda: {
            "FOMC": 2.10,
            "NFP": 1.85,
            "US_CPI": 1.70,
            "BOJ": 1.55,
            "JP_CPI": 1.15,
        }
    )

    # Approximate real-world periodicities (business days) used to
    # generate a synthetic event calendar with realistic clustering.
    # Small random jitter is added at simulation time -- see Step 1
    # docstring for why a perfectly periodic calendar is a problem.
    event_period_days: dict[str, int] = field(
        default_factory=lambda: {
            "FOMC": 46,  # ~6-7 weeks
            "NFP": 21,  # first Friday of each month, approx monthly
            "US_CPI": 21,  # mid-month, approx monthly
            "BOJ": 46,
            "JP_CPI": 21,
        }
    )
    event_jitter_days: int = 2  # uniform +/- jitter applied to each scheduled date

    seed: int = 0


# ---------------------------------------------------------------------------
# Feature engineering constants
# ---------------------------------------------------------------------------

MAX_FORWARD_FILL_DAYS = 3  # longer gaps are a data problem, not a holiday
D2N_CAP_DAYS = 15  # cap on "days to next event" feature
EVENT_STUDY_WINDOW = 10  # K = +/-10 business days around an event
INWIN_SHORT = 5
INWIN_LONG = 21
RV_WINDOWS = (5, 10, 21, 63)

VOL_PERCENT_THRESHOLD = 1.0  # values above this are assumed to be in
# percentage points (Bloomberg convention), not decimals -- used by the
# auto-rescale check in evc.data
