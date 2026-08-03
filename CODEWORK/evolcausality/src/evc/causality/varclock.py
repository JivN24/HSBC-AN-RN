"""Step 5.2: the variance-clock regression -- the centrepiece analysis
(CLAUDE.md constraint 2). Granger/TDMI/TE are supporting evidence; this
is the one that estimates the paper's omega directly from an identity,
not a correlational test.

Identity (CLAUDE.md core math), for tenor T with D_T business days to
expiry:

    IV^2(t,T) * D_T = sigma_b^2(t) * D_T + sigma_b^2(t) * sum_e (omega_e - 1) * n_e(t,T)

n_e(t,T) = count of type-e events STRICTLY AFTER t within (t, t+D_T] --
today's own event, if any, is already priced into today's snapshot,
not part of "remaining" variance to be earned over the tenor.

Two identification problems, both handled explicitly:

  (a) sigma_b^2(t) is latent and time-varying. Rather than burn a
      parameter per day on fixed effects, normalise both sides by
      IV_1Y^2(t) (the longest tenor, least sensitive to any single
      event) as a proxy for sigma_b^2(t):

          IV^2(t,T)/IV_1Y^2(t) - 1 = sum_e (omega_e - 1) * n_e(t,T)/D_T

      This is a clean linear-in-event-counts regression with the
      normalisation folded directly into y and X -- no separate
      division by an estimated sigma_b^2 is needed afterwards.

  (b) cross-tenor collinearity of event counts: a 3M window (D_T~63
      business days) contains ~3x the events of a 1M window (D_T~21),
      so without short tenors (ON/1W/2W) the individual omega_e are
      only weakly identified. Stacking ALL tenors from ON up (not just
      1M+) is what breaks this, per CLAUDE.md constraint 3.

What would falsify the hypothesis: omega_hat_e indistinguishable from
1 (bootstrap CI spanning 1) for a given event type, after the above
identification fixes are in place.

Known bias, found by checking recovery against evc.synthetic's known
ground truth (n_steps=6000, all 5 event types active): omega_hat comes
out SYSTEMATICALLY HIGH relative to omega_true for every type (e.g.
FOMC 2.26 vs true 2.10, US_CPI 2.13 vs true 1.70), though the ranking
across types is broadly preserved. Root cause, confirmed directly:
the true DGP combines simultaneous event types MULTIPLICATIVELY
(omega_A * omega_B on a day both fire), but this regression's additive
n_e(t,T) counts implicitly assume the two events contribute
(omega_A-1)+(omega_B-1) -- missing the cross term
(omega_A-1)(omega_B-1). Checked: ~3% of event days in the synthetic
calendar have 2+ coincident event types, enough to bias every
coefficient upward via the shared variance on those days getting
attributed to each type's own count feature. Real event calendars
(FOMC/NFP/CPI/BOJ rarely land on the exact same day) should show this
bias to a much smaller degree, but it is a genuine limitation of the
additive specification, not a bug -- evc.validate's recovery check
(Step 6) quantifies it explicitly rather than hiding it, and a
future refinement could add explicit pairwise interaction terms for
event-type pairs with non-trivial same-day co-occurrence rates.

Also implements the naive same-day estimator (regress dlogvar_on for
event vs non-event days) for comparison, and states explicitly why it
is biased downward: one event day is a small fraction of a longer-
tenor's variance swath, and (CLAUDE.md constraint 1) much of that
variance was already priced in via anticipation before the event day
itself, so a same-day-only comparison understates the true omega.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASELINE_TENOR = "1y"


def _business_days_to_expiry(tenor_calendar_days: dict[str, int]) -> dict[str, int]:
    """Same 5/7 calendar-to-business-day scaling used in evc.synthetic
    -- kept local here (not imported) since this module must also work
    on the REAL data's tenor set, which may not always match the
    synthetic config's dict exactly."""
    return {tenor: max(1, round(days * 5 / 7)) for tenor, days in tenor_calendar_days.items()}


def _count_events_after_within(event_positions: np.ndarray, idx_pos: np.ndarray, window: int) -> np.ndarray:
    """Count of event_positions in (i, i+window] for each i in idx_pos
    -- identical convention to evc.features' inwinK (today's own event
    doesn't count as "remaining" exposure)."""
    hi = np.searchsorted(event_positions, idx_pos + window, side="right")
    lo = np.searchsorted(event_positions, idx_pos, side="right")
    return hi - lo


def build_regression_panel(
    iv_panel: pd.DataFrame,
    events: pd.DataFrame,
    tenor_calendar_days: dict[str, int],
    event_types: list[str] | None = None,
) -> pd.DataFrame:
    """Stack (t, tenor) rows into the long-format panel the variance-
    clock regression runs on: columns y (normalised excess variance)
    and one n_e column per event type (event count, NOT yet divided by
    D_T -- that division happens per-tenor inside `variance_clock_regression`
    so each row's D_T is used correctly), plus 'tenor' and 'D_T'.
    """
    business_days = _business_days_to_expiry(tenor_calendar_days)
    index = iv_panel.index
    idx_pos = np.arange(len(index))

    if event_types is None:
        event_types = sorted(events["event_type"].unique())

    event_positions = {}
    for et in event_types:
        dates = pd.DatetimeIndex(events.loc[events["event_type"] == et, "snapshot_date"])
        positions = np.searchsorted(index.values, np.sort(dates.values))
        positions = positions[(positions >= 0) & (positions < len(index))]
        event_positions[et] = np.unique(positions)

    baseline_col = f"iv_{BASELINE_TENOR}"
    if baseline_col not in iv_panel.columns:
        raise ValueError(f"iv_panel must contain {baseline_col} to use as the sigma_b^2 proxy")
    baseline_var = iv_panel[baseline_col] ** 2

    rows = []
    for tenor, d_t in business_days.items():
        if tenor == BASELINE_TENOR:
            continue
        col = f"iv_{tenor}"
        if col not in iv_panel.columns:
            continue
        y = (iv_panel[col] ** 2) / baseline_var - 1.0
        block = pd.DataFrame({"y": y.to_numpy(), "tenor": tenor, "D_T": d_t}, index=index)
        for et in event_types:
            counts = _count_events_after_within(event_positions[et], idx_pos, d_t)
            block[f"n_{et}"] = counts
        rows.append(block)

    panel = pd.concat(rows).dropna(subset=["y"])
    return panel


def variance_clock_regression(
    panel: pd.DataFrame,
    event_types: list[str] | None = None,
    n_bootstrap: int = 500,
    rng: np.random.Generator | None = None,
    enforce_omega_ge_1: bool = False,
) -> pd.DataFrame:
    """Fit y = sum_e beta_e * (n_e/D_T) + eps (no intercept -- CLAUDE.md's
    identity has none once normalised) via OLS (or non-negative least
    squares if enforce_omega_ge_1), and report omega_hat_e = 1 + beta_hat_e
    with a block-bootstrap CI over the time dimension.

    Returns a DataFrame indexed by event_type with columns
    omega_hat, ci_lo, ci_hi, beta_hat.
    """
    rng = rng or np.random.default_rng(0)
    if event_types is None:
        event_types = sorted(c[2:] for c in panel.columns if c.startswith("n_"))

    x_cols = [f"n_{et}" for et in event_types]
    X = (panel[x_cols].to_numpy() / panel["D_T"].to_numpy()[:, None]).astype(float)
    y = panel["y"].to_numpy()

    beta_hat = _fit(X, y, enforce_omega_ge_1)

    # block bootstrap over TIME (index level), refitting each draw --
    # blocks span all tenor rows for a given time index, preserving
    # cross-tenor correlation within a bootstrap draw.
    time_index = panel.index.unique()
    n_time = len(time_index)
    block_len = max(1, round(2 * n_time ** (1 / 3)))
    boot_betas = np.full((n_bootstrap, len(event_types)), np.nan)
    for b in range(n_bootstrap):
        n_blocks = -(-n_time // block_len)
        starts = rng.integers(0, n_time, size=n_blocks)
        sampled_times = np.concatenate(
            [time_index[(np.arange(block_len) + s) % n_time] for s in starts]
        )[:n_time]
        sample = panel.loc[panel.index.isin(sampled_times)]
        if sample.empty:
            continue
        Xb = (sample[x_cols].to_numpy() / sample["D_T"].to_numpy()[:, None]).astype(float)
        yb = sample["y"].to_numpy()
        boot_betas[b] = _fit(Xb, yb, enforce_omega_ge_1)

    ci_lo = np.nanpercentile(boot_betas, 2.5, axis=0)
    ci_hi = np.nanpercentile(boot_betas, 97.5, axis=0)

    return pd.DataFrame(
        {
            "beta_hat": beta_hat,
            "omega_hat": 1 + beta_hat,
            "ci_lo": 1 + ci_lo,
            "ci_hi": 1 + ci_hi,
        },
        index=pd.Index(event_types, name="event_type"),
    )


def _fit(X: np.ndarray, y: np.ndarray, enforce_omega_ge_1: bool) -> np.ndarray:
    if enforce_omega_ge_1:
        from scipy.optimize import lsq_linear

        result = lsq_linear(X, y, bounds=(0, np.inf))
        return result.x
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def naive_same_day_estimator(panel: pd.DataFrame, events: pd.DataFrame, dlogvar: pd.Series) -> pd.DataFrame:
    """E[dlogvar | event day] vs E[dlogvar | non-event day] for each
    event type, converted to a same-day omega_naive = exp(difference).
    Documented as biased DOWNWARD relative to the variance-clock
    estimate (see module docstring): a single day's realised move
    understates the full event premium, most of which was already
    priced in via anticipation before the event day itself.
    """
    out = {}
    for et, sub in events.groupby("event_type"):
        event_days = pd.DatetimeIndex(sub["snapshot_date"])
        is_event = dlogvar.index.isin(event_days)
        mean_event = dlogvar[is_event].mean()
        mean_other = dlogvar[~is_event].mean()
        out[et] = np.exp(mean_event - mean_other)
    return pd.DataFrame({"omega_naive": out})
