"""Step 5.1: event study, K=+/-10 business days (evc.config.EVENT_STUDY_WINDOW).

What this tests: does log-variance move abnormally around event dates,
relative to a null that has no genuine event-causality? Two nulls are
computed:
  - circular block bootstrap of the vol series itself;
  - placebo event dates: shift ALL dates by one common random offset,
    preserving the vol series' own autocorrelation AND the calendar's
    own clustering, breaking only the alignment between them. This is
    the stronger null (CLAUDE.md) and is the one reported as primary.

What would falsify the event-causality hypothesis: no daily/cumulative
abnormal log-variance response distinguishable from the placebo-null
band at any lag in the window.

Diagnostic shapes (documented so a misread is caught, not just a
statistically-insignificant result): a hump building before lag 0 that
collapses after = anticipation + crush (expected, correct, matches
evc.synthetic's verified mechanism); a symmetric spike centred exactly
at lag 0 = likely looking at realised vol (proxy mode) or a broken
timezone alignment (evc.events); a permanent step with no reversion =
a regime confound, not an event effect.

Known limitation, checked directly on the isolated-FOMC synthetic
series (which HAS a verified anticipation plateau, see evc.synthetic):
this event study's baseline is the series' own unconditional mean,
which -- in a calendar dense enough that a meaningful fraction of all
days sit inside SOME event's forward window -- is itself pulled
upward by exactly the effect being measured. Result: the spike-and-
crush AT and AFTER realisation comes through clearly against both
nulls, but the PRE-event anticipation plateau (which is real, verified
separately) does not clear the null bands here. This is exactly why
CLAUDE.md treats the variance-clock regression (evc.causality.varclock)
as the centrepiece and this event study as supporting evidence, not
the primary tool for detecting anticipation specifically -- the
variance-clock regression's D_T-normalised construction does not have
this baseline-contamination problem.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evc.causality.surrogates import block_length_rule_of_thumb, circular_block_bootstrap, placebo_event_dates


def _abnormal_response(logvar: pd.Series, event_dates: pd.DatetimeIndex, window: int) -> np.ndarray:
    """Mean logvar in a +/-window band around each event date, relative
    to the series' own unconditional mean (the "abnormal" part), at
    each relative lag -window..window. Returns an array of length
    2*window+1."""
    idx_pos = {date: i for i, date in enumerate(logvar.index)}
    values = logvar.to_numpy()
    n = len(values)
    baseline = np.nanmean(values)

    lags = np.arange(-window, window + 1)
    response = np.full(len(lags), np.nan)
    valid_events = [idx_pos[d] for d in event_dates if d in idx_pos]
    for j, lag in enumerate(lags):
        positions = [p + lag for p in valid_events if 0 <= p + lag < n]
        if positions:
            response[j] = np.nanmean(values[positions]) - baseline
    return response


def event_study(
    logvar: pd.Series,
    event_dates: pd.DatetimeIndex,
    window: int,
    n_surrogates: int = 500,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Abnormal-log-variance response by relative lag, with both
    surrogate-null bands (2.5/97.5 and 0.5/99.5 percentiles, i.e. 95%
    and 99% bands).

    Returns a DataFrame indexed by relative lag (-window..window) with
    columns: response (observed), and for each null
    ('bootstrap','placebo'): {null}_lo95, {null}_hi95, {null}_lo99,
    {null}_hi99.
    """
    rng = rng or np.random.default_rng(0)
    observed = _abnormal_response(logvar, event_dates, window)

    values = logvar.to_numpy()
    block_len = block_length_rule_of_thumb(len(values))
    n_events = len(event_dates)

    bootstrap_draws = np.full((n_surrogates, 2 * window + 1), np.nan)
    for b in range(n_surrogates):
        resampled = circular_block_bootstrap(values, block_len, rng)
        resampled_series = pd.Series(resampled, index=logvar.index)
        bootstrap_draws[b] = _abnormal_response(resampled_series, event_dates, window)

    placebo_draws = np.full((n_surrogates, 2 * window + 1), np.nan)
    for b in range(n_surrogates):
        shifted = placebo_event_dates(event_dates, 5, 60, rng)
        placebo_draws[b] = _abnormal_response(logvar, shifted, window)

    lags = np.arange(-window, window + 1)
    out = pd.DataFrame({"response": observed}, index=lags)
    out.index.name = "lag"
    for name, draws in [("bootstrap", bootstrap_draws), ("placebo", placebo_draws)]:
        out[f"{name}_lo95"] = np.nanpercentile(draws, 2.5, axis=0)
        out[f"{name}_hi95"] = np.nanpercentile(draws, 97.5, axis=0)
        out[f"{name}_lo99"] = np.nanpercentile(draws, 0.5, axis=0)
        out[f"{name}_hi99"] = np.nanpercentile(draws, 99.5, axis=0)
    return out


def cumulative_response(event_study_result: pd.DataFrame) -> pd.Series:
    """Cumulative sum of the observed abnormal response from -window
    to each lag -- the standard event-study CAR-style summary."""
    return event_study_result["response"].cumsum()
