"""Null-distribution generators shared across the causality modules,
plus the multiple-testing correction (Step 5.7).

Surrogate schemes:
  - circular block bootstrap of a series (block length ~ 2*n^(1/3),
    the standard Politis-White-style rule of thumb): resamples
    contiguous blocks with wraparound, preserving short-range
    autocorrelation while destroying the specific alignment with any
    external series (e.g. the event calendar).
  - placebo/shifted event dates (common random +/-5..60 day offset):
    preserves the calendar's OWN clustering (all events move together)
    and the vol series' own autocorrelation, breaking only the
    alignment between them -- the stronger null CLAUDE.md calls for.
  - circular shift of a full series: degenerates on perfectly periodic
    input (shifting a period-P series by a multiple of P reproduces
    the original alignment) -- this is exactly why evc.synthetic
    jitters the synthetic calendar (see its module docstring) rather
    than using a perfectly periodic one.
  - IAAFT amplitude-adjusted surrogates: preserves both the amplitude
    distribution and (approximately) the power spectrum, so it is a
    strictly stronger null than a plain circular shift for anything
    with autocorrelation structure.

Multiple testing: Benjamini-Hochberg FDR across the full grid of
(event type x lag x direction x method). Both raw and BH-adjusted
p-values are always reported, never only the adjusted ones, so the
reader can see how much of the raw grid was significant before
correction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def block_length_rule_of_thumb(n: int) -> int:
    """2*n^(1/3), rounded to at least 1 -- the standard heuristic scale
    for circular block bootstrap block length."""
    return max(1, round(2 * n ** (1 / 3)))


def circular_block_bootstrap(series: np.ndarray, block_len: int, rng: np.random.Generator) -> np.ndarray:
    """One circular-block-bootstrap resample of `series`, same length,
    built from randomly-chosen contiguous (wraparound) blocks."""
    n = len(series)
    if block_len <= 0:
        raise ValueError("block_len must be positive")
    n_blocks = -(-n // block_len)  # ceil
    starts = rng.integers(0, n, size=n_blocks)
    out = np.empty(n_blocks * block_len, dtype=series.dtype)
    for i, s in enumerate(starts):
        idx = (np.arange(block_len) + s) % n
        out[i * block_len : (i + 1) * block_len] = series[idx]
    return out[:n]


def placebo_event_dates(
    event_dates: pd.DatetimeIndex, min_shift: int, max_shift: int, rng: np.random.Generator
) -> pd.DatetimeIndex:
    """Shift every event date by the SAME random offset (in business
    days, sign random too), drawn once per call. Shifting all dates
    together (rather than independently) is what preserves the
    calendar's own clustering structure -- an independent per-event
    shift would smear out any real clustering and make the null too
    weak, understating significance under the real (clustered)
    alternative.
    """
    magnitude = rng.integers(min_shift, max_shift + 1)
    sign = rng.choice([-1, 1])
    offset = pd.tseries.offsets.BusinessDay(int(magnitude * sign))
    return event_dates + offset


def iaaft_surrogate(series: np.ndarray, rng: np.random.Generator, n_iter: int = 100) -> np.ndarray:
    """Iterative Amplitude-Adjusted Fourier Transform surrogate:
    preserves the empirical amplitude distribution exactly and the
    power spectrum approximately, by alternately (a) rank-matching
    back to the original sorted values and (b) re-imposing the
    original Fourier amplitude spectrum. Strictly stronger than a
    plain circular shift (which preserves everything about the
    series except which day is "day 0") since it also destroys the
    higher-order phase structure that a real event-driven pattern
    could otherwise leak through.
    """
    n = len(series)
    sorted_vals = np.sort(series)
    target_spectrum = np.abs(np.fft.rfft(series))

    # start from a random permutation (a valid, if crude, surrogate)
    current = rng.permutation(series).astype(float)
    for _ in range(n_iter):
        # step 1: impose the target amplitude spectrum, keep phases
        spectrum = np.fft.rfft(current)
        phases = np.angle(spectrum)
        current = np.fft.irfft(target_spectrum * np.exp(1j * phases), n=n)
        # step 2: rank-match back to the true empirical values
        ranks = np.argsort(np.argsort(current))
        current = sorted_vals[ranks]
    return current


def benjamini_hochberg(pvalues: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Standard BH step-up FDR procedure. Returns a DataFrame with the
    original p-values, their BH-adjusted q-values, and a reject flag,
    indexed the same as the input -- both raw and adjusted are always
    kept side by side (see module docstring)."""
    p = pvalues.dropna().sort_values()
    m = len(p)
    ranks = np.arange(1, m + 1)
    q_raw = p.to_numpy() * m / ranks
    # BH adjustment is the running minimum from the largest p-value
    # down (ensures monotonicity of the adjusted q-values)
    q_adj = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_adj = np.clip(q_adj, 0, 1)
    reject = q_adj <= alpha

    out = pd.DataFrame({"pvalue": p.to_numpy(), "qvalue": q_adj, "reject": reject}, index=p.index)
    return out.reindex(pvalues.index)
